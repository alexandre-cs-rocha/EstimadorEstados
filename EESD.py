import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as scsp
from scipy.sparse.linalg import inv

import time

from dss import DSS as dss_engine
from Jacobiana import Jacobiana
from Residuos import Residuo

class EESD():
    def __init__(self, master_path, baseva: float = 1e8, verbose: bool = False) -> None:
        self.baseva = baseva
        self.MasterFile = master_path
        self.verbose = verbose
        self.DSSCircuit, self.DSSText, self.DSSObj, self.DSSMonitors = self.InitializeDSS()
        
        self.resolve_fluxo_carga()
        if self.verbose: print('Acabou o fluxo de carga')
        
        self.barras, self.num_medidas = self.medidas(self.baseva)
        self.vet_estados = self.iniciar_vet_estados()
        
        Ybus = scsp.csr_matrix(self.DSSObj.YMatrix.GetCompressedYMatrix())
        Ybus = scsp.lil_matrix(Ybus)

        self.Ybus, self.nodes = self.organiza_Ybus(Ybus)
        
        self.Ybus = self.Conserta_Ybus(self.Ybus)
        self.Gs = np.real(self.Ybus).toarray()
        self.Bs = np.imag(self.Ybus).toarray()
        if self.verbose: print('Matriz de adimitância modificada com sucesso')

    def resolve_fluxo_carga(self):
        self.DSSText.Command = 'Clear'
        self.DSSText.Command = f'Compile {self.MasterFile}'

        self.iniciar_medidores()

        self.DSSText.Command = 'Solve'
        if not self.DSSCircuit.Solution.Converged:
            raise 'O fluxo de carga não convergiu'

    def InitializeDSS(self):
        DSSObj = dss_engine
        flag = DSSObj.Start(0)
        if flag and self.verbose:
            print('OpenDSS COM Interface initialized succesfully.')
            
        elif self.verbose:
            print('OpenDSS COMInterface failed to start.')
            
        #Set up the interface variables - Comunication OpenDSS with Python 
        DSSText = DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSMonitors = DSSCircuit.Monitors

        return DSSCircuit, DSSText, DSSObj, DSSMonitors

    def reset(self):
        self.resolve_fluxo_carga()
        self.barras, self.num_medidas = self.medidas(self.baseva)
        self.vet_estados = self.iniciar_vet_estados()
    
    def iniciar_medidores(self) -> None:
        count = 0
        for i, barra in enumerate(self.DSSCircuit.AllBusNames):
            self.DSSCircuit.SetActiveBus(barra)
            for j, elem in enumerate(self.DSSCircuit.Buses.AllPCEatBus):
                if 'Load' in elem or 'Generator' in elem or 'Vsource' in elem or 'PVSystem' in elem:
                    self.DSSText.Command = f'New Monitor.pqi{count} element={elem}, terminal=1, mode=1, ppolar=no'
                    count += 1
            
            #Escolher elemento com maior numero de fases da barra  
            max_fases = 0
            elem = 'None'
            for pde in self.DSSCircuit.Buses.AllPDEatBus:
                self.DSSCircuit.SetActiveElement(pde)
                num_fases = len(self.DSSCircuit.ActiveCktElement.NodeOrder)
                if num_fases > max_fases:
                    elem = self.DSSCircuit.ActiveCktElement.Name
                    max_fases = num_fases
            #Colocar um medidor no elemento escolhido
            if elem != 'None':
                self.DSSCircuit.SetActiveElement(elem)
                if self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0] == barra:
                    self.DSSText.Command = f'New Monitor.v{i} element={elem}, terminal=1, mode=32'
                    
                elif self.DSSCircuit.ActiveCktElement.BusNames[1].split('.')[0] == barra:
                    self.DSSText.Command = f'New Monitor.v{i} element={elem}, terminal=2, mode=32'
                    
                else:
                    print(f'Nenhum elemento conectado na barra {barra}')

    def indexar_barras(self) -> pd.DataFrame:
        #Designa indíces às barras
        nomes = []
        bases = []
        geracao = []
        for barra in self.DSSCircuit.AllBusNames:
            #if barra.isdigit(): è possível que o sourcebus e o reg não entrem para a EE
            self.DSSCircuit.SetActiveBus(barra)
            #Base é em fase-neutro
            base = self.DSSCircuit.Buses.kVBase
            if base == 0:
                raise ValueError('Tensão base não pode ser 0')
            nomes.append(barra)
            bases.append(base)
            geracao.append(self.DSSCircuit.Buses.AllPCEatBus[0] == 'Vsource.source')

        nomes = np.concatenate([nomes[1:], [nomes[0]]])
        bases = np.concatenate([bases[1:], [bases[0]]])
        geracao = np.concatenate([geracao[1:], [geracao[0]]])

        idx = [i for i in range(len(nomes))]
        
        barras = pd.DataFrame(columns=['nome_barra', 'Bases', 'Fases', 'Inj_pot_at', 'Inj_pot_rat', 'Tensao', 'Inj_pot_at_est', 'Inj_pot_rat_est', 'Geracao'],
                            index=idx)
        
        barras['nome_barra'] = nomes
        barras.loc[idx, 'Bases'] = bases
        barras.loc[idx, 'Geracao'] = geracao

        return barras

    def gera_medida_imperfeita(self, medida: float) -> None:
        # Gerar fatores aleatórios com base na distribuição normal
        fatores = np.random.normal(medida, self.dp, self.num_medidas)
        
        for i, medidas in enumerate(self.barras['Inj_pot_at']):
            self.barras['Inj_pot_at'][i] = medidas + medidas * fatores[i*3:(i+1)*3]

    def iniciar_vet_estados(self) -> np.array:
        fases = self.barras['Fases'].tolist()
        fases = [sub_elem for elem in fases for sub_elem in elem]
        tensoes = np.array([1 for _ in fases[:-3]], dtype=np.float64)
        angulos = np.zeros(len(fases[:-3]))
        
        for i, fase in enumerate(fases[:-3]):
            if fase == 1:
                angulos[i] = -120 * 2 * np.pi / 360
            elif fase == 2:
                angulos[i] = 120 * 2 * np.pi / 360
        
        vet_estados = np.concatenate((angulos, tensoes))
                
        return vet_estados
    
    def achar_index_barra(self, barras: pd.DataFrame, barra: int) -> int:
        #Retorna o index da barra do monitor ativo
        self.DSSCircuit.SetActiveElement(self.DSSMonitors.Element)
        
        self.DSSCircuit.SetActiveBus(self.DSSCircuit.ActiveCktElement.BusNames[barra])
        nome = self.DSSCircuit.Buses.Name
        
        return barras.index[barras['nome_barra'] == nome].to_list()[0]

    def pegar_fases(self) -> np.array:
        fases = self.DSSCircuit.ActiveBus.Nodes - 1
        fases = list(dict.fromkeys(fases))
        fases = [fase for fase in fases if fase != -1]
        
        return fases

    def medidas(self, baseva: int) -> pd.DataFrame:
        #Verivicar se os indexes podem ser substituídos pelos nomes das colunas sem perda de performance
        barras = self.indexar_barras()
        
        num_medidas = 0
        for idx, bus in enumerate(barras['nome_barra']):
            self.DSSCircuit.SetActiveBus(bus)
            fases = self.pegar_fases()
            #Preenche a coluna das fases
            barras.iat[idx, 2] = fases
            if not barras['Geracao'][idx]:
                #Inicializa as medidas como 0 para injeção ativa e reativa
                barras.iat[idx, 3] = np.array([0, 0, 0, 0], dtype=np.float64)
                barras.iat[idx, 4] = np.array([0, 0, 0, 0], dtype=np.float64)
                num_medidas += len(fases)*2
        
        #Amostra e salva os valores dos medidores do sistema
        self.DSSMonitors.ResetAll()
        self.DSSMonitors.SampleAll()
        self.DSSMonitors.SaveAll()
        
        self.DSSMonitors.First
        for _ in range(self.DSSMonitors.Count):
            barra = self.DSSMonitors.Terminal - 1
            index_barra = self.achar_index_barra(barras, barra)
            
            #Pegar as fases da carga atual
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            fases = list(dict.fromkeys(fases))
            fases = [fase for fase in fases if fase != -1]
            matriz_medidas = self.DSSMonitors.AsMatrix()[0][2:]
            
            if 'pqi' in self.DSSMonitors.Name:
                medidas_at = np.zeros(4)
                medidas_rat = np.zeros(4)
                
                for i, fase in enumerate(fases):
                    medidas_at[fase] = matriz_medidas[i*2]
                    medidas_rat[fase] = matriz_medidas[i*2+1]
                
                #Atualiza as medidas de injeção ativa e reativa
                barras.iat[index_barra, 3] += -medidas_at*1000 / baseva
                barras.iat[index_barra, 4] += -medidas_rat*1000 / baseva
                
            elif 'v' in self.DSSMonitors.Name:
                if type(barras['Tensao'][index_barra]) != np.ndarray:
                    medidas = np.zeros(4)
                    
                    for i, fase in enumerate(fases):
                        medidas[fase] = matriz_medidas[i]

                    #Atualiza as medidas de tensão 
                    basekv = self.DSSCircuit.Buses.kVBase
                    barras.iat[index_barra, 5] = medidas / (basekv*1000)
                    if not barras['Geracao'][index_barra]:
                        num_medidas += len(fases)
            
            self.DSSMonitors.Next
            
        return barras, num_medidas

    def forma_matriz(self, fases: np.ndarray, fases_barra: list, Yprim: list) -> np.array:
        temp = np.zeros((len(fases_barra), len(fases_barra)), dtype=np.complex128)
        fases = [fase for fase in fases if fase != -1]
        fases_barra = list(fases_barra-1)
        idx = []
        for fase in fases:
            idx.append(fases_barra.index(fase))
            
        k = 0
        for i in idx:
            for j in idx:
                temp[i, j] = Yprim[k]
                k += 1

        if len(Yprim) == 16:
            temp = np.reshape(Yprim, (4, 4))
        
        Yprim = temp
        
        return Yprim
    
    def organiza_Ybus(self, Ybus):
        nodes = {}
        for i, node in enumerate(self.DSSCircuit.YNodeOrder):
            nodes[node.lower()] = i

        temp = Ybus.copy()
        count = 0
        for i, bus in enumerate(self.DSSCircuit.AllNodeNames):
            no = nodes[f'{bus}']
            temp[count] = Ybus[no].toarray()
            count += 1

        temp = temp.T
        Ybus = temp.copy()
        count = 0
        for i, bus in enumerate(self.DSSCircuit.AllNodeNames):
            no = nodes[f'{bus}']
            Ybus[count] = temp[no]
            count += 1
                
        #csr_matrix pode ser mais rápida para sistemas maiores
        Ybus = scsp.lil_matrix(Ybus)
        
        nodes = {}
        count = 0
        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                nodes[f'{bus}.{fase+1}'] = count
                count += 1

        return Ybus, nodes
    
    def Conserta_Ybus(self, Ybus):
        self.DSSCircuit.Transformers.First
        for _ in range(self.DSSCircuit.Transformers.Count):
            trafo = self.DSSCircuit.Transformers.Name
            self.DSSCircuit.SetActiveElement(trafo)
            num_phases = self.DSSCircuit.ActiveCktElement.NumPhases
            barras_conectadas = self.DSSCircuit.ActiveCktElement.BusNames
            self.DSSCircuit.SetActiveBus(barras_conectadas[0])
            basekv1 = self.DSSCircuit.Buses.kVBase
            self.DSSCircuit.SetActiveBus(barras_conectadas[1])
            basekv2 = self.DSSCircuit.Buses.kVBase
            if '.' in barras_conectadas[0] or '.' in barras_conectadas[1]:
                barras_conectadas[0] = barras_conectadas[0].split('.')[0]
                barras_conectadas[1] = barras_conectadas[1].split('.')[0]
                
            no1 = self.nodes[f"{barras_conectadas[0]}.{1}"]
            no2 = self.nodes[f"{barras_conectadas[1]}.{1}"]
            
            if basekv1 > basekv2:
                n = basekv1 / basekv2
                Ybus[no1:no1+num_phases, no2:no2+num_phases] = (Ybus[no1:no1+num_phases, no2:no2+num_phases])/n
                Ybus[no2:no2+num_phases, no1:no1+num_phases] = (Ybus[no2:no2+num_phases, no1:no1+num_phases])*n
            else:
                n = basekv2 / basekv1
                Ybus[no1:no1+num_phases, no2:no2+num_phases] = (Ybus[no1:no1+num_phases, no2:no2+num_phases])*n
                Ybus[no2:no2+num_phases, no1:no1+num_phases] = (Ybus[no2:no2+num_phases, no1:no1+num_phases])/n
                
            self.DSSCircuit.Transformers.Next

        #Verificar se funciona para cargas em delta
        self.DSSCircuit.Loads.First
        for _ in range(self.DSSCircuit.Loads.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.Loads.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
            
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.Loads.Next
            
        self.DSSCircuit.Generators.First
        for _ in range(self.DSSCircuit.Generators.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.Generators.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
                
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.Generators.Next
        
        self.DSSCircuit.PVSystems.First
        for _ in range(self.DSSCircuit.PVSystems.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.PVSystems.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
                
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.PVSystems.Next
            
        self.DSSCircuit.Reactors.First
        for _ in range(self.DSSCircuit.Reactors.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.Reactors.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
                
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.Reactors.Next
            
        self.DSSCircuit.SetActiveElement('Vsource.source')
        Yprim = self.DSSCircuit.ActiveCktElement.Yprim
        real = Yprim[::2]
        imag = Yprim[1::2]*1j
        Yprim = real+imag
        Yprim = np.reshape(Yprim, (6, 6))
        Ybus[:3, :3] -= Yprim[:3, :3]
        
        #Passar a matriz para pu
        basesY = np.array(self.baseva / ((self.barras['Bases']*1000) ** 2))
        basesY = np.concatenate([[basesY[-1]], basesY[:-1]])
        
        linha = 0
        for baseY, fases in zip(basesY, self.barras['Fases']):
            for fase in fases:
                Ybus[linha] = Ybus[linha] / baseY
                linha += 1
        
        return Ybus

    def Calcula_pesos(self) -> tuple:
        inj_pot_at = []
        inj_pot_rat = []
        tensoes = []
        for fases, pot_at, pot_rat, tensao in zip(self.barras['Fases'], self.barras['Inj_pot_at'], self.barras['Inj_pot_rat'], self.barras['Tensao']):
            for fase in fases:
                inj_pot_at.append(pot_at[fase])
                inj_pot_rat.append(pot_rat[fase])
                tensoes.append(tensao[fase])
        
        medidas = np.concatenate([inj_pot_at[:-3], inj_pot_rat[:-3], tensoes[:-3]])

        dp = (medidas * 0.01) / (3 * 100)
        dp[dp == 0] = 10**-5
        pesos = dp**-2
        pesos[pesos > 10**10] = 10**10
        matriz_pesos = scsp.lil_matrix((len(pesos), len(pesos)))
        matriz_pesos.setdiag(pesos)
        
        return scsp.csr_matrix(matriz_pesos), np.abs(dp)
    
    def Calcula_Residuo(self) -> np.ndarray:
        angs = self.vet_estados[:len(self.fases[:-(self.count)*3])]
        tensoes = self.vet_estados[len(self.fases[:-(self.count)*3]):]
        ang_ref = np.array([0, -120*2*np.pi / 360,  120*2*np.pi / 360])
        tensoes_ref = self.barras['Tensao'][self.DSSCircuit.NumBuses-1][:3]
        angs = np.concatenate((ang_ref, angs))
        tensoes = np.concatenate((tensoes_ref, tensoes))
        
        residuo = Residuo(self.barras, tensoes, angs)
        
        residuos = residuo.calc_res(self.Gs, self.Bs)
        
        self.inj_pot_at_est = np.array(residuo.inj_pot_at_est)
        self.inj_pot_rat_est = np.array(residuo.inj_pot_rat_est)

        return residuos

    def Calcula_Jacobiana(self) -> np.ndarray:      
        angs = self.vet_estados[:len(self.fases[:-(self.count)*3])]
        tensoes = self.vet_estados[len(self.fases[:-(self.count)*3]):]
        ang_ref = np.array([0, -120*2*np.pi / 360,  120*2*np.pi / 360])
        tensoes_ref = self.barras['Tensao'][self.DSSCircuit.NumBuses-1][:3]
        angs = np.concatenate((angs, ang_ref))
        tensoes = np.concatenate((tensoes, tensoes_ref))

        jac = Jacobiana(tensoes, angs, self.fases)
                
        jacobiana = jac.derivadas(self.Gs, self.Bs, self.inj_pot_at_est, self.inj_pot_rat_est)
        
        return scsp.csr_matrix(jacobiana)
    
    def run(self, max_error: float, max_iter: int) -> np.array:       
        self.barras, self.num_medidas = self.medidas(self.baseva)
        self.vet_estados = self.iniciar_vet_estados()
        if self.verbose: print('Vetor de estados iniciado')
        
        self.count = self.barras['Geracao'].value_counts().iloc[1]
        fases = self.barras['Fases'].tolist()
        self.fases = [sub_elem for elem in fases for sub_elem in elem]
        
        self.matriz_pesos, self.dp = self.Calcula_pesos()
        
        k = 0
        delx = 1
        while(np.max(np.abs(delx)) > max_error and max_iter > k):
            inicio = time.time()

            self.residuo = self.Calcula_Residuo()
            fim_res = time.time()

            self.jacobiana = self.Calcula_Jacobiana()
            fim_jac = time.time()

            self.matriz_pesos, self.dp = self.Calcula_pesos()
            fim_pesos = time.time()

            #Calcula a matriz ganho
            matriz_ganho = self.jacobiana.T @ self.matriz_pesos @ self.jacobiana
            
            #Calcula o outro lado da Equação normal
            seinao = self.jacobiana.T @ self.matriz_pesos @ self.residuo

            delx = np.linalg.inv(matriz_ganho.toarray()) @ seinao
            
            #Atualiza o vetor de estados
            self.vet_estados += delx
            fim = time.time()
            
            k += 1
            if np.isnan(self.vet_estados.any()):
                print('NaN na solução')
                break
            if k == 99:
                print('Não convergiu')
            if self.verbose:
                print(f'Os resíduos da iteração {k} levaram {fim_res-inicio:.3f}s')
                print(f'A jacobiana da iteração {k} levou {fim_jac-fim_res:.3f}s')
                print(f'Os pesos da iteração {k} levaram {fim_pesos-fim_jac:.3f}s')
                print(f'Atualizar o vetor de estados da iteração {k} levou {fim-fim_pesos:.3f}')
                print(f'A iteração {k} levou {fim-inicio:.3f}s')

        return self.vet_estados
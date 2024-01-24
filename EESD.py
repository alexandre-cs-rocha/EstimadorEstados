import numpy as np
import pandas as pd
import scipy as sp

from dss import DSS as dss_engine
from Jacobiana import Jacobiana
from Residuos import Residuo

class EESD():
    def __init__(self, master_path, baseva: float = 10**6) -> None:
        self.DSSCircuit, self.DSSText, self.DSSObj, self.DSSMonitors = self.InitializeDSS()
        self.baseva = baseva
        self.MasterFile = master_path
        
        self.resolve_fluxo_carga()
        
        self.barras, self.num_medidas = self.medidas(self.baseva)
        self.vet_estados = self.iniciar_vet_estados()
        
        Ybus = sp.sparse.csc_matrix(self.DSSObj.YMatrix.GetCompressedYMatrix())
        self.Ybus, self.nodes = self.organiza_Ybus(Ybus)
        self.Ybus = self.Conserta_Ybus(self.Ybus)

    def resolve_fluxo_carga(self):
        self.DSSText.Command = 'Clear'
        self.DSSText.Command = f'Compile {self.MasterFile}'

        self.iniciar_medidores()

        self.DSSText.Command = 'Solve'

    def InitializeDSS(self) -> tuple:
        DSSObj = dss_engine
        flag = DSSObj.Start(0)
        if flag:
            print('OpenDSS COM Interface initialized succesfully.')
            
        else:
            print('OpenDSS COMInterface failed to start.')
            
        #Set up the interface variables - Comunication OpenDSS with Python
        DSSText = DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSMonitors = DSSCircuit.Monitors

        return DSSCircuit, DSSText, DSSObj, DSSMonitors

    def iniciar_medidores(self) -> None:
        for i, barra in enumerate(self.DSSCircuit.AllBusNames):
            self.DSSCircuit.SetActiveBus(barra)
            for j, elem in enumerate(self.DSSCircuit.Buses.AllPCEatBus):
                if 'Load' in elem or 'Generator' in elem or 'Vsource' in elem:
                    self.DSSText.Command = f'New Monitor.pqi{i}{j} element={elem}, terminal=1, mode=1, ppolar=no'
            
            max_fases = 0
            elem = 'None'
            for pde in self.DSSCircuit.Buses.AllPDEatBus:
                self.DSSCircuit.SetActiveElement(pde)
                num_fases = len(self.DSSCircuit.ActiveCktElement.NodeOrder)
                if num_fases > max_fases:
                    elem = self.DSSCircuit.ActiveCktElement.Name
                    max_fases = num_fases
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
            nomes.append(barra)
            bases.append(base)
            geracao.append(self.DSSCircuit.Buses.AllPCEatBus[0] == 'Vsource.source')

        nomes = np.concatenate([nomes[1:], [nomes[0]]])
        bases = np.concatenate([bases[1:], [bases[0]]])
        geracao = np.concatenate([geracao[1:], [geracao[0]]])

        idx = [i for i in range(len(nomes))]
        inicial1 = [[0, 0, 0] for _ in range(len(nomes))]
        inicial2 = [[0, 0, 0] for _ in range(len(nomes))]
        
        barras = pd.DataFrame(columns=['nome_barra', 'Bases', 'Fases', 'Inj_pot_at', 'Inj_pot_rat', 'Flux_pot_at', 'Flux_pot_rat', 'Tensao', 'Inj_pot_at_est', 'Inj_pot_rat_est', 'Geracao'],
                            index=idx)
        
        barras['nome_barra'] = nomes
        barras.loc[idx, 'Bases'] = bases
        barras.loc[idx, 'Geracao'] = geracao
        
        for i in idx:
            barras['Inj_pot_at_est'][i] = inicial1[i]
            barras['Inj_pot_rat_est'][i] = inicial2[i]

        return barras

    def gera_medida_imperfeita(self, media: float) -> None:
        # Gerar fatores aleatórios com base na distribuição normal
        fatores = np.random.normal(media, self.dp, self.num_medidas)
        
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
        barras = self.indexar_barras()
        
        num_medidas = 0
        for idx, bus in enumerate(barras['nome_barra']):
            self.DSSCircuit.SetActiveBus(bus)
            fases = self.pegar_fases()
            barras['Fases'][idx] = fases
            if not barras['Geracao'][idx]:
                barras['Inj_pot_at'][idx] = np.array([0, 0, 0], dtype=np.float64)
                barras['Inj_pot_rat'][idx] = np.array([0, 0, 0], dtype=np.float64)
                num_medidas += len(fases)*2
        
        #Amostra e salva os valores dos medidores do sistema
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
            
            if 'pqij' in self.DSSMonitors.Name:
                if type(barras['Flux_pot_at'][index_barra]) != list and type(barras['Flux_pot_rat'][index_barra]) != list:
                    barras['Flux_pot_at'][index_barra] = []
                    barras['Flux_pot_rat'][index_barra] = []
                    
                elemento = self.DSSMonitors.Element
                self.DSSCircuit.ActiveCktElement.BusNames[1]
                medidas_at = np.full([3], np.NaN)
                medidas_rat = np.full([3], np.NaN)
                
                for i, fase in enumerate(fases):
                    medidas_at[fase] = matriz_medidas[i*2]*1000 / baseva
                    medidas_rat[fase] = matriz_medidas[i*2+1]*1000 / baseva
                    num_medidas += 2
                    
                barras['Flux_pot_at'][index_barra].append((elemento, medidas_at))
                barras['Flux_pot_rat'][index_barra].append((elemento, medidas_rat))
            
            elif 'pqi' in self.DSSMonitors.Name:
                medidas_at = np.zeros(3)
                medidas_rat = np.zeros(3)
                
                for i, fase in enumerate(fases):
                    medidas_at[fase] = matriz_medidas[i*2]
                    medidas_rat[fase] = matriz_medidas[i*2+1]
                    
                barras['Inj_pot_at'][index_barra] += -medidas_at*1000 / baseva
                barras['Inj_pot_rat'][index_barra] += -medidas_rat*1000 / baseva
                
            elif 'v' in self.DSSMonitors.Name:
                if type(barras['Tensao'][index_barra]) != np.ndarray:
                    medidas = np.zeros(3)
                    
                    for i, fase in enumerate(fases):
                        medidas[fase] = matriz_medidas[i]
    
                    basekv = self.DSSCircuit.Buses.kVBase
                    barras['Tensao'][index_barra] = medidas / (basekv*1000)
                    if not barras['Geracao'][index_barra]:
                        num_medidas += len(fases)
            
            self.DSSMonitors.Next
            
        return barras, num_medidas

    def forma_matriz(self, fases: list, fases_barra: list, Yprim: list) -> np.array:
        if len(fases) == 4:
            Yprim = np.reshape(Yprim, (4, 4))
        
        elif len(fases) == 3:
            Yprim = np.reshape(Yprim, (3, 3))
            
        elif len(fases) == 2 and len(fases_barra) == 3 and not -1 in fases:
            temp = np.zeros((len(fases_barra), len(fases_barra)), dtype=np.complex128)
            temp[fases[0], fases[0]] = Yprim[0]
            temp[fases[0], fases[1]] = Yprim[1]
            temp[fases[1], fases[0]] = Yprim[2]
            temp[fases[1], fases[1]] = Yprim[3]
            Yprim = temp.copy()
            
        elif len(fases) == 2 and len(fases_barra) == 2 and not -1 in fases:
            Yprim = np.reshape(Yprim, (2, 2))
            
        elif -1 in fases:
            index = np.where(fases_barra-1 == fases[0])
            temp = np.zeros((len(fases_barra), len(fases_barra)), dtype=np.complex128)
            temp[index, index] = Yprim[0]
            Yprim = temp.copy()
            
        return Yprim
    
    def organiza_Ybus(self, Ybus):
        nodes = {}
        for i, node in enumerate(self.DSSCircuit.YNodeOrder):
            nodes[node.lower()] = i
        
        temp = Ybus.copy()
        count = 0
        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                no = nodes[f'{bus}.{fase+1}']
                temp[count] = Ybus[no].toarray()
                count += 1
            
        temp = temp.T
        Ybus_org = temp.copy()
        count = 0
        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                no = nodes[f'{bus}.{fase+1}']
                Ybus_org[count] = temp[no]
                count += 1
                
        Ybus_org = sp.sparse.csc_matrix(Ybus_org)
        
        nodes = {}
        count = 0
        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                nodes[f'{bus}.{fase+1}'] = count
                count += 1

        return Ybus_org, nodes
    
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
                
        self.DSSCircuit.SetActiveElement('Vsource.source')
        Yprim = self.DSSCircuit.ActiveCktElement.Yprim
        real = Yprim[::2]
        imag = Yprim[1::2]*1j
        Yprim = real+imag
        Yprim = np.reshape(Yprim, (6, 6))
        Ybus[:3, :3] -= Yprim[:3, :3]

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
            
        matriz_pesos = np.diag(pesos)
        
        return matriz_pesos, np.abs(dp)
    
    def Calcula_Residuo(self) -> np.array:
        count = self.barras['Geracao'].value_counts()[1]
        fases = self.barras['Fases'].tolist()
        fases = [sub_elem for elem in fases for sub_elem in elem]
        
        angs = self.vet_estados[:len(fases[:-(count)*3])]
        tensoes = self.vet_estados[len(fases[:-(count)*3]):]
        ang_ref = np.array([0, -120*2*np.pi / 360,  120*2*np.pi / 360])
        tensoes_ref = self.barras['Tensao'][self.DSSCircuit.NumBuses-1]
        angs = np.concatenate((ang_ref, angs))
        tensoes = np.concatenate((tensoes_ref, tensoes))
        vet_estados_aux = np.concatenate((angs, tensoes))
        
        residuo = Residuo(self.barras)
        
        for idx, geracao in enumerate(self.barras['Geracao']):
            if not geracao:
                fases = self.barras['Fases'][idx]
                index_fase = self.barras['Fases'].tolist()
                index_fase = np.sum([len(elem) for elem in index_fase[:idx]])
                barra = self.barras['nome_barra'][idx]
                basekv = self.barras['Bases'][idx]
                baseY = self.baseva / ((basekv*1000)**2)
            
                for i, fase in enumerate(fases):
                    tensao_estimada = tensoes[int(3+index_fase+i)]
                    ang_estimado = angs[int(3+index_fase+i)]
                    
                    diff_angulos = ang_estimado - angs.copy()

                    no1 = self.nodes[barra+f'.{fase+1}']
                    Yline = self.Ybus[no1] / baseY
                    if self.barras['nome_barra'][idx] == '675' and fase == 0:
                        pass
                    
                    Gline = np.real(Yline).toarray()
                    Bline = np.imag(Yline).toarray()

                    residuo.Residuo_inj_pot_at(idx, fase, tensao_estimada, tensoes, diff_angulos, Gline, Bline)

                    residuo.Residuo_inj_pot_rat(idx, fase, tensao_estimada, tensoes, diff_angulos, Gline, Bline)
                    
                    residuo.Residuo_tensao(idx, fase, tensao_estimada)
                    
        for idx1, medidas in enumerate(self.barras['Flux_pot_at']):
            if type(medidas) == list:
                for medida in medidas:
                    elemento = medida[0]
                    fases = np.where((np.isnan(medida[1]) == False))[0]
                    residuo.Residuo_fluxo_pot_at(vet_estados_aux, fases, idx1, elemento, 
                                        self.baseva, self.barras, self.DSSCircuit, self.nodes, self.Ybus)
                
        for idx1, medidas in enumerate(self.barras['Flux_pot_rat']):
            if type(medidas) == list:
                for medida in medidas:
                    elemento = medida[0]
                    fases = np.where((np.isnan(medida[1]) == False))[0]
                    residuo.Residuo_fluxo_pot_rat(vet_estados_aux, fases, idx1, elemento, 
                                        self.baseva, self.barras, self.DSSCircuit, self.nodes, self.Ybus)
            
        return np.concatenate([residuo.vet_inj_at, residuo.vet_inj_rat, residuo.vet_tensao])

    def Calcula_Jacobiana(self) -> np.array:
        count = self.barras['Geracao'].value_counts()[1]
        fases = self.barras['Fases'].tolist()
        fases = [sub_elem for elem in fases for sub_elem in elem]
        
        angs = self.vet_estados[:len(fases[:-(count)*3])]
        tensoes = self.vet_estados[len(fases[:-(count)*3]):]
        ang_ref = np.array([0, -120*2*np.pi / 360,  120*2*np.pi / 360])
        tensoes_ref = self.barras['Tensao'][self.DSSCircuit.NumBuses-1]
        angs = np.concatenate((angs, ang_ref))
        tensoes = np.concatenate((tensoes, tensoes_ref))
        vet_estados_aux = np.concatenate((angs, tensoes))
        
        jac = Jacobiana(vet_estados_aux, self.baseva, self.barras, self.nodes, (len(fases)-3)*3)
        
        medida_atual = 0
        for idx, medida in enumerate(self.barras['Inj_pot_at']):
            if type(medida) == np.ndarray and not self.barras['Geracao'][idx]:
                medida_atual = jac.Derivadas_inj_pot_at(medida_atual, idx, self.DSSCircuit.NumBuses, self.Ybus, count)
        
        for idx, medida in enumerate(self.barras['Inj_pot_rat']):
            if type(medida) == np.ndarray and not self.barras['Geracao'][idx]:
                medida_atual = jac.Derivadas_inj_pot_rat(medida_atual, idx, self.DSSCircuit.NumBuses, self.Ybus, count)
                
        for idx1, medida in enumerate(self.barras['Flux_pot_at']):
            if type(medida) == list:
                elemento = medida[0][0]
                fases = np.where((np.isnan(medida[0][1]) == False))[0]
                medida_atual = jac.Derivadas_fluxo_pot_at(fases, medida_atual, idx1, elemento, self.barras, self.nodes, vet_estados_aux,
                                                    self.DSSCircuit, self.Ybus, self.baseva)
                
        for idx1, medida in enumerate(self.barras['Flux_pot_rat']):
            if type(medida) == list:
                elemento = medida[0][0]
                fases = np.where((np.isnan(medida[0][1]) == False))[0]
                medida_atual = jac.Derivadas_fluxo_pot_rat(fases, medida_atual, idx1, elemento, self.barras, self.nodes, vet_estados_aux,
                                                    self.DSSCircuit, self.Ybus, self.baseva)
                
        for idx, medida in enumerate(self.barras['Tensao']):
            if type(medida) == np.ndarray and not self.barras['Geracao'][idx]:
                medida_atual = jac.Derivadas_tensao(medida_atual, idx, self.DSSCircuit.NumBuses, count)
            
        return jac.jacobiana

    def run(self, max_error: float, max_iter: int) -> np.array:
        k = 0
        delx = 1
        while(np.max(np.abs(delx)) > max_error and max_iter > k):

            self.residuo = self.Calcula_Residuo()

            self.jacobiana = self.Calcula_Jacobiana()
            
            self.matriz_pesos, self.dp = self.Calcula_pesos()
            
            #Calcula a matriz ganho
            matriz_ganho = self.jacobiana.T @ self.matriz_pesos @ self.jacobiana
            
            #Calcula o outro lado da Equação normal
            seinao = self.jacobiana.T @ self.matriz_pesos @ self.residuo

            delx = np.linalg.inv(matriz_ganho) @ seinao
            
            #Atualiza o vetor de estados
            self.vet_estados += delx
            
            k += 1
            print(k)
        return self.vet_estados

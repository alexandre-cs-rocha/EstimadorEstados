import numpy as np
import pandas as pd

class Jacobiana():
    def __init__(self, vet_estados: np.array, baseva: float, barras: pd.DataFrame, nodes: dict, num_medidas: int) -> None:
        self.jacobiana = np.zeros((num_medidas, len(vet_estados)-6))
        self.vet_estados = vet_estados
        self.baseva = baseva
        self.barras = barras
        self.nodes = nodes
        
    def Derivadas_inj_pot_at(self, medida_atual: int, index_barra: int, num_buses: int, Ybus, count) -> int:
        barra1 = self.barras['nome_barra'][index_barra]
        fases = self.barras['Fases'][index_barra]
        lista_fase = self.barras['Fases'].tolist()
        index_fase = np.sum([len(elem) for elem in lista_fase[:index_barra]])
        basekv = self.barras['Bases'][index_barra]
        baseY = self.baseva / ((basekv*1000)**2)        
        
        for i, fase in enumerate(fases):
            no1 = self.nodes[barra1+f'.{fase+1}']
            tensao_estimada = self.vet_estados[int(len(self.vet_estados)//2+index_fase+i)]
            ang_estimado = self.vet_estados[int(index_fase+i)]

            #Derivada da injeção de potência ativa com relação as tensões
            for index_barra2 in range(len(self.barras['nome_barra'])):
                barra2 = self.barras['nome_barra'][index_barra2]
                fases2 = self.barras['Fases'][index_barra2]
                index_fase2 = np.sum([len(elem) for elem in lista_fase[:index_barra2]])
                
                if self.barras['Geracao'][index_barra2]:
                    continue

                for idx, m in enumerate(fases2):
                    no2 = self.nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    if no1 == no2:
                        delta = ((tensao_estimada**2)*Gs+self.barras['Inj_pot_at_est'][index_barra][fase]) / tensao_estimada
                    else:
                        ang_estimado2 = self.vet_estados[int(index_fase2+idx)]
                        delta = tensao_estimada*(Gs*np.cos(ang_estimado-ang_estimado2)+Bs*np.sin(ang_estimado-ang_estimado2))

                    self.jacobiana[medida_atual][int(len(self.vet_estados)//2+index_fase2+idx-3)] = delta
            
            #Derivadas de injeção de potência ativa com relação aos ângulos
            for index_barra2 in range(len(self.barras['nome_barra'])):
                
                if self.barras['Geracao'][index_barra2]:
                    continue

                barra2 = self.barras['nome_barra'][index_barra2]
                fases2 = self.barras['Fases'][index_barra2]
                
                index_fase2 = np.sum([len(elem) for elem in lista_fase[:index_barra2]])

                for k, m in enumerate(fases2):
                    no2 = self.nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    if no1 == no2:
                        delta = -Bs*(tensao_estimada**2)
                        delta2 = 0
                        
                        for idx in range(len(self.barras['nome_barra'])):
                            barra3 = self.barras['nome_barra'][idx]
                            fases3 = self.barras['Fases'][idx]
                            index_fase3 = np.sum([len(elem) for elem in lista_fase[:idx]])
                            
                            for j, n in enumerate(fases3):
                                no3 = self.nodes[barra3+f'.{n+1}']
                                Yij = Ybus[no1, no3] / baseY
                                if Yij != 0:
                                    tensao_estimada2 = self.vet_estados[int(len(self.vet_estados)//2+index_fase3+j)]
                                    ang_estimado2 = self.vet_estados[int(index_fase3+j)]
                                    Gs = np.real(Yij)
                                    Bs = np.imag(Yij)
                                    delta2 += tensao_estimada2*(Gs*np.sin(ang_estimado-ang_estimado2)-Bs*np.cos(ang_estimado-ang_estimado2))
                        delta = delta - tensao_estimada*delta2
                    else:
                        tensao_estimada2 = self.vet_estados[int(len(self.vet_estados)//2+index_fase2+k)]
                        ang_estimado2 = self.vet_estados[int(index_fase2+k)]
                        delta = tensao_estimada*tensao_estimada2*(Gs*np.sin(ang_estimado-ang_estimado2)-Bs*np.cos(ang_estimado-ang_estimado2))
                        
                    self.jacobiana[medida_atual][int(index_fase2+k)] = delta
                
            medida_atual += 1
            
        return medida_atual
            
    def Derivadas_inj_pot_rat(self, medida_atual: int, index_barra: int, num_buses: int, Ybus, count) -> int:
        barra1 = self.barras['nome_barra'][index_barra]
        fases = self.barras['Fases'][index_barra]
        lista_fase = self.barras['Fases'].tolist()
        index_fase = np.sum([len(elem) for elem in lista_fase[:index_barra]])
        basekv = self.barras['Bases'][index_barra]
        baseY = self.baseva / ((basekv*1000)**2)
        
        #Derivada da injeção de potência reativa com relação as tensões
        for i, fase in enumerate(fases):
            no1 = self.nodes[barra1+f'.{fase+1}']
            tensao_estimada = self.vet_estados[int(len(self.vet_estados)//2+index_fase+i)]
            ang_estimado = self.vet_estados[int(index_fase+i)]
            
            for index_barra2 in range(len(self.barras['nome_barra'])):
                if self.barras['Geracao'][index_barra2]:
                    continue
                
                barra2 = self.barras['nome_barra'][index_barra2]
                fases2 = self.barras['Fases'][index_barra2]
                
                index_fase2 = np.sum([len(elem) for elem in lista_fase[:index_barra2]])

                for idx, m in enumerate(fases2):
                    no2 = self.nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    if no1 == no2:
                        delta = ((tensao_estimada**2)*(-Bs)+self.barras['Inj_pot_rat_est'][index_barra][fase]) / tensao_estimada
                    else:
                        ang_estimado2 = self.vet_estados[int(index_fase2+idx)]
                        delta = tensao_estimada*(Gs*np.sin(ang_estimado-ang_estimado2)-Bs*np.cos(ang_estimado-ang_estimado2))
                        
                    self.jacobiana[medida_atual][int(len(self.vet_estados)//2+index_fase2+idx-3)] = delta

            #Derivadas de injeção de potência reativa com relação aos ângulos
            for index_barra2 in range(len(self.barras['nome_barra'])):
                if self.barras['Geracao'][index_barra2]:
                    continue

                barra2 = self.barras['nome_barra'][index_barra2]
                fases2 = self.barras['Fases'][index_barra2]
                
                index_fase2 = np.sum([len(elem) for elem in lista_fase[:index_barra2]])
                
                for j, m in enumerate(fases2):
                    no2 = self.nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    if no1 == no2:
                        medida_at = self.barras['Inj_pot_at_est'][index_barra][fase]
                        delta = -Gs*(tensao_estimada**2) + medida_at
                    else:
                        tensao_estimada2 = self.vet_estados[int(len(self.vet_estados)//2+index_fase2+j)]
                        ang_estimado2 = self.vet_estados[int(index_fase2+j)]
                        delta = -tensao_estimada*tensao_estimada2*(Gs*np.cos(ang_estimado-ang_estimado2)+Bs*np.sin(ang_estimado-ang_estimado2))
                        
                    self.jacobiana[medida_atual][int(index_fase2+j)] = delta
                    
            medida_atual += 1
                    
        return medida_atual

    def Derivadas_tensao(self, medida_atual: int, index_barra: int, num_buses: int, count) -> int:   
        fases = self.barras['Fases'][index_barra]
        index_fase = self.barras['Fases'].tolist()
        index_fase = np.sum([len(elem) for elem in index_fase[:index_barra]])

        for i, fase in enumerate(fases):
            self.jacobiana[medida_atual][int(len(self.vet_estados)//2+index_fase+i-3)] = 1
            medida_atual += 1
        
        return medida_atual

    def Derivadas_fluxo_pot_at(self, jacobiana: np.array, fases: np.array, medida_atual: int, index_barra1: int, elemento: str,
                            barras: pd.DataFrame, nodes: dict, vet_estados: np.array, DSSCircuit, Ybus, baseva) -> int:
        barra1 = barras['nome_barra'][index_barra1]
        basekv = barras['Bases'][index_barra1]
        baseY = baseva / ((basekv*1000)**2)
        num_buses = DSSCircuit.NumBuses
        DSSCircuit.SetActiveElement(elemento)
        Bshmatrix = np.zeros((3, 3))
        barra2 = DSSCircuit.ActiveCktElement.BusNames[1]
        index_barra2 = barras[barras['nome_barra'] == barra2].index.values[0]
        
        for fase in fases:
            no1 = nodes[barra1+f'.{fase+1}']

            tensao_estimada = vet_estados[(num_buses*3) + (index_barra1*3) + fase]
            ang_estimado = vet_estados[(index_barra1*3) + fase]

            #Derivada do fluxo de Potência ativa com relação a tensão na barra inicial
            for m in fases:
                no2 = nodes[barra2+f'.{m+1}']
                Yij = Ybus[no1, no2] / baseY
                Gs = np.real(Yij)
                Bs = np.imag(Yij)
                Bsh = Bshmatrix[fase, m] / baseY
                
                if m == fase:
                    delta = tensao_estimada*Gs
                    for n in fases:
                        no2 = nodes[barra2+f'.{n+1}']
                        Yij = Ybus[no1, no2] / baseY
                        Gs = np.real(Yij)
                        Bs = np.imag(Yij)
                        Bsh = Bshmatrix[fase, n] / baseY
                        tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra1*3) + n]
                        tensao_estimada3 = vet_estados[(num_buses*3) + (index_barra2*3) + n]
                        ang_estimado2 = vet_estados[(index_barra1*3) + n]
                        ang_estimado3 = vet_estados[(index_barra2*3) + n]
                        delta += tensao_estimada2*(Gs*np.cos(ang_estimado-ang_estimado2)+(Bs+Bsh)*np.sin(ang_estimado-ang_estimado2))
                        delta -= tensao_estimada3*(Gs*np.cos(ang_estimado-ang_estimado3)+Bs*np.sin(ang_estimado-ang_estimado3))

                else:
                    ang_estimado2 = vet_estados[(index_barra1*3) + m]
                    delta = tensao_estimada*(Gs*np.cos(ang_estimado-ang_estimado2) + (Bs+Bsh)*np.sin(ang_estimado-ang_estimado2))
                    
                jacobiana[medida_atual][(num_buses+index_barra1)*3 + m - 3] = delta
                
            if index_barra1 != 0:
                #Derivada do fluxo de Potência ativa com relação ao ângulo na barra inicial
                for m in fases:
                    no2 = nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    Bsh = Bshmatrix[fase, m] / baseY
                    if m == fase:
                        delta = -(tensao_estimada**2)*(Bs+Bsh)
                        for n in fases:
                            no2 = nodes[barra2+f'.{n+1}']
                            Yij = Ybus[no1, no2] / baseY
                            Gs = np.real(Yij)
                            Bs = np.imag(Yij)
                            Bsh = Bshmatrix[fase, n] / baseY
                            tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra1*3) + n]
                            tensao_estimada3 = vet_estados[(num_buses*3) + (index_barra2*3) + n]
                            ang_estimado2 = vet_estados[(index_barra1*3) + n]
                            ang_estimado3 = vet_estados[(index_barra2*3) + n]
                            delta -= tensao_estimada*tensao_estimada2*(Gs*np.sin(ang_estimado-ang_estimado2)-(Bs+Bsh)*np.cos(ang_estimado-ang_estimado2))
                            delta += tensao_estimada*tensao_estimada3*(Gs*np.sin(ang_estimado-ang_estimado3)-Bs*np.cos(ang_estimado-ang_estimado3))

                    else:
                        tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra1*3) + m]
                        ang_estimado2 = vet_estados[(index_barra1*3) + m]
                        delta = tensao_estimada*tensao_estimada2*(Gs*np.sin(ang_estimado-ang_estimado2) - (Bs+Bsh)*np.cos(ang_estimado-ang_estimado2))
                        
                    jacobiana[medida_atual][(index_barra1*3) + m - 3] = delta
                
            #Derivada do fluxo de Potência ativa com relação a tensão na barra final
            for m in fases:
                no2 = nodes[barra2+f'.{m+1}']
                Yij = Ybus[no1, no2] / baseY
                Gs = np.real(Yij)
                Bs = np.imag(Yij)
                Bsh = Bshmatrix[fase, m] / baseY
                ang_estimado2 = vet_estados[(index_barra2*3) + m]
                delta = -tensao_estimada*(Gs*np.cos(ang_estimado-ang_estimado2) + Bs*np.sin(ang_estimado-ang_estimado2))
                jacobiana[medida_atual][num_buses*3 + (index_barra2*3) + m - 3] = delta
                
            if index_barra2 != 0:
                #Derivada do fluxo de Potência ativa com relação ao ângulo na barra final
                for m in fases:
                    no2 = nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    Bsh = Bshmatrix[fase, m] /baseY
                    tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra2*3) + m]
                    ang_estimado2 = vet_estados[(index_barra2*3) + m]
                    delta = tensao_estimada*tensao_estimada2*(Gs*np.sin(ang_estimado-ang_estimado2) - Bs*np.cos(ang_estimado-ang_estimado2))
                    jacobiana[medida_atual][(index_barra2*3) + m - 3] = delta
            
            medida_atual += 1
            
        return medida_atual
        
    def Derivadas_fluxo_pot_rat(self, jacobiana: np.array, fases: np.array, medida_atual: int, index_barra1: int, elemento: str,
                            barras: pd.DataFrame, nodes: dict, vet_estados: np.array, DSSCircuit, Ybus, baseva) -> int:
        barra1 = barras['nome_barra'][index_barra1]
        basekv = barras['Bases'][index_barra1]
        baseY = baseva / ((basekv*1000)**2)
        num_buses = DSSCircuit.NumBuses
        DSSCircuit.SetActiveElement(elemento)
        Bshmatrix = np.zeros((3, 3))
        barra2 = DSSCircuit.ActiveCktElement.BusNames[1]
        index_barra2 = barras[barras['nome_barra'] == barra2].index.values[0]
        
        for fase in fases:  
            no1 = nodes[barra1+f'.{fase+1}']

            tensao_estimada = vet_estados[(num_buses*3) + (index_barra1*3) + fase]
            ang_estimado = vet_estados[(index_barra1*3) + fase]
            
            #Derivada do fluxo de Potência reativa com relação a tensão na barra inicial
            for m in fases:
                no2 = nodes[barra2+f'.{m+1}']
                Yij = Ybus[no1, no2] / baseY
                Gs = np.real(Yij)
                Bs = np.imag(Yij)
                Bsh = Bshmatrix[fase, m] / baseY
                if m == fase:
                    delta = -tensao_estimada*(Bs+Bsh)
                    for n in fases:
                        no2 = nodes[barra2+f'.{n+1}']
                        Yij = Ybus[no1, no2] / baseY
                        Gs = np.real(Yij)
                        Bs = np.imag(Yij)
                        Bsh = Bshmatrix[fase, n] / baseY
                        tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra1*3) + n]
                        tensao_estimada3 = vet_estados[(num_buses*3) + (index_barra2*3) + n]
                        ang_estimado2 = vet_estados[(index_barra1*3) + n]
                        ang_estimado3 = vet_estados[(index_barra2*3) + n]
                        delta += tensao_estimada2*(Gs*np.sin(ang_estimado-ang_estimado2)-(Bs+Bsh)*np.cos(ang_estimado-ang_estimado2))
                        delta -= tensao_estimada3*(Gs*np.sin(ang_estimado-ang_estimado3)-Bs*np.cos(ang_estimado-ang_estimado3))

                else:
                    ang_estimado2 = vet_estados[(index_barra1*3) + m]
                    delta = tensao_estimada*(Gs*np.sin(ang_estimado-ang_estimado2)-(Bs+Bsh)*np.cos(ang_estimado-ang_estimado2))
                    
                jacobiana[medida_atual][num_buses*3 + (index_barra1*3) + m - 3] = delta
                
            if index_barra1 != 0:
                #Derivada do fluxo de Potência reativa com relação ao ângulo na barra inicial
                for m in fases:
                    no2 = nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    Bsh = Bshmatrix[fase, m] / baseY
                    if m == fase:
                        delta = -(tensao_estimada**2)*Gs
                        for n in fases:
                            no2 = nodes[barra2+f'.{n+1}']
                            Yij = Ybus[no1, no2] / baseY
                            Gs = np.real(Yij)
                            Bs = np.imag(Yij)
                            Bsh = Bshmatrix[fase, n] / baseY
                            tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra1*3) + n]
                            tensao_estimada3 = vet_estados[(num_buses*3) + (index_barra2*3) + n]
                            ang_estimado2 = vet_estados[(index_barra1*3) + n]
                            ang_estimado3 = vet_estados[(index_barra2*3) + n]
                            delta += tensao_estimada*tensao_estimada2*(Gs*np.cos(ang_estimado-ang_estimado2)+(Bs+Bsh)*np.sin(ang_estimado-ang_estimado2))
                            delta -= tensao_estimada*tensao_estimada3*(Gs*np.cos(ang_estimado-ang_estimado3)+Bs*np.sin(ang_estimado-ang_estimado3))
                        
                    else:
                        tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra1*3) + m]
                        ang_estimado2 = vet_estados[(index_barra1*3) + m]
                        delta = -tensao_estimada*tensao_estimada2*(Gs*np.cos(ang_estimado-ang_estimado2) + (Bs+Bsh)*np.sin(ang_estimado-ang_estimado2))
                    
                    jacobiana[medida_atual][(index_barra1*3) + m - 3] = delta
                
            #Derivada do fluxo de Potência reativa com relação a tensão na barra final
            for m in fases:
                no2 = nodes[barra2+f'.{m+1}']
                Yij = Ybus[no1, no2] / baseY
                Gs = np.real(Yij)
                Bs = np.imag(Yij)
                Bsh = Bshmatrix[fase, m] / baseY
                ang_estimado2 = vet_estados[(index_barra2*3) + m]
                delta = -tensao_estimada*(Gs*np.sin(ang_estimado-ang_estimado2)-Bs*np.cos(ang_estimado-ang_estimado2))
                jacobiana[medida_atual][num_buses*3 + (index_barra2*3) + m - 3] = delta
                
            if index_barra2 != 0:
                #Derivada do fluxo de Potência reativa com relação ao ângulo na barra final
                for m in fases:
                    no2 = nodes[barra2+f'.{m+1}']
                    Yij = Ybus[no1, no2] / baseY
                    Gs = np.real(Yij)
                    Bs = np.imag(Yij)
                    Bsh = Bshmatrix[fase, m] / baseY
                    tensao_estimada2 = vet_estados[(num_buses*3) + (index_barra2*3) + m]
                    ang_estimado2 = vet_estados[(index_barra2*3) + m]
                    delta = tensao_estimada*tensao_estimada2*(Gs*np.cos(ang_estimado-ang_estimado2) + Bs*np.sin(ang_estimado-ang_estimado2))
                    jacobiana[medida_atual][(index_barra2*3) + m - 3] = delta
                
            medida_atual += 1
        
        return medida_atual

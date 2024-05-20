import numpy as np
import pandas as pd

class Jacobiana():
    def __init__(self, vet_estados: np.array, baseva: float, barras: pd.DataFrame, nodes: dict, num_medidas: int) -> None:
        self.jacobiana = np.zeros((num_medidas, len(vet_estados)-6))
        self.jac_teste = np.zeros((1, len(vet_estados)-6))
        self.vet_estados = vet_estados
        self.baseva = baseva
        self.barras = barras
        self.nodes = nodes
            
    def inj_pot_at(self, tensao_estimada, tensao_estimada2, Gs, Bs, diff_angs, medida_atual, delta_t, delta_ang, count):
        #Com relação as tensoes
        d_tensoes = tensao_estimada * (Gs * np.cos(diff_angs) + Bs * np.sin(diff_angs))
        d_tensoes[0][count] = delta_t

        #Com relação aos angulos
        d_angulos = tensao_estimada * tensao_estimada2 * (Gs * np.sin(diff_angs) - Bs * np.cos(diff_angs))
        d_angulos[0][count] = delta_ang
        
        self.jac_teste = np.concatenate([self.jac_teste, np.concatenate([d_angulos[:,:-3], d_tensoes[:,:-3]], axis=1)])
        
        return medida_atual+1

    def inj_pot_rat(self, tensao_estimada, tensao_estimada2, Gs, Bs, diff_angs, medida_atual, delta_t, delta_ang, count):
        #Com relação as tensoes
        d_tensoes = tensao_estimada * (Gs * np.sin(diff_angs) - Bs * np.cos(diff_angs))
        d_tensoes[0][count] = delta_t
        
        #Com relação aos angulos
        d_angulos = -tensao_estimada * tensao_estimada2 * (Gs * np.cos(diff_angs) + Bs * np.sin(diff_angs))
        d_angulos[0][count] = delta_ang
        
        self.jac_teste = np.concatenate([self.jac_teste, np.concatenate([d_angulos, d_tensoes], axis=1)])

        return medida_atual+1

    def tensao(self, fases_barra: list):
        diag = [1 for _ in range(len(fases_barra)-3)]
        d_tensoes = np.diag(diag)
        d_angs = np.zeros((len(fases_barra)-3, len(fases_barra)-3))
        d_total = np.concatenate([d_angs, d_tensoes], axis=1)
        
        return d_total
        
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

import numpy as np
import pandas as pd

class Residuo():
    def __init__(self, barras: pd.DataFrame, tensoes, angs) -> None:
        self.barras = barras
        self.tensoes = tensoes
        self.angs = angs
        self.matriz_tensoes, self.matriz_angs = self.ajustar_entradas(self.tensoes, self.angs)

    def calc_res(self, Gs, Bs):
        diff_angs = self.angs - self.matriz_angs.T
        diff_angs = diff_angs.T
        inj_pot_at = []
        inj_pot_rat = []
        tensoes = []
        for fases, pot_at, pot_rat, tensao in zip(self.barras['Fases'], self.barras['Inj_pot_at'], self.barras['Inj_pot_rat'], self.barras['Tensao']):
            for fase in fases:
                inj_pot_at.append(pot_at[fase])
                inj_pot_rat.append(pot_rat[fase])
                tensoes.append(tensao[fase])
        
        #res_inj_pot_at = self.barras['Inj_pot_at'].to_numpy()
        self.inj_pot_at_est = self.tensoes[3:] * np.sum(self.matriz_tensoes * (Gs * np.cos(diff_angs) + Bs * np.sin(diff_angs)), axis=1)[3:]
        res_inj_pot_at = np.array(inj_pot_at)[:-3] - self.inj_pot_at_est
        
        #res_inj_pot_rat = self.barras['Inj_pot_rat'].to_numpy()
        self.inj_pot_rat_est = self.tensoes[3:] * np.sum(self.matriz_tensoes * (Gs * np.sin(diff_angs) - Bs * np.cos(diff_angs)), axis=1)[3:]
        res_inj_pot_rat = np.array(inj_pot_rat)[:-3] - self.inj_pot_rat_est

        #res_tensao = self.barras['Tensao'].to_numpy()
        res_tensao = np.array(tensoes)[:-3] - self.tensoes[3:]
        
        return np.concatenate([res_inj_pot_at, res_inj_pot_rat, res_tensao])
    
    def ajustar_entradas(self, tensoes: np.ndarray, angs: np.ndarray):
        #Cria matrizes cujas linhas são repetições dos vetores, pois é mais fácil manipular
        matriz_tensoes = np.array([tensoes for _ in range(len(tensoes))])
        matriz_angs = np.array([angs for _ in range(len(angs))])
        
        return matriz_tensoes, matriz_angs
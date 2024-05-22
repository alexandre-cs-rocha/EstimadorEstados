import numpy as np
import pandas as pd

class Residuo():
    def __init__(self, barras: pd.DataFrame) -> None:
        self.vet_inj_at = []
        self.vet_inj_rat = []
        self.vet_tensao = []
        self.inj_pot_at_est = []
        self.inj_pot_rat_est = []
        self.barras = barras

    def Residuo_inj_pot_at(self, index_barra: int, fase: int, tensao_estimada: float, tensoes,
                           diff_angulos: np.array, Gs: np.array, Bs: np.array) -> None:
        inj_pot_est = tensao_estimada * np.sum(tensoes * (Gs * np.cos(diff_angulos) + Bs * np.sin(diff_angulos)))
        inj_pot_med = self.barras['Inj_pot_at'][index_barra][fase]
        self.barras['Inj_pot_at_est'][index_barra][fase] = inj_pot_est
        self.inj_pot_at_est.append(inj_pot_est)
        self.vet_inj_at.append(inj_pot_med - inj_pot_est)
        
    def Residuo_inj_pot_rat(self, index_barra: int, fase: int, tensao_estimada: float, tensoes,
                            diff_angulos: np.array, Gs: np.array, Bs: np.array) -> None:
        inj_pot_est = tensao_estimada * np.sum(tensoes * (Gs * np.sin(diff_angulos) - Bs * np.cos(diff_angulos)))
        inj_pot_med = self.barras['Inj_pot_rat'][index_barra][fase]
        self.barras['Inj_pot_rat_est'][index_barra][fase] = inj_pot_est
        self.inj_pot_rat_est.append(inj_pot_est)
        self.vet_inj_rat.append(inj_pot_med - inj_pot_est)

    def Residuo_tensao(self, index_barra: int, fase: int, tensao_estimada: float) -> None:
        tensao = self.barras['Tensao'][index_barra][fase]
        self.vet_tensao.append(tensao - tensao_estimada)

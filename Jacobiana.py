import numpy as np
import pandas as pd
import scipy.sparse as sp

class Jacobiana():
    def __init__(self, tensoes: np.ndarray, angs: np.ndarray, fases: list) -> None:
        self.tensoes = tensoes
        self.angs = angs
        self.fases = fases
        self.matriz_tensoes, self.matriz_angs = self.ajustar_entradas(self.tensoes, self.angs)
        
    def derivadas(self, Gs: np.ndarray, Bs: np.ndarray, inj_pot_at_est: np.ndarray, inj_pot_rat_est: np.ndarray):
        diff_angs2 = self.angs - self.matriz_angs.T
        diff_angs2 = diff_angs2.T
        diff_angs = diff_angs2[:-3, :-3]
        Gsc, Bsc = Gs[3:, 3:], Bs[3:, 3:]
        Gs = np.concatenate([Gs[3:], Gs[:3]])
        #Sistemas maiores 'morrem' nessa linha, talvez esteja consumindo muita RAM.
        #Talvez append seja mais eficiente. Boa sorte eu do futuro.
        Gs = np.concatenate([Gs[:, 3:], Gs[:, :3]], axis=1)
        Bs = np.concatenate([Bs[3:], Bs[:3]])
        Bs = np.concatenate([Bs[:, 3:], Bs[:, :3]], axis=1)
        tensoes = self.tensoes[:-3]
        matriz_tensoes = self.matriz_tensoes[:-3, :-3]
        
        #Derivada da potência ativa com relação as tensoes
        H = (tensoes * (Gsc * np.cos(diff_angs) + Bsc * np.sin(diff_angs)).T).T
        delta_t = ((tensoes**2) * Gsc.diagonal() + inj_pot_at_est) / tensoes
        np.fill_diagonal(H, delta_t)
        
        #Derivada da potência ativa com relação aos ângulos
        N = tensoes * matriz_tensoes.T * (Gsc * np.sin(diff_angs) - Bsc * np.cos(diff_angs))
        delta_ang = (-Bsc.diagonal()*(tensoes**2))-tensoes*np.sum(self.matriz_tensoes*(Gs*np.sin(diff_angs2)-Bs*np.cos(diff_angs2)), axis=1)[:-3]
        np.fill_diagonal(N, delta_ang)
        
        #Derivada da potência reativa com relação as tensoes
        M = (tensoes * (Gsc * np.sin(diff_angs) - Bsc * np.cos(diff_angs)).T).T
        delta_t = ((tensoes**2)*(-Bsc.diagonal())+inj_pot_rat_est) / tensoes
        np.fill_diagonal(M, delta_t)

        #Derivada da potência reativa com relação aos ângulos
        L = -tensoes * matriz_tensoes.T * (Gsc * np.cos(diff_angs) + Bsc * np.sin(diff_angs))
        delta_ang = -Gsc.diagonal()*tensoes**2 + inj_pot_at_est
        np.fill_diagonal(L, delta_ang)
        
        T = self.tensao(self.fases)
        
        #Junta as matrizes na ordem correta
        jacobiana = np.concatenate([np.concatenate([N, H], axis=1), np.concatenate([L, M], axis=1), T])
         
        return jacobiana

    def tensao(self, fases: list):
        #Cria a matriz com as derivadas das tensões
        diag = [1 for _ in range(len(fases)-3)]
        d_tensoes = np.diag(diag)
        d_angs = np.zeros((len(fases)-3, len(fases)-3))
        d_total = np.concatenate([d_angs, d_tensoes], axis=1)
        
        return d_total

    def ajustar_entradas(self, tensoes: np.ndarray, angs: np.ndarray):
        #Cria matrizes cujas linhas são repetições dos vetores, pois é mais fácil manipular
        matriz_tensoes = np.array([tensoes for _ in range(len(tensoes))])
        matriz_angs = np.array([angs for _ in range(len(angs))])
        
        return matriz_tensoes, matriz_angs
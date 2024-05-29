from pathlib import Path
import numpy as np
import time
import EESD


def get_gabarito(eesd: EESD.EESD) -> np.array:
    ang = np.array([])
    tensoes = np.array([])
    for barra in eesd.DSSCircuit.AllBusNames:
        eesd.DSSCircuit.SetActiveBus(barra)
        ang = np.concatenate([ang, eesd.DSSCircuit.Buses.puVmagAngle[1::2]*2*np.pi / 360])
        tensoes = np.concatenate([tensoes, eesd.DSSCircuit.Buses.puVmagAngle[::2]])

    return np.concatenate([ang[3:], tensoes[3:]])

def main():
    #Achar o path do script do OpenDSS
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '123Bus' / 'IEEE123Master.dss'
    '4_SEAUA_1''Master_DU01_20201246_4_SEAUA_1_NTMBSR1PVTTR.dss'
    'Sulgipe''Master_DU01_20201246_1_SEAUA_1_NTMBSR1PVTTR.dss'
    
    verbose = False
    
    baseva =  33.3 * 10**6

    eesd = EESD.EESD(MasterFile, baseva, verbose)
    
    inicio = time.time()
    vet_estados = eesd.run(10**-5, 100)
    fim = time.time()
    print(f'Estimador concluido em {fim-inicio}s')

    gabarito = get_gabarito(eesd)
    
    if verbose:
        print(gabarito)
        print(vet_estados)
    
if __name__ == '__main__':
    main()

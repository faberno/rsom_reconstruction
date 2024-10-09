from rsom_reconstruction import SensitivityField, saft_munich_adapter
import numpy as np
from time import time

def main():

    start = time()
    sfield = SensitivityField(clip_method='hyperbola')
    saft_munich_adapter('/home/f841r/Desktop/rsom/data/fabian_data/Fabia_231220_Biederstein/20231220162218_RSOM_Fabian_Unterarm_532nm_RSOM50.mat',
                        sfield, verbose=False)
    print('Elapsed time:', time() - start)


if __name__ == '__main__':
    main()
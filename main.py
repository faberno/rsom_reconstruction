from cupy_reconstruction import SensitivityField, saft_munich_adapter
import numpy as np
from time import time

def main():

    start = time()
    sensitivity = SensitivityField(bandpass_freq_hz=(15e6, 42e6, 90e6))
    saft_munich_adapter(r'C:\Users\fabia\PycharmProjects\rsom_reconstruction_repo\20231220162218_RSOM_Fabian_Unterarm_532nm_RSOM50.mat',
                        sensitivity, verbose=True)
    print('Elapsed time:', time() - start)


if __name__ == '__main__':
    main()
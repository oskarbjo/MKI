import numpy as np
from scipy import interpolate
import functions

spec_freq = np.load("spectrum_frequencies_2748bunches_1e-9tbl_1.15e11Int_25nsBS.dat")
spec = np.load("spectrum_data_2748bunches_1e-9tbl_1.15e11Int_25nsBS.dat")
#file=open('MKI_PostLS1_impedance.txt','r')



[positionMatrix, dataMatrix] = functions.import_CST_losses()

positionMatrix = positionMatrix.transpose()

print(np.shape(dataMatrix))


#CSTlosses = [1.176278e-21,1.017526e-20,2.724059e-20,1.218005e-20,1.643405e-20,3.388451e-20,1.025789e-19,4.682089e-19,3.696159e-18,4.403920e-18,2.013538e-18,3.553055e-19,3.207171e-19,2.873545e-19,1.798530e-19]


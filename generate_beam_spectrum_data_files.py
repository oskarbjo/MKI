import numpy as np
import functions


Npts1 = 5
nB = [2592]
tBS = 25e-9
f0 = 11245
# tBL_values = np.around(np.linspace(1e-9,1.8e-9,Npts1), decimals=13)
tBL_values = np.around([1.1e-9],decimals=13)
filepath = 'C:/Users/objorkqv/eclipse-workspace/mki_powerloss/beam_spectrum_data/'

for i in nB:
    for j in tBL_values:
        [t,y] = functions.beamCurrentGauss(i,j,tBS)
        [fPos,ySpecPos] = functions.beamSpec(t,y)
        txt1 = str(j).replace('.','p')
        filename_freq = 'freq_beamSpectrum_' + str(i) + 'bunches_' + txt1 + 'bunchLength_25nsbunchSeparation.dat'
        filename_spec = 'spec_beamSpectrum_' + str(i) + 'bunches_' + txt1 + 'bunchLength_25nsbunchSeparation.dat'
        print(filepath+filename_freq)
        fPos.dump(filepath + filename_freq)
        ySpecPos.dump(filepath + filename_spec)
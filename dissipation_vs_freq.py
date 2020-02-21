import functions
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
# impFile=r"\\cernbox-smb.cern.ch\eos\user\o\objorkqv\Documents\CST_EOS\mkiPostLS1_2000MHz_results_mostContributingf0\mkiPostLS1_imp.txt" #Remember to remove impedance file header
#impFile=r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_PostLS1_impedance.txt' #Remember to remove impedance file header
#impFile=r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_upgraded_impedance.txt' #Remember to remove impedance file header
# impFile=r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_cool_impedance.txt' #Remember to remove impedance file header
# impFile = r'E:\CST\MKIcool\MKIcool remeshed\15mm insert_IMPEDANCE SIMULATION/MKIcool_impedance_REMESHED.txt'
# impFile = r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool_redrawn_WL_study\impedance curves\250m.txt"
#impFile=r'E:\CST\MKIcool\MKIcool remeshed\Damper optimization\simulated files\10mm/MKIcool_10mm_impedance.txt'
#impFile = r"E:\CST\MKIcool\MKIcool remeshed\Damper optimization\simulated files\20mm\MKIcool_20mm_inset_imp.txt"
#impFile=r'E:\CST\MKIcool\MKIcool remeshed\Insert design suggestion\design1/insert_design1_impedance.txt'
#impFile = r'E:\CST\MKIcool\MKIcool remeshed\Insert design suggestion\design2/insert_design2_imp.txt'
# impFile = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKI cool\MKI cool resonant measurements\resonant_impedance_from_python_script.txt"
# impFile  = r"E:\CST\MKPL\Lorena_Import\Impedance simu\real_impedance.txt"
# impFile = r"E:\CST\MKIcool\MKIcool_redrawn\Original MKIcool_redrawn\simulated\fixed_negative_impedance\MKIcool_fixed_negative_impedance.txt"
impFile = r"E:\CST\MKIcool\MKIcool_redrawn\MKI cool redrawn MASTER 3D loss monitors\simulated\impedance.txt"
echarge=1.602e-19


#LHC:
N = 1.15e11              #Nominal for current LHC: 1.15
nB = 2748                 #2748
tBL = 1e-9              #1e-9  
tBS = 25e-9             #25e-9
f0 = 11245              #11245
iB = N*nB*echarge*f0

# # SPS:
# N = 1.76e11              #Nominal for current LHC: 1.15
# nB = 72                 #2748
# tBL = 3e-9              #1e-9  
# tBS = 25e-9             #25e-9
# f0 = 43450              #11245
# iB = N*nB*echarge*f0


#LHC:
[fPos,ySpecPos] = functions.loadSpectrumFromFile(nB,tBL,tBS)
#SPS:
# [fPos,ySpecPos] = functions.SPS4batchBeam(nB, tBL, tBS)

[pLoss, fInt, pLoss_spect, spectSqr, zInt, YInt] = functions.RFLosses_spect(f0,iB,ySpecPos,fPos,impFile)

loss_v_f_11khz = []
for i in range(0,np.size(pLoss_spect)):
    loss_v_f_11khz.append(np.sum(pLoss_spect[0:i]))
fInt1 = fInt
YInt1 = YInt
zInt1 = zInt
 
 
[mostContributingf0, mostContributingf0_index, fInt, pLoss_spect, spectSqr, zInt] = functions.findMostContributingf0(ySpecPos,fPos, impFile, iB, f0)
loss_v_f_11khz_SELECTED = []
npts = 50
mostContributingf0_npts = mostContributingf0[0:npts]
ascendingFreq_index = sorted(range(len(mostContributingf0_npts)),key=lambda k: mostContributingf0_npts[k])
# ascendingFreq_index = ascendingFreq_index[::-1]

#For generating list of frequencies to paste in CST VBA macro:
f = mostContributingf0[ascendingFreq_index]
for i in range(len(f)):
    print('"' + str(f[i]/1e6) + '", _')

pLoss_spect_npts = pLoss_spect[mostContributingf0_index[0:npts]]
pLoss_spect_npts = pLoss_spect_npts[ascendingFreq_index]
for i in range(0,npts):
    loss_v_f_11khz_SELECTED.append(np.sum(pLoss_spect_npts[0:i]))

print(mostContributingf0)



plt.figure()
plt.plot(fInt,pLoss_spect/np.max(pLoss_spect))
plt.plot(fInt,zInt/np.max(zInt))
plt.plot(fInt,spectSqr/np.max(spectSqr),linestyle='--')
plt.show()

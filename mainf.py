#temperature_result_plots

import functions
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#mat1 = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_5bunches.dat")
#impFile=r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_PostLS1_impedance.txt' #Remember to remove impedance file header
impFile=r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_cool_impedance.txt' #Remember to remove impedance file header
impFile2 = r'E:\CST\MKIcool\MKIcool remeshed\15mm insert_IMPEDANCE SIMULATION/MKIcool_impedance_REMESHED.txt'


echarge=1.602e-19
N = 1.8e11             #Nominal for current LHC: 1.15
nB = 2808                 #2748
tBL = 1.0e-9              #1e-9  
tBS = 25e-9             #25e-9
f0 = 11245              #11245
iB = N*nB*echarge*f0

#Temperature data
# from 1D power deposition data
# PLS1 stands for post-LS1 design
PdataPLS1 = [0, 35, 51, 127, 254, 508]
TdataPLS1Yoke = [22, 57, 73, 126, 213, 330]
# Up stands for upgraded design
PdataUp = [0, 37, 92.5, 185, 370]
TdataUpRing = [22, 117, 212, 318, 462]
TdataUpYoke = [22, 63, 107, 154, 218]




#[t,y] = functions.beamCurrentGauss(nB,tBL,tBS)
#print('Time domain beam function computed')
#[fPos,ySpecPos] = functions.beamSpec(t,y)
#print('Beam spectrum computed')
#[pLoss, fImp, zImp] = functions.RFLosses(f0,iB,ySpecPos,fPos,impFile)
#print('Total power losses computed')

#print(pLoss)
 
#[P_mat, tBL_values, N_values]=functions.tBL_and_N_sweep(nB,tBS,f0,impFile)
#print('P matrix computed')
#T_mat=functions.T_vs_tBL_and_N(P_mat,PdataPLS1,TdataPLS1Yoke)
#print('T matrix computed')
#X_tbl, Y_N = np.meshgrid(tBL_values,N_values)
#X_tbl=X_tbl.transpose()
#Y_N=Y_N.transpose()  

#[P_array, N] = functions.nB_sweep(N,tBS,f0,impFile)
#T_array = functions.getTempFromP(2.5*P_array,PdataUp,TdataUpRing)
#P_array.dump("P_PostLS1_P_vs_nB_mike_plot.dat")

#Write data:
#P_mat.dump("P_mat_PostLS1_yoke_2748bunches.dat")
#T_mat.dump("T_mat_PostLS1_yoke_2748bunches.dat")
#tBL_values.dump("tBL_PostLS1_yoke_2748bunches.dat")
#N_values.dump("N_PostLS1_yoke_2748bunches.dat")


[fPos,ySpecPos] = functions.loadSpectrumFromFile(nB,tBL,tBS)
[pLoss1, fImp1, zImp1] = functions.RFLosses(f0, iB, ySpecPos, fPos, impFile)
[pLoss2, fImp2, zImp2] = functions.RFLosses(f0, iB, ySpecPos, fPos, impFile2)

[fInt2,zInt2]=functions.importImpedanceData(impFile)
[fInt3,zInt3]=functions.importImpedanceData(impFile2)

fig2 = plt.figure(1)
plt.plot(fInt2,zInt2)
plt.plot(fInt3,zInt3)

 
#fig4 = plt.figure(3)
#plt.plot(fImp,zImp)
#plt.xlim(0,np.max(fImp))
  
#fig5 = plt.figure(4)
#plt.plot(PdataPLS1,TdataPLS1Yoke)
#plt.plot(PdataUp,TdataUpYoke)
  
# fig = plt.figure(5)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X_tbl, Y_N, T_mat, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
 
plt.show()



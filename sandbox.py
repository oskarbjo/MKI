
import matplotlib.pyplot as plt
import numpy as np
import functions
import pandas as pd
echarge=1.602e-19
N = 1.8e11              #Nominal for current LHC: 1.15
nB = 72                 #2748
tBL = 2e-9              #1e-9  
tBS = 25e-9             #25e-9
f0 = 43450              #11245
iB = N*nB*echarge*f0

# SPS 4 batch fill:
beam1 = functions.beamCurrentGauss(nB, tBL, tBS)
beam2 = np.zeros(round(len(beam1[1][:])/72)*8)
beam3 = np.concatenate((beam1[1][:],beam2))
beam4 = np.concatenate((beam3,beam3))
beam5 = np.concatenate((beam4,beam4))
beam6 = np.zeros(round(len(beam1[1][:])/72)*612)
beam7 = np.concatenate((beam5,beam6))

t1 = np.linspace(0,25*925e-9,len(beam7))
t2 = np.linspace(0,2*25*80e-9,len(beam4))


plt.figure()
plt.plot(t1,beam7)

[fPos,ySpecPos] = functions.beamSpec(t1,beam7)
[fPos,ySpecPos] = functions.beamSpec(beam1[0][:],beam1[1][:])

plt.figure()
plt.plot(fPos,np.abs(ySpecPos))
plt.show()







# def makeComplexArray(real,imag):
#     imag_unit=np.array([1j])
#     complex = real + imag_unit*imag
#     return complex
# 
# filepath1 = r"E:\CST\MKIcool\MKIcool remeshed\15mm insert_IMPEDANCE SIMULATION\MKIcool_impedance_REMESHED.txt"
# filepath2 = r"E:\CST\MKIcool\MKIcool remeshed\Crack model\MKI_cool_crack_model_simplified_IMP.txt"
# 
# [f1,r1] = functions.importImpedanceData(filepath1)
# [f2,r2] = functions.importImpedanceData(filepath2)
# 
# 
# plt.figure()
# plt.plot(f1,r1)
# plt.plot(f2,r2)
# plt.legend(['No crack','With crack'])
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Longitudinal impedance [Ohm]')
# plt.grid()
# 
# 
# Ltot = 2*np.pi*0.043
# mu0=1.25663706e-06
# muf=745
# A=0.01*0.06
# Lag=np.linspace(0,0.002,100)
# Lf=Ltot-Lag
# Rag=Lag/(mu0*A)
# Rf=Lf/(mu0*muf*A)
# Rtot=Rag+Rf
# mutot = Ltot/(mu0*Rtot*A)
# 
# 
# 
# plt.figure()
# plt.plot(Lag*1000,mutot)
# plt.xlabel('Crack width [mm]')
# plt.ylabel('Equivalent mu')
# plt.grid()
# 
# 
# 
# #Same thing with complex mu
# mu_prim_data = pd.read_excel("E:\CST\MKIcool\MKIcool remeshed\Crack model\mu_prim.xlsx")
# mu_prim = np.asarray(mu_prim_data.iloc(1)[1])
# mu_freq = np.asarray(mu_prim_data.iloc(1)[0])
# 
# mu_bis_data = pd.read_excel("E:\CST\MKIcool\MKIcool remeshed\Crack model\mu_bis.xlsx")
# mu_bis = np.asarray(mu_bis_data.iloc(1)[1])
# 
# mu_compl = makeComplexArray(mu_prim,-mu_bis)
# 
# 
# Ltot = 2*np.pi*0.043
# mu0=1.25663706e-06
# muf=mu_compl
# A=0.01*0.06
# Lag=0.0005
# Lf=Ltot-Lag
# Rag=Lag/(mu0*A)
# Rf=Lf/(mu0*muf*A)
# Rtot=Rag+Rf
# mutot = Ltot/(mu0*Rtot*A)
# 
# 
# 
# plt.figure()
# plt.plot(mu_freq,np.abs(muf))
# plt.plot(mu_freq,np.abs(mutot))
# 
# 
# 
# plt.show()

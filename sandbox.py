
import matplotlib.pyplot as plt
import numpy as np
import functions
import pandas as pd
Nbunches = 1100
Fmin = 0
Fmax = 2e9
Npts = Nbunches*1000-900
sigma = 1e-9

def makeComplexArray(real,imag):
    imag_unit=np.array([1j])
    complex = real + imag_unit*imag
    return complex

filepath1 = r"E:\CST\MKIcool\MKIcool remeshed\15mm insert_IMPEDANCE SIMULATION\MKIcool_impedance_REMESHED.txt"
filepath2 = r"E:\CST\MKIcool\MKIcool remeshed\Crack model\MKI_cool_crack_model_simplified_IMP.txt"

[f1,r1] = functions.importImpedanceData(filepath1)
[f2,r2] = functions.importImpedanceData(filepath2)


plt.figure()
plt.plot(f1,r1)
plt.plot(f2,r2)
plt.legend(['No crack','With crack'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Longitudinal impedance [Ohm]')
plt.grid()


Ltot = 2*np.pi*0.043
mu0=1.25663706e-06
muf=745
A=0.01*0.06
Lag=np.linspace(0,0.002,100)
Lf=Ltot-Lag
Rag=Lag/(mu0*A)
Rf=Lf/(mu0*muf*A)
Rtot=Rag+Rf
mutot = Ltot/(mu0*Rtot*A)



plt.figure()
plt.plot(Lag*1000,mutot)
plt.xlabel('Crack width [mm]')
plt.ylabel('Equivalent mu')
plt.grid()



#Same thing with complex mu
mu_prim_data = pd.read_excel("E:\CST\MKIcool\MKIcool remeshed\Crack model\mu_prim.xlsx")
mu_prim = np.asarray(mu_prim_data.iloc(1)[1])
mu_freq = np.asarray(mu_prim_data.iloc(1)[0])

mu_bis_data = pd.read_excel("E:\CST\MKIcool\MKIcool remeshed\Crack model\mu_bis.xlsx")
mu_bis = np.asarray(mu_bis_data.iloc(1)[1])

mu_compl = makeComplexArray(mu_prim,-mu_bis)


Ltot = 2*np.pi*0.043
mu0=1.25663706e-06
muf=mu_compl
A=0.01*0.06
Lag=0.0005
Lf=Ltot-Lag
Rag=Lag/(mu0*A)
Rf=Lf/(mu0*muf*A)
Rtot=Rag+Rf
mutot = Ltot/(mu0*Rtot*A)



plt.figure()
plt.plot(mu_freq,np.abs(muf))
plt.plot(mu_freq,np.abs(mutot))



plt.show()

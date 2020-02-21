
import matplotlib.pyplot as plt
import numpy as np
import functions
import pandas as pd


c0 = 299792458
eps0 = 8.85418782e-12



def skin(f):
    mu0 = 1.25663706e-6
    gamma = 5.8e7 #copper
    skinD = np.sqrt(1/(np.pi*f*mu0*gamma))
    return skinD


def calculateResonantImpedance(filepath):
    
    
    data = np.loadtxt(filepath, comments='"', delimiter=',')
    
    f = data[:,0]
    QL = data[:,1]
    S21db = data[:,2]
    
    # choose 0 for the outer conductor loss increase since that's what we want
    # to measure
    rho = 17.24e-9 # copper resisitivity [Ohm*m] at 20 degrees celsius
    # resistivity values in literature between 1.5% (altes taschenbuch der hf)
    # and 4% lower (web) the given by fontolliet
    rho_oc = 1/3.3e7 # RESISTIVITY OF OUTER CONDUCTOR [Ohm*m], alu
    
    D = 0.045           # pipe diameter [m]
    d = 0.0005          # wire diameter [m]
    Lwire = 3.551       # Wire length
    Lcomponent = 3.55   # DUT length
    Z0 = 60*np.log(D/d) # Characteristic impedance of beam pipe
    
    S21 = np.power(10,S21db/20) # Linear S21
    
    # coupling coefficient
    k = S21/(1-S21)
    
    # unloaded quality factor
    Q0 = QL * (1+k)
    
    # measured attenuation [dB/m]
    alpha_m = 8.686*np.pi*f/(c0*Q0)
    
    # skin depth [m]
    skin_depth = skin(f)
    
    # finite skin depth correction for small frequencies according to altes
    # taschenbuch der hf, page 10, valid for copper for d>4*skindepth => f>2e5 Hz
    skin_corr = (d+skin_depth)/d
    
    # line attenuation of copper [dB/m]
    alpha_c = skin_corr * 8.686 * np.sqrt(np.pi*eps0*rho*f) / np.log(D/d) * ((1/d) + (1/D)*np.sqrt(rho/rho_oc)) # Fontolliet
    
    R0 = 4*Lwire*rho/(np.pi*d*d)
    R = R0*d/(4*skin_depth)*skin_corr
    
    Z = R*(alpha_m-alpha_c)/alpha_c # Real part of longitudinal impedance    
    
    return [f,Z]


def main():    
#     filepath = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKI cool\MKI cool resonant measurements\MKIcool_resonant_1600pts"
#     [f,Z] = calculateResonantImpedance(filepath)
#     filepath2 = r"F:\MKIcool\Modified MKI cool measurement\MKI_modified.txt"
#     [f0,Z0] = calculateResonantImpedance(filepath2)
#     [f1,R1]=functions.importImpedanceData(r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_cool_impedance.txt')
#     [f2,R2]=functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\Original MKIcool_redrawn\simulated\initial/origina_MKIcool_redrawn_impedance.txt")
#     [f3,R3]=functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\Original MKIcool_redrawn\simulated\improved\MKIcool_redrawn_improved_impedance.txt")
#     [f4,R4]=functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\Original MKIcool_redrawn\simulated\with metal ring\long_impedance.txt")
#     [f5,R5]=functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\Alumina tube reduced epsilon\simulated\MKIcool_reduced_epsilon_impedance.txt")
#     [f6,R6]=functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\Original MKIcool_redrawn\simulated\fixed_negative_impedance\MKIcool_fixed_negative_impedance.txt")
#     plt.figure(num=None, figsize=(9, 5), dpi=140, facecolor='w', edgecolor='k')
# #     plt.plot(f/1e9,Z)
# #     plt.plot(np.array(f)/1e9,Z)
# #     plt.plot(np.asarray(f1)/1e9,R1)
#     plt.plot(f0/1e9,Z0)
#     plt.plot(np.asarray(f5)/1e9,R5)
#     plt.plot(np.asarray(f6)/1e9,R6)
# #     plt.plot(f3,R3)
#     plt.grid()
#     plt.ylabel('Impedance [Ohms]')
#     plt.xlabel('Frequency [GHz]')
#     plt.title('MKI cool longitudinal impedance')
#     plt.legend(['Measurement','Simulation (epsilon = 7.5)','Simulation (epsilon = 10)'])
    WLstudy()
    plt.show()


def WLstudy():
#     [f0,R0] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool_redrawn_WL_study\impedance curves\100m.txt")
#     [f1,R1] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool_redrawn_WL_study\impedance curves\150m.txt")
#     [f2,R2] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool_redrawn_WL_study\impedance curves\200m.txt")
#     [f3,R3] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool_redrawn_WL_study\impedance curves\250m.txt")


    [f0,R0] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool_redrawn_WL_study\impedance curves\250m.txt")
    [f1,R1] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool endcap study\64mm\simulated\impedance_64mm_endcap_hole.txt")
    [f2,R2] = functions.importImpedanceData(r"E:\CST\MKIcool\MKIcool_redrawn\MKIcool endcap study\68mm\simulated\impedance_68mm_endcap_hole.txt")
    
    plt.figure(num=None, figsize=(10, 6), dpi=140, facecolor='w', edgecolor='k')
    plt.plot(np.asarray(f0)/1e6,R0)
    plt.plot(np.asarray(f1)/1e6,R1)
    plt.plot(np.asarray(f2)/1e6,R2)
#     plt.plot(np.asarray(f3)/1e6,R3)
    plt.legend(['Original cutout (60 mm)','64 mm','68 mm'])
    plt.grid()
    plt.ylabel('Impedance [Ohms]')
    plt.xlabel('Frequency [MHz]')
    
    
if __name__ == "__main__":
    main()
    

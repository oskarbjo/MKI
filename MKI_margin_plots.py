

import functions
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate

Nint = 100

tbl = np.load('tBL_mike_plot.dat')
tbl_int = np.linspace(1e-9,1.8e-9,Nint)
T_PostLS1_T_vs_tbl = np.load('T_PostLS1_yoke_T_vs_tbl_mike_plot.dat')
T_upgraded_yoke_T_vs_tbl = np.load('T_Upgraded_yoke_T_vs_tbl_mike_plot.dat')
T_upgraded_ring_T_vs_tbl = np.load('T_Upgraded_ring_T_vs_tbl_mike_plot.dat')
T_PostLS1_T_temp_T_vs_tbl = interpolate.interp1d(tbl,T_PostLS1_T_vs_tbl[:,0])
T_upgraded_yoke_temp_T_vs_tbl = interpolate.interp1d(tbl,T_upgraded_yoke_T_vs_tbl[:,0])
T_upgraded_ring_temp_T_vs_tbl = interpolate.interp1d(tbl,T_upgraded_ring_T_vs_tbl[:,0])
T_PostLS1_int_T_vs_tbl = T_PostLS1_T_temp_T_vs_tbl(tbl_int)
T_upgraded_yoke_int_T_vs_tbl = T_upgraded_yoke_temp_T_vs_tbl(tbl_int)
T_upgraded_ring_int_T_vs_tbl = T_upgraded_ring_temp_T_vs_tbl(tbl_int)


nB = np.load('nB_PostLS1_mike_plot.dat')
nB_int = np.linspace(np.min(nB),np.max(nB),Nint)
T_PostLS1_T_vs_nB = np.load('T_PostLS1_yoke_T_vs_nB_mike_plot.dat')
T_Upgraded_yoke_T_vs_nB = np.load('T_Upgraded_yoke_T_vs_nB_mike_plot.dat')
T_Upgraded_ring_T_vs_nB = np.load('T_Upgraded_ring_T_vs_nB_mike_plot.dat')
T_PostLS1_T_vs_nB_temp = interpolate.interp1d(nB,T_PostLS1_T_vs_nB[:,0])
T_Upgraded_yoke_T_vs_nB_temp = interpolate.interp1d(nB,T_Upgraded_yoke_T_vs_nB[:,0])
T_Upgraded_ring_T_vs_nB_temp = interpolate.interp1d(nB,T_Upgraded_ring_T_vs_nB[:,0])
T_PostLS1_T_vs_nB_int = T_PostLS1_T_vs_nB_temp(nB_int)
T_Upgraded_yoke_T_vs_nB_int = T_Upgraded_yoke_T_vs_nB_temp(nB_int)
T_Upgraded_ring_T_vs_nB_int = T_Upgraded_ring_T_vs_nB_temp(nB_int)

N = np.load('N_T_vs_N_mike_plot.dat')
N_int = np.linspace(np.min(N),np.max(N),Nint)
T_PostLS1_T_vs_N = np.load('T_PostLS1_yoke_T_vs_N_mike_plot.dat')
T_Upgraded_yoke_T_vs_N = np.load('T_Upgrade_yoke_T_vs_N_mike_plot.dat')
T_Upgraded_ring_T_vs_N = np.load('T_Upgrade_ring_T_vs_N_mike_plot.dat')
T_PostLS1_T_vs_N_temp = interpolate.interp1d(N,T_PostLS1_T_vs_N[:,0])
T_Upgraded_yoke_T_vs_N_temp = interpolate.interp1d(N,T_Upgraded_yoke_T_vs_N[:,0])
T_Upgraded_ring_T_vs_N_temp = interpolate.interp1d(N,T_Upgraded_ring_T_vs_N[:,0])
T_PostLS1_T_vs_N_int = T_PostLS1_T_vs_N_temp(N_int)
T_Upgraded_yoke_T_vs_N_int = T_Upgraded_yoke_T_vs_N_temp(N_int)
T_Upgraded_ring_T_vs_N_int = T_Upgraded_ring_T_vs_N_temp(N_int)


for i in range(0,Nint):
    if T_PostLS1_int_T_vs_tbl[i] > 125:
        T_PostLS1_int_T_vs_tbl[i] = 125
    if T_upgraded_yoke_int_T_vs_tbl[i] > 125:
        T_upgraded_yoke_int_T_vs_tbl[i] = 125
    if T_upgraded_ring_int_T_vs_tbl[i] > 250:
        T_upgraded_ring_int_T_vs_tbl[i] = 250
    if T_PostLS1_T_vs_nB_int[i] > 125:
        T_PostLS1_T_vs_nB_int[i] = 125
    if T_Upgraded_yoke_T_vs_nB_int[i] > 125:
        T_Upgraded_yoke_T_vs_nB_int[i] = 125
    if T_Upgraded_ring_T_vs_nB_int[i] > 250:
        T_Upgraded_ring_T_vs_nB_int[i] = 250
    if T_PostLS1_T_vs_N_int[i] > 125:
        T_PostLS1_T_vs_N_int[i] = 125
    if T_Upgraded_yoke_T_vs_N_int[i] > 125:
        T_Upgraded_yoke_T_vs_N_int[i] = 125
    if T_Upgraded_ring_T_vs_N_int[i] > 250:
        T_Upgraded_ring_T_vs_N_int[i] = 250
        
lineWidth = 3.0
plt.figure(num=0, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
plt.plot(tbl_int*1e9,T_PostLS1_int_T_vs_tbl, linewidth=lineWidth)
plt.plot(tbl_int*1e9,T_upgraded_ring_int_T_vs_tbl, linewidth=lineWidth)
plt.plot(tbl_int*1e9,T_upgraded_yoke_int_T_vs_tbl, linewidth=lineWidth)
plt.legend(['PostLS1-Yokes','Upgraded-Rings','Upgraded-Yokes'])
plt.grid()
plt.title('1.8e11 ppb, 2808 bunches')
plt.xlabel('Bunch length $(4\sigma)$ [ns]')
plt.ylabel('Max temp [C]')
plt.xlim([1,1.8])
plt.ylim([75, 275])


plt.figure(num=1, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
plt.plot(nB_int,T_PostLS1_T_vs_nB_int, linewidth=lineWidth)
plt.plot(nB_int,T_Upgraded_ring_T_vs_nB_int, linewidth=lineWidth)
plt.plot(nB_int,T_Upgraded_yoke_T_vs_nB_int, linewidth=lineWidth)
plt.legend(['PostLS1-Yokes','Upgraded-Rings','Upgraded-Yokes'])
plt.grid()
plt.title('1.8e11 ppb, 1.2ns')
plt.xlabel('Number of bunches')
plt.ylabel('Max temp [C]')
plt.xlim([np.min(nB_int),np.max(nB_int)])
plt.ylim([75, 250])


plt.figure(num=2, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
plt.plot(N_int/1e11,T_PostLS1_T_vs_N_int, linewidth=lineWidth)
plt.plot(N_int/1e11,T_Upgraded_ring_T_vs_N_int, linewidth=lineWidth)
plt.plot(N_int/1e11,T_Upgraded_yoke_T_vs_N_int, linewidth=lineWidth)
plt.legend(['PostLS1-Yokes','Upgraded-Rings','Upgraded-Yokes'])
plt.grid()
plt.title('2592 bunches, 1.1ns')
plt.xlabel('Intensity (1e11)')
plt.ylabel('Max temp [C]')
plt.xlim([np.min(N_int/1e11),np.max(N_int/1e11)])
plt.ylim([75, 250])


plt.show()
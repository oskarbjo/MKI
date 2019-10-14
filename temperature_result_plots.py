import functions
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate
from matplotlib.pyplot import xlim
#Temperature data
# from 1D power deposition data
# PLS1 stands for post-LS1 design
PdataPLS1 = [0, 35, 51, 127, 254, 508]
TdataPLS1Yoke = [22, 57, 73, 126, 213, 330]
# Up stands for upgraded design
PdataUp = [0, 37, 92.5, 185, 370]
TdataUpRing = [22, 117, 212, 318, 462]
TdataUpYoke = [22, 63, 107, 154, 218]

P_mat_PostLS1Yoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_PostLS1_yoke_2000bunches.dat")
T_mat_PostLS1Yoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/T_mat_PostLS1_yoke_2000bunches.dat")
tBL_PostLS1Yoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/tBL_PostLS1_yoke_2000bunches.dat")
N_PostLS1Yoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/N_PostLS1_yoke_2000bunches.dat")

P_mat_PostLS1Yoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_PostLS1_yoke_2748bunches.dat")
T_mat_PostLS1Yoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/T_mat_PostLS1_yoke_2748bunches.dat")
tBL_PostLS1Yoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/tBL_PostLS1_yoke_2748bunches.dat")
N_PostLS1Yoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/N_PostLS1_yoke_2748bunches.dat")

P_mat_UpgradedYoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_upgraded_yoke_2748bunches.dat")
T_mat_UpgradedYoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/T_mat_upgraded_yoke_2748bunches.dat")
tBL_UpgradedYoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/tBL_upgraded_yoke_2748bunches.dat")
N_UpgradedYoke_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/N_upgraded_yoke_2748bunches.dat")

P_mat_UpgradedYoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_upgraded_yoke_2000bunches.dat")
T_mat_UpgradedYoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/T_mat_upgraded_yoke_2000bunches.dat")
tBL_UpgradedYoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/tBL_upgraded_yoke_2000bunches.dat")
N_UpgradedYoke_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/N_upgraded_yoke_2000bunches.dat")

P_mat_UpgradedRing_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_upgraded_ring_2000bunches.dat")
T_mat_UpgradedRing_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/T_mat_upgraded_ring_2000bunches.dat")
tBL_UpgradedRing_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/tBL_upgraded_ring_2000bunches.dat")
N_UpgradedRing_2000b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/N_upgraded_ring_2000bunches.dat")

P_mat_UpgradedRing_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/P_mat_upgraded_ring_2748bunches.dat")
T_mat_UpgradedRing_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/T_mat_upgraded_ring_2748bunches.dat")
tBL_UpgradedRing_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/tBL_upgraded_ring_2748bunches.dat")
N_UpgradedRing_2748b = np.load("C:/Users/objorkqv/eclipse-workspace/mki_powerloss/N_upgraded_ring_2748bunches.dat")



lim1 = 1.05
lim2 = 0.95
yokeCurie = 125
ringCurie = 250


#Include 2.5 factor
T_mat_PostLS1Yoke_2748b=functions.T_vs_tBL_and_N(np.multiply(P_mat_PostLS1Yoke_2748b,2.5*0.8),PdataPLS1,TdataPLS1Yoke) #Factor 0.8 for all magnets but the previous 8D one that also was a PostLS1
T_mat_PostLS1Yoke_2000b=functions.T_vs_tBL_and_N(np.multiply(P_mat_PostLS1Yoke_2000b,2.5*0.8),PdataPLS1,TdataPLS1Yoke)
T_mat_UpgradedYoke_2748b=functions.T_vs_tBL_and_N(np.multiply(P_mat_UpgradedYoke_2748b,2.5),PdataUp,TdataUpYoke)
T_mat_UpgradedYoke_2000b=functions.T_vs_tBL_and_N(np.multiply(P_mat_UpgradedYoke_2000b,2.5),PdataUp,TdataUpYoke)
T_mat_UpgradedRing_2748b=functions.T_vs_tBL_and_N(np.multiply(P_mat_UpgradedRing_2748b,2.5),PdataUp,TdataUpRing)
T_mat_UpgradedRing_2000b=functions.T_vs_tBL_and_N(np.multiply(P_mat_UpgradedRing_2000b,2.5),PdataUp,TdataUpRing)


T_mat_PostLS1Yoke_2000b_limit = np.int_(T_mat_PostLS1Yoke_2000b > yokeCurie)
T_mat_PostLS1Yoke_2748b_limit = np.int_(T_mat_PostLS1Yoke_2748b > yokeCurie)
T_mat_UpgradedYoke_2748b_limit = np.int_(T_mat_UpgradedYoke_2748b > yokeCurie)
T_mat_UpgradedYoke_2000b_limit = np.int_(T_mat_UpgradedYoke_2000b > yokeCurie)
T_mat_UpgradedRing_2748b_limit = np.int_(T_mat_UpgradedRing_2748b > ringCurie)
T_mat_UpgradedRing_2000b_limit = np.int_(T_mat_UpgradedRing_2000b > ringCurie)

PostLS1Yoke_limit_indices_2748b = []
PostLS1Yoke_limit_indices_2000b = []
UpgradedYoke_limit_indices_2748b = []
UpgradedYoke_limit_indices_2000b = []
UpgradedRing_limit_indices_2748b = []
UpgradedRing_limit_indices_2000b = []


for i in range(np.size(T_mat_PostLS1Yoke_2748b[:][0])):
    for j in range(0,np.size(T_mat_PostLS1Yoke_2748b[0][:])):
        if T_mat_PostLS1Yoke_2748b[i][j]/yokeCurie < lim1 and T_mat_PostLS1Yoke_2748b[i][j]/yokeCurie > lim2: 
            PostLS1Yoke_limit_indices_2748b.append([i,j])

temp_ind = np.array(PostLS1Yoke_limit_indices_2748b)
PostLS1Yoke_limit_indices_2748b = temp_ind.transpose()

for i in range(np.size(T_mat_PostLS1Yoke_2000b[:][0])):
    for j in range(0,np.size(T_mat_PostLS1Yoke_2000b[0][:])):
        if T_mat_PostLS1Yoke_2000b[i][j]/yokeCurie < lim1 and T_mat_PostLS1Yoke_2000b[i][j]/yokeCurie > lim2: 
            PostLS1Yoke_limit_indices_2000b.append([i,j])

temp_ind = np.array(PostLS1Yoke_limit_indices_2000b)
PostLS1Yoke_limit_indices_2000b = temp_ind.transpose()

for i in range(np.size(T_mat_UpgradedYoke_2748b[:][0])):
    for j in range(0,np.size(T_mat_UpgradedYoke_2748b[0][:])):
        if T_mat_UpgradedYoke_2748b[i][j]/yokeCurie < lim1 and T_mat_UpgradedYoke_2748b[i][j]/yokeCurie > lim2: 
            UpgradedYoke_limit_indices_2748b.append([i,j])

temp_ind = np.array(UpgradedYoke_limit_indices_2748b)
UpgradedYoke_limit_indices_2748b = temp_ind.transpose()

for i in range(np.size(T_mat_UpgradedYoke_2000b[:][0])):
    for j in range(0,np.size(T_mat_UpgradedYoke_2000b[0][:])):
        if T_mat_UpgradedYoke_2000b[i][j]/yokeCurie < lim1 and T_mat_UpgradedYoke_2000b[i][j]/yokeCurie > lim2: 
            UpgradedYoke_limit_indices_2000b.append([i,j])

temp_ind = np.array(UpgradedYoke_limit_indices_2000b)
UpgradedYoke_limit_indices_2000b = temp_ind.transpose()

for i in range(np.size(T_mat_UpgradedRing_2748b[:][0])):
    for j in range(0,np.size(T_mat_UpgradedRing_2748b[0][:])):
        if T_mat_UpgradedRing_2748b[i][j]/ringCurie < lim1 and T_mat_UpgradedRing_2748b[i][j]/ringCurie > lim2: 
            UpgradedRing_limit_indices_2748b.append([i,j])

temp_ind = np.array(UpgradedRing_limit_indices_2748b)
UpgradedRing_limit_indices_2748b = temp_ind.transpose()

for i in range(np.size(T_mat_UpgradedRing_2000b[:][0])):
    for j in range(0,np.size(T_mat_UpgradedRing_2000b[0][:])):
        if T_mat_UpgradedRing_2000b[i][j]/ringCurie < lim1 and T_mat_UpgradedRing_2000b[i][j]/ringCurie > lim2: 
            UpgradedRing_limit_indices_2000b.append([i,j])

temp_ind = np.array(UpgradedRing_limit_indices_2000b)
UpgradedRing_limit_indices_2000b = temp_ind.transpose()

X_tbl_PostLS1Yoke_2748b, Y_N_PostLS1Yoke_2748b = np.meshgrid(tBL_PostLS1Yoke_2748b,N_PostLS1Yoke_2748b)
X_tbl_PostLS1Yoke_2748b=X_tbl_PostLS1Yoke_2748b.transpose()
Y_N_PostLS1Yoke_2748b=Y_N_PostLS1Yoke_2748b.transpose() 

X_tbl_PostLS1Yoke_2000b, Y_N_PostLS1Yoke_2000b = np.meshgrid(tBL_PostLS1Yoke_2000b,N_PostLS1Yoke_2000b)
X_tbl_PostLS1Yoke_2000b=X_tbl_PostLS1Yoke_2000b.transpose()
Y_N_PostLS1Yoke_2000b=Y_N_PostLS1Yoke_2000b.transpose() 

X_tbl_UpgradedYoke_2748b, Y_N_UpgradedYoke_2748b = np.meshgrid(tBL_UpgradedYoke_2748b,N_UpgradedYoke_2748b)
X_tbl_UpgradedYoke_2748b=X_tbl_UpgradedYoke_2748b.transpose()
Y_N_UpgradedYoke_2748b=Y_N_UpgradedYoke_2748b.transpose() 

X_tbl_UpgradedYoke_2000b, Y_N_UpgradedYoke_2000b = np.meshgrid(tBL_UpgradedYoke_2000b,N_UpgradedYoke_2000b)
X_tbl_UpgradedYoke_2000b=X_tbl_UpgradedYoke_2000b.transpose()
Y_N_UpgradedYoke_2000b=Y_N_UpgradedYoke_2000b.transpose()

X_tbl_UpgradedRing_2748b, Y_N_UpgradedRing_2748b = np.meshgrid(tBL_UpgradedRing_2748b,N_UpgradedRing_2748b)
X_tbl_UpgradedRing_2748b=X_tbl_UpgradedRing_2748b.transpose()
Y_N_UpgradedRing_2748b=Y_N_UpgradedRing_2748b.transpose() 

X_tbl_UpgradedRing_2000b, Y_N_UpgradedRing_2000b = np.meshgrid(tBL_UpgradedRing_2000b,N_UpgradedRing_2000b)
X_tbl_UpgradedRing_2000b=X_tbl_UpgradedRing_2000b.transpose()
Y_N_UpgradedRing_2000b=Y_N_UpgradedRing_2000b.transpose()




fig = plt.figure(0)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_PostLS1Yoke_2000b, Y_N_PostLS1Yoke_2000b, T_mat_PostLS1Yoke_2000b, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
ax.set_zlabel('Temperature [C]')
plt.title('PostLS1 MKI peak yoke temperature [2000 bunches]')

fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_PostLS1Yoke_2000b, Y_N_PostLS1Yoke_2000b, T_mat_PostLS1Yoke_2000b_limit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
plt.title('PostLS1 MKI yoke safe region of operation [2000 bunches]')

fig = plt.figure(num=2, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
x1=X_tbl_PostLS1Yoke_2000b[PostLS1Yoke_limit_indices_2000b[0],PostLS1Yoke_limit_indices_2000b[1]]
y1=Y_N_PostLS1Yoke_2000b[PostLS1Yoke_limit_indices_2000b[0],PostLS1Yoke_limit_indices_2000b[1]]
y1_interp = interpolate.interp1d(x1,y1)
x1 = [np.min(x1), (np.min(x1)+np.max(x1))/2, np.max(x1)]
plt.plot(x1,y1_interp(x1)) 
plt.ylabel('Bunch intensity [ppb]')
plt.title('PostLS1 MKI yoke temp. limit [2000 bunches]')
ax = fig.gca()
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.fill_between(x1, 0, y1_interp(x1),facecolor='green')
ax.fill_between(x1, y1_interp(x1), np.max(y1_interp(x1)),facecolor='red')


fig = plt.figure(3)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_PostLS1Yoke_2748b, Y_N_PostLS1Yoke_2748b, T_mat_PostLS1Yoke_2748b, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
ax.set_zlabel('Temperature [C]')
plt.title('PostLS1 MKI peak yoke temperature [2748 bunches]')

fig = plt.figure(4)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_PostLS1Yoke_2748b, Y_N_PostLS1Yoke_2748b, T_mat_PostLS1Yoke_2748b_limit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
plt.title('PostLS1 MKI yoke safe region of operation [2748 bunches]')

fig = plt.figure(num=5, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
x2=X_tbl_PostLS1Yoke_2748b[PostLS1Yoke_limit_indices_2748b[0],PostLS1Yoke_limit_indices_2748b[1]]
y2=Y_N_PostLS1Yoke_2748b[PostLS1Yoke_limit_indices_2748b[0],PostLS1Yoke_limit_indices_2748b[1]]
y2_interp = interpolate.interp1d(x2,y2)
x2 = [np.min(x2), (np.min(x2)+np.max(x2))/2, np.max(x2)]
plt.plot(x2,y2_interp(x2))
plt.ylabel('Bunch intensity [ppb]')
plt.title('PostLS1 MKI yoke temp. limit  [2748 bunches]')
ax = fig.gca()
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.fill_between(x2, 0, y2_interp(x2),facecolor='green')
ax.fill_between(x2, y2_interp(x2), np.max(y2_interp(x2)),facecolor='red')


fig = plt.figure(6)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedYoke_2748b, Y_N_UpgradedYoke_2748b, T_mat_UpgradedYoke_2748b, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
ax.set_zlabel('Temperature [C]')
plt.title('Upgraded MKI peak yoke temperature [2748 bunches]')

fig = plt.figure(7)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedYoke_2748b, Y_N_UpgradedYoke_2748b, T_mat_UpgradedYoke_2748b_limit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI yoke safe region of operation [2748 bunches]')

fig = plt.figure(num=8, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
x3=X_tbl_UpgradedYoke_2748b[UpgradedYoke_limit_indices_2748b[0],UpgradedYoke_limit_indices_2748b[1]]
y3=Y_N_UpgradedYoke_2748b[UpgradedYoke_limit_indices_2748b[0],UpgradedYoke_limit_indices_2748b[1]]
y3_interp = interpolate.interp1d(x3,y3)
x3 = [np.min(x3), (np.min(x3)+np.max(x3))/2, np.max(x3)]
plt.plot(x3,y3_interp(x3))
plt.ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI yoke temp. limit  [2748 bunches]')
ax = fig.gca()
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.fill_between(x3, 0, y3_interp(x3),facecolor='green')
ax.fill_between(x3, y3_interp(x3), np.max(y3_interp(x3)),facecolor='red')


fig = plt.figure(9)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedYoke_2000b, Y_N_UpgradedYoke_2000b, T_mat_UpgradedYoke_2000b, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
ax.set_zlabel('Temperature [C]')
plt.title('Upgraded MKI peak yoke temperature [2000 bunches]')

fig = plt.figure(10)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedYoke_2000b, Y_N_UpgradedYoke_2000b, T_mat_UpgradedYoke_2000b_limit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI yoke safe region of operation [2000 bunches]')

fig = plt.figure(num=11, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
x4=X_tbl_UpgradedYoke_2000b[UpgradedYoke_limit_indices_2000b[0],UpgradedYoke_limit_indices_2000b[1]]
y4=Y_N_UpgradedYoke_2000b[UpgradedYoke_limit_indices_2000b[0],UpgradedYoke_limit_indices_2000b[1]]
y4_interp = interpolate.interp1d(x4,y4)
x4 = [np.min(x4), (np.min(x4)+np.max(x4))/2, np.max(x4)]
plt.plot(x4,y4_interp(x4))
ax=fig.gca()
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
plt.ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI yoke temp. limit  [2000 bunches]')
ax.fill_between(x4, 0, y4_interp(x4),facecolor='green')
ax.fill_between(x4, y4_interp(x4), np.max(y4_interp(x4)),facecolor='red')


fig = plt.figure(12)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedRing_2748b, Y_N_UpgradedRing_2748b, T_mat_UpgradedRing_2748b, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
ax.set_zlabel('Temperature [C]')
plt.title('Upgraded MKI peak Ring temperature [2748 bunches]')

fig = plt.figure(13)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedRing_2748b, Y_N_UpgradedRing_2748b, T_mat_UpgradedRing_2748b_limit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI Ring safe region of operation [2748 bunches]')

fig = plt.figure(num=14, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
x5=X_tbl_UpgradedRing_2748b[UpgradedRing_limit_indices_2748b[0],UpgradedRing_limit_indices_2748b[1]]
y5=Y_N_UpgradedRing_2748b[UpgradedRing_limit_indices_2748b[0],UpgradedRing_limit_indices_2748b[1]]
y5_interp = interpolate.interp1d(x5,y5)
x5 = [np.min(x5), (np.min(x5)+np.max(x5))/2, np.max(x5)]
plt.plot(x5,y5_interp(x5))
plt.ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI Ring temp. limit  [2748 bunches]')
ax = fig.gca()
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.fill_between(x5, 0, y5_interp(x5),facecolor='green')
ax.fill_between(x5, y5_interp(x5), np.max(y5_interp(x5)),facecolor='red')


fig = plt.figure(15)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedRing_2000b, Y_N_UpgradedRing_2000b, T_mat_UpgradedRing_2000b, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
ax.set_zlabel('Temperature [C]')
plt.title('Upgraded MKI peak Ring temperature [2000 bunches]')

fig = plt.figure(16)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_tbl_UpgradedRing_2000b, Y_N_UpgradedRing_2000b, T_mat_UpgradedRing_2000b_limit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
ax.set_ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI Ring safe region of operation [2000 bunches]')

fig = plt.figure(num=17, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 18})
x6=X_tbl_UpgradedRing_2000b[UpgradedRing_limit_indices_2000b[0],UpgradedRing_limit_indices_2000b[1]]
y6=Y_N_UpgradedRing_2000b[UpgradedRing_limit_indices_2000b[0],UpgradedRing_limit_indices_2000b[1]]
y6_interp = interpolate.interp1d(x6,y6)
x6 = [np.min(x6), (np.min(x6)+np.max(x6))/2, np.max(x6)]
plt.plot(x6,y6_interp(x6))
ax=fig.gca()
ax.set_xlabel(r'Bunch length $(4\sigma)$ [s]')
plt.ylabel('Bunch intensity [ppb]')
plt.title('Upgraded MKI Ring temp. limit  [2000 bunches]')
ax.fill_between(x6, 0, y6_interp(x6),facecolor='green')
ax.fill_between(x6, y6_interp(x6), np.max(y6_interp(x6)),facecolor='red')


fig = plt.figure(18)

x7 = X_tbl_PostLS1Yoke_2748b[:,16]
y7 = T_mat_PostLS1Yoke_2748b[:,16]
x8 = X_tbl_UpgradedRing_2748b[:,16]
y8 = T_mat_UpgradedRing_2748b[:,16]
x9 = X_tbl_UpgradedYoke_2748b[:,16]
y9 = T_mat_UpgradedYoke_2748b[:,16]
plt.plot(x7,y7)
plt.plot(x8,y8)
plt.plot(x9,y9)
plt.xlim(1e-9,1.8e-9)
plt.ylim(50,350)
plt.grid()
plt.show()
## 2D plot

import numpy as np
import matplotlib.pyplot as plt

npts = 100
a=np.linspace(0,2*np.pi,npts)
b=np.sin(a)
plt.figure(0)
plt.plot(a,b)
plt.plot(a,b,marker='d')

##3D plot

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

N=100
a=np.linspace(0,2*np.pi,N)
a_mat = np.tile(a, (N,1))
b_mat=a_mat.transpose()
z_mat = np.sin(a_mat) + np.sin(b_mat)
#Better way to create 2D grid:
X = np.arange(0, 2*np.pi, 0.05)
Y = np.arange(0, 2*np.pi, 0.05)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X) + np.sin(Y)
#
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(a_mat, b_mat, z_mat, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
plt.show()
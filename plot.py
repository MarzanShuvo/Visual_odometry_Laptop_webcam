import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = np.loadtxt('/media/marzan/workspace/visual_odometry/position.csv', delimiter=',', unpack=True)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x,y,z,color='orange',marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('z')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
plt.show()
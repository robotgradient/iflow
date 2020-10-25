import numpy as np
import matplotlib.pyplot as plt


x0 = np.array([1.])
a = np.array([-10.])
dt = 0.0001


T = 1000


def velocity(x):
    #return a*x
    return -10*np.tanh(100*x)
##forward
trj = x0[None, :]
for i in range(T):
    vel = velocity(x0)
    x1 = vel*dt + x0
    x0 = x1
    trj = np.concatenate((trj, x1[None, :]),0)

##backward
trj2 = x0[None, :]
for i in range(T):
    vel = -velocity(x0)
    x1 = vel*dt + x0
    x0 = x1
    trj2 = np.concatenate((trj2, x1[None, :]),0)

plt.plot(trj)
plt.plot(np.flip(trj2))
plt.show()


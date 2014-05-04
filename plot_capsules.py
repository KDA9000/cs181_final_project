import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

data_pickle = "capsule_data_labeled.npy"
with open(data_pickle, "rb") as fp:
    X_orig,t_orig = pickle.load(fp)

print("finished loading in pickled data...")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t = t_orig
indices0 = np.where(t==0)
indices1 = np.where(t==1)

#s = StandardScaler(copy=False)
#X_orig = s.fit_transform(X_orig)

X0 = X_orig[indices0]
X1 = X_orig[indices1]

#X0 = np.square(X0)
#X1 = np.square(X1)

N0 = len(X0)
N1 = len(X1)
N0 = 10000
N1 = 10000
real = np.array([[0.57992318,1.32338916,0.94076627],
[-1.14773678,-4.82263876,1.21043419],
[-0.04505671,-1.27733715,-0.11816285],
[-2.43469675,-1.13362012,1.39733327],
[-0.45958987,-1.02964365,0.63902567]]) 

ax.plot(X0[0:N0,0], X0[0:N0,1], X0[0:N0,2], 'o', color='b', alpha=0.3)
ax.plot(X1[0:N1,0], X1[0:N1,1], X1[0:N1,2], 'o', color='r', alpha=0.3)
ax.plot(real[:,0], real[:,1], real[:,2], 'o', color='m')

plt.show()

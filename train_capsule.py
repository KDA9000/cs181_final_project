import cPickle as pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

data_pickle = "capsule_data_labeled.npy"
with open(data_pickle, "rb") as fp:
    X,t = pickle.load(fp)

real = np.array([[0.57992318,1.32338916,0.94076627],
[-1.14773678,-4.82263876,1.21043419],
[-0.04505671,-1.27733715,-0.11816285],
[-2.43469675,-1.13362012,1.39733327],
[-0.45958987,-1.02964365,0.63902567]]) 

real2 = np.array([[-1.75668676, -3.93176841,  1.0796569 ],
[-2.35268908, -2.75135884,  1.72056535],
[-2.92623291, -3.0556334,   0.43412184],
[-4.60226967, -2.69694684,  1.21777208],
[ 1.14752187, -0.22815669,  1.06197207]])

real3 = np.array([[-3.76847372, -1.47269873, 1.59132488],
[-0.05430889, -0.82014282,  2.1256671 ],
[-3.75748022, -2.62226653,  0.25661983],
[-2.72486727, -4.0332927,   1.19162643],
[-0.71077942, -3.6285072,   1.00463513]])


print("finished loading in pickled data...")

'''
X_train, X_test, t_train, t_test = sklearn.cross_validation.train_test_split(X, t, test_size = 0.4)
print("learning...")
#clf = LogisticRegression()
#clf = svm.SVC(kernel='linear')
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, t_train)
print("testing...")
score = clf.score(X_test, t_test)
print(score)
'''
'''
Not compatible with scikit-learn v0.10 :(
s = StandardScaler(copy=False)
'''
# For scikit-learn v0.10
s = preprocessing.Scaler(copy=False)


X = s.fit_transform(X)
real = s.transform(real)
real2 = s.transform(real2)
real3 = s.transform(real3)

'''
Not compatible with scikit-learn v0.10
km = KMeans(n_clusters=3)
'''
km = KMeans(k=3)
km.fit(X)
centers = km.cluster_centers_
labels = km.labels_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['b', 'r', 'y']

for i in xrange(3):
    indices = np.where(labels==i)
    new_X = X[indices]
    ax.plot(new_X[:,0], new_X[:,1], new_X[:,2], 'o', color=colors[i], alpha = 0.7)

ax.plot(centers[:,0], centers[:,1], centers[:,2], 'o', color='m')
ax.plot(real[:,0], real[:,1], real[:,2], 'o', color='g')
ax.plot(real2[:,0], real2[:,1], real2[:,2], 'o', color='c')
ax.plot(real3[:,0], real3[:,1], real3[:,2], 'o', color='w')
print(km.predict(real))
print(km.predict(real2))
print(km.predict(real3))
plt.show()

with open('kmeans_params', 'wb') as fp:
    pickle.dump(km, fp)

with open('normalization_params', 'wb') as fp2:
    pickle.dump(s, fp2)

import cPickle as pickle
import numpy as np
import os
from scipy.stats import pearsonr


pickle_name = "pickled_ghost_train1"
data = pickle.load(open(pickle_name, "rb"))
print "finished loading in pickled data..."

epsilon = 0.05
for cl in [0,1,2,3,5]:
    for feat1 in xrange(1,14):
        for feat2 in xrange(feat1,14):
            if feat1 == feat2:
                continue
            cl_feat1 = np.array(data[(cl,feat1)])
            cl_feat2 = np.array(data[(cl,feat2)])
            res = pearsonr(cl_feat1, cl_feat2)
            if res[1] < epsilon:
                print("Correlation between features " + str(feat1) + " and " +
                        str(feat2) + " in class " + str(cl)+" is significant!")
                print(res)

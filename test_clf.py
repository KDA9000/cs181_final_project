import cPickle as pickle
import numpy as np
from sklearn.externals import joblib
from scipy.stats import itemfreq

test_file = 'SVM_multi_linear_size_488162_011.pkl'

pickle_name = "ghost_sklearn_data"
with open(pickle_name, "rb") as fp:
    X_all,y = pickle.load(fp)
print "finished loading in pickled data..."
t = y[:,0]
N = X_all.shape[0]

clf = joblib.load(test_file)

print "testing..."
pred = clf.predict(X_all)
pred = pred.astype(np.int_, copy=False)
for i in [0,1,2,3,5]:
    print "for class "+str(i)+":"
    indices = np.where(t==i)
    freq = itemfreq(pred[indices])
    freq[:,1] = freq[:,1]*100 / np.sum(freq[:,1])
    print freq
# print "the classifier was "+str(sco*100)+"% accurate on a dataset of size "+str(N)
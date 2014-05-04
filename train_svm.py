import cPickle as pickle
import numpy as np
import os
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib

suffix = "11"
multi = True
c = 0
ker = 'linear'
test_fac = 0.4

data_pickle = "ghost_sklearn_data_test"
with open(data_pickle, "rb") as fp:
    X,y = pickle.load(fp)
print "finished loading in pickled data..."
N = X.shape[0]
mul_name = "multi" if multi else "two"

def correct(n):
    rest = [0,1,2,3,5]
    rest.remove(c)
    if n in rest:
        return 0
    else:
        assert(n == c)
        return 1
correctv = np.vectorize(correct)

t = y[:,0].astype(np.int_)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, t, test_size=test_fac)
n = y_test.shape[0]
Xy_test = np.concatenate((X_test,y_test.reshape(n,1)),axis=1)

print "learning..."
clf = svm.SVC(kernel=ker)
clf.fit(X_train, y_train)
print "testing..."
sco = clf.score(X_test,y_test)
print "the "+mul_name+"-class "+ker+" SVM was "+str(sco*100)+ \
    "% accurate with "+str(test_fac*100)+"% of the sample (size "+str(N)+") as tests"
for i in [0,1,2,3,5]:
    indices = np.where(y_test==i)
    print "the score for class "+str(i)+" is "+str(100*clf.score(X_test[indices],y_test[indices]))+"%"

print "saving..."
joblib.dump(clf, 'SVM_'+mul_name+'_'+ker+'_size_'+str(N)+"_"+suffix+'.pkl', compress=9)
print "done!"

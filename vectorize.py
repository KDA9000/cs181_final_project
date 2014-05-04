import cPickle as pickle
import numpy as np
import os

data_path = "data/ghost_train1.csv"
save_filename = "ghost_sklearn_data_noquad_test"

N = 488162
N = 10000 #comment out if loading the whole file
temp = np.empty((N,16))

fp = open(data_path,"rb")

i = 0
for line in fp:
    line = np.array(line.split(),dtype=np.float_)
    temp[i] = line
    if i < N-1:
        i += 1
    else:
        break
fp.close()

X = temp[:,range(3,16)]
y = temp[:,[1,2]]

with open(save_filename,"wb") as fp:
    pickle.dump((X,y),fp)
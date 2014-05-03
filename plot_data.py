# included in this file are functions to plot and visualize the data 
# and find the gaussian parameters for the specific gaussian distributions. 
# All data shoud be pickled and dumped to directory.

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os

f = open("data/ghost_train1.csv")

pickle_name = "pickled_ghost_train1"
gaussian_param_name = "gaussian_parameters_ghost_train1"

# hashmap of all data points where the keys are pairs (class, feature_index) where
# class is the latent class (0,1,2,3,5) and feature_index are (0,1,...,13), and 
# featuer_index 0 is score, and feature_index (1,2,...,13) is the feature vector

data = dict()

# reads in training data and stores them into a dictionary with 
# key = (class, feature_index) and pickles and dumps it onto harddrive
def load_dump_data():
    for line in f:
        line_arr = line.split(' ')
        latent = float(line_arr[1])
        for i in xrange(14):
            feature_data = line_arr[i+2]
            key = (latent, i)
            if not data.has_key(key):
                data[key] = []
            else:
                data[key].append(float(feature_data))
    pickle.dump(data, open(pickle_name, "wb"))

# plots the five latent classes of feature_index in a plot
def plot_pair(feature_index):
    if not os.path.exists(pickle_name):
        print "Cannot find pickled file. Starting the pickling process..."
        load_dump_data()
        print "finished pickling! ;)"
    else:
        print "pickle already exists"
    data = pickle.load(open(pickle_name, "rb"))
    print "finished loading in pickled data..."
    
    for i in xrange(1,6):
        plt.subplot(5, 1, i)
        if i == 5:
            arr = np.array(data[5.,feature_index])
        else:
            arr = np.array(data[float(i-1),feature_index])
        plt.hist(arr, bins = 100)
    plt.show()
'''    
    arr1 = np.array(data[0, feature_index])
    arr2 = np.array(data[1, feature_index])
    axarr1.hist(arr1, bins = 100)
    axarr2.hist(arr2, bins = 100)
#    plt.ylabel('scores')
'''

# reads in data dict and outputs a dict with (latent, feature_index) that is filled wiht
# (u, std), u = mean, std = standard deviation
def fit_gaussian():
    if not os.path.exists(pickle_name):
        print "Cannot find pickled file. Starting the pickling process..."
        load_dump_data()
        print "finished pickling! ;)"
    else:
        print "pickle already exists"
    data = pickle.load(open(pickle_name, "rb"))
    gaussian_params = dict()
    for i in xrange(0,5):
        if i == 4:
            latent = 5.
        else:
            latent = float(i)
        for j in xrange(14):
            arr = np.array(data[(latent, j)])
            mean = np.mean(arr)
            std = np.std(arr)
            gaussian_params[(latent, j)] = (mean, std)
    pickle.dump(gaussian_params, open(gaussian_param_name, "wb"))



if  __name__ == "__main__":
    # plots a particular feature for all five latent classes
    plot_pair(5.)


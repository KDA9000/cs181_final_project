import cPickle as pickle
import math
import numpy as np
gaussian_param_name = "gaussian_parameters_ghost_train1"

gaussian_dict = pickle.load(open(gaussian_param_name, "rb"))



    

# finds the PDF of input x given (mean, std) pair for Gaussian distribution
def gaussian_pdf((mean, std) ,x):
    var = float(std) ** 2
    denom = std * ((math.pi * 2) ** 0.5)
    num = math.exp(-((float(x) - float(mean))**2) / (2*var))
    return num / denom


def classify_ghost(gaussian_dict, feature_vector):
        max_prob = -1
        most_likely_class = -1
        all_prob = []
        for i in xrange(5):
            if i == 4:
                latent_class = 5
            else:
                latent_class = i
            prob = 1.
            # only consider the first 10 features cause they're the only Gaussian ones
            for feature_index in xrange(1, 11):
                prob *= gaussian_pdf(gaussian_dict[(latent_class, feature_index)], feature_vector[feature_index])
            if prob > max_prob:
                max_prob = prob
                most_likely_class = latent_class
            all_prob.append(prob)
        return most_likely_class, max_prob
    
f = open("data/ghost_train1.csv")
for i in xrange(20):
    line = f.readline()
    line_arr = line.split(' ')
    latent = float(line_arr[1])
    list = []
    for j in xrange(3, 15):
        list.append(float(line_arr[j]))
    fv = np.array(list)
    print classify_ghost(gaussian_dict, fv), latent        

    

import cPickle as pickle
import numpy as np

data_path = "data/capsule_data_labeled.txt"
save_path = "capsule_data_labeled.npy"

n_lines = 663038 
n_real = 163037
#n_lines = 6
#n_real = 1

temp = np.empty((n_lines-n_real,4))
fp = open(data_path, "rU")

file_data = fp.readlines()
fp.close()

counter = 0
lines = 0
while lines < n_lines:
    if file_data[lines].strip() == "Maximum data points collected!":
        break
    if file_data[lines] == "Real!":
        pass
    else:
        nums = file_data[lines].split(",")
        if file_data[lines+1].strip() == "Real!":
            nums.append(1)
            lines += 2
        else:
            nums.append(0)
            lines += 1
        temp[counter] = np.array(nums, dtype=np.float_)
        counter += 1

X = temp[:,[0,1,2]]
t = np.ravel(temp[:,[3]])
print(X)
print(t)

with open(save_path, "wb") as fp:
    pickle.dump((X,t),fp)

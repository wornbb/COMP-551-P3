import numpy as np
import sklearn as sl
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.misc # to visualize only
x = np.loadtxt("..\\data\\train_x.csv", delimiter=",") # load from text
x = x.reshape(-1, 64, 64) # reshape
pickle.dump(x,open('train_x','wb'))
y = np.loadtxt("..\\data\\train_y.csv", delimiter=",")
y = y.reshape(-1, 1)
pickle.dump(y,open('train_y','wb'))


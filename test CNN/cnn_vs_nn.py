import numpy as np
import pickle
import keras
from NN import nn_new
if __name__ == "__main__":
    input_layer_size = 64*64
    hidden_layer_size = 1
    hidden_layer_struc = [8]
    output_layer = 40
    nn = nn_new.my_nn(input_layer_size,hidden_layer_size,hidden_layer_struc,output_layer)
    # x = np.array([[0,0],[0,0],[0,0],[0,0]])
    # y = np.array([0,0,0,0])
    #nn = pickle.load(open('trained_nn.nn','rb'))
    x = pickle.load(open('..\\text segmentation\\train_x','rb'))
    y = pickle.load(open('..\\train_y_preprocessed','rb'))
    x = x[0:1000]
    y = y[0:1000]
    y = nn_new.map_y(y,output_layer)
    nn.train(x,y,200)
    for k in range(4):
        p = nn.predict(x[k])
        p = nn.readable(p)


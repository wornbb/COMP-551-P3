import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
class sigmoid:
    @staticmethod
    def cal(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def dif(x):
        return 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))
class default:
    @staticmethod
    def cal(y,predict):
        return 0.5*(y-predict)*(y-predict)
    @staticmethod
    def dif(y,predict):
        return -(y-predict)
def gaussian(row,col):
    return np.random.normal(0,0.1,[row,col])
class my_nn:
    """

    """
    def __init__(self,
                 neuron_input, hidden_layers,
                 neuron_per_layer,
                 neuron_output,
                 neuron_function=sigmoid,
                 gradient_step = 0.15,
                 la = 0.05,
                 random_method = gaussian,
                 loss_function = default):
        """
        IMPORTANT: the weight corresponding to each layer locates before the layer. for example,
        the weight for the first hidden layer is the weight used to multiply the input-layer-output

        :param hidden_layers: number of hidden layers (integer)
        :param neuron_per_layer: number of neurons per hidden layer (integer list)
        :param neuron_output: number of neuron in output layer (integer)
        :param random_method: the method used to initialize weight. Must be callable with 2 input arguments (a function)
        :param neuron_function: sigmoid or other neuron function (a function)

                            w = copy.deepcopy(self.w)
                    p1 = self.gradient_check_forward(x,w)
                    w[layer][next_node] = w[layer][next_node] + 0.0001
                    p2 = self.gradient_check_forward(x,w)
                    self.delta[layer][next_node] = (p2-p1)/0.0001
        """
        # dimension match check
        if len(neuron_per_layer) > hidden_layers:
            raise ValueError('A very stupid mistake, not enough layers!!! ')
        elif len(neuron_per_layer) < hidden_layers:
            raise ValueError('A very stupid mistake, there are layers without neuron!!!')
        self.structure = copy.deepcopy(neuron_per_layer)
        self.structure.insert(0, neuron_input)
        self.structure.append(neuron_output)
        # initialize weight for hidden layers and output layer
        self.w = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]   # one more iteration for output layer
        self.dw = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]   # one more iteration for output layer
        self.neuron = neuron_function
        #self.neuron_v = np.vectorize(neuron_function)
        # store output for each layer, used for backprop
        self.delta = [np.zeros([neurons]) for neurons in self.structure[1:len(self.structure)]]
        self.b = [np.zeros([neurons]) for neurons in self.structure[1:len(self.structure)]]
        self.o = [np.zeros([neurons]) for neurons in self.structure[0:len(self.structure)]]
        self.step = gradient_step
        self.loss_function = loss_function
        self.la = la
        #self.avg_cost = 0 # structure need to be changed to make plot available
    def predict(self, x):
        """
        :param input: 2d array, representing an image
        :return: prediction
        """
        # dimension  check
        x = x.flatten()
        if len(x) != self.structure[0]:
            raise ValueError('Input dimension mismatch!!!')
        #self.h = [np.dot(input,self.w[layer]) for layer in range(len(self.structure)-1)]
        self.o[0] = x
        for layer in range(len(self.structure) - 1):
            wx = np.dot(x, self.w[layer])
            a = np.add(self.b[layer],wx)
            x = self.neuron.cal(a)
            self.o[layer+1] = x
        return x # a row of output
    def train(self,x,y,
              iteration = 5000):
        cnt = 0
        self.avg_cost = np.zeros(int(iteration/100))
        sample_size = len(y)
        for current_iteration in range(iteration):
            for train_sample in range(sample_size):
                output = self.predict(x[train_sample])
                self.backprop(x[train_sample],y[train_sample])
                self.gradient_descent(x[train_sample])
     # Plotting
            if current_iteration%100 == 0:
                print(current_iteration/100)
                index = int(current_iteration/100)
                error = default.cal(y[train_sample],self.o[2])
                #self.avg_cost[index]=error
            print(current_iteration)
        pickle.dump(self,open('trained_nn.nn','wb'))
        return 0
    def gradient_descent(self,x):
        step_delta = [np.multiply(self.step,d) for d in self.delta]
        for layer in range(len(self.structure) - 2, -1, -1):
            for node in range(self.structure[layer]):
                self.dw[layer][node,:] = np.multiply(self.neuron.cal(self.o[layer][node]),self.delta[layer])
        #buffer = [np.multiply(self.la, w) for w in self.w]
        #self.dw = [np.add(dw, b) for dw, b in zip(self.dw, buffer)]
        self.dw = [np.multiply(self.step,dw) for dw in self.dw]
        self.w = [np.subtract(s_w,dwdw) for s_w,dwdw in zip(self.w,self.dw)]
        self.b = [np.subtract(s_b,step_d) for s_b,step_d in zip(self.b,step_delta)]
        return 0
    def backprop(self,
                 x,
                 y
                 ):
        for layer in range(len(self.structure) - 2,-1,-1): # start from the second laste layer, end in the input layer
            for next_node in range(self.structure[layer+1]):
                if layer == (len(self.structure)-2):
                    #self.delta[layer][next_node] = self.loss_function.dif(y,self.h[layer][next_node])*self.neuron.dif(self.h[layer][next_node])
                    self.delta[layer][next_node] = self.loss_function.dif(y[next_node],self.o[layer+1][next_node])*self.neuron.dif(self.o[layer+1][next_node])
                else:
                    #self.delta[layer][next_node] = np.dot(self.w[layer+1][next_node,:], self.delta[layer+1])*self.neuron.dif(self.h[layer][next_node])
                    for sum_loop in range(self.structure[layer+2]):
                        self.delta[layer][next_node] += self.delta[layer+1][sum_loop]*self.w[layer+1][next_node,sum_loop]*self.neuron.dif(self.o[layer+1][next_node])
        return 0
    @staticmethod
    def readable(y):
        return y.index(max(y))+1

def map_y(y,size):
    a = [[]]*len(y)
    for index in range(len(y)):
        a[index] = [0]*size
        try: a[index][int(y[index][0])-1] = 1
        except: pass
    return a
if __name__ == "__main__":
    input_layer_size = 64*64
    hidden_layer_size = 1
    hidden_layer_struc = [8]
    output_layer = 40
    #nn = my_nn(input_layer_size,hidden_layer_size,hidden_layer_struc,output_layer)
    # x = np.array([[0,0],[0,0],[0,0],[0,0]])
    # y = np.array([0,0,0,0])
    #nn = pickle.load(open('trained_nn.nn','rb'))
    x = pickle.load(open('..\\text segmentation\\train_x','rb'))
    y = pickle.load(open('..\\train_y_preprocessed','rb'))
    x = x[0:1000]
    y = y[0:1000]
    y = map_y(y,output_layer)
    #nn.train(x,y,200)
    nn = pickle.load(open('trained_nn.nn','rb'))
    for k in range(4):
        p = nn.predict(x[k])
        #p = nn.readable(p)
        print(p)







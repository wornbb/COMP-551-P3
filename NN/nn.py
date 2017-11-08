import numpy as np
import matplotlib.pyplot as plt
import pickle
class sigmoid:
    @staticmethod
    def cal(x):
        return x*2
    @staticmethod
    def dif(x):
        return x*3
class default:
    @staticmethod
    def cal(x):
        return x*2
    @staticmethod
    def dif(y,predict):
        return y*3
class my_nn:
    """

    """
    def __init__(self,
                 neuron_input, hidden_layers,
                 neuron_per_layer,
                 neuron_output,
                 neuron_function=sigmoid,
                 gradient_step = 0.25,
                 random_method = np.random.rand,
                 loss_function = default):
        """
        IMPORTANT: the weight corresponding to each layer locates before the layer. for example,
        the weight for the first hidden layer is the weight used to multiply the input-layer-output

        :param hidden_layers: number of hidden layers (integer)
        :param neuron_per_layer: number of neurons per hidden layer (integer list)
        :param neuron_output: number of neuron in output layer (integer)
        :param random_method: the method used to initialize weight. Must be callable with 2 input arguments (a function)
        :param neuron_function: sigmoid or other neuron function (a function)
        """
        # dimension match check
        if len(neuron_per_layer) > hidden_layers:
            raise ValueError('A very stupid mistake, not enough layers!!! ')
        elif len(neuron_per_layer) < hidden_layers:
            raise ValueError('A very stupid mistake, there are layers without neuron!!!')
        self.structure = neuron_per_layer[:]
        self.structure.insert(0, neuron_input)
        self.structure.append(neuron_output)
        # initialize weight for hidden layers and output layer
        self.w = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]   # one more iteration for output layer
        self.b = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]
        self.delta = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]
        self.neuron = neuron_function
        #self.neuron_v = np.vectorize(neuron_function)
        # store output for each layer, used for backprop
        self.h = [np.empty([neurons]) for neurons in self.structure[1:len(self.structure)]]
        self.a = [np.empty([neurons]) for neurons in self.structure[1:len(self.structure)]]
        self.step = gradient_step
        self.loss_function = loss_function
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
        for layer in range(len(self.structure) - 1):
            a = np.dot(x, self.w[layer])
            x = self.neuron(a)
            self.h[layer] = x
            self.a[layer] = a
        return x # a row of output
    def train(self,x,y,
              iteration = 5000):
        cnt = 0
        self.avg_cost = []*int(iteration/100)
        sample_size = len(y)
        for current_iteration  in range(iteration):
            for train_sample in range(sample_size):
                output = self.predict(x[train_sample])
                self.backprop(y)
                self.gradient_descent()
     # Plotting
            if current_iteration%100 == 0:
                print(current_iteration/100)
                p = plt.plot(range(iteration),self.avg_cost)
                p.clear()
                plt.show()
        pickle.dump(self,open('trained_nn.nn','wb'))
        return

    def gradient_descent(self):
        step_delta = np.multiply(self.step,self.delta)
        dw = np.multiply(step_delta,self.a)
        self.w = np.subtract(self.w,dw)
        self.b = np.subtract(self.b,step_delta)
        return 0
    def backprop(self,
                 y
                 ):
        for layer in range(len(self.structure) - 2,-1,-1): # start from the second laste layer, end in the input layer
            for node in range(self.structure[layer]):
                for next_node in range(self.structure[layer+1]):
                    if layer == (len(self.structure)-2):
                        self.delta[layer][node,next_node] = self.loss_function.dif(y,self.h[layer][next_node])\
                                                            *self.neuron.dif(self.h[layer][next_node])\
                                                            #*self.a[layer][next_node]
                    else:
                        self.delta[layer][node,next_node] = np.dot(self.w[layer+1][next_node,:],self.delta[layer+1][next_node,:])\
                                                            *self.neuron.dif(self.h[layer][next_node])\
                                                            #*self.a[layer][next_node]
        return 0







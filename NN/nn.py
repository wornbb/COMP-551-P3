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
                 gradient_step = 0.25,
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
        self.neuron = neuron_function
        #self.neuron_v = np.vectorize(neuron_function)
        # store output for each layer, used for backprop
        self.delta =  [np.empty([neurons]) for neurons in self.structure[1:len(self.structure)]]
        self.b = [np.empty([neurons]) for neurons in self.structure[1:len(self.structure)]]
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
            self.a[layer] = x
            a = np.dot(x, self.w[layer])
            a = np.add(self.b[layer],a)
            x = self.neuron.cal(a)
            self.h[layer] = x
            #self.a[layer] = a
        return x # a row of output
    def train(self,x,y,
              iteration = 5000):
        cnt = 0
        self.avg_cost = np.zeros(int(iteration/100))
        sample_size = len(y)
        for current_iteration in range(iteration):
            for train_sample in range(sample_size):
                output = self.predict(x[train_sample])
                self.backprop(y[train_sample])
                self.gradient_descent(x[train_sample])
                self.avg_cost
     # Plotting
                if current_iteration%100 == 0:
                    print(current_iteration/100)
                    index = int(current_iteration/100)
                    error = default.cal(y[train_sample],self.h[1])
                    self.avg_cost[index]=error
                    p = plt.plot(range(iteration//100),self.avg_cost)
                    p.clear()
                    plt.show(block=False)
        pickle.dump(self,open('trained_nn.nn','wb'))
        return 0
    def gradient_check_forward(self,x,w):
        for layer in range(len(self.structure) - 1):
            a = np.dot(x, w[layer])
            x = self.neuron.cal(a)
        return x
    def gradient_descent(self,x):
        step_delta = [np.multiply(self.step,d) for d in self.delta]
        dw = [np.multiply(step_d,np.vstack(a_view)) for step_d,a_view in zip(step_delta,self.a)]
        self.w = [np.subtract(s_w,dwdw) for s_w,dwdw in zip(self.w,dw)]
        self.b = [np.subtract(s_b,step_d) for s_b,step_d in zip(self.b,step_delta)]
        return 0
    def backprop(self,
                 y
                 ):
        for layer in range(len(self.structure) - 2,-1,-1): # start from the second laste layer, end in the input layer
            for next_node in range(self.structure[layer+1]):
                if layer == (len(self.structure)-2):
                    self.delta[layer][next_node] = self.loss_function.dif(y,self.h[layer][next_node])*self.neuron.dif(self.h[layer][next_node])
                else:
                    self.delta[layer][next_node] = np.dot(self.w[layer+1][next_node,:],self.delta[layer+1][next_node])*self.neuron.dif(self.h[layer][next_node])
        return 0

if __name__ == "__main__":
    nn = my_nn(2,1,[1],1)
    #x = np.array([[1,0,0,0,0,0,0,0],
    ##             [0, 0, 1, 0, 0, 0, 0, 0],
      #            [0, 0, 0, 1, 0, 0, 0, 0],
       #           [0, 0, 0, 0, 1, 0, 0, 0],
        #          [0, 0, 0, 0, 0, 1, 0, 0],
         #         [0, 0, 0, 0, 0, 0, 1, 0],
          #        [0, 0, 0, 0, 0, 0, 0, 1]])
    x = np.array([[1,1],[1,0],[0,1],[0,0]])
    y = np.array([1,0,0,0])
    #nn = pickle.load(open('trained_nn.nn','rb'))
    nn.train(x,y,50000)
    for k in range(4):
        p = nn.predict(x[k])
        print(p)
    plt.show()






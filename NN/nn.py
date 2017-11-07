import numpy as np
class sigmoid:
    def cal(self,x):
        return x*2
    def dif(self,x):
        return x*3
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
                 loss_function = 'default'):
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
        self.structure = neuron_per_layer
        self.structure.insert(0, neuron_input)
        self.structure.append(neuron_output)
        # initialize weight for hidden layers and output layer
        self.w = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]   # one more iteration for output layer
        self.b = [random_method(self.structure[selected_layer],
                                self.structure[selected_layer + 1])
                    for selected_layer in range(hidden_layers + 1)]
        self.neuron = neuron_function
        #self.neuron_v = np.vectorize(neuron_function)
        # store output for each layer, used for backprop
        self.h = [np.empty([neurons]) for neurons in self.structure[1:len(self.structure)]]
        self.step = gradient_step
    def predict(self, x):
        """
        :param input: 1 row, not vector
        :return: prediction
        """
        # dimension  check
        if len(x) != self.structure[0]:
            raise ValueError('Input dimension mismatch!!!')
        #self.h = [np.dot(input,self.w[layer]) for layer in range(len(self.structure)-1)]
        for layer in range(len(self.structure) - 1):
            intermediate_x = np.dot(x, self.w[layer])
            x = self.neuron(intermediate_x)
            self.h[layer] = x
        return x
    def train(self,x,y,
              iteration = 5000,):
        output = self.predict(x)





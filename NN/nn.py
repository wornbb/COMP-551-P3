import numpy as np
class my_nn:
    """

    """
    def __init__(self,
                 neuron_input,hidden_layers,
                 neuron_per_layer,
                 neuron_output,
                 random_method = np.random.rand,
                 neuron_function = sigmoid):
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
            raise  ValueError('A very stupid mistake, there are layers without neuron!!!')
        self.structure = neuron_per_layer
        self.structure.insert(0,neuron_input)
        self.structure.append(neuron_output)
        # initialize weight for hidden layers and output layer
        for selected_layer in range(hidden_layers + 1): # one more iteration for output layer
            self.w[selected_layer] = random_method(self.structure[selected_layer],
                                                   self.structure[selected_layer+1])
        self.neuron_type = neuron_function

    def predict(self, input):
        # dimension  check
        if len(input) != self.structure[0]:
            raise ValueError('Input dimension mismatch!!!')




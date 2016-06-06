import logging
from neuron import Neuron
from layer import Layer
import uuid

class NeuralNetError(Exception):
    def __init__(self, message):
        super(ValidationError, self).__init__(message)

class NeuralNet(object):
    def __init__(self, depth = 3):
        self.depth = depth
        self.layers = [None] * depth

        self.input_layer = 0
        self.output_layer = self.depth - 1

    def create_layer(self, layer_size):
        new_layer = []
        for i in range(0, layer_size):
            new_neuron = Neuron(uuid.uuid4())
            new_layer.append(new_neuron)
        return Layer(new_layer)

    def add_layer(self, layer, layer_position):
        self.layers[layer_position] = layer

    def attach_layers(self):
        for layer_position, layer in enumerate(self.layers):
            if layer_position > self.input_layer:
                layer.attach_input(self.layers[layer_position-1])

            if layer_position < self.output_layer:
                layer.attach_output(self.layers[layer_position+1])

    def set_input_values(self, values):
        input_layer = self.layers[self.input_layer]
        input_layer.set_output_of_layer(values)
        input_layer.describe_layer()

    def activate_net(self, values):
        self.set_input_values(values)

        for layer in self.layers:
            layer.activate_layer()

    def describe_net(self):
        return {
            "layers" : len(self.layers),
            "nodes" : sum([len(layer.get_neurons()) for layer in self.layers]),
            "inputs" :  len(self.layers[self.input_layer].get_neurons()),
            "outputs" : len(self.layers[self.output_layer].get_neurons()),
        }

    def describe_output(self):
        return {
            "output_value" : self.layers[self.output_layer].describe_layer()
        }

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    neural_net = NeuralNet(4)

    neural_net.add_layer(neural_net.create_layer(9), 0)
    neural_net.add_layer(neural_net.create_layer(3), 1)
    neural_net.add_layer(neural_net.create_layer(3), 2)
    neural_net.add_layer(neural_net.create_layer(1), 3)
    neural_net.attach_layers()

    print("Created Neural Network.")
    print(neural_net.describe_net())
    neural_net.activate_net([0,1,0,1,1,1,0,1,0])
    print(neural_net.describe_output())

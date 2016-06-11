import logging
import numpy
from neural_net_error import NeuralNetError

DEFAULT_WEIGHT = 0.5
class Neuron(object):
    def __init__(self, id):
        self.node_id = id
        self.input_weights = {}
        self.input_neurons = []
        self.output_neurons = []
        self.output_value = 0

    def trigger(self):
        if not self.input_neurons:
            logging.debug("No input nodes - assuming neuron is part of input layer.")
            return

        self.do_activation()

    def attach_to_neuron_as_input(self, neuron):
        self.input_neurons.extend([neuron])
        self.input_weights[neuron.get_node_id()] = DEFAULT_WEIGHT

    def attach_to_neuron_as_output(self, neuron):
        self.output_neurons.extend([neuron])

    def do_activation(self):
        sigma = self.calculate_sigma_of_inputs()

        self.output_value = self.activation_function(sigma)
        logging.info("Result of activation : {0}".format(self.output_value))

    def calculate_sigma_of_inputs(self):
        return sum([
            self.get_weight_adjusted_input(input_neuron)
            for input_neuron in self.input_neurons
        ])

    def get_weight_adjusted_input(self, neuron):
        return (neuron.get_output_value() * self.input_weights[neuron.get_node_id()])

    def activation_function(self, sigma):
        logging.debug("Activation of {0} being calculated".format(sigma))
        return 1 / (1 + numpy.exp(-sigma))

    def get_node_id(self):
        return self.node_id

    def set_output_value(self, output_value):
        self.output_value = output_value

    def get_output_value(self):
        return self.output_value

    def set_input_weight(self, node_id, new_input_weight):
        if node_id in self.input_weights:
            self.input_weights[node_id] = new_input_weight
        else:
            raise NeuralNetError("ABORT: Unable to add input weight - node does not exist.")

    def get_input_weight(self, node_id):
        return self.input_weights.get(node_id)

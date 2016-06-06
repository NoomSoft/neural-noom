import logging
import uuid
import numpy

DEFAULT_WEIGHT = 0.5
class Neuron(object):
    def __init__(self):
        self.node_id = uuid.uuid4()
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
        sigma = 0
        for input_neuron in self.input_neurons:
            sigma += input_neuron.get_output_value()

        self.output_value = self.activation_function(sigma)
        logging.info("Result of activation : {0}".format(self.output_value))

    def activation_function(self, sigma):
        logging.debug("Activation of {0} being calulated".format(sigma))
        return 1 / (1 + numpy.exp(-sigma))

    def get_node_id(self):
        return self.node_id

    def set_output_value(self, output_value):
        self.output_value = output_value

    def get_output_value(self):
        return self.output_value

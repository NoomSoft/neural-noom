import logging

class Layer(object):
    def __init__(self, neurons):
        self.neurons = neurons

    def get_neurons(self):
        return self.neurons

    def attach_output(self, layer):
        if layer is None:
            return

        for local_neuron in self.get_neurons():
            for foreign_neuron in layer.get_neurons():
                logging.debug("Attaching {0} to {1} as output.".format(foreign_neuron.get_node_id(), local_neuron.get_node_id()))
                local_neuron.attach_to_neuron_as_output(foreign_neuron)

    def attach_input(self, layer):
        if layer is None:
            return

        for local_neuron in self.get_neurons():
            for foreign_neuron in layer.get_neurons():
                logging.debug("Attaching {0} to {1} as input.".format(foreign_neuron.get_node_id(), local_neuron.get_node_id()))
                local_neuron.attach_to_neuron_as_input(foreign_neuron)

    def set_output_of_layer(self, values):
        for input_count, neuron in enumerate(self.neurons):
            if input_count < len(values):
                neuron.set_output_value(values[input_count])
            else:
                neuron.set_output_value(0)

    def activate_layer(self):
        for neuron in self.neurons:
            neuron.trigger()

    def describe_layer(self):
        neuron_descriptions = {}

        for neuron in self.neurons:
            logging.debug("Neuron {0} - {1}".format(neuron.get_node_id(), neuron.get_output_value()))
            neuron_descriptions[neuron.get_node_id()] = neuron.get_output_value()
        return neuron_descriptions

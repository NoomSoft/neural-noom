import unittest
from neural_net import Neuron

class NeuronTests(unittest.TestCase):
    def test_get_node_id(self):
        neuron = Neuron("TEST_ID")

        self.assertEqual(neuron.get_node_id(), "TEST_ID", "Should return the node_id passed.")

    def test_set_output_value(self):
        neuron = Neuron("TEST_ID")

        neuron.set_output_value(123456)
        self.assertEqual(neuron.output_value, 123456, "Should have set self.output_value.")

    def test_get_output_value(self):
        neuron = Neuron("TEST_ID")

        neuron.output_value = 123456
        self.assertEqual(neuron.get_output_value(), 123456, "Should return newly set value.")

    def test_attach_to_neuron_as_input(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")

        neuron_1.attach_to_neuron_as_input(neuron_2)
        self.assertEqual(neuron_1.input_neurons, [neuron_2], "Should contain neuron_2 as input.")

    def test_calculate_sigma_of_inputs(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")
        neuron_2.set_output_value(2)
        neuron_3 = Neuron("TEST_3")
        neuron_3.set_output_value(4)

        neuron_1.attach_to_neuron_as_input(neuron_2)
        neuron_1.attach_to_neuron_as_input(neuron_3)

        result = neuron_1.calculate_sigma_of_inputs()
        self.assertEqual(result, 6, "Should have returned the sigma of 2, 4.")

    def test_attach_to_neuron_as_input___add_multiple_neurons_as_inputs(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")
        neuron_3 = Neuron("TEST_3")

        neuron_1.attach_to_neuron_as_input(neuron_2)
        neuron_1.attach_to_neuron_as_input(neuron_3)
        self.assertEqual(neuron_1.input_neurons, [neuron_2, neuron_3], "Should contain neuron_2 and neuron_3 as input.")

    def test_attach_to_neuron_as_output(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")

        neuron_1.attach_to_neuron_as_output(neuron_2)
        self.assertEqual(neuron_1.output_neurons, [neuron_2], "Should contain neuron_2 as output.")

    def test_attach_to_neuron_as_output___add_multiple_neurons_as_outputs(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")
        neuron_3 = Neuron("TEST_3")

        neuron_1.attach_to_neuron_as_output(neuron_2)
        neuron_1.attach_to_neuron_as_output(neuron_3)
        self.assertEqual(neuron_1.output_neurons, [neuron_2, neuron_3], "Should contain neuron_2 and neuron_3 as outputs.")

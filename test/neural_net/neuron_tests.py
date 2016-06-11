import unittest
from neural_net import Neuron
from neural_net import NeuralNetError

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
        neuron_1.set_input_weight("TEST_2", 2)

        neuron_1.attach_to_neuron_as_input(neuron_3)
        neuron_1.set_input_weight("TEST_3", 0.5)

        result = neuron_1.calculate_sigma_of_inputs()
        self.assertEqual(result, 6, "Should have returned the sigma of 2*2, 4*0.5.")

    def test_get_weight_adjusted_input(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")
        neuron_2.set_output_value(2)

        neuron_1.attach_to_neuron_as_input(neuron_2)
        result = neuron_1.get_weight_adjusted_input(neuron_2)

        self.assertEqual(result, 1, "Should have returned the output value adjusted by weight 0.5.")

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

    def test_get_input_weight___neuron_exists(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")

        neuron_1.attach_to_neuron_as_input(neuron_2)
        self.assertEqual(neuron_1.get_input_weight("TEST_2"), 0.5, "Should return the default weight of 0.5.")

    def test_get_input_weight___neuron_does_not_exist(self):
        neuron_1 = Neuron("TEST_1")
        self.assertEqual(neuron_1.get_input_weight("TEST_2"), None, "Should return None.")

    def test_set_input_weight(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")

        neuron_1.attach_to_neuron_as_input(neuron_2)
        neuron_1.set_input_weight("TEST_2", 2)

        self.assertEqual(neuron_1.input_weights, {"TEST_2" : 2}, "Should set new input weight of TEST_2 be 2")

    def test_set_input_weight___neuron_does_not_exist(self):
        neuron_1 = Neuron("TEST_1")

        with self.assertRaises(NeuralNetError):
            neuron_1.set_input_weight("TEST_2", 2)

    def test_set_input_weight___multiple_weights_set(self):
        neuron_1 = Neuron("TEST_1")
        neuron_2 = Neuron("TEST_2")
        neuron_3 = Neuron("TEST_3")

        neuron_1.attach_to_neuron_as_input(neuron_2)
        neuron_1.set_input_weight("TEST_2", 2)
        neuron_1.attach_to_neuron_as_input(neuron_3)
        neuron_1.set_input_weight("TEST_3", 4)

        self.assertEqual(neuron_1.input_weights, {"TEST_2" : 2, "TEST_3" : 4}, "Should set input weights for TEST_2 and TEST_3 to 2 and 4")

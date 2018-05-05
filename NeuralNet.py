# import libraries
import numpy
# import special math lib for the sigmoid function expit()
import scipy.special



# neural network class definition
class neuralNetwork:
    # initiale the neural newtwork
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set the network geometry
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set the learning rate
        self.lr = learningrate

        # set the initial random weights
        # random uniformly distributed weights [-0.5; 0.5]
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        # better choise: Normal distribution with mean = 0 and variance = 1/sqrt(input_nodes)
        self.wih = (numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)))

        # set sigmoid as the activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert input and target lists to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate values for the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # apply activation function to calculate hidden outputs
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate values for the final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # apply activation function to calculate output outputs
        final_outputs = self.activation_function(final_inputs)
        # calculate the error which is targets - actual values
        output_errors = targets - final_outputs
        # backpropagate final error to hidden nodes by splitting according to weights
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # update the weights between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        # convert input list to 2D arrays
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate values for the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # apply activation function to calculate hidden outputs
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate values for the final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # apply activation function to calculate output outputs
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# define the number of nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
# define the learning rate
learning_rate = 0.3
# create the neural network instance

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(n.query([1.0, 2.0, -1.2]))

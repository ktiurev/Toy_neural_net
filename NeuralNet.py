# import libraries
import numpy
# import special math lib for the sigmoid function expit()
import scipy.special
# import matplotlib for plotting the symbols
import matplotlib.pyplot


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

        # calculate input values for the hidden layer
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
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# define the learning rate
learning_rate = 0.2

#  create the neural network instance
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load training datafiles
training_data_file = open("./mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# pre-process each entry and train the neural network
epochs  = 2

for e in range(epochs):
    for record in training_data_list:
        # split entries with commas
        all_values = record.split(',')
        # scale and shift the entries
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create target values
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# load testing datafiles

test_data_file = open("./mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# all_values = test_data_list[0].split(',')
# print(n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))

# test the correctness
scorecard = []

for record in test_data_list:
    # split the record with commas
    all_values = record.split(',')
    # correct value is the first entry
    correct_label = int(all_values[0])
    print("correct label is", correct_label)
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # compute the output
    outputs = n.query(inputs)
    # find the label with the highest value
    label = int(numpy.argmax(outputs))
    print("network answer is", label)
    # add score 1 if correct, 0 if incorrect
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass


# print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print("Performance is", 100*scorecard_array.sum()/scorecard_array.size, "%")


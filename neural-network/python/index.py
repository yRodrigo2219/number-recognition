import mnist_loader
import network
import serialize
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data)

#commenting/uncomenting as needed

#print(training_data[0])
#x, y = training_data[0]
#serialize.saveInputJson(x)
#print(np.argmax(y))

#input = serialize.loadInputJson('testInput.json')

#net = serialize.readJson('networks/network-69_37.json')
#net.evaluate(test_data, True)
#print(np.argmax(net.feedforward(input)))
#net.SGD(training_data, 1000, 10, 3.0, test_data)

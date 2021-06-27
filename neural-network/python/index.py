import mnist_loader
import network
import serialize

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

#net = network.Network([784, 50, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data)

net = serialize.readJson('networks/network-96_13.json')
net.SGD(training_data, 1000, 10, 3.0, test_data)
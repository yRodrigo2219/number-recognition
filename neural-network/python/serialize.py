import codecs, json
import network
import numpy as np

def saveJson(net, acc = None):
  newNet = network.Network(net.sizes)
  newNet.biases = [x.tolist() for x in net.biases]
  newNet.weights = [x.tolist() for x in net.weights]

  file_path = "networks/network.json"
  if(acc):
    file_path = "networks/network-" + acc.replace('.', '_') + ".json"
  json.dump(newNet.__dict__, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=2)

def readJson(file_path):
  net_text = codecs.open(file_path, 'r', encoding='utf-8').read()
  net_dict = json.loads(net_text)
  net = network.Network(net_dict["sizes"])
  net.biases = [np.array(x) for x in net_dict["biases"]]
  net.weights = [np.array(x) for x in net_dict["weights"]]

  return net
  

#with open('network.json', 'w') as outfile:
#  json.dump(net.__dict__, outfile)
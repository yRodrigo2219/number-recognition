import codecs, json
import network

def saveJson(net, acc = None):
  newNet = network.Network(net.sizes)
  newNet.biases = [x.tolist() for x in net.biases]
  newNet.weights = [x.tolist() for x in net.weights]

  file_path = "networks/network.json"
  if(acc):
    file_path = "networks/network-" + acc.replace('.', '_') + ".json"
  json.dump(newNet.__dict__, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=2)

#with open('network.json', 'w') as outfile:
#  json.dump(net.__dict__, outfile)
import Matrix from "./Matrix.js";

export default class Network {
  constructor(data) {
    this.sizes = data.sizes;
    this.num_layers = data.num_layers;
    this.biases = data.biases;
    this.weights = data.weights;
  }

  feedforward(input) {
    let a = input;

    for (let i = 0; i < this.biases.length; i++) {
      const w = this.weights[i];
      const b = this.biases[i];

      a = Network.sigmoid(Matrix.add(Matrix.dot(w, a), b));
    }

    return a;
  }

  static sigmoid(z) {
    if (typeof z === "object") return z.map((v) => Network.sigmoid(v));
    else return 1 / (1 + Math.exp(-z));
  }
}

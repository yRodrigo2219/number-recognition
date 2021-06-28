const gaussian = require("gaussian");
const math = require("mathjs");
// limpar imports (economizar memoria)

class Network {
  constructor(sizes) {
    const distribution = gaussian(0, 1);

    this.num_layers = sizes.length;
    this.sizes = sizes;
    this.biases = sizes
      .slice(1) // skips the first
      .map((size) => [...Array(size)].map(() => distribution.random(1)));
    this.weights = sizes
      .slice(1) // skips the first
      .map((size, i) =>
        [...Array(size)].map(() => distribution.random(sizes[i]))
      );
  }

  feedforward(a) {
    this.biases.forEach((b, i) => {
      const w = this.weights[i];

      a = Network.sigmoid(
        math.add(math.multiply(w, a), b) // (w . a) + b
      );
    });

    return a;
  }

  SGD(training_data, epochs, mini_batch_size, eta, test_data) {
    const n_test = test_data?.length;
    const n = training_data.length;
    const mini_batches = [];

    for (let j = 0; j < epochs; j++) {
      console.time("epoch" + j);
      shuffleArray(training_data);
      for (let k = 0; k < n; k += mini_batch_size)
        mini_batches.push(training_data.slice(k, k + mini_batch_size));

      mini_batches.forEach((mini_batch) => {
        this.update_mini_batch(mini_batch, eta);
      });

      if (test_data)
        console.log(`Epoch ${j}: ${this.evaluate(test_data)} / ${n_test}`);
      else console.log(`Epoch ${j}: complete`);
      console.timeEnd("epoch" + j);
    }
  }

  update_mini_batch(mini_batch, eta) {
    let nabla_b = this.biases.map((bias) =>
      math.zeros(math.matrix(bias)._size)
    );
    let nabla_w = this.weights.map((weight) =>
      math.zeros(math.matrix(weight)._size)
    );

    for (const [x, y] of mini_batch) {
      const [delta_nabla_b, delta_nabla_w] = this.backprop(x, y);
      nabla_b = nabla_b.map((nb, i) => math.add(nb, delta_nabla_b[i]));
      nabla_w = nabla_w.map((nw, i) => math.add(nw, delta_nabla_w[i]));

      this.weights = this.weights.map((w, i) =>
        math.subtract(w, math.dotMultiply(eta / mini_batch.length, nabla_w[i]))
      );
      this.biases = this.biases.map((b, i) =>
        math.subtract(b, math.dotMultiply(eta / mini_batch.length, nabla_b[i]))
      );
    }
  }

  backprop(x, y) {
    const nabla_b = this.biases.map((bias) =>
      math.zeros(math.matrix(bias)._size)
    );
    const nabla_w = this.weights.map((weight) =>
      math.zeros(math.matrix(weight)._size)
    );

    let activation = x;
    const activations = [x];
    const zs = [];

    this.biases.map((b, i) => {
      const w = this.weights[i];
      //const z = b.map(([bi]) => math.add(math.multiply(w, activation), bi));
      const z = math.add(math.multiply(w, activation), b);
      zs.push(z);
      activation = Network.sigmoid(z);
      activations.push(activation);
    });

    let delta = math.dotMultiply(
      this.cost_derivative(activations[activations.length - 1], y),
      Network.sigmoid_prime(zs[zs.length - 1])
    );

    nabla_b[nabla_b.length - 1] = delta;
    nabla_w[nabla_w.length - 1] = math.multiply(
      delta,
      math.transpose(activations[activations.length - 2])
    );

    for (let l = 2; l < this.num_layers; l++) {
      const z = zs[zs.length - l];
      const sp = Network.sigmoid_prime(z);
      delta = math.dotMultiply(
        math.multiply(
          math.transpose(this.weights[this.weights.length - l + 1]),
          delta
        ),
        sp
      );
      nabla_b[nabla_b.length - l] = delta;
      nabla_w[nabla_w.length - l] = math.multiply(
        delta,
        math.transpose(activations[activations.length - l - 1])
      );
    }

    return [nabla_b, nabla_w];
  }

  evaluate(test_data) {
    const test_results = test_data.map(([x, y]) => [
      this.feedforward(x).reduce(reducerArgmax, 0),
      y.reduce(reducerArgmax, 0),
    ]);

    return test_results.reduce((sum, [x, y]) => (x === y ? ++sum : sum), 0);
  }

  cost_derivative(output_activations, y) {
    return math.subtract(output_activations, y);
  }

  static sigmoid(z) {
    if (typeof z === "object") return z.map((v) => Network.sigmoid(v));
    else return 1 / (1 + Math.exp(-z));
  }

  static sigmoid_prime(z) {
    return math.dotMultiply(
      Network.sigmoid(z),
      math.subtract(1, Network.sigmoid(z))
    );
  }
}

const reducerArgmax = (iMax, x, i, arr) => (x[0] > arr[iMax][0] ? i : iMax);

function shuffleArray(array) {
  for (var i = array.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

module.exports = Network;

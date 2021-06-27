const Network = require("./src/Network");
const mnist = require("mnist");
const { reshape } = require("mathjs");

console.time("setLoad");
const set = mnist.set(100, 10);
console.timeEnd("setLoad");

console.time("reshape");
const trainingSet = set.training.map(({ input, output }) => [
  reshape(input, [784, 1]),
  reshape(output, [10, 1]),
]);

const testSet = set.test.map(({ input, output }) => [
  reshape(input, [784, 1]),
  reshape(output, [10, 1]),
]);
console.timeEnd("reshape");

const net = new Network([784, 30, 10]);
net.SGD(trainingSet, 30, 10, 3.0, testSet);

// TO-DO:
//  * Performance Increase (some 100's times)
//  * Training
//  * Serialize
//  * Github pages

console.log("app is loading ...");
const tf = require("@tensorflow/tfjs");

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Generate some synthetic data for training.
// house price is 50K + 50 per bed room
const xs = tf.tensor1d([0,1, 2, 3, 4]);
const ys = tf.tensor1d([50, 100, 150, 200, 250]);

// Train the model using the data.
model.fit(xs, ys,{epochs: 1000}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor1d([7])).print(); //
});

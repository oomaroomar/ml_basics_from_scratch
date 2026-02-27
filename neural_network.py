from math import floor
import numpy as np
import pandas as pd

data = pd.read_csv("data/clean_weather.csv", index_col=0)
data = data.ffill()
data = data.dropna(axis=0)

if data.isna().any().any():
    raise ValueError("Data contains missing values")


def mean_square_error(actual, predicted):
    return (actual - predicted) ** 2 / actual.shape[0]


def mse_grad(actual, predicted):
    return (predicted - actual) / actual.shape[0]


class neural_network:
    def __init__(self, layer_config=None, lr=1e-4):
        self.lr = lr
        if layer_config == None:
            layer_config = [3, 10, 10, 1]
        self.layers = []
        for i in range(1, len(layer_config)):
            self.layers.append(
                {
                    "w": np.random.rand(layer_config[i - 1], layer_config[i]),
                    "b": np.ones((1, layer_config[i])),
                }
            )

    def forward(self, batch):
        activations = [batch.copy()]
        for i in range(0, len(self.layers)):
            batch = batch @ self.layers[i]["w"] + self.layers[i]["b"]
            if i < len(self.layers) - 1:  # Relu activation for hidden layers
                batch = np.maximum(batch, 0)
            activations.append(batch.copy())
        return batch, activations

    def backward(self, activations, dL):
        """
        :param activations: activations of a forward pass
        :param dL: dL/da (derivative of loss function w.r.t. last layer activations)
        """
        for i in range(len(self.layers) - 1, -1, -1):
            # da/dz
            da = (
                np.heaviside(activations[i + 1], 1)
                if i != len(self.layers) - 1
                else np.ones(activations[i + 1].shape)
            )
            # dz/dw
            dz = activations[i]

            w_nudge = dz.T @ (dL * da)
            b_nudge = np.mean(dL * da)
            self.layers[i]["w"] -= w_nudge * self.lr
            self.layers[i]["b"] -= b_nudge * self.lr
            dL = (dL * da) @ self.layers[i]["w"].T

    def train(self, X, y, epochs=10, batch_size=8):
        for ep in range(epochs):
            mean_epoch_loss = 0
            for i in range(0, floor(X.shape[0] / batch_size)):
                x_batch = X[(i * batch_size) : ((i + 1) * batch_size)]
                y_batch = y[(i * batch_size) : ((i + 1) * batch_size)]
                pred, activations = self.forward(x_batch)
                loss = mse_grad(y_batch, pred)
                mean_epoch_loss = mean_epoch_loss + mean_square_error(y_batch, pred)
                self.backward(activations, loss)
            mean_epoch_loss = mean_epoch_loss / (i + 1)
            print(f"Epoch: {ep}, loss: {mean_epoch_loss}")


if __name__ == "__main__":
    features = ["tmax", "tmin", "tmax_tomorrow"]
    labels = ["rain"]
    X = data[features].to_numpy()
    y = data[labels].to_numpy()
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    # y_normalized = (y - y.mean(axis=0)) / y.std(axis=0)
    neural_net = neural_network()
    pred, _ = neural_net.forward(X_normalized)
    print(
        f"Error before training: {mean_square_error(y, pred)}"
    )
    neural_net.train(X_normalized, y)
    pred, _ = neural_net.forward(X_normalized)
    print(
        f"Error after training: {mean_square_error(y, pred)}"
    )

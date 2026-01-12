import matplotlib

matplotlib.use("module://matplotlib-backend-sixel")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv("data/clean_weather.csv", index_col=0)
data = data.ffill()
data = data.dropna(axis=0)

if data.isna().any().any():
    raise ValueError("Data contains missing values")


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true, y_pred):
    grad = (y_pred - y_true) / y_true.shape[0]
    return grad


def forward(X, w, b):
    return X @ w + b


def backward(X: np.ndarray, w, b, learning_rate, grad):
    if np.isnan(X).any():
        raise ValueError("X contains missing values")
    if np.isnan(grad).any():
        raise ValueError("grad contains missing values")
    wd = X.T @ grad
    bd = np.sum(grad, axis=0)

    w = w - learning_rate * wd
    b = b - learning_rate * bd

    return w, b


def init_params(n_columns):
    k = math.sqrt(1 / n_columns)
    np.random.seed(0)
    w = np.random.rand(n_columns, 1) * 2 * k - k
    b = np.ones((1, 1)) * 2 * k - k
    return w, b


def gradient_descent(X, y, *, learning_rate=1e-4, epochs=500000):
    w, b = init_params(X.shape[1])
    for t in range(epochs):
        y_pred = forward(X, w, b)
        grad = mse_grad(y, y_pred)
        w, b = backward(X, w, b, learning_rate, grad)
        if t % 10000 == 0:
            y_true = forward(X, w, b)
            valid_loss = mean_squared_error(y, y_true)
            print(f"Epoch {t} validation loss: {valid_loss}")
    return w, b


features = ["tmax", "tmin"]
labels = ["tmax_tomorrow", "rain"]
X = data[features].to_numpy()
y = data[labels].to_numpy()
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
y_normalized = (y - y.mean(axis=0)) / y.std(axis=0)

w, b = gradient_descent(X_normalized, y_normalized)

plt.scatter(data["tmax"], data["tmax_tomorrow"])
plt.plot(data["tmax"], forward(X, w, b), color="red")
plt.show()

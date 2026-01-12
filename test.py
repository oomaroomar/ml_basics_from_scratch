import matplotlib

matplotlib.use("module://matplotlib-backend-sixel")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv("data/clean_weather.csv", index_col=0)
data = data.ffill()
data = data.dropna(axis=0)

print(data.head())

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font="IPAexGothic", style="white")

# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv", header=None)
print("Data Shapes")
print("Train:", train.shape, "Test:", test.shape, "Sample:", sample.shape)

# %%

from Eclat import fim
import pandas as pd
import numpy as np
import json

data = pd.read_csv("data_groceries.csv")
data.drop("Item(s)", axis=1, inplace=True)

transactions = []

for i in range(0, data.shape[0]):
    arr = []
    for j in range(0, 32):
        if data.values[i, j] is not np.nan:
            arr.append(data.values[i, j])
    transactions.append(arr)

rules = fim(transactions)
print(rules)



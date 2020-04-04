from apyori import apriori
import pandas as pd
import numpy as np
import json

data = pd.read_csv("data_groceries.csv")
data.drop("Item(s)", axis=1, inplace=True)

# print(data)
# print("NULL: ", data.isnull().sum())

transactions = []

for i in range(0, data.shape[0]):
    arr = []
    for j in range(0, 32):
        if data.values[i, j] is not np.nan:
            arr.append(data.values[i, j])
    transactions.append(arr)

min_support = 0.05
min_confidence = 0.2
min_lift = 1.5

rules = apriori(
    transactions,
    min_support=min_support,
    min_confidence=min_confidence,
    min_lift=min_lift,
    min_length=2,
)

results = list(rules)
print(results)
results_list = {}
for i in range(0, len(results)):
    results_list_sub = {}
    for j in range(0, len(results[i][2])):
        results_list_sub[j] = {
            "BASE ITEM": str(results[i][2][j][0]),
            "CONSEQUENT ITEM": str(results[i][2][j][1]),
            "CONFIDENCE": str(results[i][2][j][2]),
            "LIFT": str(results[i][2][j][3]),
        }
    results_list[i] = {
        "RULE": str(results[i][0]),
        "SUPPORT": str(results[i][1]),
        "RESULTS": results_list_sub,
    }

open("results.json", "w").write(json.dumps(results_list))


from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')

year = []
gdp_per_capita = []

for i in range(0, 264):
    year.append(data.values[i][1]-1987)
    gdp_per_capita.append((data.values[i][10]-1859)/1387)
# Standardization
# print(data[:264, 10]-1859)
# print(np.mean(data[:264, 10])) # 1859
# print(np.std(data[:264, 10], ddof= 1)) # 1387

slope, intercept, rval, pval, stderr = stats.linregress(year, gdp_per_capita)
print(slope, intercept, rval, pval, stderr)
plt.plot(np.array(year)+1987, (np.array(year)*float(slope) + intercept)*1387+1859, 'b', label = 'fitted line')
plt.plot(np.array(year)+1987, (np.array(gdp_per_capita))*1387+1859, '*',c='r', label = 'original data')
plt.title("Albania GDP growth")
plt.legend(title="Legend")
plt.xlabel("year")
plt.ylabel("gdp per capita")
plt.show()


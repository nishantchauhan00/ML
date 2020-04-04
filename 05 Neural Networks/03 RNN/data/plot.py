import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

data = pd.read_csv("data_stock.csv")
# data["Date"] = list(date.split("/") for date in data["Date"])
data["Volume"] = list(int(vol.replace(",", "")) for vol in data["Volume"])

plt.figure(1)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.plot(data["Date"].values, data["Open"].values)

# # similar to open
# plt.figure(2)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
# plt.plot(data["Date"].values, data["High"].values)

# similar to open
# plt.figure(3)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
# plt.plot(data["Date"], data["Low"])

# linear and similar to volume-fig5
# plt.figure(4)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
# plt.plot(data["Date"], data["Close"])

# linear
plt.figure(5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.plot(data["Date"], data["Volume"])

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("plantData/2025-04-08_14-11-25.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Plot the green percentage over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["green_percentage"], label="Green Percentage")
plt.xlabel("Time")
plt.ylabel("Green Percentage (%)")
plt.title("Plant Green Trend Over Time")
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D:/Uni shit/Thesis/Code_New/education_framework/runs/metrics.csv")
w = 100  # same as paper-like smoothing

df["reward_ma"] = df["reward"].rolling(w).mean()
df["steps_ma"]  = df["steps"].rolling(w).mean()

plt.figure()
plt.plot(df["episode"], df["reward_ma"])
plt.xlabel("Episode"); plt.ylabel(f"Mean reward (last {w})")
plt.show()

plt.figure()
plt.plot(df["episode"], df["steps_ma"])
plt.xlabel("Episode"); plt.ylabel(f"Mean steps/episode (last {w})")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# ler ficheiro
df = pd.read_csv(r"C:\Users\joaov\Desktop\TFM\knee_angles.csv")

# garantir nomes corretos
print(df.head())

# gráfico
plt.figure()
plt.plot(df["time_sec"], df["knee_angle"])

plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo joelho (°)")
plt.title("Ângulo do joelho ao longo do tempo")

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carica i dati
df1 = pd.read_csv("atr42_thermal_power_train_data.csv", quotechar='"', delimiter=";")
df2 = pd.read_csv("ATR42_power_train_data_morepower.csv", quotechar='"', delimiter=";")

# Pulisce le intestazioni
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Estrai le colonne
colonna1 = df1["turboshaft_1 fuel_consumption [kg/h]"].to_numpy()
colonna2 = df2["turboshaft_1 fuel_consumption [kg/h]"].to_numpy()



colonna1 = df1["turboshaft_1 fuel_consumed_t [kg]"].to_numpy()
colonna2 = df2["turboshaft_1 fuel_consumed_t [kg]"].to_numpy()
#Crea assi temporali (se non presenti nei dati)
x1 = np.arange(len(colonna1))
x2 = np.arange(len(colonna2))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x1, colonna1, label="Turboshaft Fuel Consumption [kg/h], full thermal", color='tab:blue')
plt.plot(x2, colonna2, label="Turboshaft Fuel Consumption [kg/h], hybrid", color='tab:orange')
plt.xlabel("Timestep")
plt.ylabel("Valore")
plt.title("Confronto tra efficienza dell’elica e consumo del turboshaft")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


colonna1 = df1["turboshaft_1 fuel_consumed_t [kg]"].to_numpy()
colonna2 = df2["turboshaft_1 fuel_consumed_t [kg]"].to_numpy()
# Crea assi temporali (se non presenti nei dati)
x1 = np.arange(len(colonna1))
x2 = np.arange(len(colonna2))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x1, colonna1, label="Turboshaft Fuel Consumption [kg], full thermal", color='tab:blue')
plt.plot(x2, colonna2, label="Turboshaft Fuel Consumption [kg], hybrid", color='tab:orange')
plt.xlabel("Timestep")
plt.ylabel("Valore")
plt.title("Confronto tra efficienza dell’elica e consumo del turboshaft")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


colonna1 = df1["turboshaft_1 specific_fuel_consumption [kg/kW/h]"].to_numpy()
colonna2 = df2["turboshaft_1 specific_fuel_consumption [kg/kW/h]"].to_numpy()
# Crea assi temporali (se non presenti nei dati)
x1 = np.arange(len(colonna1))
x2 = np.arange(len(colonna2))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x1, colonna1, label="Turboshaft specific Fuel Consumption [kg/kW/h], full thermal", color='tab:blue')
plt.plot(x2, colonna2, label="Turboshaft specific Fuel Consumption [kg/kW/h], hybrid", color='tab:orange')
plt.xlabel("Timestep")
plt.ylabel("Valore")
plt.title("Confronto tra efficienza dell’elica e consumo del turboshaft")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





colonna1 = df1["turboshaft_1 power_required [kW]"].to_numpy()
colonna2 = df2["turboshaft_1 power_required [kW]"].to_numpy()
# Crea assi temporali (se non presenti nei dati)
x1 = np.arange(len(colonna1))
x2 = np.arange(len(colonna2))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x1, colonna1, label="Turboshaft turboshaft_1 power_required [kW], full thermal", color='tab:blue')
plt.plot(x2, colonna2, label="Turboshaft turboshaft_1 power_required [kW], hybrid", color='tab:orange')
plt.xlabel("Timestep")
plt.ylabel("Valore")
plt.title("Confronto tra efficienza dell’elica e consumo del turboshaft")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

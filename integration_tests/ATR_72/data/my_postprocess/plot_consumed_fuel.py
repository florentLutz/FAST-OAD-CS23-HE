import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Carica il file CSV
df1 = pd.read_csv("fuel_propulsion_ATR_median_all.csv", quotechar='"', delimiter=";")
df2 = pd.read_csv("flight_points_RTA_Correct8MODIFIED.csv", quotechar='"', delimiter=";")

# Estrai colonne
colonna_time1 = df1["time"].to_numpy() + 360
colonna_fuel_consumed1 = df1["mass"].to_numpy()
colonna_time2 = df2["time"].to_numpy()
colonna_fuel_consumed2 = df2["consumed_fuel"].to_numpy()

# Inverti mass -> fuel consumed
colonna_fuel_consumed1 = colonna_fuel_consumed1[0] - colonna_fuel_consumed1

# Crea un array di tempi comuni fino a 11000
common_times = np.sort(np.unique(np.concatenate([colonna_time1, colonna_time2])))
common_times_filtered = common_times[common_times <= 11000]

# Interpolazione
interp_func1 = interp1d(colonna_time1, colonna_fuel_consumed1, kind="linear", bounds_error=False, fill_value="extrapolate")
interp_func2 = interp1d(colonna_time2, colonna_fuel_consumed2, kind="linear", bounds_error=False, fill_value="extrapolate")

interp_fuel1 = interp_func1(common_times_filtered)
interp_fuel2 = interp_func2(common_times_filtered)

# Calcolo errore
rmse = np.sqrt(np.mean((interp_fuel1 - interp_fuel2) ** 2))
rmse_percent = (rmse / np.mean(np.concatenate([interp_fuel1, interp_fuel2]))) * 100

# --- PLOT ---

plt.figure(figsize=(12, 6))

# Linee
plt.plot(colonna_time1, colonna_fuel_consumed1, label="HE", color="blue", marker="o")
plt.plot(colonna_time2, colonna_fuel_consumed2, label="RTA", color="red", linestyle="--")

# Punti specifici
highlight_points = [
    (colonna_time1[30], colonna_fuel_consumed1[30]),
    (1596.227431, 340.0923999),
    (colonna_time1[60], colonna_fuel_consumed1[60]),
    (9715.243209, 1979.014255),
    (colonna_time1[79], colonna_fuel_consumed1[79]),
    (10539.56616, 2032.905317),
]
for x, y in highlight_points:
    plt.scatter(x, y, color="red", zorder=5)

# Etichette
plt.xlabel("TIME (s)")
plt.ylabel("FUEL (kg)")
plt.title("Comparison of Fuel Consumed in RTA and HE Models")

# RMSE nel grafico
plt.text(
    500,
    max(max(colonna_fuel_consumed1), max(colonna_fuel_consumed2)) * 0.85,
    f"RMSE: {rmse:.2f} kg\nRMSE %: {rmse_percent:.2f}%",
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5"),
)

plt.legend()
plt.grid(True)
plt.xlim(0, 11000)
plt.tight_layout()
plt.show()

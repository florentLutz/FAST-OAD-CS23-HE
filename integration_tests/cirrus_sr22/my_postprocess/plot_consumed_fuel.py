# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.interpolate import interp1d
#
# # Carica il file CSV
# df1 = pd.read_csv(
#     "fuel_propulsion_ATR_median_all.csv", quotechar='"', delimiter=";"
# )  # Usa il delimitatore giusto se necessario
# df2 = pd.read_csv(
#     "flight_points_RTA_Correct8MODIFIED.csv", quotechar='"', delimiter=";"
# )  # Usa il delimitatore giusto se necessario
# # Stampa i nomi delle colonne
# print(df2.columns)
#
#
# # Supponiamo che il nome della colonna sia 'Altitude' e 'Fuel_Consumed'
# colonna_time2 = df2["time"].tolist()  # Usa il nome esatto della colonna
# colonna_fuel_consumed2 = df2["consumed_fuel"].to_numpy()  # Usa il nome esatto della colonna
#
# # Supponiamo che il nome della colonna sia 'Altitude' e 'Fuel_Consumed'
# colonna_time1 = df1["time"].tolist()  # Usa il nome esatto della colonna
# colonna_fuel_consumed1 = df1["mass"].to_numpy()  # Usa il nome esatto della colonna
#
# colonna_fuel_consumed1 = colonna_fuel_consumed1[0] - colonna_fuel_consumed1
# # Crea un grafico
# plt.figure(figsize=(10, 6))  # Puoi regolare la dimensione del grafico se necessario
#
# # Traccia la prima funzione (ad esempio, altitudine)
# plt.plot(colonna_time1, colonna_fuel_consumed1, label="HE", color="blue", marker="o")
#
# # Se vuoi tracciare una seconda funzione (ad esempio, potresti avere un'altra colonna o calcolare qualcosa di diverso)
# # Per esempio, se hai una seconda colonna chiamata 'true_airspeed'
#
# plt.plot(colonna_time2, colonna_fuel_consumed2, label="RTA", color="red", linestyle="--")
#
# specific_x = colonna_time1[30]
# specific_y = colonna_fuel_consumed1[30]
#
# plt.scatter(specific_x, specific_y, color="red", zorder=5)  # Punto rosso
#
# specific_x = [1596.227431]
# specific_y = [340.0923999]
# plt.scatter(specific_x, specific_y, color="red", zorder=5)
#
# specific_x = colonna_time1[60]
# specific_y = colonna_fuel_consumed1[60]
# plt.scatter(specific_x, specific_y, color="red", zorder=5)
#
# specific_x = [9715.243209]
# specific_y = [1979.014255]
# plt.scatter(specific_x, specific_y, color="red", zorder=5)
#
# specific_x = colonna_time1[79]
# specific_y = colonna_fuel_consumed1[79]
# plt.scatter(specific_x, specific_y, color="red", zorder=5)
#
# specific_x = [10539.56616]
# specific_y = [2032.905317]
# plt.scatter(specific_x, specific_y, color="red", label="change phase", zorder=5)
#
#
# # Aggiungi etichette e titolo
# plt.xlabel("TIME(s)")
# plt.ylabel("FUEL(kg)")
# plt.title("Comparison fuel consumed in RTA and HE model")
#
# # Aggiungi una legenda per distinguere le linee
# plt.legend()
#
#
# # Mostra il grafico
# plt.show()



# # Crea un array comune di timestamp
# # Uniamo i timestamp di entrambi i set di dati e ordiniamo
# common_times = np.sort(np.unique(np.concatenate([colonna_time1, colonna_time2])))
#
# # Definisci il timestamp di arresto
# stop_time = 11000  # Modifica con il valore desiderato, ad esempio, fermarsi a time = 100
#
# # Filtro dei timestamp fino a stop_time
# common_times_filtered = common_times[common_times <= stop_time]
#
# # Interpolazione per df1 senza estrapolazione
# interp_func1 = interp1d(
#     colonna_time1,
#     colonna_fuel_consumed1,
#     kind="linear",
#     bounds_error=False,
#     fill_value="extrapolate",
# )
# interp_fuel1 = interp_func1(common_times)
#
# # Interpolazione per df2 senza estrapolazione
# interp_func2 = interp1d(
#     colonna_time2,
#     colonna_fuel_consumed2,
#     kind="linear",
#     bounds_error=False,
#     fill_value="extrapolate",
# )
# interp_fuel2 = interp_func2(common_times)
#
# # Ora che i dati sono allineati, calcoliamo la distanza Euclidea
# distance = np.sqrt(np.sum((interp_fuel1 - interp_fuel2) ** 2))
#
# print(f"Distanza Euclidea tra le due curve (dopo interpolazione): {distance}")
#
# # Calcolare l'errore quadratico medio (RMSE)
# rmse = np.sqrt(np.mean((interp_fuel1 - interp_fuel2) ** 2))
#
# print(f"Errore Quadratico Medio (RMSE): {rmse}")
#
# # Calcolare la media dei valori osservati (può essere la media dei valori interpolati)
# mean_fuel = np.mean(np.concatenate([interp_fuel1, interp_fuel2]))
#
# # Calcolare l'RMSE percentuale
# rmse_percent = (rmse / mean_fuel) * 100
#
# print(f"Errore Quadratico Medio (RMSE): {rmse}")
# print(f"RMSE Percentuale: {rmse_percent:.2f}%")
#
#
# # # Creiamo il grafico delle curve interpolate
# # plt.figure(figsize=(10, 6))
# #
# # # Traccia le curve
# # plt.plot(
# #     common_times,
# #     interp_fuel1,
# #     label="Curva Interpolata df1",
# #     color="blue",
# #     linestyle="-",
# #     marker="o",
# # )
# # plt.plot(
# #     common_times,
# #     interp_fuel2,
# #     label="Curva Interpolata df2",
# #     color="red",
# #     linestyle="-",
# #     marker="x",
# # )
# #
# # # Etichetta del grafico
# # plt.title("Curve Interpolate di df1 e df2")
# # plt.xlabel("Time")
# # plt.ylabel("Fuel Consumed")
# # plt.legend()
# #
# # # Mostra il grafico
# # plt.grid(True)
# # plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Filtro dei timestamp fino a 1050 secondi
# plot_time_limit = 10500
# common_times_plot = common_times[common_times <= plot_time_limit]
#
# # Interpolazioni limitate per il plot
# interp_fuel1_plot = interp_func1(common_times_plot)
# interp_fuel2_plot = interp_func2(common_times_plot)
#
# # Creiamo il grafico delle curve interpolate
# plt.figure(figsize=(10, 6))
#
# # Traccia le curve interpolate fino a 1050s
# plt.plot(
#     common_times_plot,
#     interp_fuel1_plot,
#     label="Interpolated HE (df1)",
#     color="blue",
#     linestyle="-",
#     marker="o",
# )
# plt.plot(
#     common_times_plot,
#     interp_fuel2_plot,
#     label="Interpolated RTA (df2)",
#     color="red",
#     linestyle="-",
#     marker="x",
# )
#
# # Inserisci il valore di RMSE nel grafico
# plt.text(
#     50,  # x position
#     max(max(interp_fuel1_plot), max(interp_fuel2_plot)) * 0.9,  # y position
#     f"RMSE: {rmse:.2f} kg\nRMSE %: {rmse_percent:.2f}%",
#     fontsize=12,
#     bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5"),
# )
#
# # Etichetta del grafico
# plt.title("Interpolated Fuel Consumption up to 1050s")
# plt.xlabel("Time (s)")
# plt.ylabel("Fuel Consumed (kg)")
# plt.legend()
# plt.grid(True)
# plt.xlim(0, plot_time_limit)
#
# # Mostra il grafico
# plt.show()
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

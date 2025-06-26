import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV
df1 = pd.read_csv(
    "BM=0.5.csv", quotechar='"', delimiter=";"
)  # Usa il delimitatore giusto se necessario
df2 = pd.read_csv(
    "BM=1.csv", quotechar='"', delimiter=";"
)  # Usa il delimitatore giusto se necessario
df3 = pd.read_csv(
    "BM=1.5.csv", quotechar='"', delimiter=";"
)  # Usa il delimitatore giusto se necessario
df4 = pd.read_csv(
    "BM=2.csv", quotechar='"', delimiter=";"
)  # Usa il delimitatore giusto se necessario

# Rimuovi gli spazi dai nomi delle colonne
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df3.columns = df3.columns.str.strip()
df4.columns = df4.columns.str.strip()

# Stampa i nomi delle colonne
print(df2.columns)


# Supponiamo che il nome della colonna sia 'Altitude' e 'Fuel_Consumed'
colonna_frequenza_05 = df1["x"].to_numpy()  # Usa il nome esatto della colonna
colonna_sp_iron_losses_05 = df1["y"].to_numpy()  # Usa il nome esatto della colonna

colonna_frequenza_1 = df2["x"].to_numpy()  # Usa il nome esatto della colonna
colonna_sp_iron_losses_1 = df2["y"].to_numpy()  # Usa il nome esatto della colonna

colonna_frequenza_15 = df3["x"].to_numpy()  # Usa il nome esatto della colonna
colonna_sp_iron_losses_15 = df3["y"].to_numpy()  # Usa il nome esatto della colonna

colonna_frequenza_2 = df4["x"].to_numpy()  # Usa il nome esatto della colonna
colonna_sp_iron_losses_2 = df4["y"].to_numpy()  # Usa il nome esatto della colonna


# Dati di input (esempio)
# Bm = np.array([0.5, 1.0, 1.5, 2])      # Tesla
Bm = 1.5

# Prepariamo le feature combinando tutte le potenze di sqrt(Bm) e sqrt(f) fino a grado 4
X = []
for i in range(len(colonna_frequenza_05)):
    sqrt_Bm = np.sqrt(Bm)
    sqrt_f = np.sqrt(colonna_frequenza_05[i])
    features = []
    for i_pow in range(1, 5):
        for j_pow in range(1, 5):
            features.append((sqrt_f**i_pow) * (sqrt_Bm**j_pow))
    X.append(features)

X = np.array(X)

# Regressione lineare
model = LinearRegression(fit_intercept=False)  # niente termine noto, solo a_ij
model.fit(X, colonna_sp_iron_losses_05)

# Coefficienti a_ij
coeffs = model.coef_.reshape((4, 4))  # Matrice 4x4 di a_ij


# Salva la matrice in un file .npy
np.savetxt("coeffs05.txt", coeffs, delimiter="\t", fmt="%.6f")

# Output
print("Matrice dei coefficienti a_ij:")
print(coeffs)


# Previsione dei valori di Pf usando i coefficienti calcolati
Pf_pred = model.predict(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    colonna_frequenza_05, colonna_sp_iron_losses_05, color="red", label="Dati di partenza", zorder=5
)  # Dati effettivi
plt.plot(
    colonna_frequenza_05, Pf_pred, color="blue", label="Curva di regressione", zorder=10
)  # Curva di regressione
plt.title("Fitting della curva ai dati di partenza")
plt.xlabel("P_f (W/kg)")
plt.yticks([])  # Rimuovi l'asse y, poiché non serve
plt.legend(loc="best")

# Aggiungi una griglia
plt.grid(True)

# Mostra il grafico
plt.show()

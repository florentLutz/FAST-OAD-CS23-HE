import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def predici_perdite(model, f, Bm):
    sqrt_f = np.sqrt(f)
    sqrt_Bm = np.sqrt(Bm)

    # Costruisci le feature come nel training
    features = []
    for i_pow in range(1, 5):
        for j_pow in range(1, 5):
            features.append((sqrt_Bm**j_pow) * (sqrt_f**i_pow))

    # Converti in array 2D per il modello
    features = np.array(features).reshape(1, -1)

    # Predizione
    Pf_pred = model.predict(features)[0]
    return Pf_pred


# Carica i file CSV
df1 = pd.read_csv("BM=0.5.csv", quotechar='"', delimiter=";")
df2 = pd.read_csv("BM=1.csv", quotechar='"', delimiter=";")
df3 = pd.read_csv("BM=1.5.csv", quotechar='"', delimiter=";")
df4 = pd.read_csv("BM=2.csv", quotechar='"', delimiter=";")

# Rimuovi gli spazi dai nomi delle colonne
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df3.columns = df3.columns.str.strip()
df4.columns = df4.columns.str.strip()

# Estrazione dei dati per ogni file
colonna_frequenza_05 = df1["x"].to_numpy()
colonna_sp_iron_losses_05 = df1["y"].to_numpy()

colonna_frequenza_1 = df2["x"].to_numpy()
colonna_sp_iron_losses_1 = df2["y"].to_numpy()

colonna_frequenza_15 = df3["x"].to_numpy()
colonna_sp_iron_losses_15 = df3["y"].to_numpy()

colonna_frequenza_2 = df4["x"].to_numpy()
colonna_sp_iron_losses_2 = df4["y"].to_numpy()


# Combiniamo tutti i dati in un'unica matrice per la regressione
frequenze = np.concatenate(
    [colonna_frequenza_05, colonna_frequenza_1, colonna_frequenza_15, colonna_frequenza_2]
)
iron_losses = np.concatenate(
    [
        colonna_sp_iron_losses_05,
        colonna_sp_iron_losses_1,
        colonna_sp_iron_losses_15,
        colonna_sp_iron_losses_2,
    ]
)

# Dati per Bm (tutti e 4 i valori)
Bms = np.array([0.5, 1.0, 1.5, 2.0])


# Lista dei valori di Bm
Bm_values = [0.5, 1.0, 1.5, 2.0]

# Liste di array per frequenze e iron losses
frequenze_list = [
    df1["x"].to_numpy(),
    df2["x"].to_numpy(),
    df3["x"].to_numpy(),
    df4["x"].to_numpy(),
]

iron_losses_list = [
    df1["y"].to_numpy(),
    df2["y"].to_numpy(),
    df3["y"].to_numpy(),
    df4["y"].to_numpy(),
]

# Prepariamo le feature combinando le potenze necessarie di sqrt(Bm) e sqrt(f) fino a grado 4
X = []

for i in range(len(Bm_values)):
    f_i = frequenze_list[i]
    Pf_i = iron_losses_list[i]
    Bm_i = Bm_values[i]
    for freq in f_i:
        sqrt_f = np.sqrt(freq)
        # Aggiungiamo le combinazioni di (sqrt(Bm))^j e (sqrt(f))^i per ogni Bm
        features = []
        for i_pow in range(1, 5):  # i va da 1 a 4
            for j_pow in range(1, 5):  # j va da 1 a 4
                features.append((np.sqrt(Bm_i) ** j_pow) * (sqrt_f**i_pow))

        X.append(features)


X = np.array(X)

# Regressione lineare per trovare i coefficienti a_ij
model = LinearRegression(fit_intercept=False)  # Senza intercept, solo i coefficienti a_ij
model.fit(X, iron_losses)

# Coefficienti a_ij
coeffs = model.coef_

# Verifica il numero di coefficienti
print(f"Numero di coefficienti: {len(coeffs)}")

# Reshape dei coefficienti in una matrice 4x4 (poiché i,j vanno da 1 a 4)
coeffs_reshaped = coeffs.reshape(4, 4)
print("Matrice dei coefficienti reshaped (4x4):")
print(coeffs_reshaped)
np.save("coeffs_reshaped.npy", coeffs_reshaped)
# Colori e label per ciascun valore di Bm
Bm_values = [0.5, 1.0, 1.5, 2.0]
labels = [f"Bₘ = {bm}" for bm in Bm_values]
colors = ["red", "green", "orange", "purple"]

# Griglia comune di frequenze per il plot
frequenze_plot = np.linspace(min(frequenze), max(frequenze), 100)

# Plot dei dati originali
plt.scatter(frequenze, iron_losses, label="Dati originali", color="blue", alpha=0.5)

# Genera e plotta le curve per ogni Bm
for bm, label, color in zip(Bm_values, labels, colors):
    sqrt_Bm = np.sqrt(bm)
    X_plot = []
    for freq in frequenze_plot:
        sqrt_f = np.sqrt(freq)
        features_plot = []
        for i_pow in range(1, 5):
            for j_pow in range(1, 5):
                features_plot.append((sqrt_Bm**j_pow) * (sqrt_f**i_pow))
        X_plot.append(features_plot)

    X_plot = np.array(X_plot)
    predictions = model.predict(X_plot)

    plt.plot(frequenze_plot, predictions, label=label, color=color)

# Dettagli del plot
plt.xlabel("Frequency (Hz)")
plt.ylabel("Iron losses (W/kg)")
plt.title("Iron losses fitting for different values of Bₘ")
plt.legend()
plt.grid(True)
plt.show()

# Esempio: predizione per f = 120 Hz e Bm = 1.2 T
f_test = 15970 / 30
Bm_test = 0.9

Pf_stimato = predici_perdite(model, f_test, Bm_test) * 224.88
print(f"Per f = {f_test} Hz e Bm = {Bm_test} T, le perdite stimate sono: {Pf_stimato:.2f} W/kg")


def predici_perdite(model, f, Bm):
    sqrt_f = np.sqrt(f)
    sqrt_Bm = np.sqrt(Bm)

    # Costruisci le feature come nel training
    features = []
    for i_pow in range(1, 5):
        for j_pow in range(1, 5):
            features.append((sqrt_Bm**j_pow) * (sqrt_f**i_pow))

    # Converti in array 2D per il modello
    features = np.array(features).reshape(1, -1)

    # Predizione
    Pf_pred = model.predict(features)[0]
    return Pf_pred

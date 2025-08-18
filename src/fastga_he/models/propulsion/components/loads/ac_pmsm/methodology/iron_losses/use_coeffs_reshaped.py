import numpy as np

# Carica i coefficienti salvati (matrice 4x4)
coeffs_reshaped = np.load("coeffs_reshaped.npy")

if __name__ == "__main__":
    # Valori di test
    f = 15970 / 30  # Frequenza in Hz
    Bm = 0.9  # Induzione magnetica massima in Tesla

    sqrt_f = np.sqrt(f)
    sqrt_Bm = np.sqrt(Bm)

    Pf_pred = 0

    for i in range(4):  # i = potenza di sqrt(f)
        for j in range(4):  # j = potenza di sqrt(Bm)
            coeff = coeffs_reshaped[i][j]
            term = coeff * (sqrt_f ** (i + 1)) * (sqrt_Bm ** (j + 1))
            Pf_pred += term

    # Se necessario, moltiplica per un fattore di scala (es. 224.88)
    Pf_pred *= 224.88

    print(f"Per f = {f:.2f} Hz e Bm = {Bm} T, le perdite stimate sono: {Pf_pred:.2f} W/kg")

import numpy as np
import matplotlib.pyplot as plt

# Vettori di esempio: tempo in minuti, fuel in kg
powershare = np.array([1305, 1350, 1400, 1450, 1600])  # tempo [min]
fuel = np.array([2203, 2183.70, 2210.6, 2226.581, 2246.419])  # carburante residuo [kg]
EMISSION8FACTOR = np.array([1.596, 1.4970, 1.4821, 1.45716, 1.39746])
# Plot
plt.figure(figsize=(8, 5))
plt.plot(powershare, fuel, marker='o', color='blue', linestyle='-')

# Annotazioni
plt.title("fuel  vs powershare")
plt.xlabel("powershare [kw]")
plt.ylabel("fuel [kg]")
plt.grid(True)
plt.tight_layout()

plt.show()

plt.figure(figsize=(8, 5))
plt.plot(powershare, EMISSION8FACTOR, marker='o', color='blue', linestyle='-')

# Annotazioni
plt.title("EMISSION8FACTOR  vs powershare")
plt.xlabel("powershare [kw]")
plt.ylabel("emissions [kgCO2/kg/km]")
plt.grid(True)
plt.tight_layout()

plt.show()
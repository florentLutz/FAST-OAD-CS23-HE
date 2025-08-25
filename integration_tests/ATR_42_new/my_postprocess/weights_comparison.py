import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio (sostituiscili con i tuoi valori reali)
labels = ["MTOW", "OWE", "Payload", "Fuel - Mission", "Powertrain"]
# config1 = [17843, 11496, 4560, 1787, 1182]
# config2 = [17843, 11949, 4155, 1739, 1567]
# config3 = [18390, 12008, 4560, 1822, 1586]

config1 = [17843, 11496, 4560, 1787, 1182]
config2 = [18390, 12008, 4560, 1822, 1586]
config3 = [21034, 14326, 4560, 2269, 3576]


# Impostazioni
x = np.arange(len(labels))
width = 0.25

# Creazione del grafico
fig, ax = plt.subplots(figsize=(6, 6))
bars1 = ax.bar(x - width, config1, width, label="Conventional, Fuel: 1787 kg", color="royalblue")
bars2 = ax.bar(x, config2, width, label="Hybrid Parallel, Fuel: 1822 kg", color="mediumseagreen")
bars3 = ax.bar(x + width, config3, width, label="Pure Series, Fuel: 2269 kg", color="mediumorchid")

# Etichette e titolo con font size maggiore
ax.set_ylabel("[kg]", fontsize=24)
ax.set_title("Maximum Take-Off Weight Breakdown", fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24)
ax.legend(fontsize=24)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)

# Sfondo azzurrino chiaro
ax.set_facecolor("#eaf2fa")

# Asse Y più leggibile
ax.tick_params(axis="y", labelsize=24)

plt.tight_layout()
plt.show()

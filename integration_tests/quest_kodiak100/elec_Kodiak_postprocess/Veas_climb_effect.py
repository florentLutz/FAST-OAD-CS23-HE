import matplotlib.pyplot as plt

# Esempio di dati
Veas = [91, 96, 101, 106, 111, 116, 121, 126, 131, 136]      # Asse X
MTOW = [7793, 7546, 7368, 7260, 7204, 7194, 7075, 7115, 8684, 8730]    # Asse Y
Shaft_power_rating = [1096, 1054, 1023, 1002, 990, 985, 986, 994, 1367, 1388]    # Asse Y

# Plot
plt.figure(1)
plt.plot(Veas, MTOW, marker='o', linestyle='-', color='b')
plt.xlabel('Veas (knots)')
plt.ylabel('MTOW (kg))')
plt.title('Veas effect on MTOW')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(2)
plt.plot(Veas, Shaft_power_rating, marker='o', linestyle='-', color='b')
plt.xlabel('Veas (knots)')
plt.ylabel('Shaft Poxer Rating (Kw)')
plt.title('Veas effect on Shaft Power rating')
plt.grid(True)
plt.legend()
plt.show()

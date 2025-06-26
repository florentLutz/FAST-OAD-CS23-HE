import matplotlib.pyplot as plt

# Esempio di dati
Veas = [91, 96, 101, 106, 111, 116, 121, 126, 131, 136]      # Asse X
MTOW = [4200, 4184, 4176, 4174, 4177, 4186, 4200, 4219, 4243, 4275]    # Asse Y
Shaft_power_rating = [656.7, 650.8, 648.8, 650.1, 654.6, 661.5, 670.9, 682.8, 697.2, 713.9]    # Asse Y

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

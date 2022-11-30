import numpy as np
import scipy.constants as sc
import openmdao.api as om
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit

##### 40% potassium formate density calculation, webplot digitzer data

T = [
    -34.72868217054261,
    -13.023255813953469,
    -3.4108527131782864,
    7.441860465116292,
    16.124031007751938,
    23.25581395348837,
    37.82945736434108,
]
rho = [
    1279.8571428571427,
    1273.5714285714284,
    1270.4285714285713,
    1265.7142857142856,
    1262.5714285714284,
    1259.4285714285713,
    1253.142857142857,
]

T = [x + 273.15 for x in T]
T = np.array(T)

rho = np.array(rho)

# define the true objective function
def objective(T, a, b, c):
    return a * T**2 + b * T + c


# curve fit
popt, pcov = curve_fit(objective, T, rho)
# summarize the parameter values
a, b, c = popt
print("y = %.5f * x**2 + %.5f * x + %.5f" % (a, b, c))

# plot
plt.plot(T, rho)
plt.plot(T, a * T**2 + b * T + c)
plt.xlabel("Temperature (K)")
plt.ylabel("Density (J/kg/K)")
plt.title("Potassium formate")
plt.show()

# getting R²
residuals = rho - objective(T, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((rho - np.mean(rho)) ** 2)

r_squared = 1 - (ss_res / ss_tot)
print("R squared value:", r_squared)

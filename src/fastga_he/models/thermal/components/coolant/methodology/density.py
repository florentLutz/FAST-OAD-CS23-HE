import numpy as np
import scipy.constants as sc
import openmdao.api as om
from sklearn.linear_model import LinearRegression


##  40% potassium formate density calculation, webplot digitzer data

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
print(T)
# lin = LinearRegression()
# lin.fit(T, rho)

# print(lin)

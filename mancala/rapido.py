#distribucion donde el 1 ocurra el 75% de las veces, random entre 0 y 1
import numpy as np
np.random.choice([0, 1], p=[0.25, 0.75])
Q=Q+lr*(1-Q)

#converger q a .75
"""
lr=1
lr=0.1
lr=0.01
lr=0.001
"""

#graficar la evolucion de q
import matplotlib.pyplot as plt

#flta que sea estadictuca significativo
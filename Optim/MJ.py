import numpy as np
import PyMoosh as pm
from PyMoosh.optimization import optimization
import matplotlib.pyplot as plt
"""
DON'T FORGET TO INSTALL THE CORRESPONDING LIBRARIES:
- PyMoosh
- matplotlib
- numpy
- Refractiveindex
- concurrent.futures
"""

### IN PROGRESS... ###

# Fixing the structure:
n = 50
mat = [1.0, 1.5**2, 2.0**2, "SiA"]

stack =  [0] + [2, 1] * n + [2] + [0]
thickness = [100.0 for _ in stack]
which_layers = np.array([i != 1 for i in stack])#np.bool_(np.ones_like(stack)[0:]) #np.array([i != 3 for i in stack])
min_th, max_th = 0.0, 150.0

#X_max = np.linspace(200, 400, len(stack))
X_min = min_th * np.ones_like(stack)
X_max = max_th * np.ones_like(stack)

# Light parameters:
wl_domain = np.linspace(400, 600, 400)
incidence = 0.0
polar = 0

# Optimization paramters:
def objective(wl):
    wl1 = 500
    return np.heaviside(wl - wl1, 0)

nb, cut = 50, 10
left, right = np.linspace(400, 500 - cut, nb), np.linspace(500+cut, 600, nb)
cw = np.concatenate((left, right))
budget = 10000
nb_runs = 1

# Finally, start optimization:

optim = optimization(mat, stack, thickness, incidence, polar, X_min, X_max, cw, budget, nb_runs, wl_domain,
                     objective, ['R', 'A'], optimizer='QODE', wl_plot_stack=600)

struct = optim.run()
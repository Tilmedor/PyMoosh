import numpy as np
import PyMoosh as pm
from PyMoosh.optimization import optimization
"""
DON'T FORGET TO INSTALL THE CORRESPONDING LIBRARIES:
- PyMoosh
- matplotlib
- numpy
- Refractiveindex
- concurrent.futures
"""
# Fixing the structure:
n = 30
mat = [1.0, 1.5**2, 2.0**2]
stack =  [0] + [1, 2] * n + [1, 0]
thickness = [999.0 for _ in stack]
min_th, max_th = 50.0, 100.0
X_min = min_th * np.ones_like(stack) 
X_max = max_th * np.ones_like(stack)  

# Light parameters:
wl_domain = np.linspace(400, 800, 400)
incidence = 0.0
polar = 0

# Optimization paramters:
def objective(wl):
    wl1, wl2 = 550, 650
    return np.heaviside(wl - wl1, 0) * np.heaviside(wl2 - wl, 0)

computation_window = np.array((600.0))
budget = 100000
nb_runs = 1

def custom_function(struct, domain):
    return np.real(pm.coefficient(struct, domain, incidence, polar, wavelength_opti=True)[0])
## Finally, start optimization:

optim = optimization(mat, stack, thickness, incidence, polar, X_min, X_max, computation_window, budget, nb_runs, wl_domain, objective, ['R'], wl_plot_stack=600, verbose=False)

optim.run()
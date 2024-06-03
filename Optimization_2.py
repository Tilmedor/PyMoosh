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
which_layers = np.bool_(np.ones_like(stack))
min_th, max_th = 50.0, 100.0
min_n, max_n = 1.5**2, 2.0**2
X_min = min_th * np.ones_like(stack) 
X_max = max_th * np.ones_like(stack)  

# Light parameters:
wl_domain = np.linspace(400, 800, 1000)
incidence = 0.0
polar = 0

# Optimization paramters:
def objective(wl):
    wl1, wl2 = 550, 650
    return np.heaviside(wl - wl1, 0) * np.heaviside(wl2 - wl, 0)

computation_window = np.array((600.0))
objective_vector = objective(computation_window)

def cost(layers):
    structure = pm.Structure(mat, stack, list(layers), verbose=False) # 6.8e-5
    R_vector = pm.coefficient(structure, computation_window, incidence, polar, wavelength_opti=True)[2] # 6.4e-3
    return np.linalg.norm(objective_vector - R_vector, ord=None)/len(computation_window)

cost_function = cost
bound_min = X_min
bound_max = X_max
indices = False
budget = 10000
nb_runs = 1
optimizer = 'QNDE'

## Plot parameters:

def f_draw(struct):
    return pm.coefficient(struct, wl_domain, incidence, polar, wavelength_opti=True)[2]

progression = True
objective_title = 'Reflectance in function of wavelength, one run.'
objective_ylabel = 'reflectance'
wl_plot_stack = 600.0
precision = 3

## Finally, start optimization:

optim = optimization(mat, stack, thickness, wl_domain, incidence, polar,
                     indices, cost_function, X_min, X_max, computation_window, which_layers, budget, nb_runs, optimizer,  
                     objective_vector, f_draw, progression, objective_title, objective_ylabel, wl_plot_stack, precision=3, verbose=False)

optim.run()
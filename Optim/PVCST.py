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

"""# Fixing the structure:
n = 15
mat = [1.0, 1.5**2, 2.0**2]
stack =  [0] + [2, 1] * n + [2, 0]
thickness = [100.0 for _ in stack]
which_layers = np.array([i != 1 for i in stack])#np.bool_(np.ones_like(stack)[0:]) #np.array([i != 3 for i in stack])
min_th, max_th = 0.0, 150.0
min_n, max_n = 1.4**2, 2.1**2
X_min = min_th * np.ones_like(stack)
X_max = max_th * np.ones_like(stack)
X_min = np.concatenate((min_th * np.ones_like(stack), min_n * np.ones_like(stack)))
X_max = np.concatenate((max_th * np.ones_like(stack), max_n * np.ones_like(stack)))
print(X_min)
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
computation_window = np.concatenate((left, right))
budget = 25000
nb_runs = 1

# Finally, start optimization:

optim = optimization(mat, stack, thickness, incidence, polar, X_min, X_max, computation_window, budget, nb_runs, wl_domain,
                     objective, ['R'], optimizer='QODE', wl_plot_stack=600, verbose=False,  indices=True, which_layers=which_layers) #which_layers=which_layers,

#optim.run()"""
"""
th = [148.5781637,   92.13281683, 165.71673389, 169.58497928, 208.33550383,
 129.49516222, 160.25020626, 204.35465038, 127.1035741 , 223.90682296,
  65.66300478, 131.57729693, 223.56012969, 104.34269643, 209.92703928,
 236.61041649, 223.14115951, 324.27472217, 176.20105922, 194.98226572,
 183.58120561, 224.8319449 , 208.11442658, 214.88112391, 194.20691089,
 336.63025276, 211.32949468, 212.65861585, 181.98667139, 308.51859726,
 201.2053048 , 127.77283009, 211.71859123, 120.28252388, 205.06809235,
 132.57614369, 193.9770684 ,  72.10225141, 174.58326465, 319.35489472,
  24.69346818, 116.44418528, 370.10053467, 204.16847833, 169.58309371,
 206.00983926, 186.22829224, 215.59964331, 395.90893316, 196.65567508,
 198.1483292 , 146.76378753, 201.95808283, 187.22465526, 189.54925118,
 299.38748245, 201.11281397, 136.57488206, 198.46905537, 329.30869288,
 220.28977288, 230.44112153, 267.62590586]

s = pm.Structure(mat, stack, th, verbose=False)
R = pm.coefficient(s, wl_domain, incidence, polar, wavelength_opti=True)[2]
plt.plot(wl_domain, R)
plt.show()"""

# Fixing the structure:

"""mat = [1.0, pm.Material(['main', 'SiO2', 'Malitson'], 'RII')]
stack = [0, 1, 0]
thickness = [100.0, 100.0, 100.0]
"""

n = 50

mat = [1.0, pm.Material(['main', 'SiO2', 'Malitson'], 'RII'), 2.0**2]
stack =  [0] + [2, 1] * n + [2, 0]
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
budget = 6000
nb_runs = 1

# Finally, start optimization:

optim = optimization(mat, stack, thickness, incidence, polar, X_min, X_max, cw, budget, nb_runs, wl_domain,
                     ['R'], objective, optimizer='QNDE', wl_plot_stack=600)

struct, info = optim.run()
info = info[0]
print(f'population:{info[0]}')
print(f'cluster_count:{info[1]}')
print(f'noise_count:{info[2]}')
print(f'best_in_cluster_list:{info[3]}')
print(f'cost_best_list:{info[4]}')
print(f'density_list:{info[5]}')


#struct = optim.run()

#optim.robustness(struct, distance=2, budget=100)
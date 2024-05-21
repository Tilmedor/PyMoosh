import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import concurrent.futures
import sys
plt.rcParams['figure.dpi'] = 150

class optimization:
    """
    One code to optimize them all, one code to find solutions,
    One code to try them all, and in the convergence bind them;
    In the Land of PyMoosh where the multilayers lie.
    
    Possible choices for the optimizer:    | name (str)
    ---------------------------------------|-------------
    differential evolution                 | 'DE'
    quasi opposite differential evolution  | 'QODE'
    quasi newtonian differential evolution | 'QNDE'
    bfgs                                   | 'BFGS'
    """
    def __init__(
        self,
        struct: pm.Structure,
        which_layers: np.ndarray,
        wl_domain: np.ndarray, # only for plot
        incidence: float,
        polar: int,
        cost_function,
        computation_window: np.ndarray, # only for computation
        objective_vector: np.ndarray,
        bound_min: np.ndarray,
        bound_max: np.ndarray,
        indices: bool = False,
        budget: int = 1000,
        nb_runs: int = 1,
        optimizer: str = 'DE',
        progression: bool = False,
        objective_title: str = "default title",
        objective_ylabel: str = "default",
        wl_plot_stack: float = 500,
    ):  
        # Structure to optimize. Thicknesses are supposed to be optimized by default.
        # Optimizing optical indices is also possible, then the chosen materials are 
        # not impacting the result, but the boundaries are. Stack is always fixed.

        """
        if isinstance(self.struct.materials, np.ndarray) or isinstance(self.struct.materials, list):
            self.materials = np.asarray(self.struct.materials) #, dtype=float)
        else:
            print(f'Required material should be a list or an array, but a {type(struct.materials)} were given.')
            sys.exit()

        if isinstance(self.struct.layer_type, np.ndarray) or isinstance(self.struct.layer_type, list):
            self.stack = np.asarray(self.struct.layer_type, dtype=int)
        else:
            print(f'Required stack should be a list or an array, but a {type(struct.layer_type)} were given.')
            sys.exit()

        if isinstance(self.struct.thickness, np.ndarray) or isinstance(self.struct.thickness, list):
            self.thickness = np.asarray(self.struct.thickness, dtype=float)
        else:
            print(f'Required material should be a list or an array, but a {type(struct.thickness)} were given')
            sys.exit()
        """
        self.which_layers = np.bool_(np.ones_like(which_layers)) # which layer to optimize

        # Light parameters:
        self.wl_domain = wl_domain # only for plots
        self.incidence = incidence
        self.polar = polar

        # Computation parameters:
        self.cost_function = cost_function
        self.computation_window = computation_window
        self.objective_vector = objective_vector
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.indices = indices
    
        if indices:
            # we change the stack-material relation by a one-by-one link, 
            # because the optical indices will change over all optimized
            # layers. 
            new_materials = np.ones_like(struct.layer_type, dtype=pm.Material)
            for count, mat in enumerate(struct.materials):
                mask = (np.asarray(struct.layer_type)[:] == count)
                np.putmask(new_materials, mask, np.full((1,len(new_materials)), mat))
            struct.materials = new_materials
            struct.layer_type = np.arange(len(struct.layer_type), dtype=int)
        
        self.budget = budget
        self.nb_runs = nb_runs
        self.optimizer = optimizer
        self.progression = progression

        # Plots parameters:
        self.progression = progression
        #self.f = f # could be a function to plot or a list of function to plot
        self.objective_title = objective_title
        self.objective_ylabel = objective_ylabel
        self.wl_plot_stack = wl_plot_stack

    def wrapper_optimize(self):
        if self.optimizer == 'DE':
            return pm.differential_evolution(self.cost_function, self.budget, self.bound_min, self.bound_max, progression=self.progression)
        
        elif self.optimizer == 'QODE':
            return pm.QODE(self.cost_function, self.budget, self.bound_min, self.bound_max, progression=self.progression)
        
        elif self.optimizer == 'QNDE':
            return pm.QNDE(self.cost_function, self.budget, self.bound_min, self.bound_max, progression=self.progression)
        
        elif self.optimizer == 'BFGS':
            return pm.bfgs(self.cost_function, self.budget, struct.thickness, [self.bound_min, self.bound_max])
        
        else:
            print('Unknown optimizer. See docstring:')
            print(self.__doc__)
            sys.exit()

    def run(self):
        if self.nb_runs == 1:
            # Start time.
            print("Current Time =", datetime.now().strftime("%H:%M:%S"))
            start = time.perf_counter()
            
            # Optimize.
            best, convergence = self.wrapper_optimize()

            # Finish time.
            perf = round(time.perf_counter()-start, 2)
            print(f'Finished in {perf // 60} min {round(perf % 60, 2)} seconds.')

            print('Best guess found:')                
            if self.indices:
                lim = len(best)//2
                th = np.asarray(struct.thickness, dtype=float)
                mat = np.asarray(struct.materials)
                np.putmask(th, self.which_layers, best[:lim])
                np.putmask(mat, self.which_layers, best[lim:])
                struct.update(th, mat)
                print('thicknesses:')
                print(struct.thickness)
                print('materials:')
                for i in struct.materials:
                    print(i)

            else:
                th = np.asarray(struct.thickness, dtype=float)
                np.putmask(th, self.which_layers, best)
                struct.update(th)
                print('thicknesses:')
                print(struct.thickness)

            # Plot convergence.
            plt.plot(convergence)
            plt.suptitle(f'Convergence curve, one run, {self.budget} budget.')
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.show()

            R_best = pm.coefficient(struct, self.wl_domain, self.incidence, self.polar, wavelength_opti=True)[2]

            # Plot objective.
            plt.plot(self.computation_window, self.objective_vector, label='objective')
            plt.plot(self.wl_domain, R_best, label='best guess')
            plt.suptitle(self.objective_title)
            plt.xlabel("wavelength, nm")
            plt.ylabel(self.objective_ylabel)
            plt.legend(loc='best')
            plt.show()

            # Plot stack
            struct.plot_stack(wavelength=self.wl_plot_stack, lim_eps_colors=[1.5, 4])

        elif self.nb_runs != 1:
            def stats(_):
                best, convergence = pm.differential_evolution(self.cost_function, self.budget, self.bound_min, self.bound_max)
                return best, convergence
            best_list, convergence_list = [], []

            # Time.
            print("Current Time =", datetime.now().strftime("%H:%M:%S"))
            start = time.perf_counter()

            # Statistics.
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(stats, _) for _ in range(self.runs)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    best_list.append(result[0])
                    convergence_list.append(result[1]) 

            # Finish time.
            perf = round(time.perf_counter()-start, 2)
            print(f'Finished in {perf // 60} min {round(perf % 60, 2)} seconds.')

            # Plot convergence & stats.
            for convergence in convergence_list:
                plt.plot(convergence)
            plt.suptitle(f'Consistency curve.')
            plt.title(f'{self.runs} runs, {self.budget} budget.')
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.show()

            # Statistics plot.
            convergence_value_list = [convergence[-1] for convergence in convergence_list]
            plt.hist(convergence_value_list)
            plt.suptitle(f'Histograms of convergent cost values.')
            plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
            plt.xlabel("Convergence Cost value")
            plt.ylabel("Occurance")
            plt.show()

            # Printing best.
            costs = [self.cost_function(best) for best in best_list]
            index_best = np.argmin(costs)
            best = best_list[index_best]
            print('Best guess found:')                
            if self.indices:
                np.putmask(self.thickness, self.which_layers, best[len(best)//2:])
                print('thicknesses:')
                print(self.thickness)
                np.putmask(self.thickness, self.which_layers, best[:len(best)//2])
                print('optical indices:')
                print(mat)
            else:
                np.putmask(self.thickness, self.which_layers, best)
                print('thicknesses:')
                print(thickness)

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    ## Fixing the structure:
    n = 5
    mat = [1.4, 1.7, 2.0, 1.0]
    stack =  [0, 1] * n # [3] + + [2]
    thickness = [300 for _ in range(2*n)] # not so important #[0] ++ [100]
    struct = pm.Structure(mat, stack, thickness, verbose=False)

    which_layers = np.ones_like(stack) #np.bool_(np.ones_like(stack))#np.bool_(np.concatenate(([0],np.ones(2*n),[0]))) # array of boolean
    min_th, max_th = 0, 600
    min_n, max_n = 1.0, 2.0
    X_min = min_th * np.ones(2*n) 
    X_max = max_th * np.ones(2*n)  
    #X_min = np.append( min_th * np.ones(2*n), min_n * np.ones(2*n) )
    #X_max = np.append( max_th * np.ones(2*n), max_n * np.ones(2*n) )
    
    ## Light parameters:
    wl_domain = np.linspace(400, 800, 1000)
    incidence = 0.0
    polar = 0

    ## Optimization paramters:

     # First define objective function:
    def objective(wl):
        wl1, wl2 = 550, 600
        return np.heaviside(wl - wl1, 0) * np.heaviside(wl2 - wl, 0)
    
    computation_window = np.linspace(551, 599, 1)
    objective_vector = objective(computation_window)

     # Then define cost function:
    ### LE BANDEAU DOIT IMPERATIVEMENT ETRE DE LA FORME SUIVANTE : LAYERS / MAT / STACK / THICKNESS / WHICH_LAYERS
    
    def cost(layers): # total = 6.6e-3
        # layers == thicknesses. Update the thickness:
        #np.putmask(np.asarray(thickness, dtype=float), which_layers, layers)
        # Create the new structure:
        structure = pm.Structure(mat, stack, layers, verbose=False) # 6.8e-5
        # Compute the new Reflectance coefficient:
        R_vector = pm.coefficient(structure, computation_window, incidence, polar, wavelength_opti=True)[2] # 6.4e-3
        # How far the new reflectance is with respect to the objective ?
        #print(R_vector)
        return np.linalg.norm(R_vector - objective_vector, ord=None)
    """  
    def cost(layers):
        new_thickness = np.zeros_like(thickness, dtype=float)
        # layers == thicknesses. Update the thickness:
        np.putmask(new_thickness, which_layers, layers)
        # Update structure:
        struct.update(new_thickness)
        # Compute the new Reflectance coefficient:
        R_vector = pm.coefficient(struct, computation_window, incidence, polar, wavelength_opti=True)[2] 
        # How far the new reflectance is with respect to the objective ?
        return np.linalg.norm(R_vector - objective_vector, ord=None)
    """
    """
    def cost_indices(layers_and_indices, mat, stack, thickness, which_layers):
        # layers[:lim] == thicknesses, and layers[lim:] == optical indices.
        lim = len(layers_and_indices)//2
        # Update the thickness:
        np.putmask(np.asarray(thickness, dtype=float), which_layers[:lim], np.asarray(layers_and_indices[:lim]))
        # Update the optical indices: 
        print(mat)
        np.putmask(np.asarray(mat, dtype=float), which_layers[lim:], np.asarray(layers_and_indices[lim:]))
        # Create the new structure:
        structure = pm.Structure(mat, stack, thickness, verbose=False)
        # Compute the new Reflectance coefficient:
        R_vector = pm.coefficient(structure, computation_window, incidence, polar, wavelength_opti=True)[2]
        # How far the new reflectance is with respect to the objective ?
        return np.linalg.norm(R_vector - objective_vector, ord=None)
    """
    def cost_indices(layers_and_indices):
        lim = len(layers_and_indices)//2
        new_thickness = np.zeros_like(struct.thickness, dtype=float)
        new_materials = np.zeros_like(struct.materials, dtype=float)

        # layers[:lim] == thicknesses, and layers[lim:] == optical indices.
        
        # Update the thickness:
        np.putmask(new_thickness, which_layers, np.asarray(layers_and_indices[:lim]))
        # Update the optical indices: 
        np.putmask(new_materials, which_layers, np.asarray(layers_and_indices[lim:]))
        # Update structure:
        struct.update(new_thickness, new_materials) #= new_materials
        # Compute the new Reflectance coefficient:
        R_vector = pm.coefficient(struct, computation_window, incidence, polar, wavelength_opti=True)[2]
        # How far the new reflectance is with respect to the objective ?
        return np.linalg.norm(R_vector - objective_vector, ord=None)
    
    cost_function = cost
    bound_min = X_min
    bound_max = X_max
    indices = False
    budget = 30000
    nb_runs = 1
    optimizer = 'QNDE'

    ## Plot parameters:
    progression = True
    draw_convergence = True
    draw_objective = True
    draw_stack = True
    objective_title = 'Reflectance in function of wavelength, one run.'
    objective_ylabel = 'reflectance'

    ## Finally, start optimization:
    optim = optimization(struct, which_layers, wl_domain, incidence, polar, cost_function, 
                         computation_window, objective_vector, bound_min, bound_max, indices, 
                         budget, nb_runs, optimizer, progression, objective_title, objective_ylabel)

    optim.run()
    """
    structure = pm.Structure(mat, stack, thickness, verbose=False) # 3.4e-5
    structure.thickness = thickness
    R_vector = pm.coefficient(structure, computation_window, incidence, polar, wavelength_opti=True)[2] # 6.4e-3
    obj_vector = objective(computation_window) # 2.8e-5
    np.linalg.norm(R_vector - obj_vector, ord=None) # 2.9e-5 ord=None
    """
    """
    start = time.perf_counter()
   
    perf = time.perf_counter()-start
    print(perf)
    """
    """
    thickness = np.array(thickness)
    which_layers = np.bool_(which_layers)
    print(which_layers)
    print(np.logical_not(which_layers))
    layers = np.arange(90)
    np.putmask(np.asarray(thickness), np.asarray(which_layers), layers)
    print(thickness)"""

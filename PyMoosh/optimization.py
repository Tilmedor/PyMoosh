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
    
    */* PHYSICS *\*

    - mat (list or array of floats or pm.Materials): list of materials*.

    - stack (list or array of integers)*

    - thickness (list or array of floats)*

    - wl_domain (nunmpy ndarray of floats): The wavelength region to plot.
      Only for plots, and not computations (see below).

    - incidence (float): incidence of the light beam above surface (in degree °, 
      between 0° (normal) and 90 ° (tangent) ).

    - polar (0 or 1): field polarization. 0 == TE, other is TM.

    */* OPTIMIZATION *\*

    - indices (boolean): If True, the optical indices are also
      optimized. Optimized layers are indicated by 'which_layers' (see above).
      WARNING : if True, the stack needs to be 'np.arange(len(stack))' to work properly.

    - which_layers (nunmpy ndarray of boolean): Same length of 'stack'.
      It indicates which layers are optimized in the stack (which_layer[i] = True).
      Neither the thicknesses nor the optical indices are optimized for the others
      (which_layer[i] = False). 

    - cost_function (function): function to minimize. It requires only
      the argument 'layers'.

    - X_min (nunmpy ndarray of floats): lower boundaries of the optimization
      domain, a vector with the same size as the argument of the cost function.

    - X_max (nunmpy ndarray of floats): upper boundaries, see just above.

    - computation_window (nunmpy ndarray of floats): The wavelength
      optimization region. Only for computations.

    - computation window : NOT FINISHED

    - budget (integer, default = 1000): number of iteration for optimization.

    - nb_runs (integer, default = 1): number of time the optimization is done.

    - optimizer (string, default = 'DE'): Global Optimization algorithm used.

      Possible choices for the optimizer**:  | name (string)
      ---------------------------------------|--------------
      differential evolution                 | 'DE'
      quasi opposite differential evolution  | 'QODE'
      quasi newtonian differential evolution | 'QNDE'
      bfgs                                   | 'BFGS'

    */* PLOTS *\*

    Depending on the number of runs, several plots can show:
    If 'nb_runs' = 1, a 'convergence plot' will be shown first. Then a'comparison plot'
    will appear between the 'objective_vector' and 'f_draw'. Otherwise, if 'nb_runs' is 
    higher than 1, a 'consistency plot' will be shown, which is a superposition of the
    'nb_runs' 'consistency' plots. Independently of 'nb_runs' value, a diagram of the
    thicknesses and the optical indices will appear.

    - objective_vector (nunmpy ndarray of floats): A vector to visually compare
    to 'f_draw'.

    - f_draw (function): function to print and compare to the 'objective_function'.

    - progression (boolean, default = False): If True, prints the optimization progression
      as a percentage of computation.

    - objective_title (string, default = "comparison plot"): Title for comparison plot
       between the desired objective and the actual optimization.

    - objective_ylabel (string, default = "default"): y label for comparisaon plot.

    - wl_plot_stack (float, default = 500): wavelength for optical indices for diagram plot.

    - precision  (integer, default = 3): printing precision for optical indices.

    - verbose (boolean, default = False)

    *see 'PyMoosh_Basics.ipynb' tutorial.

    **see 'optim_algo.py' for code.
    """
    def __init__(
        self,
        # Physics
        mat,
        stack,
        thickness,
        wl_domain: np.ndarray,
        incidence: float,
        polar: int,
        # Optimization
        indices: bool,
        which_layers: np.ndarray,
        cost_function,
        X_min: np.ndarray,
        X_max: np.ndarray,
        computation_window: np.ndarray,
        budget: int = 1000,
        nb_runs: int = 1,
        optimizer: str = 'DE',
        # Plots
        objective_vector: np.ndarray = None,
        f_draw = None, # SAME SIZE AS WL_DOMAIN
        progression: bool = False,
        objective_title: str = "comparison curve",
        objective_ylabel: str = "default",
        wl_plot_stack: float = 500,
        precision: int = 3,
        verbose: bool = False
    ):  
        # Structure to optimize. Thicknesses are supposed to be optimized by default.
        # Optimizing optical indices is also possible, then initial materials are 
        # not impacting the result, but the boundaries are. Stack is always fixed.
        self.mat = mat

        #if (isinstance(stack, list) or isinstance(stack, np.ndarray)) and isinstance(np.asarray(stack).all, int):
        self.stack = stack
        #else:
        #    print(f'WARNING: stack is expected to be a list or an array of integers, but {stack} were given.')
        
        #if (isinstance(stack, list) or isinstance(stack, np.ndarray)) and isinstance(np.asarray(stack).all, float):
        self.thickness = thickness
        #else:
        #    print(f'WARNING: thickness is expected to be a list or an array of floats, but {thickness} were given.')

        # Light parameters:
        self.wl_domain = wl_domain # only for plots
        self.incidence = incidence
        self.polar = polar

        # Computation parameters:
        self.indices = indices
        """
        if indices:
            # we change the stack-material relation by a one-by-one link, 
            # because the optical indices will change over all optimized
            # layers. 
            new_materials = np.ones_like(self.stack, dtype=pm.Material)
            for count, mat in enumerate(mat):
                mask = (np.asarray(self.stack)[:] == count)
                np.putmask(new_materials, mask, np.full((1,len(new_materials)), pm.Material(mat)))
            self.mat = new_materials
            self.stack = np.arange(len(self.stack), dtype=int)
        """
        self.which_layers = which_layers
        self.cost_function = cost_function
        self.X_min = X_min
        self.X_max = X_max
        self.computation_window = computation_window
        self.budget = budget
        self.nb_runs = nb_runs
        self.optimizer = optimizer     

        # Plots parameters:
        self.objective_vector = objective_vector
        self.f_draw = f_draw # could be a function to plot or a list of function to plot
        #if isinstance(f_draw, list):
        #    draw_multiple = True
        #else:
        #    draw_multiple = False
        self.progression = progression
        self.objective_title = objective_title
        self.objective_ylabel = objective_ylabel
        self.wl_plot_stack = wl_plot_stack
        self.precision = precision

        if verbose:
            for i in self.mat:
                print(f'mat: {i}')
            print(f'stack: {self.stack}')
            print(f'thickness: {self.thickness}')
            print(f'which_layers: {self.which_layers}')
            print(f'wl_domain: {self.wl_domain}')
            print(f'incidence: {self.incidence}')
            print(f'polar: {self.polar}')
            print(f'cost_function: {self.cost_function}')
            print(f'computation_window: {self.computation_window}')
            print(f'objective_vector: {self.objective_vector}')
            print(f'X_min: {self.X_min}')
            print(f'X_max: {self.X_max}')
            print(f'indices: {self.indices}')
            print(f'budget: {self.budget}')
            print(f'nb_runs: {self.nb_runs}')
            print(f'optimizer: {self.optimizer}')
            print(f'progression: {self.progression}')
            print(f'f_draw: {self.f_draw}')
            print(f'objective_title: {self.objective_title}')
            print(f'objective_ylabel: {self.objective_ylabel}')
            print(f'wl_plot_stack: {self.wl_plot_stack}')

    def wrapper_optimizer(self):
        if self.optimizer == 'DE':
            return pm.differential_evolution(self.cost_function, self.budget, self.X_min, self.X_max, population=30, progression=self.progression)
        
        elif self.optimizer == 'QODE':
            return pm.QODE(self.cost_function, self.budget, self.X_min, self.X_max, population=30, progression=self.progression)
        
        elif self.optimizer == 'QNDE':
            return pm.QNDE(self.cost_function, self.budget, self.X_min, self.X_max, population=30, progression=self.progression)
        
        elif self.optimizer == 'BFGS':
            return pm.bfgs(self.cost_function, self.budget, self.thickness, self.X_min, self.X_max)
        
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
            best, convergence = self.wrapper_optimizer()
            
            # Finish time.
            perf = round(time.perf_counter()-start, 2)
            print(f'Finished in {perf // 60} min {round(perf % 60, 2)} seconds.')

            print('Best guess found:')                
            if self.indices:
                lim = len(best)//2
                print('thicknesses:')
                print(best[:lim])
                print('materials:')
                for i in best[lim:]:
                    print(i)
                struct = pm.Structure(best[lim:], self.stack, best[:lim])

            else:
                print('thicknesses:')
                print(best)
                struct = pm.Structure(self.mat, self.stack, best)

            # Plot convergence.
            plt.plot(convergence)
            plt.suptitle(f'Convergence curve, one run, {self.budget} budget.')
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.show()

            #R_best = pm.coefficient(struct, self.wl_domain, self.incidence, self.polar, wavelength_opti=True)[2]

            # Plot objective.
            plt.plot(self.computation_window, self.objective_vector, label='objective')
            plt.plot(self.wl_domain, self.f_draw(struct), label='best guess')
            plt.suptitle(self.objective_title)
            plt.xlabel("wavelength, nm")
            plt.ylabel(self.objective_ylabel)
            plt.legend(loc='best')
            plt.show()

            # Plot stack
            struct.plot_stack(wavelength=self.wl_plot_stack, lim_eps_colors=[1.5, 4], precision=self.precision)

            # Future implementation about tolerance:
            """
            instant = True
            tol = 0.1 # percent
            step = 1 # in nm
            if instant:
                parameter_space = []
                for state in parameter_space:
                    True
            """
            
        elif self.nb_runs != 1:
            def stats(_):
                best, convergence = pm.differential_evolution(self.cost_function, self.budget, self.X_min, self.X_max, progression=self.progression)
                return best, convergence
            best_list, convergence_list = [], []

            # Time.
            print("Current Time =", datetime.now().strftime("%H:%M:%S"))
            start = time.perf_counter()

            # Statistics.
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(stats, _) for _ in range(self.nb_runs)]
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
            plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
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
                lim = len(best)//2
                print('thicknesses:')
                print(best[:lim])
                print('materials:')
                for i in best[lim:]:
                    print(i)
                struct = pm.Structure(best[lim:], self.stack, best[:lim])

            else:
                print('thicknesses:')
                print(best)
                struct = pm.Structure(self.mat, self.stack, best)
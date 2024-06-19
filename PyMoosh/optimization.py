import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import concurrent.futures
import sys
plt.rcParams['figure.dpi'] = 150

def norm(a, b, c, ord=None):
    return np.linalg.norm(a - b, ord=ord)/len(c)


def constant(wl):
    return 1.0


def attributeDrawFunction(draw_functions, incidence, polar, active_layer):

    is_reference_list = []
    function_list = []
    short_name_list = []
    name_list = []

    for f in draw_functions:
        """
        For plot purpose. From the list of functions to draw, return differents informations for plots.
        Inputs:
        - draw_functions : list of functions or 'keywords' to use built-in functions
        - wl_domain      : *see 'optimization' class docstring.
        - incidence      : *
        - polar          : *
        - active_layer   : *

        Returns:
        - is_reference_list (list of bool)      : Tell if the function is a reference function or not
        - function_list     (list of functions) : List of the functions to draw. The first function is 
                                                  expected to be one of the reference function, otherwise
                                                  the optimization could returns nonsense.
        - short_name_list   (list of strings)   : List of the shortnames of the optical properties.
        - name_list         (list of strings)   : List of the names of the optical properties.


        """

        is_reference = True
        short_name = f

        if f == 'R':
            name = 'Reflectance'
            ref_function = lambda struct, domain : pm.absorption(struct, domain, incidence, polar, wavelength_opti=True)[3]

        elif f == 'T':
            name = 'Transmittance'
            ref_function = lambda struct, domain : pm.absorption(struct, domain, incidence, polar, wavelength_opti=True)[4]

        elif f == 'A':
            name = 'Absorption'
            ref_function = lambda struct, domain : pm.absorption(struct, domain, incidence, polar, wavelength_opti=True)[0][:,active_layer]

        elif f == 'C':
            name = 'Short-circuit current'
            ref_function = lambda struct, domain : pm.opti_photo(struct, incidence, polar, domain[0], domain[-1], active_layer, len(domain))[1]

        elif f == 'CM':
            name = 'Maximum short-circuit current'
            ref_function = lambda struct, domain : pm.opti_photo(struct, incidence, polar, domain[0], domain[-1], active_layer, len(domain))[2]

        elif f == 'joker':
            sys.exit() # for future implementations

        else:
            is_reference = False
            ref_function = f
            short_name = f.__name__[0]
            name = f.__name__
        
        is_reference_list.append(is_reference)
        function_list.append(ref_function)
        short_name_list.append(short_name)
        name_list.append(name)
    
    return [is_reference_list, function_list, short_name_list, name_list]



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

    - cost_function (function): function to minimize. It requires only
      the argument 'layers'.

    - X_min (nunmpy ndarray of floats): lower boundaries of the optimization
      domain, a vector with the same size as the argument of the cost function.

    - X_max (nunmpy ndarray of floats): upper boundaries, see just above.

    - computation_window (nunmpy ndarray of floats): The wavelength
      optimization region. Only for computations.

    - which_layers (nunmpy ndarray of boolean): Same length of 'stack'.
      It indicates which layers are optimized in the stack (which_layer[i] = True).
      Neither the thicknesses nor the optical indices are optimized for the others
      (which_layer[i] = False). 

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
    will appear between the 'objective_vector' and 'draw_function'. Otherwise, if 'nb_runs' is 
    higher than 1, a 'consistency plot' will be shown, which is a superposition of the
    'nb_runs' 'consistency' plots. Independently of 'nb_runs' value, a diagram of the
    thicknesses and the optical indices will appear.

    - objective_vector (nunmpy ndarray of floats): A vector to visually compare
    to 'draw_function'.

    - draw_function (function): function to print and compare to the 'objective_function'.

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
        # Basic parameters
        mat,
        stack,
        thickness,
        incidence: float,
        polar: int,
        X_min: np.ndarray,
        X_max: np.ndarray,
        computation_window: np.ndarray,
        budget: int,
        nb_runs: int,
        wl_domain: np.ndarray = np.linspace(400, 800, 100),
        objective_function = constant,
        draw_functions = 'R',
        # Advanced parameters
        active_layer: float = -1,
        cost_function = None,
        which_layers: np.ndarray = None,
        indices: bool = False,
        optimizer: str = 'QNDE',
        # Plot parameters   
        progression: bool = True,
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
        self.stack = stack
        self.thickness = thickness
        self.incidence = incidence
        self.polar = polar
        self.X_min = X_min
        self.X_max = X_max
        self.computation_window = computation_window
        self.budget = budget
        self.nb_runs = nb_runs
        self.wl_domain = wl_domain
        self.objective_function = objective_function
        # Create a vector from the function.
        objective_vector = objective_function(computation_window) # Do not erase this line!
        self.objective_vector = objective_vector
        self.draw_functions = list(draw_functions)
        
        # Advanced parameters:

        self.active_layer = active_layer
        self.cost_function = cost_function
        self.optimizer = optimizer
        ## Check the layer to optimize.
        if type(which_layers) == None:
            self.which_layers = np.bool_(np.ones_like(self.stack))
        else:
            self.which_layers = which_layers

        ## Check the indices to optimize.
        self.indices = indices
        if indices:
            # we change the stack-material relation by a one-by-one link, 
            # because the optical indices will change over all optimized
            # layers. 
            new_materials = np.ones_like(self.stack, dtype=pm.Material)
            for count, mat in enumerate(mat):
                mask = (np.asarray(self.stack)[:] == count)
                np.putmask(new_materials, mask, np.full((1,len(new_materials)), pm.Material(mat)))
            mat = new_materials # variable to use later
            self.mat = mat
            stack = np.arange(len(self.stack), dtype=int)
            self.stack = stack # variable to use later       
         
        # Plots parameters:

        self.progression = progression
        self.objective_title = objective_title
        self.objective_ylabel = objective_ylabel
        self.wl_plot_stack = wl_plot_stack
        self.precision = precision

        print(locals()) if verbose else None

        # Internal computations for default cost functions
        # 3 - Define a default draw function
           
        is_reference_list, function_list, short_name_list, name_list = attributeDrawFunction(draw_functions, incidence, polar, active_layer)

        if len(draw_functions) == 1 :
            self.objective_title = f'{name_list[0]} in function of wavelength, number of runs:{nb_runs}, bugdet:{budget}.'
            self.objective_ylabel = name_list[0]

        else:
            self.objective_title = f'Optical properties in function of wavelength, number of runs:{nb_runs}, bugdet:{budget}.'
            self.objective_ylabel = short_name_list

        self.function_list = function_list

        # 4 - Define a default cost function

        f = function_list[0]
        if is_reference_list[0] and not indices:

            def default_cost_function(layers):
                structure = pm.Structure(mat, stack, list(layers), verbose=False)             
                return norm(objective_vector, f(structure, computation_window), computation_window)
            
            self.cost_function = default_cost_function
            
        elif is_reference_list[0] and indices:
            
            lim = len(X_min)//2
            def default_cost_function_indices(param):
                layers = param[::lim]
                mat = param[lim::]
                structure = pm.Structure(mat, stack, list(layers), verbose=False)
                return norm(objective_vector, f(structure), computation_window)
            
            self.cost_function = default_cost_function_indices
        

    def wrapper_algorithms(self):
        if self.optimizer == 'DE':
            return pm.differential_evolution(self.cost_function, self.budget, self.X_min, self.X_max, population=30, progression=self.progression)
        
        elif self.optimizer == 'QODE':
            return pm.QODE(self.cost_function, self.budget, self.X_min, self.X_max, population=50, progression=self.progression)
        
        elif self.optimizer == 'QNDE':
            return pm.QNDE(self.cost_function, self.budget, self.X_min, self.X_max, population=30, progression=self.progression)
        
        elif self.optimizer == 'BFGS':
            return pm.bfgs(self.cost_function, self.budget, self.thickness, self.X_min, self.X_max)
        
        else:
            print('Unknown optimizer. See docstring:')
            print(self.__doc__)
            sys.exit()


    def do_optimize(self):

        def iterate_runs(_):
            best, convergence = self.wrapper_algorithms()
            return best, convergence
        best_list, convergence_list = [], []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(iterate_runs, _) for _ in range(self.nb_runs)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                best_list.append(result[0])
                convergence_list.append(result[1]) 
        
        return [best_list, convergence_list]


    def plot_convergence(self, convergence_list):
        for convergence in convergence_list:
            plt.plot(convergence)
        suptitle = 'Convergence curve' if self.nb_runs == 1 else 'Convergence curves'
        plt.suptitle(suptitle)
        plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.show()


    def plot_consistency(self, convergence_list):
        convergence_value_list = [convergence[-1] for convergence in convergence_list]
        convergence_value_list.sort() # sort the list in ascending order
        plt.plot(convergence_value_list, marker="s")
        plt.suptitle(f'Consistency curve.')
        plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
        plt.xlabel("")
        plt.ylabel("Convergence cost value")
        plt.show()
        median = np.median(convergence_value_list)
        mean = np.mean(convergence_value_list)
        sigma = np.std(convergence_value_list)
        print(f'Median :{median}')
        print(f'Mean :{mean}')
        print(f'Sigma :{sigma}')    


    def best_structure(self, best_list):
        costs = [self.cost_function(best) for best in best_list]
        index_best = np.argmin(costs)
        best = best_list[index_best]
        print('Best guess found:')
        if self.indices:
            lim = len(best)//2
            print(f'thicknesses: {best[:lim]}')
            print('materials:')
            for i in best[lim:]:
                print(i)
            return pm.Structure(best[lim:], self.stack, best[:lim])
        else:
            print(f'thicknesses: {best}')
            return pm.Structure(self.mat, self.stack, best)
        

    def plot_objective(self, struct):
        plt.scatter(self.computation_window, self.objective_vector, label='objective', marker='+')
        for i, f in enumerate(self.function_list):
            plt.plot(self.wl_domain, f(struct, self.wl_domain), label=self.objective_ylabel[i])
        plt.suptitle(self.objective_title)
        plt.xlabel("wavelength, nm")
        plt.ylabel(self.objective_ylabel)
        plt.legend(loc='best')
        plt.show()


    def run(self):
        # Start time counter.
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        start = time.perf_counter()

        best_list, convergence_list = self.do_optimize()

        # Stop counter.
        perf = round(time.perf_counter()-start, 2)
        print(f'Finished in {perf // 60} min {round(perf % 60, 2)} seconds.')

        self.plot_convergence(convergence_list)

        if self.nb_runs != 1:
            self.plot_consistency(convergence_list)

        struct = self.best_structure(best_list)
        
        # Plot objective.
        self.plot_objective(struct)

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


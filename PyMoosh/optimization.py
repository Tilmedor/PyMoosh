import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import concurrent.futures
import sys
plt.rcParams['figure.dpi'] = 150

def norm(a, b, c, ord=None):
    """
    Takes the norm of a vector (a - b) which corresponds to the mathematical
    norm || a - b || of order "ord".
    """
    return np.linalg.norm(a - b, ord=ord)/len(c)


def constant(wl):
    return 1.0


def attributeFunction(functions, incidence, polar, active_layer):
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

    is_reference_list = []
    function_list = []
    short_name_list = []
    name_list = []

    for f in functions:

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



def wrapper_cost_function(f, reference, mat, stack, thickness, indices, which_layers, objective_vector, computation_window, X_min, cost_function):
    """
    Returns a default cost function or a customized cost function.
    """
    if reference and not indices and which_layers.all():

        def default_cost_function(layers):
            structure = pm.Structure(mat, stack, list(layers), verbose=False)             
            return norm(objective_vector, f(structure, computation_window), computation_window)
            
        return default_cost_function
            
    elif reference and indices and which_layers.all():
            
        lim = len(X_min)//2
        def default_cost_function_indices(param):
            layers = param[:lim]
            mat = param[lim:]
            structure = pm.Structure(mat, stack, list(layers), verbose=False)
            return norm(objective_vector, f(structure, computation_window), computation_window)

        return default_cost_function_indices
            
    elif reference and not indices and not which_layers.all():

        mask = np.logical_not(which_layers)
        def default_cost_function(layers):
            np.putmask(layers, mask, thickness)
            structure = pm.Structure(mat, stack, list(layers), verbose=False)             
            return norm(objective_vector, f(structure, computation_window), computation_window)
            
        return default_cost_function
    
    elif reference and indices and not which_layers.all():
            
        mask = np.logical_not(which_layers)
        lim = len(X_min)//2
        #default_structure = pm.Structure(mat, stack, thickness, verbose=False)

        def default_cost_function_indices(param):
            layers = param[:lim]
            np.putmask(layers, mask, thickness)
            optical_indices = param[lim:]
            np.putmask(optical_indices, mask, mat)
            structure = pm.Structure(optical_indices, stack, list(layers), verbose=False)
            return norm(objective_vector, f(structure, computation_window), computation_window)
           
        return default_cost_function_indices

    else:
        return cost_function


class optimization:
    """
    One code to optimize them all, one code to find solutions,
    One code to test them all, and in the convergence bind them;
    In the Parametre Space where the One Solution lie.
    
    */* PHYSICS *\*

    - mat (list or array of floats or pm.Materials): list of materials*.

    - stack (list or array of integers)*

    - thickness (list or array of floats)*

    - incidence (float): incidence of the light beam above surface (in degree °, 
      between 0° (normal) and 90 ° (tangent) ).

    - polar (0 or 1): field polarization. 0 == TE, other is TM.

    - X_min (nunmpy ndarray of floats): lower boundaries of the optimization
      domain, a vector with the same size as the argument of the cost function.

    - X_max (nunmpy ndarray of floats): upper boundaries, see just above.

    - computation_window (nunmpy ndarray of floats): The wavelength
      optimization region. Only for computations.

    - budget (integer, default = 1000): number of iteration for optimization.

    - nb_runs (integer, default = 1): number of time the optimization is done.

    - wl_domain (nunmpy ndarray of floats): The wavelength region to plot.
      Only for plots, and not computations (see below).

    - objective_function (function of floats): A function to draw the objective we
      work with. Used to build the 'objective_vector' to compare it to 'draw_function'.

    - draw_function (function): function to print and compare to the 'objective_vector'.
    
    */* ADVANCED PARAMETERS *\*

    - active_layer (float) : indicate the active layer in the stack for photovoltaic
      purpose.

    - cost_function (function): function to minimize. It requires only
      the argument 'layers'.

    - which_layers (nunmpy ndarray of boolean): Same length of 'stack'.
      It indicates which layers are optimized in the stack (which_layer[i] = True).
      Neither the thicknesses nor the optical indices are optimized for the others
      (which_layer[i] = False). 

    - indices (boolean): If True, the optical indices are also
      optimized. Optimized layers are indicated by 'which_layers' (see above).
      WARNING : if True, the stack needs to be 'np.arange(len(stack))' to work properly.

    - optimizer (string, default = 'DE'): Global Optimization algorithm used.

      Possible choices for the optimizer**:  | name (string)
      ---------------------------------------|--------------
      differential evolution                 | 'DE'
      quasi opposite differential evolution  | 'QODE'
      quasi newtonian differential evolution | 'QNDE'
      bfgs                                   | 'BFGS'
      QODE + distance counter                | 'QODEd'
      QNDE + distance counter                | 'QNDEd'
      QODE + bfgs for each individuals in pop| 'superDE' (coming soon...)

    */* PLOTS *\*

    Depending on the number of runs, several plots can show:
    If 'nb_runs' = 1, a 'convergence plot' will be shown first. Then a'comparison plot'
    will appear between the 'objective_vector' and 'draw_function'. Otherwise, if 'nb_runs' is 
    higher than 1, a 'consistency plot' will be shown, which is a superposition of the
    'nb_runs' 'consistency' plots. Independently of 'nb_runs' value, a diagram of the
    thicknesses and the optical indices will appear.

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
        which_layers = None,
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
        if type(which_layers) != np.ndarray:
            self.which_layers = np.bool_(np.ones_like(self.stack))
        else:
            self.which_layers = which_layers

        ## Check the indices to optimize.
        self.indices = indices
        if indices:
            # we change the stack-material relation by a one-by-one link, 
            # because the optical indices will change over all optimized
            # layers. 
            new_materials = np.ones_like(self.stack, dtype=float) #pm.Material
            for count, mat in enumerate(mat):
                mask = (np.asarray(self.stack)[:] == count)
                np.putmask(new_materials, mask, np.full((1,len(new_materials)), mat)) #pm.Material(mat)
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
           
        is_reference_list, function_list, short_name_list, name_list = attributeFunction(draw_functions, incidence, polar, active_layer)

        if len(draw_functions) == 1 :
            self.objective_title = f'{name_list[0]} in function of wavelength, number of runs:{nb_runs}, bugdet:{budget}.'
            self.objective_ylabel = name_list[0]

        else:
            self.objective_title = f'Optical properties in function of wavelength, number of runs:{nb_runs}, bugdet:{budget}.'
            self.objective_ylabel = short_name_list

        self.function_list = function_list

        # 4 - Define a default cost function
              
        self.cost_function = wrapper_cost_function(function_list[0], is_reference_list[0], self.mat, self.stack, self.thickness, 
                                                   self.indices, self.which_layers, objective_vector, computation_window, X_min, cost_function)

    def wrapper_algorithms(self, cost_function, budget, X_min, X_max, progression):
        """
        Return the algorithm we use for the optimization.
        """
        if self.optimizer == 'DE':
            return pm.differential_evolution(cost_function, budget, X_min, X_max, population=30, progression=progression)
        
        elif self.optimizer == 'QODE':
            return pm.QODE(cost_function, budget, X_min, X_max, population=30, progression=progression)
        
        elif self.optimizer == 'QNDE':
            return pm.QNDE(cost_function, budget, X_min, X_max, population=30, progression=progression)
        
        elif self.optimizer == 'BFGS':
            return pm.bfgs(cost_function, budget, self.thickness, X_min, X_max)
        
        elif self.optimizer == 'QODEd':
            return pm.QODE_distance(cost_function, budget, X_min, X_max, population=30, progression=progression)
        
        elif self.optimizer == 'QNDEd':
            return pm.QNDE_distance(cost_function, budget, X_min, X_max, population=30, progression=progression)
        
        else:
            print('Unknown optimizer. See docstring:')
            print(self.__doc__)
            sys.exit()


    def do_optimize(self, cost_function, nb_runs, budget, X_min, X_max, progression):
        """
        Do the optimization for a given algorithm, budget and number of runs.
        Whenever nb_runs is more than one, multiple cores can be used to run
        some optimization all at once. 
        """
        # number of cores we use.
        number_of_worker = 8
        if self.optimizer == 'QODEd' or self.optimizer == 'QNDEd':
            def iterate_runs(_):
                best, convergence, distances = self.wrapper_algorithms(cost_function, budget, X_min, X_max, progression)
                return best, convergence, distances
        else:
            def iterate_runs(_):
                best, convergence = self.wrapper_algorithms(cost_function, budget, X_min, X_max, progression)
                return best, convergence
            
        best_list, convergence_list, distances_list = [], [], []

        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_worker) as executor:
            futures = [executor.submit(iterate_runs, _) for _ in range(nb_runs)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if self.optimizer == 'QODEd' or self.optimizer == 'QNDEd':
                    distances_list.append(result[2])
                best_list.append(result[0])
                convergence_list.append(result[1])
        
        return [best_list, convergence_list, distances_list]


    def plot_convergence(self, convergence_list):
        """
        Plot the convergence curves (costs VS iteration). Iteration are roughly the
        budget divided by the number of population used in the chosen algorithm.
        """
        for convergence in convergence_list:
            plt.plot(convergence)
        suptitle = 'Convergence curve' if self.nb_runs == 1 else 'Convergence curves'
        plt.suptitle(suptitle)
        plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.show()


    def plot_consistency(self, convergence_list):
        """
        Plot the consistency curve (the sorted costs) and the main statistical
        indicators (media, mean and standard deviation).
        """
        convergence_value_list = [convergence[-1] for convergence in convergence_list]
        convergence_value_list.sort() # sort the list in ascending order

        plt.plot(convergence_value_list, marker="s")
        plt.suptitle(f'Consistency curve.')
        plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
        plt.xlabel("")
        plt.ylabel("Convergence cost value")
        plt.show()

        print(f'Median :{np.median(convergence_value_list)}')
        print(f'Mean :{np.mean(convergence_value_list)}')
        print(f'Sigma :{np.std(convergence_value_list)}')    


    def best_structure(self, best_list):
        """
        Returns the thicknesses (and the material if we optimize the
        indices) of the best structure the algorithm has found.
        """
        costs = [self.cost_function(best) for best in best_list]
        index_best = np.argmin(costs)
        best = best_list[index_best]
        print(f'cost: {np.min(costs)}')
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
        

    def worst_structure(self, worst_list):
        """
        Returns the thicknesses (and the material if we optimize the
        indices) of the worst structure the algorithm has found.
        """
        costs = [self.cost_function(worst) for worst in worst_list]
        index_worst = np.argmax(costs)
        worst = worst_list[index_worst]
        print(f'cost: {np.max(costs)}')
        print('Worst guess found:')
    
        if self.indices:
            lim = len(worst)//2
            print(f'thicknesses: {worst[:lim]}')
            print('materials:')
            for i in worst[lim:]:
                print(i)
            return pm.Structure(worst[lim:], self.stack, worst[:lim])
        else:
            print(f'thicknesses: {worst}')
            return pm.Structure(self.mat, self.stack, worst)
        

    def plot_objective(self, struct):
        """
        Plots the physical property VS its objective.
        """
        plt.scatter(self.computation_window, self.objective_vector, label='objective', marker='+')
        for i, f in enumerate(self.function_list):
            plt.plot(self.wl_domain, f(struct, self.wl_domain), label=self.objective_ylabel[i])
        plt.suptitle(self.objective_title)
        plt.xlabel("wavelength, nm")
        plt.ylabel(self.objective_ylabel)
        plt.legend(loc='best')
        plt.show()


    def dont_use(self, distances_list, d):
        """
        Returns the distances between the physical structures.
        """
        for distances in distances_list:
            #print(distances)
            plt.plot(distances, color='b')
            #envelop = np.sum(distances, axis=1)/30 # population
        #envelop = [np.sum(distances, axis=1) for distances in distances_list]
        #plt.plot(envelop, color='r', label='envelop')
        #plt.plot(d*np.ones_like(envelop))
        suptitle = 'Distance-to-the-best curve' if self.nb_runs == 1 else 'Distance-to-the-best curves'
        plt.suptitle(suptitle)
        plt.title(f'{self.nb_runs} runs, {self.budget} budget.')
        plt.xlabel("Iterations")
        plt.ylabel("Distance (nm)")
        plt.show()


    def plot_distance(self, distances_runs, d):
        """
        Returns the distances between the physical structures.
        """
        for distances in distances_runs:
            distances_curve = []
            for distances_generations in distances:
                values = np.zeros_like(distances_generations)
                mask = (distances_generations[:] < d*np.ones_like(distances_generations))
                np.putmask(values, mask, np.ones_like(values))
                sum = np.sum(values)
                distances_curve.append(sum)
            X = np.arange(len(distances_curve))
            plt.plot(X, distances_curve)
        suptitle = 'How many individuals are near to the best ?' if self.nb_runs == 1 else 'Distance-to-the-best curves'
        plt.suptitle(suptitle)
        plt.title(f'{self.nb_runs} runs, {self.budget} budget. Radius={d} nm.')
        plt.xlabel("Iterations")
        plt.ylabel("Distance (nm)")
        plt.show()


    def run(self, plot_convergence=True, plot_consistency=True, plot_objective=True, plot_stack=True):
        """
        Do run the previous methods.
        Method for users to do the optimization.
        RETURNS A STRUCTURE.
        """
        # Start time counter.
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        start = time.perf_counter()

        best_list, convergence_list, distances_list = self.do_optimize(self.cost_function, self.nb_runs, self.budget, self.X_min, self.X_max, self.progression)

        # Stop counter.
        perf = round(time.perf_counter()-start, 2)
        print(f'Finished in {perf // 60} min {round(perf % 60, 2)} seconds.')

        # Plot convergence curve(s).
        if plot_convergence:
            self.plot_convergence(convergence_list)

        # Plot consistency curve.
        if self.nb_runs != 1 and plot_consistency:
            self.plot_consistency(convergence_list)

        # Create the optimized structure.
        structure = self.best_structure(best_list)
        
        # Plot objective.
        if plot_objective:
            self.plot_objective(structure)

        # Plot stack.
        if plot_stack:
            structure.plot_stack(wavelength=self.wl_plot_stack, lim_eps_colors=[1.5, 4], precision=self.precision)

        # Plot distances to the best.
        if self.optimizer == 'QODEd' or self.optimizer == 'QNDEd':
            self.plot_distance(distances_list, 7.0)

        return structure


    def robustness(self, structure, distance: float, budget: int = 5000, nb_runs: int =1):
        """
        Method to test the robustness of a structure.
        Returns the worst cost for a structure near to a given structure (an optimized one usually).
        """

        # We define smaller boundaries for the "opposite" optimization.
        th = structure.thickness
        xmin = th - distance*np.ones_like(self.X_min)
        xmax = th + distance*np.ones_like(self.X_min)

        # We create the opposite cost_function to maximize the cost, instead.
        opposite_cost_function = lambda param: 1 - self.cost_function(param)

        # Start time counter.
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        start = time.perf_counter()

        worst_list, _, _ = self.do_optimize(opposite_cost_function, nb_runs, budget, xmin, xmax, progression=True)

        # Stop counter.
        perf = round(time.perf_counter()-start, 2)
        print(f'Finished in {perf // 60} min {round(perf % 60, 2)} seconds.')

        # Create the least optimized structure in the allowed space.
        structure = self.worst_structure(worst_list)
        
        # Plot objective.
        self.plot_objective(structure)
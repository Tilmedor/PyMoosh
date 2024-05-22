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
        mat,
        stack,
        thickness,
        which_layers: np.ndarray,
        wl_domain: np.ndarray, # only for plot
        incidence: float,
        polar: int,
        cost_function,
        computation_window: np.ndarray, # only for computation
        objective_vector: np.ndarray,
        X_min: np.ndarray,
        X_max: np.ndarray,
        indices: bool = False,
        budget: int = 1000,
        nb_runs: int = 1,
        optimizer: str = 'DE',
        progression: bool = False,
        objective_title: str = "default title",
        objective_ylabel: str = "default",
        wl_plot_stack: float = 500,
        precision: int = 3,
        verbose: bool = False
    ):  
        # Structure to optimize. Thicknesses are supposed to be optimized by default.
        # Optimizing optical indices is also possible, then the chosen materials are 
        # not impacting the result, but the boundaries are. Stack is always fixed.
        self.mat = mat
        self.stack = stack
        self.thickness = thickness
        self.which_layers = which_layers # which layer to optimize

        # Light parameters:
        self.wl_domain = wl_domain # only for plots
        self.incidence = incidence
        self.polar = polar

        # Computation parameters:
        self.cost_function = cost_function
        self.computation_window = computation_window
        self.objective_vector = objective_vector
        self.X_min = X_min
        self.X_max = X_max
        self.indices = indices
    
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
    
        self.budget = budget
        self.nb_runs = nb_runs
        self.optimizer = optimizer

        # Plots parameters:
        self.progression = progression
        #self.f = f # could be a function to plot or a list of function to plot
        self.objective_title = objective_title
        self.objective_ylabel = objective_ylabel
        self.wl_plot_stack = wl_plot_stack
        self.precision = precision

        if verbose:
            for i in self.mat:
                print(f'DEBUG mat: {i}')
            print(f'DEBUG stack: {self.stack}')
            print(f'DEBUG thickness: {self.thickness}')
            print(f'DEBUG which_layers: {self.which_layers}')
            print(f'DEBUG wl_domain: {self.wl_domain}')
            print(f'DEBUG incidence: {self.incidence}')
            print(f'DEBUG polar: {self.polar}')
            print(f'DEBUG cost_function: {self.cost_function}')
            print(f'DEBUG computation_window: {self.computation_window}')
            print(f'DEBUG objective_vector: {self.objective_vector}')
            print(f'DEBUG X_min: {self.X_min}')
            print(f'DEBUG X_max: {self.X_max}')
            print(f'DEBUG indices: {self.indices}')
            print(f'DEBUG budget: {self.budget}')
            print(f'DEBUG nb_runs: {self.nb_runs}')
            print(f'DEBUG optimizer: {self.optimizer}')
            print(f'DEBUG progression: {self.progression}')
            print(f'DEBUG objective_title: {self.objective_title}')
            print(f'DEBUG objective_ylabel: {self.objective_ylabel}')
            print(f'DEBUG wl_plot_stack: {self.wl_plot_stack}')

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
            struct.plot_stack(wavelength=self.wl_plot_stack, lim_eps_colors=[1.5, 4], precision=self.precision)

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
# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
from refractiveindex import RefractiveIndexMaterial
from PyMoosh.anisotropic_functions import get_refraction_indices

class Material:

    """
        Types of material (default): / format:                 / specialType:

            - simple_perm            / complex                 / 'Default'
            - magnetic               / list(complex, float)    / 'Default'
            - CustomFunction         / function                / 'Default'
            - BrendelBormann         / string                  / 'Default'

        Types of material (special): / format:                 / specialType:

            - ExpData                / ???                     / 'ExpData'
            - RefractiveIndexInfo    / list(shelf, book, page) / 'RII'
            - Anisotropic            / list(shelf, book, page) / 'ANI'
    """

    def __init__(self, mat, specialType="Default", verbose=False):

        if specialType == "Default":
            self.specialType = specialType
            if mat.__class__.__name__ == 'function':
                self.type = "CustomFunction"
                self.permittivity_function = mat
                self.name = "CustomFunction: "+mat.__name__
                if verbose :
                    print("Custom dispersive material. Epsilon=",mat.__name__,"(wavelength in nm)")
            elif not hasattr(mat, '__iter__'):
            # no func / not iterable --> single value, convert to complex by default
                self.type = "simple_perm"
                self.name = "SimplePermittivity:"+str(mat)
                self.permittivity = complex(mat)
                if verbose :
                    print("Simple, non dispersive: epsilon=",self.permittivity)
            # elif isinstance(mat,list) or isinstance(mat,tuple) or isinstance(mat,np.ndarray):
            elif isinstance(mat, list) and isinstance(mat[0], float) and isinstance(mat[1], float): # magnetic == [float, float]
            # iterable: if list or similar --> magnetic
                self.type = "magnetic"
                self.permittivity = mat[0]
                self.permeability = mat[1]
                self.name = "MagneticPermittivity:"+str(mat[0])+"Permability:"+str(mat[1])
                print("hello")
                if verbose :
                    print("Magnetic, non dispersive: epsilon=", mat[0]," mu=",mat[1])
                if len(mat)>2:
                    print(f'Warning: Magnetic material should have 2 values (epsilon / mu), but {len(mat)} were given.')

            elif isinstance(mat,str):
            # iterable: string --> database material
            # from file in shipped database
                import pkgutil
                f = pkgutil.get_data(__name__, "data/material_data.json")
                f_str = f.decode(encoding='utf8')
                database = json.loads(f_str)
                if mat in database:
                    material_data = database[mat]
                    model = material_data["model"]
                    """
                    if model == "ExpData":
                        self.type = "ExpData"
                        self.name = "ExpData: "+ str(mat)

                        wl=np.array(material_data["wavelength_list"])
                        epsilon = np.array(material_data["permittivities"])
                        if "permittivities_imag" in material_data:
                            epsilon = epsilon + 1j*np.array(material_data["permittivities_imag"])

                        self.wavelength_list = np.array(wl, dtype=float)
                        self.permittivities  = np.array(epsilon, dtype=complex)
                    """
                    if model == "BrendelBormann":
                        self.type = "BrendelBormann"
                        self.name = "BrendelBormann model: " + str(mat)
                        self.f0 = material_data["f0"]
                        self.Gamma0 = material_data["Gamma0"]
                        self.omega_p = material_data["omega_p"]
                        self.f = np.array(material_data["f"])
                        self.gamma = np.array(material_data["Gamma"])
                        self.omega = np.array(material_data["omega"])
                        self.sigma = np.array(material_data["sigma"])

                    elif model == "CustomFunction":
                        self.type = "CustomDatabaseFunction"
                        self.name = "CustomDatabaseFunction: " + str(mat)
                        permittivity = material_data["function"]
                        self.permittivity_function = authorized[permittivity]

                    else:
                        print(model," not an existing model (yet).")
                        #sys.exit()

                    if verbose :
                        print("Database material:",self.name)
                else:
                    print(mat,"Unknown material (for the moment)")
                    #print("Known materials:\n", existing_materials())
                    #sys.exit()

            else:
                print(f'Warning: Given data is not in the right format for a \'Default\' specialType. You should check the data format or specify a specialType. You can refer to the following table:')
                print(self.__doc__)

        elif specialType == "RII":
            if len(mat) != 3:
                print(f'Warning: Material RefractiveIndex Database is expected to be a list of 3 values, but {len(mat)} were given.')
            self.type = "RefractiveIndexInfo"
            self.specialType = specialType
            self.name = "MaterialRefractiveIndexDatabase: " + str(mat)
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf: {}, book: {}, page: {}".format(shelf, book, page) # not necessary ?
            material = RefractiveIndexMaterial(shelf, book, page) # create object
            self.material = material
            if verbose :
                print("Hello there ;)")
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(f'Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given.')

        elif specialType == "ExpData":
            import pkgutil
            f = pkgutil.get_data(__name__, "data/material_data.json")
            f_str = f.decode(encoding='utf8')
            database = json.loads(f_str)
            if mat in database:
                material_data = database[mat]
                model = material_data["model"]
                if model == "ExpData":
                    self.type = "ExpData"
                    self.name = "ExpData: "+ str(mat)
                    self.specialType = specialType

                    wl=np.array(material_data["wavelength_list"])
                    epsilon = np.array(material_data["permittivities"])
                    if "permittivities_imag" in material_data:
                        epsilon = epsilon + 1j*np.array(material_data["permittivities_imag"])

                    self.wavelength_list = np.array(wl, dtype=float)
                    self.permittivities  = np.array(epsilon, dtype=complex)
                else:
                    print('Warning: Used model should be "ExpData", but {} were given.'.format(model))

        elif specialType == "ANI" :
            if len(mat) != 3:
                print(f'Warning: Anisotropic material from Refractiveindex.info is expected to be a list of 3 values, but {len(mat)} were given.')
            self.type = "Anisotropic"
            self.specialType = specialType
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf: {}, book: {}, page: {}".format(shelf, book, page) # not necessary ?
            material_list = wrapper_anisotropy(shelf, book, page) # A list of three materials
            self.material_list = material_list
            self.material_x = material_list[0]
            self.material_y = material_list[1]
            self.material_z = material_list[2]
            self.name = "Anisotropic material from Refractiveindex.info: " + str(mat)
            if verbose :
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(f'Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given.')

        elif specialType == "Unspecified":
            self.specialType = specialType
            print(specialType, "Unknown type of material (for the moment)")
            # sys.exit()

        else:
            print(f'Warning: Unknown type : {specialType}')

    def __str__(self):
        return self.name

    def get_permittivity(self,wavelength):
        if self.type == "simple_perm":
            return self.permittivity
        elif self.type == "magnetic":
            return self.permittivity
        elif self.type == "CustomFunction":
            print('hello there')
            return self.permittivity_function(wavelength)
        elif self.type == "BrendelBormann":
            w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength
            a = np.sqrt(w * (w + 1j * self.gamma))
            x = (a - self.omega) / (np.sqrt(2) * self.sigma)
            y = (a + self.omega) / (np.sqrt(2) * self.sigma)
            # Polarizability due to bound electrons
            chi_b = np.sum(1j * np.sqrt(np.pi) * self.f * self.omega_p ** 2 /
                        (2 * np.sqrt(2) * a * self.sigma) * (wofz(x) + wofz(y)))
            # Equivalent polarizability linked to free electrons (Drude model)
            chi_f = -self.omega_p ** 2 * self.f0 / (w * (w + 1j * self.Gamma0))
            epsilon = 1 + chi_f + chi_b
            return epsilon
        elif self.type == "RefractiveIndexInfo":
            try:
                k = self.material.get_extinction_coefficient(wavelength)
                return self.material.get_epsilon(wavelength)
            except:
                n = self.material.get_refractive_index(wavelength)
                return n**2
        elif self.type == "ExpData":
            return np.interp(wavelength, self.wavelength_list, self.permittivities)
        elif self.type == "Anisotropic":
            print(f'Warning: Functions for anisotropic materials generaly requires more information than isotropic ones. You probably want to use \'get_permittivity_ani()\' function.')

    def get_permeability(self,wavelength, verbose=False):
        if self.type == "magnetic":
            return self.permeability
        elif self.type == "RefractiveIndexInfo":
            if verbose:
                print('Warning: Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0 .')
            return 1.0
        elif self.type == "Anisotropic":
            if verbose:
                print('Warning: Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0 .')
            return [1.0, 1.0, 1.0]
        return 1.0

# Anisotropic method
    def get_permittivity_ani(self, wavelength, elevation_beam, precession, nutation, spin):
        # We have three permittivities to extract
        refraction_indices_medium = []
        for material in self.material_list:
            try:
                k = material.get_extinction_coefficient(wavelength)
                refraction_indices_medium.append(material.material.get_epsilon(wavelength))
            except:
                n = material.get_refractive_index(wavelength)
                refraction_indices_medium.append(n**2)
        return np.sqrt(get_refraction_indices(elevation_beam, refraction_indices_medium, precession, nutation, spin))

def existing_materials():
    import pkgutil
    f = pkgutil.get_data(__name__, "data/material_data.json")
    f_str = f.decode(encoding='utf8')
    database = json.loads(f_str)
    for entree in database:
        if "info" in database[entree]:
            print(entree,"::",database[entree]["info"])
        else :
            print(entree)

# Sometimes materials can be defined not by a well known model
# like Cauchy or Sellmeier or Lorentz, but have specific formula.
# That may be convenient.

def permittivity_glass(wl):
    #epsilon=2.978645+0.008777808/(wl**2*1e-6-0.010609)+84.06224/(wl**2*1e-6-96)
    epsilon = (1.5130 - 3.169e-9*wl**2 + 3.962e3/wl**2)**2
    return epsilon

# Declare authorized functions in the database. Add the functions listed above.

authorized = {"permittivity_glass":permittivity_glass}

def wrapper_anisotropy(shelf, book, page):
    if page.endswith("-o") or page.endswith("-e"):
        if page.endswith("-e"):
            page_e, page_o = page, page.replace("-e", "-o")
        elif page.endswith("-o"):
            page_e, page_o = page.replace("-o", "-e"), page

        # create ordinary and extraordinary object.
        material_o = RefractiveIndexMaterial(shelf, book, page_o)
        material_e = RefractiveIndexMaterial(shelf, book, page_e)
        return [material_o, material_o, material_e]
    
    elif page.endswith("-alpha") or page.endswith("-beta") or page.endswith("-gamma"):
        if page.endswith("-alpha"):
            page_a, page_b, page_c = page, page.replace("-alpha", "-beta"), page.replace("-alpha", "-gamma")
        elif page.endswith("-beta"):
            page_a, page_b, page_c = page.replace("-beta", "-alpha"), page, page.replace("-beta", "-gamma")
        elif page.endswith("-gamma"):
            page_a, page_b, page_c = page.replace("-gamma", "-alpha"), page.replace("-gamma", "-beta"), page
        
        # create ordinary and extraordinary object.
        material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
        material_beta = RefractiveIndexMaterial(shelf, book, page_b)
        material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
        return [material_alpha, material_beta, material_gamma]
    
    else:
        # there may better way to do it.
        print("no")
        try:
            page_e, page_o = "".join(page, "-e"), "".join(page, "-e")
            material_o = RefractiveIndexMaterial(shelf, book, page_o)
            material_e = RefractiveIndexMaterial(shelf, book, page_e)
            return [material_o, material_o, material_e]
        except:
            print("neither that way")
            try:
                page_a, page_b, page_c = page + "-alpha", page + "-beta", page + "-gamma"
                print(page_a)
                material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
                material_beta = RefractiveIndexMaterial(shelf, book, page_b)
                material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
                return [material_alpha, material_beta, material_gamma]
            except:
                print(f'Warning: Given material is not known to be anisotropic in the Refractiveindex.info database. You should try to remove "ANI" keyword in material definition or to spellcheck the given path.')
import numpy as np
# for get_refraction_indices
def ani_refraction_indices(u_vector, refraction_indices_medium, optical_axis='biaxis', verbose=False):
    """
    Computes refraction indices m1 and m2 in an anisotropic medium for an incident plane wave.
    It works for both uni-axis and bi-axis birefringent material.

    Inputs:
    - u_vector : direction of propagation in space, given in the material eigen base.
    - refraction_indices_medium : refraction indices associated to material eigen base.
    - isotropy : special case for fast computing: uniaxis or biaxis

    Returns:
    A list containing m1 and m2.
    """
    if len(u_vector) != 3:
        print(f'Warning : A list of {len(u_vector)} items is given, but a list of exactly 3 float is expected.')

    ux, uy, uz = float(u_vector[0][0]), float(u_vector[1][0]), float(u_vector[2][0])
    u = np.sqrt((ux**2) + (uy**2) + (uz**2))
    if u == 0:
        print('Warning: a null vector is given, where a non zero vector is expected.')
    elif (u != 1 and verbose==True):
        u_vector = u_vector/u
        print(f"Warning : a vector of {u} length is given, where a unitary vector is expected. Normalized version is used instead => {u_vector}")
    
    if optical_axis == 'biaxis':
        if len(refraction_indices_medium) != 3:
            print(f'Warning : A list of {len(refraction_indices_medium)} items is given, but a list of exactly 3 float is expected.')
        else:
            nx, ny, nz = refraction_indices_medium[0], refraction_indices_medium[1], refraction_indices_medium[2]
            
            # We build a quadratic polynomial from the Fresnel equation, with the following coefficients:
            a = (nx**2)*(ux**2) + (ny**2)*(uy**2) + (nz**2)*(uz**2)
            b = -((nx**2)*(ux**2)*(ny**2 + nz**2) + (ny**2)*(uy**2)*(nx**2 + nz**2) + (nz**2)*(uz**2)*(ny**2 + nx**2))
            c = (nx**2)*(ny**2)*(nz**2)*((ux**2)+(uy**2)+(uz**2))
 
    elif optical_axis == 'uniaxis':
        if len(refraction_indices_medium) != 2:
            print(f'Warning : A list of {len(refraction_indices_medium)} items is given, but a list of exactly 2 float is expected.')
        else:
            nx, nz = refraction_indices_medium[0], refraction_indices_medium[1]
                        
            # We build a quadratic polynomial from the Fresnel equation, with the following coefficients:
            a = (nx**2)*(ux**2 + uy**2) + (nz**2)*(uz**2)
            b = -((nx**2)*(ux**2 + uy**2)*(nx**2 + nz**2) + 2*(nz**2)*(uz**2)*(nx**2))
            c = (nx**4)*(nz**2)*((ux**2)+(uy**2)+(uz**2))

    else:
        print(f'Warning : {optical_axis} is an invalid optical axis type. Please choose \'uniaxis\' or \'biaxis\'.')

    # The indices we want are roots of this polynomial
    sol = np.roots([a, b, c])
    if len(sol) == 1:
        sol = np.append(sol, sol)
    return np.sqrt(sol)

def change_of_basis_matrix(precession, nutation, spin):
    """
    Transition matrix from canonical basis to matrix eigen basis, using Euler's angles.

    Inputs: euler's angles to pass from canoncial basis to main basis

    Returns: 3x3 matrix with a unit determinant, orthogonal matrix
    """
    a,b,c = precession*(np.pi/180), nutation*(np.pi/180), spin*(np.pi/180)
    x = [np.cos(a)*np.cos(c)-np.sin(a)*np.cos(b)*np.sin(c), -np.cos(a)*np.sin(c)-np.sin(a)*np.cos(b)*np.cos(c), np.sin(a)*np.sin(b)]
    y = [np.sin(a)*np.cos(c)+np.cos(a)*np.cos(b)*np.sin(c), -np.sin(a)*np.sin(c)+np.cos(a)*np.cos(b)*np.cos(c), -np.cos(a)*np.sin(b)]
    z = [np.sin(b)*np.sin(c), np.sin(b)*np.cos(c), np.cos(b)]
    result = np.matrix([x,y,z])
    return result

def incident_vector_in_eigen_basis(alpha, P, verbose=False):
    """
    Write the direction of propagation vector of a beam in the material eigen basis.

    Inputs:
    - alpha == incident elevation of the beam, with respect to the canonical basis.
    - P == change of basis matrix, from canonical basis to eigen basis.

    Returns: direction propagation vector of the beam in the eigen basis.
    """
    if (alpha < 0 or alpha > 90) and verbose==True:
        print(f'Warning : an elevation of {alpha}° is given, but a value between 0° and 90° is expected.')
    else:
        alpha = alpha*(np.pi/180)
        vector = np.array([np.sin(alpha), 0, -np.cos(alpha)])
        vector.shape = (3,1)
        result_vector = (P.transpose()).dot(vector)
        return result_vector

# functions to use
def get_refraction_indices(elevation_beam, refraction_indices_medium, precession, nutation, spin):
    """
    Computes the refraction indices of an anisotropic medium given its geometry (main directions and main indices) and the beam orientation.
    
    Inputs:
    - elevation_beam : incident elevation of the beam with respect to the material surface. Be careful, the normal is pointing up, and the wavevector is pointing down
    - azimuth_beam :
    - refraction_indices_medium : refraction indices associated to material eigen base. A list of 3 floats.
    - elevation_eigen_basis : incident elevation of the main direction pz, degree.
    - azimuth_eigen_basis : incident azimuth of the main direction pz, degree.

    Returns: Two refraction indices. For isotropic situation, returned indices are exactly the same. 
    """
    # Transition matrix first:
    P = change_of_basis_matrix(precession, nutation, spin)
    # Then we write the direction of propagation vector into the main basis:
    u_vector = incident_vector_in_eigen_basis(elevation_beam, P)
    # And finally we obtain the desired refraction indices coefficients:
    indices = ani_refraction_indices(u_vector, refraction_indices_medium)

    return indices

def get_direction_propagation(incident_refraction_index, elevation_beam, refraction_indices_medium, precession, nutation, spin):
    """
    Compute the plane wave's direction of propagation vector u for both ordinary and extraordinary beams.

    Inputs:
    - incident_refraction_index : n of the previous material, needed to apply generalized Descartes' law.
    - elevation_beam : incident elevation of the beam with respect to the material surface.
    - refraction_indices_medium : refraction indices associated to material eigen base. A list of 3 floats.
    - elevation_eigen_basis : incident elevation of the main direction pz, degree.
    - azimuth_eigen_basis : incident azimuth of the main direction pz, degree.

    Returns : a list of two unit vectors, arrays.
    """
    # refracted indices
    n = incident_refraction_index
    [m1, m2] = get_refraction_indices(elevation_beam, refraction_indices_medium, precession, nutation, spin)
    # refracted angles theta1 and theta2 for these beams
    elevation_beam = elevation_beam*(np.pi/180)
    u1 = (n/m1)*np.array([np.sin(elevation_beam), 0,-np.sqrt((m1/n)**2 - (np.sin(elevation_beam))**2)])
    u2 = (n/m2)*np.array([np.sin(elevation_beam), 0,-np.sqrt((m2/n)**2 - (np.sin(elevation_beam))**2)])
    return [u1, u2]

if (__name__ == '__main__'):
    import matplotlib.pyplot as plt
    import numpy as np

    ## Test 1 one value : ok
    """
    # beam
    elevation_beam = 45
    # material
    refraction_indices_medium = [1.0, 2.0, 2.0]
    # main basis
    precession, nutation, spin = 0,0,0
    indices = get_refraction_indices(elevation_beam, refraction_indices_medium, precession, nutation, spin)
    print(indices)
    """
    ## Test 2 isotropy : ok
    # We expect to have the same result for each plot : indices == 2.0
    """
    # beam
    elevation_beam_range = np.linspace(90,180,90)
    # material
    refraction_indices_medium = [2.0, 2.0, 2.0]
    # main basis
    precession_range = np.linspace(0,359,360)
    nutation_range = np.linspace(0,359,360)
    spin_range = np.linspace(0,359,360)

    a1 = [get_refraction_indices(_, refraction_indices_medium, 0,0,0)[0] for _ in elevation_beam_range]
    a2 = [get_refraction_indices(_, refraction_indices_medium, 0,0,0)[1] for _ in elevation_beam_range]    
    c = [get_refraction_indices(120, refraction_indices_medium, _,0,0) for _ in precession_range]
    d = [get_refraction_indices(120, refraction_indices_medium, 0,_,0) for _ in nutation_range]
    e = [get_refraction_indices(120, refraction_indices_medium, 0,0,_) for _ in spin_range]

    plt.plot(elevation_beam_range, a1)
    plt.plot(elevation_beam_range, a2)
    plt.plot(precession_range, c)
    plt.plot(nutation_range, d)
    plt.plot(spin_range, e)
    plt.show()
    """

    ## Test anisotropy 3 : ok
    # UNI-AXIS MATERIAL + NORMAL IS OPTICAL AXIS
    # We expect to see the same optical index (1.0) for all azimuth of the beam
    # + for all spin (precession = nutation = 0)
    # BUT a variation for elevation (indices surface)
    """
    # beam
    elevation_beam_range = np.linspace(0,90,100)
    # material
    refraction_indices_medium = [1.0, 1.0, 2.0]
    # main basis
    spin_range = np.linspace(0,359,360)

    a = [get_refraction_indices(_, refraction_indices_medium, 0,0,0) for _ in elevation_beam_range]
    c = [get_refraction_indices(90, refraction_indices_medium, 0,0,_) for _ in spin_range]
    plt.title('indices optiques $n_o$ et $n_e$ en fonction de $\\theta$.')
    plt.suptitle('AO vertical, [1,1,2], azimuth nulle')
    plt.ylabel('indice optique')
    plt.xlabel('$\\theta$ sphérique (°)')
    plt.plot(elevation_beam_range, a)
    plt.show()
    plt.title('indices optiques $n_o$ et $n_e$ en fonction de l\'azimuth.')
    plt.suptitle('AO vertical, [1,1,2], $\\theta = 90°$')
    plt.ylabel('indice optique')
    plt.xlabel('$\\phi$ sphérique (°)')
    plt.plot(spin_range, c)
    plt.show()
    #plt.plot(spin_range, c)
    #plt.show()
    """

    ## Test anisotropy 3 : ok
    # UNI-AXIS MATERIAL + NORMAL IS TANGENT TO THE SURFACE
    # We expect to see a variation for elevation of the beam (indices surface) and azimuth fixed
    # + 2 constants for azimuth variation and for elevation = 180
    """
    # beam
    elevation_beam_range = np.linspace(90,180,90)
    # material
    refraction_indices_medium = [1.0, 1.0, 2.0]

    a = [get_refraction_indices(_, refraction_indices_medium, 0,90,0) for _ in elevation_beam_range]
    
    plt.plot(elevation_beam_range, a)
    plt.show()
    """
    ## Test anisotropy 4: to do
    # BI AXIS ??

    ## Test direction of propagation beams
    """
    # beam
    elevation_beam = 60
    # material
    refraction_indices_medium = [2.0, 1.0, 3.0]
    # main basis
    precession, nutation, spin = 0,0,0
    answer = get_direction_propagation(1.0, elevation_beam, refraction_indices_medium, precession, nutation, spin)
    print(answer)
    """
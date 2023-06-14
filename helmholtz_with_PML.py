import numpy as np
import ufl
from dolfinx import geometry
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, dirichletbc, locate_dofs_topological, form, Constant, VectorFunctionSpace, TensorFunctionSpace ,Expression
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, meshtags
from ufl import dx, grad, inner, outer, dot, ds, Measure, as_matrix, as_tensor, inv, det, log
from mpi4py import MPI
from petsc4py import PETSc
import time
from autogen_PML import PML_Functions
from ufl.algorithms.compute_form_data import estimate_total_polynomial_degree

# The code will be executed in frequency-multiprocessing mode:
# the entire domain will be assigned to one process for every frequency.
# Set the number of processes to use in parallel:
n_processes = 4

#frequency range definition
f_axis = np.arange(50, 3005, 50)

#Mic position definition
mic = np.array([0.1, 0.1, 0.1]) 

# fluid quantities definition
c0 = 340
rho_0 = 1.225
# omega = Constant(msh, PETSc.ScalarType(1))
# k0 = Constant(msh, PETSc.ScalarType(1))

# approximation space polynomial degree
deg = 1

# cad name
CAD_name = 'air.step'

# mesh name 
mesh_name_prefix =  CAD_name.rpartition('.')[0] 

# PML 
PML_FLAG      = 1       # 1 if PML_Functions needs to be used, 0 if not
Num_layers    = 8       # number of PML elements layers
d_PML         = 0.08    # total thickness of the PML layer
mesh_size_max = 0.005   # the mesh will be created entirely in gmsh. this sets its maximum
                        # size
# PML_surfaces = [4,6]  # vector of tags, identifying the surfaces from which the PML
                        # layer gets automatically extruded. Comment this line to extrude
                        # all the 2D surfaces of the system

# PML check
if PML_FLAG == 1:
    LAMBDA_PML, detJ, omega, k0, msh, V, VV, cell_tags, facet_tags = PML_Functions(CAD_name, mesh_size_max, Num_layers, d_PML)
else:
    msh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_name_prefix + "TOT.msh", MPI.COMM_SELF, 0, gdim=3)
    V = FunctionSpace(msh, ("CG", deg))
    VV = VectorFunctionSpace(msh, ("CG", deg))
    omega = Constant(V, PETSc.ScalarType(1))
    k0 = Constant(V, PETSc.ScalarType(1))

# Source amplitude
Q = 0.0001

#Source definition position = (Sx,Sy)
Sx = 0.05
Sy = 0.02
Sz = 0.02

# Test and trial function space
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)  
f = Function(V)

#Narrow normalized gauss distribution (quasi-monopole)
alfa          = c0/max(f_axis)/10 
delta_tmp     = Function(V)
delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2+(x[2]-Sz)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) 
delta         = delta_tmp/int_delta_tmp
int_delta     = assemble_scalar(form(delta*dx))


# weak form definition: 
f  = 1j*rho_0*omega*Q*delta

dx = Measure("dx", domain=msh, subdomain_data=cell_tags, metadata={"quadrature_degree": 6})

a  = inner(grad(u), grad(v)) * dx(1) - k0**2 * inner(u, v) * dx(1)  \
        + inner(LAMBDA_PML*grad(u), grad(v)) * dx(2) - detJ * k0**2 * inner(u, v) * dx(2)

L  = inner(f, v) * dx(1)

#building the problem
uh      = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"})

# creation of the mesh without the PML on which the soultion is projected
msh_INT, cell_tags_INT, facet_tags_INT = gmshio.read_from_msh(mesh_name_prefix + "INT.msh", MPI.COMM_SELF, 0, gdim=3)
V_INT = FunctionSpace(msh_INT, ("CG", deg))
uh_NOPML = Function(V_INT)

# spectrum initialization
p_mic = np.zeros((len(f_axis),1),dtype=complex)

# frequency loop
print(" ")

start_time = time.time()
def frequency_loop(nf):

    freq = f_axis[nf]
    print("Computing frequency: " + str(freq) + " Hz...")  

    # Compute solution
    omega.value = f_axis[nf]*2*np.pi
    k0.value    = 2*np.pi*f_axis[nf]/c0
    problem.solve()

    uh_NOPML.interpolate(uh)
    
    #Export field for multiple of 100 Hz frequencies
    if freq%100 == 0:
        with XDMFFile(msh.comm,"Solution_" + str(freq) + "Hz.xdmf", "w") as xdmf:
             xdmf.write_mesh(msh)
             xdmf.write_function(uh)
        with XDMFFile(msh_INT.comm,"Solution_" + str(freq) + "Hz_NOPML.xdmf", "w") as xdmf:
             xdmf.write_mesh(msh_INT)
             xdmf.write_function(uh_NOPML)
    
    # Microphone pressure at specified point evaluation
    points = np.zeros((3,1))
    points[0][0] = mic[0]
    points[1][0] = mic[1]
    points[2][0] = mic[2]
    bb_tree = geometry.BoundingBoxTree(msh, msh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)
    
    #building of sound pressure vector
    return u_values

# MultiProcessing: every frequency the entire domain gets assigned to on one process
from multiprocessing import Pool
nf = range(0,len(f_axis))

if __name__ == '__main__':
    print("Computing...")  
    pool = Pool(n_processes)                        # Create a multiprocessing Pool
    p_mic = pool.map(frequency_loop, nf)  # process data_inputs iterable with pool
    pool.close()
    pool.join()


print ("\033[A                             \033[A")
end_time = time.time()
print("JOB COMPLETED in " + str(end_time - start_time)+ " s")


#creation of file [frequency pressure]
Re_p = np.real(p_mic)
Im_p = np.imag(p_mic)
arr  = np.column_stack((f_axis,Re_p,Im_p))
print("Saving data to .csv file...")
np.savetxt("p_mic.csv", arr, delimiter=",")

#spectrum plot
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f_axis, 20*np.log10(np.abs(p_mic)/2e-5), "k", linewidth=1, label="Sound pressure [dB]")
plt.grid(True)
plt.xlabel("frequency [Hz]")
plt.legend()
plt.show()

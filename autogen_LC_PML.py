# AUTO-GENERATING LOCALLY-CONFORMAL PERFECTLY MATCHED LAYER 
#
# Copyright (C) 2022-2023 Antonio Baiano Svizzero, Undabit
# www.undabit.com

# This function takes CAD file and PML quantities as inputs, giving as output all the necessary 
# functions to build the weak form of the Helmholtz equations in complex coordinates such as
# Lambda tensor, det(J_PML), etc.



# The Lambda_PML formulas are the implementation of the following papers:
# Y. Mi, X. Yu          - Isogeometric locally-conformal perfectly matched layer 
#                         for time-harmonic acoustics, 2021 
# H. Beriot, A. Modave  - An automatic perfectly matched layer for acoustic finite
#                         element simulations in convex domains of general shape, 2020
# O. Ozgun, M. Kuzoglus - Locally-Conformal Perfeclty Matched Layer implementation 
#                         for finite element mesh truncation, 2006





# description of the inputs: 

# def PML_Functions( CAD_name        = name of the CAD file from which the pml has to be extruded. Tested with step files,
#                    mesh_size_max   = maximum mesh size used by gmsh,
#                    Num_layers      = number of PML element sublayers built in the extrusion,
#                    d_pml           = total thickness of the PML layer, 
#                    PML_surfaces    = vector containing the id of the surfaces on which the extrusion is performed. 
#                                      if = -1 gmsh will perform the extrusion on all the 2D surfaces 
# ):



def PML_Functions(CAD_name, mesh_size_max, Num_layers, d_pml, PML_surfaces=-1):

        import gmsh
        import sys
        import os
        import numpy as np
        
        #initizalizing gmsh
        gmsh.initialize(sys.argv)
        gmsh.option.setString('Geometry.OCCTargetUnit', 'M')
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)

        # merge STEP file\
        mesh_name_prefix = CAD_name.rpartition('.')[0] 
        path             = os.path.dirname(os.path.abspath(__file__))
        gmsh.merge(os.path.join(path, CAD_name))

        # set volume physical group, create volume mesh and export it in .msh
        gmsh.model.addPhysicalGroup(3, [1], 1, "air_int")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_name_prefix + "INT.msh")
        gmsh.model.mesh.clear()
        gmsh.model.occ.synchronize()

        # set PML interface tags if = -1 every 2D surface gets extruded
        if PML_surfaces != -1:
                
                PML_interface_entities = [(2,s) for s in PML_surfaces]
        else:
                PML_interface_entities = gmsh.model.getEntities(2)

        # extrude PML layer of thickness d_PML and Num_layers layers
        gmsh.option.setNumber('Geometry.ExtrudeReturnLateralEntities', 0)
        e = gmsh.model.geo.extrudeBoundaryLayer(PML_interface_entities, [Num_layers], [d_pml], False)

        # retrieve tags from PML interface (bottom surf), PML subvolumes
        bottom_surf  = [s[1] for s in PML_interface_entities]
        pml_entities = [s for s in e if s[0] == 3]
        pml_volumes  = [s[1] for s in pml_entities]

        # set PML volume and PML interface physical groups, generate the entire 3D mesh
        # and export it in .msh file
        gmsh.model.addPhysicalGroup(3,pml_volumes, 2, "PML")
        gmsh.model.addPhysicalGroup(2, bottom_surf, 3, "pml_int")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_name_prefix + "TOT.msh")
       

        # compute and extract the needed 2D quantities using gmsh API: 
        #        k2, k3 = principal curvatures of the PML interface
        #        t2, t3 = principal direction of curvature
        #        n      = normal of the PML interface

        TAGS  = []
        K2    = []
        K3    = []
        T2    = []
        T3    = []
        N     = []
        COORD = []

        for sub_surf in bottom_surf:
                tags, coord, param = gmsh.model.mesh.getNodes(2, sub_surf, True)
                k_2, k_3, t_2, t_3 = gmsh.model.getPrincipalCurvatures(sub_surf,param)
                n_1 = gmsh.model.getNormal(sub_surf,param)

                COORD.append(coord)     # GMSH doesn't allow to extract the coordinate and
                K2.append(np.abs(k_2))  # parametrization in one step for all the mesh.
                K3.append(np.abs(k_3))  # The extraction is performed on the single 
                T2.append(t_2)          # sub-surface and then all the quantities are
                T3.append(t_3)          # appended in one vector.
                TAGS.append(tags)       
                N.append(n_1)

        COORD2 = np.concatenate(COORD)  # With concatenate we have one single vector of 
        K22    = np.concatenate(K2)     # quantities for the entire mesh
        K33    = np.concatenate(K3)     # WARNING! 
        T22    = np.concatenate(T2)     # Since we retrieved the quantities for every
        T33    = np.concatenate(T3)     # single sub-surface, we now have duplicates 
        NN     = np.concatenate(N)      # on the interface between the subsurfaces.
        TAGSS  = np.concatenate(TAGS)   # np.unique() will solve this

        # delete the duplicate nodes and related quantities retrieve sorted node tags
        TAGSS_NODUP, indexes = np.unique(TAGSS, return_index=True) 

        # remove duplicates and sort by node tag
        XX = COORD2[3*indexes]  
        YY = COORD2[3*indexes+1]
        ZZ = COORD2[3*indexes+2]

        K22_NODUP = K22[indexes]
        K33_NODUP = K33[indexes]

        T22_x = T22[3*indexes]  
        T22_y = T22[3*indexes+1]
        T22_z = T22[3*indexes+2]

        T33_x = T33[3*indexes]  
        T33_y = T33[3*indexes+1]
        T33_z = T33[3*indexes+2]

        N_x = NN[3*indexes]  
        N_y = NN[3*indexes+1]
        N_z = NN[3*indexes+2]


        # extract 3D nodes coordinates and tags
        tags3, coord3, param3 = gmsh.model.mesh.getNodes(3, -1, includeBoundary = True, returnParametricCoord = True)
        tags3_nodup, indexes3 = np.unique(tags3, return_index=True)
        tags3_PML, coord3_PML = gmsh.model.mesh.getNodesForPhysicalGroup(3,2)
        
        # gain for 3D remove duplicates and sort
        xx=coord3[3*indexes3]  
        yy=coord3[3*indexes3+1]
        zz=coord3[3*indexes3+2]
        gmsh.finalize()

        # start working in FEniCSx: import mesh, define FunctionSpaces, 
        #                           retrieve dolfinx reordered indexes.
        import numpy as np
        from dolfinx.fem import Function, FunctionSpace, Constant, VectorFunctionSpace, TensorFunctionSpace
        from dolfinx.io import XDMFFile, gmshio
        from ufl import as_matrix, ln
        from mpi4py import MPI
        from petsc4py import PETSc


        msh, cell_tags, facet_tags  = gmshio.read_from_msh(mesh_name_prefix + "TOT.msh", MPI.COMM_SELF, 0, gdim=3)
        V                           = FunctionSpace(msh, ("CG", 1))
        VV                          = VectorFunctionSpace(msh, ("CG", 1))
         
        indexes_dolfinx             = msh.geometry.input_global_indices
 
        omega                       = Constant(V, PETSc.ScalarType(1))
        k0                          = Constant(V, PETSc.ScalarType(1))
        
        # initialize variables
        k2_PML                      = np.zeros([len(tags3_nodup)])
        k3_PML                      = np.zeros([len(tags3_nodup)])
        d                           = np.zeros([len(tags3_nodup)])
        t2_x_PML                    = np.zeros([3*len(tags3_nodup)])
        t2_y_PML                    = np.zeros([3*len(tags3_nodup)])
        t2_z_PML                    = np.zeros([3*len(tags3_nodup)])
        t3_x_PML                    = np.zeros([3*len(tags3_nodup)])
        t3_y_PML                    = np.zeros([3*len(tags3_nodup)])
        t3_z_PML                    = np.zeros([3*len(tags3_nodup)])
        n_x_PML                     = np.zeros([3*len(tags3_nodup)])
        n_y_PML                     = np.zeros([3*len(tags3_nodup)])
        n_z_PML                     = np.zeros([3*len(tags3_nodup)])

        # Assign the 2D quantities computed on the PML interface to the nodes in the
        # PML volume. Every node of the PML volume, gets assigned the quantities in
        # the nearest PML interface node.
        for n in range(len(tags3_nodup)):
                if n in tags3_PML:
                        d_square    = np.sqrt((xx[n] - XX)**2 + (yy[n] - YY)**2 + (zz[n] - ZZ)**2)
                        min_d       = np.min(d_square)
                        min_idx     = np.argmin(d_square)

                        k2_PML[n]   = K22_NODUP[min_idx]
                        k3_PML[n]   = K33_NODUP[min_idx]

                        d[n]        = min_d
                        
                        t2_x_PML[n] = T22_x[min_idx]
                        t2_y_PML[n] = T22_y[min_idx]
                        t2_z_PML[n] = T22_z[min_idx]

                        t3_x_PML[n] = T33_x[min_idx]
                        t3_y_PML[n] = T33_y[min_idx]
                        t3_z_PML[n] = T33_z[min_idx]

                        n_x_PML[n]  = N_x[min_idx]
                        n_y_PML[n]  = N_y[min_idx]
                        n_z_PML[n]  = N_z[min_idx]

        # assign the gmsh quantities to the dolfinx functions:
        k2                    = Function(V)
        k2.x.array[:]         = k2_PML[indexes_dolfinx]  

        k3                    = Function(V)
        k3.x.array[:]         = k3_PML[indexes_dolfinx]  

        phi_domain            = Function(V)
        phi_domain.x.array[:] = d[indexes_dolfinx]/d_pml

        t2                    = Function(VV)
        t2_PML                = []
        for a, b, c in zip(t2_x_PML[indexes_dolfinx], t2_y_PML[indexes_dolfinx], t2_z_PML[indexes_dolfinx]):
                t2_PML.append([a, b, c])
        t2.x.array[:]         =  np.concatenate(t2_PML)

        t3                    = Function(VV)
        t3_PML                = []
        for a, b, c in zip(t3_x_PML[indexes_dolfinx], t3_y_PML[indexes_dolfinx], t3_z_PML[indexes_dolfinx]):
                t3_PML.append([a, b, c])
        t3.x.array[:]         =  np.concatenate(t3_PML)

        n                     = Function(VV)
        n_PML                 = []
        for a, b, c in zip(n_x_PML[indexes_dolfinx], n_y_PML[indexes_dolfinx], n_z_PML[indexes_dolfinx]):
                n_PML.append([a, b, c])
        n.x.array[:]          =  np.concatenate(n_PML)


        # Building the model. Definition of all the functions needed to compute
        # the stretching functions 

        # normal parametric coordinate
        csi   = phi_domain*d_pml 

        #### Absorbing function ####

        # ## Polynomial ## 
        # C = 10*k0 #constant of the sigma function

        # # polynomial absorbtion function f(csi) (other functions can be chosen)
        # N_stretching = 2 # degree of the function
        # sigma        = C*phi_domain**N_stretching
        # f_csi        = d_pml*C/(N_stretching+1)*phi_domain**(N_stretching+1)

        ## Hyperbolic ##
        sigma        = 1/(d_pml- csi)
        f_csi        = - ln(1 - csi/d_pml)


        # scale factors
        h2    = 1 + k2*csi      
        h3    = 1 + k3*csi

        # stretching functions
        s1    = 1 + 1/(1j*k0)*sigma
        s2    = 1 + k2 * f_csi/(1j*k0*h2) 
        s3    = 1 + k3 * f_csi/(1j*k0*h3) 

        # diadic tensors (outer product of the principal directions)
        nnT   = as_matrix(np.outer(n,n))
        t2t2T = as_matrix(np.outer(t2,t2))
        t3t3T = as_matrix(np.outer(t3,t3))

        # determinant of J_PML
        detJ  = s1*s2*s3

        # PML tensor 
        LAMBDA_PML = s2*s3/s1 * nnT + s1*s3/s2 * t2t2T + s1*s2/s3 * t3t3T
   
        # Outputs:

        # LAMBDA_PML = PML tensor, to be used in the weak form 
        # detJ       = determinant of J_PML, to be used in the weak form
        # omega      = angular frequency
        # k0         = acoustic wavenumber
        # msh        = dolfinx msh 
        # V          = scalar functionspace
        # VV         = vector functionspace
        # cell_tags  = dolfinx cell tags
        # facet_tags = dolfinx facet tags

        return LAMBDA_PML, detJ, omega, k0, msh, cell_tags, facet_tags

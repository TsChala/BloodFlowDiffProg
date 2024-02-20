import sv
import vtk
import sys
import os
import re

print("SV imported")

def create_model_from_stl(stl_file, script_path):
    modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA)  
    # Read STL file
    try:
        model = modeler.read(stl_file)
        print("Model read from", stl_file)
        # Create face IDs for the model ('ModelFaceID' array).
        face_ids = model.compute_boundary_faces(angle=60.0)
        print("Model face IDs: " + str(face_ids))
        # Remesh the STL model if needed.
        model_surf = model.get_polydata()
        remesh_model = sv.mesh_utils.remesh_faces(model.get_polydata(),face_ids = face_ids, edge_size = 0.005)
        print("Remeshing surfaces is done")
        model.set_surface(surface=remesh_model)
        model.compute_boundary_faces(angle=60.0)
        # Write out the model as a .vtp file.
        file_format = "vtp"
        model.write(file_name = script_path, format=file_format)
        # Reset the model loaded by the mesher.
#        model_file = Path(str(new_file) + '.vtp')
        
    except Exception as e:
        print("Error reading STL file: ", e)
        sys.exit(1)


    return model

def mesh_model(model, script_path):

    mesher = sv.meshing.create_mesher(sv.meshing.Kernel.TETGEN) 
    # Set meshing options - these are just example parameters
    meshing_options = sv.meshing.TetGenOptions()
    meshing_options.surface_mesh_flag = True
    meshing_options.volume_mesh_flag = True
    meshing_options.global_edge_size = 0.035  # decent mesh with this
    #meshing_options.global_edge_size = 0.085   # use this for testing to be faster
 #   meshing_options.radius_meshing_on = True
#    meshing_options.radius_meshing_compute_centerlines = True
    meshing_options.surface_mesh_flag = True
    meshing_options.use_mmg = False
    mesher.set_boundary_layer_options(number_of_layers = 8, edge_size_fraction = 1.50, layer_decreasing_ratio = 0.80, constant_thickness = False)
    
    #set model
    mesher.set_model(model)
    # compute the boundary faces with an angle of 50 degrees
    mesher.compute_model_boundary_faces(50)
    # set walls to faceID = 1, this should be checked before to make sure the walls are really ID = 1
    mesher.set_walls([1])
    print("Walls set")
    # Generate mesh
    try:      

        # Retry with different edge size if mesher crashes
        max_retries = 10
        for retry_count in range(max_retries):
            try:
                mesher.generate_mesh(meshing_options)
                break  # Break out of the loop if meshing succeeds
            except Exception as e:
                if retry_count < max_retries - 1:
                    meshing_options.global_edge_size = meshing_options.global_edge_size-0.001
                    print("Meshing failed, retrying with new edge size:", meshing_options.global_edge_size)
                else:
                    print("Max retries reached. Unable to generate mesh.")
                    raise  # Re-raise the exception if max retries reached.


        mesh = mesher.get_mesh()
        print("Mesh:");
        print("  Number of nodes: {0:d}".format(mesh.GetNumberOfPoints()))
        print("  Number of elements: {0:d}".format(mesh.GetNumberOfCells()))

        ## Write the mesh.
        mesh_file = script_path + 'mesh-complete-mesh.vtu'
        mesher.write_mesh(file_name=str(mesh_file))    
        
        
        # save the different faces walls, inlet, outlet to vtp files
        
        modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA) 
        # save the walls as a vtp file
        face1_polydata = mesher.get_face_polydata(1)
        face1_name = script_path + '/mesh-surfaces/wall'
        
        # create mesh-surfaces directory if it doesn't exists
        isExist = os.path.exists(script_path + '/mesh-surfaces/')
        if not isExist:
    	    os.makedirs(script_path + '/mesh-surfaces/')
    	    print("New directory created:", script_path + 'mesh-surfaces')
    	
        model.set_surface(face1_polydata)
        model.write(str(face1_name),'vtp') 
        
        # simvascular needs two wall files for some reason, they are the same, just different name
        model.write(str(script_path) + 'walls_combined','vtp')
        
        
        modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA) 
        # save the inlet and outlet as a vtp file
        # select based which is closer to zero in x
        face2_polydata = mesher.get_face_polydata(2)
        bounds_2 = face2_polydata.GetBounds()

        face3_polydata = mesher.get_face_polydata(3)
        bounds_3 = face3_polydata.GetBounds()
        print("Bounds face 2:", bounds_2)
        print("Bounds face 3:", bounds_3)

        # check which X_min is smaller, that should be the inlet
        if bounds_2[0] < bounds_3[0]:
            face2_name = script_path + '/mesh-surfaces/inlet'
            face3_name = script_path + '/mesh-surfaces/outlet'
            inlet_id = 2
            outlet_id = 3
            print("Inlet is face number #2")
            print("Outlet is face number #3")
        else:
            face2_name = script_path + '/mesh-surfaces/outlet'
            face3_name = script_path + '/mesh-surfaces/inlet'
            inlet_id = 3
            outlet_id = 2
            print("Inlet is face number #3")
            print("Outlet is face number #2")



        model.set_surface(face2_polydata)
        model2 = model
        model2.write(str(face2_name),'vtp') 
        
        modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA) 
        # save the other face as a vtp file
        face3_polydata = mesher.get_face_polydata(3)

        model.set_surface(face3_polydata)
        model3 = model
        model3.write(str(face3_name),'vtp') 
        
        
        modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA) 
        # save the outlet as a vtp file
        face4_polydata = mesher.get_surface()
        face4_name = script_path + '/mesh-complete-exterior'
        model.set_surface(face4_polydata)
        model4 = model
        model4.write(str(face4_name),'vtp') 
        
        
        
            
    except Exception as e:
        print("Error during meshing: ", e)
        sys.exit(1)

    return mesh, inlet_id, outlet_id


def create_svpre(file_path, mesh_path, exterior_vtp, inlet_vtp, inlet_id, outlet_vtp, outlet_id, walls_vtp, inlet_flow, path):
    # create svpre text file from scratch
    
    # file_path - path of the created .svpre file
    # mesh_path - the mesh.vtu file
    # exterior_vtp - vtp file containing all exterior faces
    # inlet_vtp - vtp file containing the inlet face
    # inlet_id - inlet face ID (2 or 3)
    # outlet_vtp - vtp file containing the outlet face
    # outlet_id - outlet face ID (3 or 2)
    # walls_vtp - vtp file containing the walls
    # inlet_flow - path to the inlet.flow file defining the inlet flow rate boundary condition
    # path - absolute path to the mesh-complete folder
    
    # the other parameters are left unchanged for all cases
    
    content = """mesh_and_adjncy_vtu {}
set_surface_id_vtp {} 1
set_surface_id_vtp {} {}
set_surface_id_vtp {} {}
fluid_density 1.06
fluid_viscosity 0.04
initial_pressure 0
initial_velocity 0.0001 0.0001 0.0001
prescribed_velocities_vtp {}
bct_analytical_shape parabolic
bct_period 1
bct_point_number 201
bct_fourier_mode_number 10
bct_flip
bct_create {} {}
bct_write_dat {}bct.dat
bct_write_vtp {}bct.vtp
pressure_vtp {} 0
deformable_wall_vtp {}
deformable_thickness 0.05
deformable_E 15000000
deformable_nu 0.5
deformable_kcons 0.833333
deformable_pressure 133322
deformable_solve_displacements
wall_displacements_write_vtp {}displacement.vtp
write_geombc {}geombc.dat.1
write_restart {}restart.0.1
""".format(mesh_path, exterior_vtp, inlet_vtp, inlet_id, outlet_vtp, outlet_id, inlet_vtp, inlet_vtp, inlet_flow, path, path, outlet_vtp, walls_vtp, path, path, path)

    with open(file_path, 'w') as file:
        file.write(content)

    print("File {} has been created.".format(file_path))



def create_solver_inp(solver_path, outlet_id):

    # solver_path - location to save solver.inp file
    # outlet_id - oulet face ID (2 or 3 in this case)

    content = """Density: 1.06
Viscosity: 0.04

Number of Timesteps: 3000
Time Step Size: 1e-3

Number of Timesteps between Restarts: 10
Number of Force Surfaces: 1
Surface ID's for Force Calculation: 1
Force Calculation Method: Velocity Based
Print Average Solution: True
Print Error Indicators: False

Time Varying Boundary Conditions From File: True

Step Construction: 0 1 0 1 0 1

Number of Resistance Surfaces: 1
List of Resistance Surfaces: {}
Resistance Values: 109236.90537

Deformable Wall: True
Thickness of Vessel Wall: 0.05
Young Mod of Vessel Wall: 15000000
Density of Vessel Wall: 1.075
Poisson Ratio of Vessel Wall: 0.5
Shear Constant of Vessel Wall: 0.833333
Pressure Coupling: Implicit
Number of Coupled Surfaces: 1

Backflow Stabilization Coefficient: 0.2
Residual Control: True
Residual Criteria: 0.01
Minimum Required Iterations: 3
svLS Type: GMRES
Number of Krylov Vectors per GMRES Sweep: 100
Number of Solves per Left-hand-side Formation: 1
Tolerance on Momentum Equations: 0.01
Tolerance on Continuity Equations: 0.01
Tolerance on svLS NS Solver: 0.01
Maximum Number of Iterations for svLS NS Solver: 3
Maximum Number of Iterations for svLS Momentum Loop: 2
Maximum Number of Iterations for svLS Continuity Loop: 400
Time Integration Rule: Second Order
Time Integration Rho Infinity: 0.5
Flow Advection Form: Convective
Quadrature Rule on Interior: 2
Quadrature Rule on Boundary: 3""".format(outlet_id)

    with open(solver_path, 'w') as file:
        file.write(content)

    print("File {} has been created.".format(solver_path))

def main():
    
    # loop over all geometries
    for case_i in range(1,11):
        # set waveform counter to zero
        waveform_i = 0
        print('Case:',case_i)
        
        # stl file containing the vessel geometry
        stl_file = '/home/hunor/PhD/LANL/data/geometries/stenosis_geometries/stenosis_' + str(case_i) + '_cm.stl'
        
        folder_path = '/home/hunor/PhD/LANL/data/geometries/stenosis_geometries/case_'+ str(case_i) + '_' + str(waveform_i) + '/mesh-complete/'
    
        # check if path exists
        isExist = os.path.exists(folder_path)
        # if path doesn't exist create the directories
        if not isExist:
    	    os.makedirs(folder_path)
    	    print("New directory created:", folder_path)
    
        # create model from STL and remesh the surface
        model = create_model_from_stl(stl_file, folder_path)
    
        #save model to vtp file
        model.write('/home/hunor/PhD/LANL/data/geometries/stenosis_geometries/case_'+ str(case_i) + '_' + str(waveform_i) + '/surface_mesh','vtp')
        #creata and save mesh
        mesh, inlet_id, outlet_id = mesh_model(model, folder_path)


        path = '/home/hunor/PhD/LANL/data/geometries/stenosis_geometries/case_'+ str(case_i) + '_' + str(waveform_i)+'/'
	    # Set variables for geometry
        # they don't change with waveforms so can be set outside the loop
        mesh_path = path + 'mesh-complete/mesh-complete-mesh.vtu'
        exterior_vtp = path + 'mesh-complete/mesh-complete-exterior.vtp'
        inlet_vtp = path + 'mesh-complete/mesh-surfaces/inlet.vtp'
        outlet_vtp = path + 'mesh-complete/mesh-surfaces/outlet.vtp'
        walls_vtp = path + 'mesh-complete/walls_combined.vtp'
    

        # create cases with different inlet waveforms
        for waveform_i in range(0,10):
            print("Waveform number #", waveform_i)
            # inlet BC location
            inlet_flow = '/home/hunor/PhD/LANL/data/geometries/waveforms/test10_flow_' + str(waveform_i) + '.txt'

            #set the path
            path = '/home/hunor/PhD/LANL/data/geometries/stenosis_geometries/case_'+ str(case_i) + '_' + str(waveform_i)+'/'

                    # check if path exists
            isExist = os.path.exists(path)
            # if path doesn't exist create the directories
            if not isExist:
    	        os.makedirs(path)
    	        print("New directory created:", path)
        
            # write the new mesh files and their names into job.svpre file
            file_path = path + 'job.svpre'
            
            # create the svpre file
            create_svpre(file_path, mesh_path, exterior_vtp, inlet_vtp, inlet_id, outlet_vtp, outlet_id, walls_vtp, inlet_flow, path)


            # solver.inp path
            solver_path = path +'solver.inp'
            create_solver_inp(solver_path, outlet_id)
            
       
            # run the SimVascular presolver to create the necessary input files to the simulation
            command = '/usr/local/sv/svsolver/2022-07-22/svpre {}'.format(path + 'job.svpre')
            os.system(command)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

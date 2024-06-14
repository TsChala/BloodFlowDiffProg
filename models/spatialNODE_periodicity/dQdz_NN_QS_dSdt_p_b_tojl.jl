begin
    using LinearAlgebra 
    using FFTW
    using FileIO
    using JLD2
    using DiffEqFlux
    using OrdinaryDiffEq
    using BlockArrays
    using LaTeXStrings
    using SparseArrays
    using BSON
    using Distances
    using DifferentialEquations
    using Optimization
    using OptimizationPolyalgorithms
    using Zygote
    using OptimizationOptimJL
    using OptimizationOptimisers
    using DiffEqSensitivity
    using DelimitedFiles
    using HDF5
    using InvertedIndices
    using Random
    using Statistics
    using LineSearches
    using CSV
    using DataFrames
    using ChainRulesCore
    using NPZ
end 

# Define the dimensions
num_geomIDs = 10  # number of geometries
num_waveIDs = 10  # number of waveforms

# Define the matrix to store the data
all_data = []

# Iterate over geomID and waveID
for geomID in 1:num_geomIDs
    for waveID in 0:num_waveIDs-1  # Adjust the range as needed
        filename = "/uufs/chpc.utah.edu/common/home/u1447794/LANL/data/case_$(geomID)_$(waveID)/averagedTo1DFrom0_case_$(geomID)_$(waveID).npy"
        try
            # Use np.load() to read data from the .npy file
            data = npzread(filename)
            push!(all_data, data)
        catch
            println("File not found: ", filename)
        end
    end
end

data_mat = permutedims(reduce((x, y) -> cat(x, y, dims=4), all_data),(2,3,4,1))
println("Size of 1D data matrix:",size(data_mat))
println("Shape: [timesteps,spatial locations,cases, variables]")
println("Variable order: flow rate, pressure*area, undef area, deformed area, radial displacement, wss")

# define global variables
begin
    global dz = 0.04                  # x step
    global L = 4.00                   # total length
    global zspan = (0,L)              # end points of spatial integration for neural ODE momentum eqn
    global train_maxiters = 5         # number of internal iterations of learning (this is NOT the epochs)
    global learning_rate  = 0.01      # learning rate
    global dt = 0.01                  # time step, has to be smaller or equal to saveat
    global tcycles = 6                  # total time
	global T = tcycles - 0.01; 
    global saveat = 0.01              # ground truth data time resolution
    global tsteps = 0.0:dt:T          # discretized time dimension
    global tspan = (0,T);             # end points of time integration for continuity eqn

end;



path_to_working_directory="/uufs/chpc.utah.edu/common/home/u1447794/LANL/"

# import auxiliary functions for training and numerical derivative calculations

include("$path_to_working_directory/src/numerical_derivatives.jl");
include("$path_to_working_directory/src/train_utils.jl");

#first order derivatives for time
∂t1_center = f1_secondOrder_central(size(tsteps)[1],dt);
∂t1_forward = f1_secondOrder_forward(size(tsteps)[1],dt);
∂t1_backward = f1_secondOrder_backward(size(tsteps)[1],dt);

# use central difference for the majority
∂t1 = ∂t1_center
# use forward and backward difference near the boundaries
∂t1[1,:]=∂t1_forward[1,:]
∂t1[end,:] = ∂t1_backward[end,:];

# Set the seed for reproducibility
Random.seed!(123)

total_waveforms = size(data_mat,3)

indices = 1:total_waveforms;

# Divide the shuffled indices into training and test sets
# This is done somewhat arbitrarily
test_indices_geom = [indices[91:100]]   # new stenosis blockage ratio
test_indices_wave = [indices[10:10:90]]  # new waveforms

# Convert StepRange objects to arrays
test_indices_geom = collect(test_indices_geom[1])
test_indices_wave = collect(test_indices_wave[1])

# Find indices not in test_indices_ex
# shuffle training indices randomly
train_indices = shuffle(setdiff(indices, vcat(test_indices_wave, test_indices_geom)));

# select field variable ID
# 1 -flow rate, 2- pressure*area, 3- undef area, 4- deformed area, 5-radial displacement, 6-wss
pID = 1;

#define IC's
u0 = data_mat[1,1:end,train_indices,pID];         # train
u01 = data_mat[1,1:end,1,pID];

# define test set for interpolation and extrapolation cases
u0_test_wave = data_mat[1,1:end,test_indices_wave,pID];     # new waveform
u0_test_geom = data_mat[1,1:end,test_indices_geom,pID];    # new stenosis blockage ratio

#ground truth data
ytrain2 = data_mat[:,1:end,train_indices,pID];    #train
ytrain21 = data_mat[:,1:end,1,pID];

ytest_wave = data_mat[:,1:end,test_indices_wave,pID];      # new waveform
ytest_geom = data_mat[:,1:end,test_indices_geom,pID];     # new stenosis blockage ratio

# inlet boundary conditions 
bc_flow = data_mat[:,1,train_indices,pID];        # train
bc_flow1 = bc_flow[:,1];

bc_flow_test_wave = data_mat[:,1,test_indices_wave,pID];    # new waveform
bc_flow_test_geom  = data_mat[:,1,test_indices_geom,pID];  # new stenosis blockage ratio

Nspace = size(u01,1)   # number of points in space
x = 0.0 : dz : L        # discretized spatial dimension 

# finite-difference schemes for spatial derivatives

#first order derivatives for space
∂x1_center = f1_secondOrder_central(Nspace,dz);
∂x1_forward = f1_secondOrder_forward(Nspace,dz);
∂x1_backward = f1_secondOrder_backward(Nspace,dz);

# use central difference for the majority
∂x1 = ∂x1_center
# use forward and backward difference near the boundaries
∂x1[1,:]=∂x1_forward[1,:]
∂x1[end,:] = ∂x1_backward[end,:]

#second order derivatives
∂x2_center = f2_secondOrder_central(Nspace,dz);
∂x2_forward = f2_secondOrder_forward(Nspace,dz);
∂x2_backward = f2_secondOrder_backward(Nspace,dz);

# use central difference for the majority
∂x2 = ∂x2_center;
# use forward and backward difference near the boundaries
∂x2[1,:]=∂x2_forward[1,:];
∂x2[end,:] = ∂x2_backward[end,:];

# create arrays with area data

aID = 4 # deformed area
# aID-1 = 3 is undeformed area

#ground truth data for area
Atrain_org = data_mat[:,1:end,train_indices,aID]; # ground-truth area

# undeformed area will initilize the model
global Atrain
global Atest_wave
global Atest_geom

Atrain = data_mat[:,1:end,train_indices,aID-1] #"undeformed area" for initalization
Atrain_undef = Atrain;
Atrain1 = data_mat[:,1:end,1,aID];

# the same areas will from be used for all 3 sets
Atest_wave_org = data_mat[:,1:end,test_indices_wave,aID]; # ground-truth area
Atest_wave =  data_mat[:,1:end,test_indices_wave,aID-1] #undeformed area for initalization
Atest_wave_undef = Atest_wave;

Atest_geom_org = data_mat[:,1:end,test_indices_geom,aID]; # ground-truth area
Atest_geom = data_mat[:,1:end,test_indices_geom,aID-1]; #undeformed area for initalization
Atest_geom_undef = Atest_geom;


N = size(bc_flow1,1)
# NN(Q,S) embedded in PDE for Differential programming
# Define the network architecture with initialization
hidden_dim = 10

# network takes as input Q and S, output size is same as Q
ann = Chain(
    Dense(2*N, hidden_dim, tanh, init = Flux.glorot_uniform),
    Dense(hidden_dim, hidden_dim, tanh, init = Flux.glorot_uniform),
    Dense(hidden_dim, hidden_dim, tanh, init = Flux.glorot_uniform),
    Dense(hidden_dim, N, init = Flux.glorot_uniform)
)

# flatten parameters in NN for learning.                
p, re = Flux.destructure(ann);
ps = deepcopy(p)
p_size = size(p,1);
println("Number of parameters in neural network for momentum equation: $p_size");
flush(stdout) 

# parameters for continuity equation                
p_cont = ones(2 .* size(ytrain2[1,:,1]))
ps_cont = deepcopy(p_cont)
pcont_size = size(p_cont,1);
println("Number of parameters in p vector for continuity equation: $pcont_size"); 

# # Define time-dependent variables
function interpolate_variables(t, vector, dz, L)
    # t - dependent variable, could be time or space (z) too
    #      if t is time then use dt, if it's space use dz, same with T <--> L
    # vector - data vector with values at distinct t locations
    # dz - steps at which we have data in vector
    # L - (physical) length of t
    #
    # This function interpolates values such that we can access the values from vector
    # not just at the original data points but anywhere in between
    

    # Find the two closest points in vector
    #caculate the time index that's closest to time t

    t_index = Int(floor(t / dz)) + 1

    # calculate local time fraction between the grid points
    t_frac = (t - (t_index - 1) * dz) / dz

    # Perform linear interpolation between data points in vector
    # if we are at the last timesteps just copy the value cause t_index+1 will not exist
    if t == L
        vector_interp = vector[:,:,t_index]
        
    else
        vector_interp = (1 - t_frac) * vector[:,:,t_index] + t_frac * vector[:,:,t_index + 1]
    end
    
    # return the interpolated value of vector at time(space) = t
    return vector_interp
end

# parabolic flow profile constant
# https://simvascular.github.io/docs1DSimulation.html#solver
δ = 1/3;
# viscosity in CGS units
ν = 0.04;
# velocity profile constant
Nprof = -8*π*ν;

function learn_1DBlood(u, p, z, interp_func)
    # u -  field variable we are solving for (flow rate)
    # p - neural network parameters
    # z - dependent variable (z -space, or t - time)
    # interp_func - area values for interpolation as a function of z
    
    
    Φ = re(p)  # restructure flattened parameter vector into NN architecture.
    
   
    # add some physics term to the right-hand-side
    # 1/2*Q/A*dA/dz + N/(2*(1+δ))
    
    #
    #.+ 1/2 .* u ./ interp_func(z) .* dadz_interp_func(z) .+ Nprof/(2*(1+δ))
    #.-1/(2*(1+δ)) .* interp_func(z) ./ u .* (∂t1 * u) 
    
    
    # prints for debugging
#     println("space z: ", z)
#     println("term 1: 1/2*Q/S*dS/dz ", sum(1/2 .* u ./ interp_func(z) .* dadz_interp_func(z)))
#     println("term 2 N/2(1+delta): ", sum(Nprof/(2*(1+δ))))
#     println("term 3 -1/2(1+d)*S/Q*dQ/dt: ", sum(-1/(2*(1+δ)) .* interp_func(z) ./ u .* (∂t1 * u)))
#     println("Mean area S: ", mean(interp_func(z)), "size S: ", size(interp_func(z)))
    
#     println("Mean area dS/dz: ", mean(dadz_interp_func(z)))
#     println("Mean Previous flow rate Q:", mean(u), ", std Q:", std(u), "size Q: ", size(u))
#     println("Mean Q/S:", mean(u ./ interp_func(z)))
#     println("Mean S/Q:", mean(interp_func(z) ./ u))
#     println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#     flush(stdout)
    
#     plotQ = plot(u)
#     display(plot(plotQ))
#     sleep(1)
    
    
    v = vcat(u,interp_func(z))   # concatenate Q and S
    return Φ(v)
end

# #define learning problem.
learn_1DBlood_prob(u01,zspan) =  ODEProblem((u, p, z) -> learn_1DBlood(u, p, z, bc_left_func), u01, zspan, p)

function learn_1DBlood_continuity(u, p, t, interp_func)
    # u -  field variable we are solving for (flow rate)
    # p - learnable parameters parameters
    # t - dependent variable (z -space, or t - time)
    # interp_func - flow rate values for interpolation as a function of t
    
    flow_rate = interp_func(t) #flow rate at the interpolated timestep
    return -p[1:size(∂x1,1)] .* (∂x1 * flow_rate) .+ p[size(∂x1,1)+1:end]
end

ode_solver = "Tsit5"
output_dir = "./output/"
working_dir = output_dir


#prob = ODEProblem((u, p, z) -> learn_1DBlood(u, p, z,bc_left_func,dadz_interp_func), bc_flow1, zspan, p) ;
function predict(θ,prob)
    if ode_solver == "Tsit5"
        Array(solve(prob,Tsit5(),p=θ,dt=dz,saveat=dz,adaptive=true))
    elseif ode_solver == "RK4"
        Array(solve(prob,RK4(),p=θ,dt=dz/10,saveat=dz,adaptive=false,alg_hints=[:stiff]))
    elseif ode_solver == "Rosenbrock23"
        Array(solve(prob,Rosenbrock23(),p=θ,dt=dz/10,saveat=dz,adaptive=false,alg_hints=[:stiff]))
    end
end 

prob_S = ODEProblem((u, p, z) -> learn_1DBlood_continuity(u, p, t,interp_func), Atrain1[1,:], tspan, p) ;
function predict_S(θ,prob_S)
    if ode_solver == "Tsit5"
        Array(solve(prob_S,Tsit5(),p=θ,dt=dt,saveat=dt,adaptive=false))
    elseif ode_solver == "RK4"
        Array(solve(prob_S,RK4(),p=θ,dt=dt/10,saveat=dt,adaptive=false,alg_hints=[:stiff]))
    elseif ode_solver == "Rosenbrock23"
        Array(solve(prob_S,Rosenbrock23(),p=θ,dt=dt/10,saveat=dt,adaptive=false,alg_hints=[:stiff]))
    end
end 

function loss(θ, ytrain21, prob)
    # θ - NN parameters
    # ytrain21 - ground truth flow rate values
    # prob - ODE problem
    
    
    # calculate neural ODE predicted Q
    pred = predict(θ, prob)

    # calculate loss between predicted and ground truth Q
    l = sum(abs2,(pred - ytrain21))
    
    # L1 regularization could be added as:  + 1e-3*sum(abs.(θ))
    
    return l, pred
end


# training optimizer definition
adtype = Optimization.AutoZygote() ;
#optf = Optimization.OptimizationFunction((x,p)->loss(x,ytrain21,prob),adtype) ;

# Runge-Kutta solver
# this will be used for the continuity eqn
function rk4_solve_1step(prob, θ, dt,tf,u0)
    # set initial condition
    u = u0
    
    # set problem right-hands side function
    f = prob.f
    
    # set initial time, one step before the final time
    ti = tf-dt
    
    
    #calculate Runge-Kutta step
    k1 = f(u, θ, ti)
    k2 = f(u .+ dt/2 .* k1, θ, ti + dt/2)
    k3 = f(u .+ dt/2 .* k2, θ, ti + dt/2)
    k4 = f(u .+ dt .* k3, θ, ti + dt)

    u_new = u .+ dt/6 .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)
    
    return u_new
end;

function loss_S(θ, ytrain21, prob, atrain1)
    # solve system and calculate loss for continuity equation
    
    # θ -  parameters
    # ytrain21 - input flow rate data - shape: [space, batch, time]
    # prob - ODE problem formulation
    # atrain1 - ground truth area data - shape [space, batch, time]
	
    pred = predict_S(θ, prob)
	
    # MSE loss from data
	# accumulate loss only after 2nd(or 3rd) cycle
    MSEloss = sum(abs2,(pred[:,:,201:end] - atrain1[:,:,201:end]))
	
    # periodicity loss from second cycle
    per_loss = 0.0
    for i in 3:tcycles-1
        per_loss = per_loss +  10 .* sum(abs2,pred[:,:,((i-1)*100)+1:(i)*100] - pred[:,:,(i*100)+1:(i+1)*100])
    end

    l = MSEloss + per_loss
    #return the total loss at the end
    return l, pred
end


#function to replicate vectors
function replicate_array(array, times, dim)
    return cat([array for _ in 1:times]..., dims=dim)
end

# training optimizer definition
adtype = Optimization.AutoZygote() ;


# training optimizer definition
adtype = Optimization.AutoZygote() ;

path_checkpoint=nothing
optimizer_choice1 = "ADAM";
optimizer_choice2 = "BFGS";

# process user choices

if !isdir(working_dir)
    mkdir(working_dir)
end

cd(working_dir) #switch to working directory
if !isdir(output_dir)
    mkdir(output_dir)
end
println("optimizer 1 is $optimizer_choice1")
if !isnothing(optimizer_choice2)
    println("optimizer 2 is $optimizer_choice2 optimizer")
end

println("ODE spatial integrator selected:", ode_solver)


# restart training from previaous checkpoint if it exists, else start 
# fresh training.
transfer_learning = false
global uinit
if transfer_learning
    #load learnt parameters from file
    p_learn = load("/home/tschala/research/LANL/summer2024/dQdz_NN_QS_dSdt_p_b_shuffled/ptrained_BFGS_Q.jld2")
    uinit = p_learn["p"];
    
    p_cont_learn = load("/home/tschala/research/LANL/summer2024/dQdz_NN_QS_dSdt_p_b_shuffled/ptrained_BFGS_S.jld2")
    uinit_cont = p_cont_learn["p"];
    println("Transfer learning - loaded weights from file")
else
    uinit_cont = deepcopy(ps_cont)
    uinit = deepcopy(ps)
    println("Fresh training initialized")
end

zspan = (0.0, L)
println("Spatial domain size:",zspan)
flush(stdout)
n_epochs = 40
n_epochs_cont = 1

train_block = 5
plot_flag = false
#set batch size
batch_size = 10
println("Batch size:", batch_size)
println("############################################")
#training batches
batch_iterations = Int(ceil(size(ytrain2,3)/batch_size))
#testing batches
test_wave_batch_iterations = Int(ceil(size(ytest_wave,3)/batch_size));
test_geom_batch_iterations = Int(ceil(size(ytest_geom,3)/batch_size));

list_loss_train = []
list_loss_epoch = []
list_loss_test_wave = []
list_loss_epoch_test_wave = []
list_loss_test_geom = []
list_loss_epoch_test_geom = []


list_S_loss_train = []
list_S_loss_epoch = []
list_S_loss_test_wave = []
list_S_loss_epoch_test_wave = []
list_S_loss_test_geom = []
list_S_loss_epoch_test_geom = []


pred_train = []
pred_test_wave = []
pred_test_geom = []



for j in 1:n_epochs
    println("Start training epoch ",j)
    loss_tot = 0.0
    loss_tot_test_wave = 0.0
    loss_tot_test_geom = 0.0
    
    
    # Change learning rate for ADAM optimizer, BFGS doesn't use it
    if j % 3 == 0 && learning_rate > 1e-6
        global learning_rate = learning_rate * 0.1
        println("Changing learning rate to:",learning_rate)
    end
            # loop over different waveforms
            for i in 1:batch_iterations
                
                println("waveform batch: ",i, "/",batch_iterations)
                flush(stdout)
                #reorder ytrain, atrain and dAdz to [time, batch_size, spatial location]
                # batch size should be second column
        
                #reorder ytrain to (spatial location, batch_size, time)
                if i!=batch_iterations
                    ytrain = permutedims(ytrain2[:,:,batch_size*(i-1)+1:batch_size*i],(1,3,2))
                    atrain = permutedims(Atrain[:,:,batch_size*(i-1)+1:batch_size*i],(1,3,2))
                else
                    ytrain = permutedims(ytrain2[:,:,batch_size*(i-1)+1:end],(1,3,2))
                    atrain = permutedims(Atrain[:,:,batch_size*(i-1)+1:end],(1,3,2))
                end
                
                
                #define function for interpolating area and dA/dz to the actual spatial location for the ODE
                interp_func(z) = interpolate_variables(z, atrain, dz, L)
                dadz_interp_func(z) = interpolate_variables(z, dadztrain, dz, L)
            
                #define optimization problem
                prob = ODEProblem((u, p, z) -> learn_1DBlood(u, p, z, interp_func), ytrain[:,:,1], zspan, p);
                optf = Optimization.OptimizationFunction((x,p)->loss(x,ytrain[:,:,:],prob),adtype) ;

                println("Using $optimizer_choice1 optimizer")
                println("Sum of params:", sum(uinit))

				
#                 uinit = train_loop(uinit,adtype,optf,train_maxiters,learning_rate,optimizer_choice1,0)
#                 println("Sum of params:", sum(uinit))

                if !isnothing(optimizer_choice2)
                    println("Switching to $optimizer_choice2 optimizer")

                    global uinit = train_loop(uinit,adtype,optf,train_maxiters*1,learning_rate,optimizer_choice2,0)

                    println("Sum of params:", sum(uinit))

                end


                #calculate final loss and push it to the list
                prob = ODEProblem((u, p, z) -> learn_1DBlood(u, p, z, interp_func), ytrain[:,:,1], zspan, p);
                l , pred = loss(uinit,ytrain[:,:,:],prob)
                loss_tot = loss_tot + l
                
                # save loss to list
                push!(list_loss_train, l)
                # save prediction for continuity equation every 'train_block' epochs
                if j%train_block==0
                    push!(pred_train, pred)
                end
        
                if plot_flag
                    # plot results for visual check
                    plot1 = heatmap(pred[:,1,:]', color=:viridis, title = "neural ODE flow rate")
                    xlabel!("time")
                    ylabel!("x")

                    plot2 = heatmap(ytrain[:,1,:]', title="1D - flow rate", color=:viridis)
                    xlabel!("time")
                    ylabel!("x")
    #                 display(plot(plot1,plot2,layout = (2, 1)))
    #                 sleep(1)

                    plot3 = heatmap(∂x1 * pred[:,1,:]', color=:viridis, title = "∂Q∂z - neural ODE", clims = (-0.05,0.05))
                    xlabel!("time")
                    ylabel!("x")

                    plot4 = heatmap(∂x1 * ytrain[:,1,:]', title="∂Q∂z", color=:viridis, clims = (-0.05,0.05))
                    xlabel!("time")
                    ylabel!("x")

                    display(plot(plot1, plot2, plot3, plot4,layout = (2, 2)))
                    sleep(1)
                end

            end    
    
            #testing loop - new geom
            println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            println("Testing - geometry - stenosis blockage ratio:")

            for i in 1:test_geom_batch_iterations

                println("waveform batch: ",i, "/",test_geom_batch_iterations)

                #reorder ytrain to (spatial location, batch_size, time)
                if i!=test_geom_batch_iterations
                    Ytest_geom = permutedims(ytest_geom[:,:,batch_size*(i-1)+1:batch_size*i],(1,3,2))
                    atest_geom = permutedims(Atest_geom[:,:,batch_size*(i-1)+1:batch_size*i],(1,3,2))
                else
                    Ytest_geom = permutedims(ytest_geom[:,:,batch_size*(i-1)+1:end],(1,3,2))
                    atest_geom = permutedims(Atest_geom[:,:,batch_size*(i-1)+1:end],(1,3,2))
                end

        
                #define function for interpolating area and dA/dz to the actual spatial location for the ODE
                interp_func(z) = interpolate_variables(z, atest_geom, dz ,L)
                dadz_interp_func(z) = interpolate_variables(z, dadztest_geom, dz, L)


                #calculate final loss and push it to the list
                prob = ODEProblem((u, p, t) -> learn_1DBlood(u, p, t, interp_func), Ytest_geom[:,:,1], zspan, p);
                l , pred = loss(uinit,Ytest_geom[:,:,:],prob)
                loss_tot_test_geom = loss_tot_test_geom + l

                push!(list_loss_test_geom, l)
                println("Test loss - new stenosis blockage ratio:",l )
                if j%train_block==0
                    push!(pred_test_geom, pred)
                end
        
                if plot_flag
                    # plot solution for comparison
                    plot1 = heatmap(pred[:,1,:]', color=:viridis, title = "neural ODE flow rate")
                    xlabel!("time")
                    ylabel!("x")

                    plot2 = heatmap(Ytest_geom[:,1,:]', title="3D - flow rate", color=:viridis)
                    xlabel!("time")
                    ylabel!("x")
                    display(plot(plot1,plot2,layout = (2, 1)))
                    sleep(1)
                end
            end

            #testing loop - new waveforms
            println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            println("Testing - waveform:")

            for i in 1:test_wave_batch_iterations
                
                println("waveform batch: ",i, "/",test_wave_batch_iterations)

                #reorder ytrain to (spatial location, batch_size, time)
                if i!=test_wave_batch_iterations
                    Ytest_wave = permutedims(ytest_wave[:,:,batch_size*(i-1)+1:batch_size*i],(1,3,2))
                    atest_wave = permutedims(Atest_wave[:,:,batch_size*(i-1)+1:batch_size*i],(1,3,2))
                else
                    Ytest_wave = permutedims(ytest_wave[:,:,batch_size*(i-1)+1:end],(1,3,2))
                    atest_wave = permutedims(Atest_wave[:,:,batch_size*(i-1)+1:end],(1,3,2))
                end

        
                #define function for interpolating area and dA/dz to the actual spatial location for the ODE
                interp_func(z) = interpolate_variables(z, atest_wave, dz, L)
                dadz_interp_func(z) = interpolate_variables(z, dadztest_wave, dz, L)


                #calculate final loss and push it to the list
                prob = ODEProblem((u, p, t) -> learn_1DBlood(u, p, t, interp_func), Ytest_wave[:,:,1], zspan, p);
                l , pred = loss(uinit,Ytest_wave[:,:,:],prob)
                loss_tot_test_wave = loss_tot_test_wave + l

                push!(list_loss_test_wave, l)
                println("Test loss - new wavform:",l )
        
                if j%train_block==0
                    push!(pred_test_wave, pred)
                end
                if plot_flag
                    # plot solution for comparison
                    plot1 = heatmap(pred[:,1,:]', color=:viridis, title = "neural ODE flow rate")
                    xlabel!("time")
                    ylabel!("x")

                    plot2 = heatmap(Ytest_wave[:,1,:]', title="3D - flow rate", color=:viridis)
                    xlabel!("time")
                    ylabel!("x")
                    display(plot(plot1,plot2,layout = (2, 1)))
                    sleep(1)
                end
            end
    
    # save and print losses
    push!(list_loss_epoch, loss_tot/(size(ytrain2,3)))
    push!(list_loss_epoch_test_wave, loss_tot_test_wave/(size(ytest_wave,3)))
    push!(list_loss_epoch_test_geom, loss_tot_test_geom/(size(ytest_geom,3)))
    println("Epoch ", j, " mean train loss:", loss_tot/(size(ytrain2,3)))
    println("Epoch ", j, " mean test loss - new waveform:", loss_tot_test_wave/(size(ytest_wave,3)))
    println("Epoch ", j, " mean test loss - new geom:", loss_tot_test_geom/(size(ytest_geom,3)))
    


    pred_S_train = []
    pred_S_test_wave = []
    pred_S_test_geom = []
                
                
    if j%train_block == 0
        # get predicted flow rates and shape them as we need
        pred_train_mat = reduce((x, y) -> cat(x, y, dims=2), pred_train)
        pred_test_wave_mat = reduce((x, y) -> cat(x, y, dims=2), pred_test_wave);
        pred_test_geom_mat = reduce((x, y) -> cat(x, y, dims=2), pred_test_geom);
            
    
        println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        for k in 1:n_epochs_cont
            println("Start continuity training epoch ",k)
            loss_tot = 0.0
            loss_tot_test_wave = 0.0
            loss_tot_test_geom = 0.0


            # Change learning rate for ADAM optimizer, BFGS doesn't use it
            if k % 3 == 0 && learning_rate > 1e-6
                global learning_rate = learning_rate * 0.1
                println("Changing learning rate to:",learning_rate)
            end
                    # loop over different waveforms
                    for i in 1:batch_iterations

                        println("waveform batch: ",i, "/",batch_iterations)
                        flush(stdout)
                        # batch size should be second column

                        #reorder ytrain to (spatial location, batch_size, time)
                        if i!=batch_iterations
                            ytrain = permutedims(pred_train_mat[:,batch_size*(i-1)+1:batch_size*i,:],(3,2,1))
                            atrain = permutedims(Atrain_org[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))
                            atrain_undef = permutedims(Atrain_undef[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))

                            #repeat for tcycles cardiac cycles
                            ytrain = replicate_array(ytrain, tcycles, 3)
                            atrain = replicate_array(atrain, tcycles, 3)
                        else
                            ytrain = permutedims(pred_train_mat[:,batch_size*(i-1)+1:end,:],(3,2,1))
                            atrain = permutedims(Atrain_org[:,:,batch_size*(i-1)+1:end],(2,3,1))
                            atrain_undef = permutedims(Atrain_undef[:,:,batch_size*(i-1)+1:end],(2,3,1))


                            ytrain = replicate_array(ytrain, tcycles, 3)
                            atrain = replicate_array(atrain, tcycles, 3)
                        end


                        #interpolate flow rate values as a function of time
                        interp_func(t) = interpolate_variables(t, ytrain, dt, T)

                         #define optimization problem
                        prob = ODEProblem((u, p_cont, t) -> learn_1DBlood_continuity(u, p_cont, t, interp_func), atrain_undef[:,:,1], tspan, p_cont);
                        optf = Optimization.OptimizationFunction((x,p_cont)->loss_S(x,ytrain,prob,atrain),adtype) ;

                        println("Using $optimizer_choice1 optimizer")
                        println("Sum of params:", sum(uinit_cont))

						global uinit_cont
                        uinit_cont = train_loop(uinit_cont,adtype,optf,train_maxiters,learning_rate,optimizer_choice2,1)
                        println("Sum of params:", sum(uinit_cont))



                        #calculate final loss and push it to the list
                        prob = ODEProblem((u, p_cont, t) -> learn_1DBlood_continuity(u, p_cont, t, interp_func), atrain_undef[:,:,1], tspan, p_cont);
                        l , pred = loss_S(uinit_cont,ytrain,prob, atrain)
                        loss_tot = loss_tot + l

                        # save loss to list
                        push!(list_S_loss_train, l)

                        # Initialize the resulting matrix
                        pred_S_matrix = zero(atrain)

                        # Restructure the matrices into the result matrix
						pred_S_matrix = pred
                        
						pred = nothing;
                        push!(pred_S_train, pred_S_matrix)    

                        if plot_flag
                            # plot results for visual check
                            plot1 = heatmap(pred_S_matrix[:,1,:]', color=:viridis, title = "neural ODE area ")
                            xlabel!("time")
                            ylabel!("x")

                            plot2 = heatmap(atrain[:,1,:]', title="3D - area", color=:viridis)
                            xlabel!("time")
                            ylabel!("x")

                            display(plot(plot1, plot2, layout = (1,2)))
                            sleep(1)
                        end

                    end    

                    #testing loop - new geom
                    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    println("Testing - geometry - stenosis blockage ratio:")

                    for i in 1:test_geom_batch_iterations

                        println("batch: ",i, "/",test_geom_batch_iterations)

                        #reorder ytrain to (spatial location, batch_size, time)
                        if i!=test_geom_batch_iterations
    #                         Ytest_geom = permutedims(ytest_geom[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))
                            Ytest_geom = permutedims(pred_test_geom_mat[:,batch_size*(i-1)+1:batch_size*i,:],(3,2,1))
                            atest_geom = permutedims(Atest_geom_org[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))
                            atest_geom_undef = permutedims(Atest_geom_undef[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))


                            # repeat for tcycles cardiac cycles
                            Ytest_geom = replicate_array(Ytest_geom, tcycles, 3)
                            atest_geom = replicate_array(atest_geom, tcycles, 3)
                        else
    #                         Ytest_geom = permutedims(ytest_geom[:,:,batch_size*(i-1)+1:end],(2,3,1))
                            Ytest_geom = permutedims(pred_test_geom_mat[:,batch_size*(i-1)+1:end,:],(3,2,1))
                            atest_geom = permutedims(Atest_geom_org[:,:,batch_size*(i-1)+1:end],(2,3,1))
                            atest_geom_undef = permutedims(Atest_geom_undef[:,:,batch_size*(i-1)+1:end],(2,3,1))

                            # repeat for tcycles cardiac cycles
                            Ytest_geom = replicate_array(Ytest_geom, tcycles, 3)
                            atest_geom = replicate_array(atest_geom, tcycles, 3)
                        end


                        #define function for interpolating area and dA/dz to the actual spatial location for the ODE
                        interp_func(t) = interpolate_variables(t, Ytest_geom, dt, T)


                        #calculate final loss and push it to the list
                        prob = ODEProblem((u, p_cont, t) -> learn_1DBlood_continuity(u, p_cont, t, interp_func,), atest_geom_undef[:,:,1], tspan, p_cont);
                        optf = Optimization.OptimizationFunction((x,p_cont)->loss_S(x,Ytest_geom,prob,atest_geom),adtype) ;
                        l , pred = loss_S(uinit_cont,Ytest_geom,prob,atest_geom)
                        loss_tot_test_geom = loss_tot_test_geom + l

                        push!(list_S_loss_test_geom, l)
                        println("Test loss - new stenosis blockage ratio:",l )

                        # Initialize the resulting matrix
                        pred_S_test_geom_matrix = zero(atest_geom)

                        pred_S_test_geom_matrix = pred

                        pred = nothing

                        push!(pred_S_test_geom, pred_S_test_geom_matrix)

                        if plot_flag
                            # plot solution for comparison
                            plot1 = heatmap(pred_S_test_geom_matrix[:,1,:]', color=:viridis, title = "neural ODE area")
                            xlabel!("time")
                            ylabel!("x")

                            plot2 = heatmap(atest_geom[:,1,:]', title="3d - area", color=:viridis)
                            xlabel!("time")
                            ylabel!("x")
                            display(plot(plot1,plot2,layout = (2, 1)))
                            sleep(1)
                        end
                    end

                    #testing loop - new waveforms
                    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    println("Testing - waveform:")

                    for i in 1:test_wave_batch_iterations

                        println("batch: ",i, "/",test_wave_batch_iterations)

                        #reorder ytrain to (spatial location, batch_size, time)
                        if i!=test_wave_batch_iterations
    #                         Ytest_wave = permutedims(ytest_wave[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))
                            Ytest_wave = permutedims(pred_test_wave_mat[:,batch_size*(i-1)+1:batch_size*i,:],(3,2,1))
                            atest_wave = permutedims(Atest_wave_org[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))
                            atest_wave_undef = permutedims(Atest_wave_undef[:,:,batch_size*(i-1)+1:batch_size*i],(2,3,1))

                            # repeat for tcycles cardiac cycles
                            Ytest_wave = replicate_array(Ytest_wave, tcycles, 3)
                            atest_wave = replicate_array(atest_wave, tcycles, 3)

                        else
    #                         Ytest_wave = permutedims(ytest_wave[:,:,batch_size*(i-1)+1:end],(2,3,1))
                            Ytest_wave = permutedims(pred_test_wave_mat[:,batch_size*(i-1)+1:end,:],(3,2,1))
                            atest_wave = permutedims(Atest_wave_org[:,:,batch_size*(i-1)+1:end],(2,3,1))
                            atest_wave_undef = permutedims(Atest_wave_undef[:,:,batch_size*(i-1)+1:end],(2,3,1))

                            # repeat tcycles cardiac cycles    
                            Ytest_wave = replicate_array(Ytest_wave, tcycles, 3)
                            atest_wave = replicate_array(atest_wave, tcycles, 3)
                        end


                        #define function for interpolating area and dA/dz to the actual spatial location for the ODE
                        interp_func(t) = interpolate_variables(t, Ytest_wave, dt, T)


                        #calculate final loss and push it to the list
                        prob = ODEProblem((u, p_cont, t) -> learn_1DBlood_continuity(u, p_cont, t, interp_func,), atest_wave_undef[:,:,1], tspan, p_cont);
                        optf = Optimization.OptimizationFunction((x,p_cont)->loss_S(x,Ytest_wave,prob,atest_wave),adtype) ;
                        l , pred = loss_S(uinit_cont,Ytest_wave,prob,atest_wave)
                        loss_tot_test_wave = loss_tot_test_wave + l

                        push!(list_S_loss_test_wave, l)
                        println("Test loss - new waveform:",l )

                        # Initialize the resulting matrix
                        pred_S_test_wave_matrix = zero(atest_wave)

                        pred_S_test_wave_matrix = pred

                        pred = nothing

                        push!(pred_S_test_wave, pred_S_test_wave_matrix)
                    
                        if plot_flag
                            # plot solution for comparison
                            plot1 = heatmap(pred_S_test_wave_matrix[:,1,:]', color=:viridis, title = "neural ODE area")
                            xlabel!("time")
                            ylabel!("x")

                            plot2 = heatmap(atest_wave[:,1,:]', title="3d - area", color=:viridis)
                            xlabel!("time")
                            ylabel!("x")
                            display(plot(plot1,plot2,layout = (2, 1)))
                            sleep(1)
                        end
                    end

            # save and print losses
            push!(list_S_loss_epoch, loss_tot/(size(ytrain2,3)))
            push!(list_S_loss_epoch_test_wave, loss_tot_test_wave/(size(ytest_wave,3)))
            push!(list_S_loss_epoch_test_geom, loss_tot_test_geom/(size(ytest_geom,3)))
            println("Epoch ", k, " continuity mean train loss:", loss_tot/(size(ytrain2,3)))
            println("Epoch ", k, " continuity mean test loss - new waveform:", loss_tot_test_wave/(size(ytest_wave,3)))
            println("Epoch ", k, " continuity mean test loss - new geom:", loss_tot_test_geom/(size(ytest_geom,3)))

            # update area values
            # these will be the next input to the NN(Q,S)
            pred_S_matrix = reduce((x, y) -> cat(x, y, dims=2), pred_S_train)
            pred_S_test_wave_matrix = reduce((x, y) -> cat(x, y, dims=2), pred_S_test_wave)
            pred_S_test_geom_matrix = reduce((x, y) -> cat(x, y, dims=2), pred_S_test_geom);                    
			
			pred_S_train = nothing
			pred_S_test_wave = nothing
			pred_S_test_geom = nothing
			
			
            global Atrain = permutedims(pred_S_matrix,(3,1,2))[end-99:end,:,:]
            global Atest_wave = permutedims(pred_S_test_wave_matrix,(3,1,2))[end-99:end,:,:]
            global Atest_geom = permutedims(pred_S_test_geom_matrix, (3,1,2))[end-99:end,:,:]
			
			pred_S_matrix = nothing
			pred_S_test_wave_matrix = nothing
			pred_S_test_geom_matrix = nothing


            if k ≠ n_epochs_cont
            # reset list variable
                pred_S_train = []
                pred_S_test_wave = []
                pred_S_test_geom = []
            end  

        end
    end

            if j ≠ n_epochs
            # reset list variable
                global pred_train = []
                global pred_test_wave = []
                global pred_test_geom = []
            end                        

    end

# plot loss as a function of epochs
#p3 = plot([list_loss_epoch,list_loss_epoch_test_wave,list_loss_epoch_test_geom], yaxis=:log, label = ["train" "new wave" "new geom"])
#ylabel!("loss")
#xlabel!("epochs")
#png("dQdz_dSdt_p_b.png")
#display(p3)

# plot loss as a function of epochs
#p4 = plot([list_S_loss_epoch,list_S_loss_epoch_test_wave,list_S_loss_epoch_test_geom], yaxis=:log, label = ["train" "new wave" "new geom"])
#ylabel!("loss")
#xlabel!("epochs")
#png("dQdz_dSdt_p_b_S_loss.png")
#display(p4)

# write loss to CSV file
df = DataFrame([list_loss_epoch,list_loss_epoch_test_wave, list_loss_epoch_test_geom], ["train", "wave", "geom"])
CSV.write("dQdz_NN_QS_dSdt_loss_output.csv", df, writeheader=true)

filename1D = "/uufs/chpc.utah.edu/common/home/u1447794/LANL/1D_data/case10_1D_waveforms_results.h5"
file = h5open(filename1D,"r")
data1d = read(file["data"])
close(file)

println("Size of 1D data matrix:", size(data1d))
println("Shape: [timesteps, spatial locations, waveforms, variables]")
    
#variables: 1 - flow rate, 2 - pressure , 3 - area, 4 - WSS

data1d_fix = data1d[200:end-1,Not(11:11:end),:,:];

flow_GT = permutedims(ytest_geom,(1,3,2));
flow_GT_wave = permutedims(ytest_wave,(1,3,2));
flow_GT_train = permutedims(ytrain2,(1,3,2));


#reshape prediction matrix
pred_train_mat = zero(flow_GT_train);
pred_test_wave_mat = zero(flow_GT_wave);
pred_test_geom_mat = zero(flow_GT);

for i in range(1,size(pred_train)[1])
    if i==size(pred_train)[1]
        pred_train_mat[:,batch_size*(i-1)+1:end,:] = pred_train[i]
    else
        pred_train_mat[:,batch_size*(i-1)+1:batch_size*i,:] = pred_train[i] 
    end
end


for i in range(1,size(pred_test_wave)[1])
    if i==size(pred_test_wave)[1]
        pred_test_wave_mat[:,batch_size*(i-1)+1:end,:] = pred_test_wave[i]
    else
        pred_test_wave_mat[:,batch_size*(i-1)+1:batch_size*i,:] = pred_test_wave[i] 
    end
end



for i in range(1,size(pred_test_geom)[1])
    if i==size(pred_test_geom)[1]
        pred_test_geom_mat[:,batch_size*(i-1)+1:end,:] = pred_test_geom[i]
    else
        pred_test_geom_mat[:,batch_size*(i-1)+1:batch_size*i,:] = pred_test_geom[i] 
    end
end



flow_err_node = norm(pred_test_geom_mat[:, 1:end-1, :] - flow_GT[:, 1:end-1, :])/norm(flow_GT[:,1:end-1, :])
flow_err_1d = norm(data1d_fix[:,1:4:end,:,1] - ytest_geom[:, 1:end-1, :])/norm(ytest_geom[:,1:end-1, :])
print("Neural ODE flow rate prediction relative error: ", flow_err_node ,"\n")
print("1D FEM flow rate prediction relative error: ", flow_err_1d)

flow_err_node_wave = norm(pred_test_wave_mat[:, 1:end, :] - flow_GT_wave[:, 1:end, :])/norm(flow_GT_wave[:,1:end, :])
print("Neural ODE flow rate prediction relative error - new waveform: ", flow_err_node_wave ,"\n")

flow_err_node_train = norm(pred_train_mat[:, 1:end, :] - flow_GT_train[:, :, :])/norm(flow_GT_train[:,:, :])
print("Neural ODE flow rate prediction relative error - training: ", flow_err_node_train ,"\n")

area_err_node = norm(Atest_geom[:,6:end-5, :] - Atest_geom_org[:,6:end-5, :]) / norm(Atest_geom_org[:,6:end-5, :])
area_err_1d = norm(data1d_fix[:,21:4:end-16,:,3] - Atest_geom_org[:,6:end-5, :]) / norm(Atest_geom_org[:,6:end-5, :])
print("Neural ODE area prediction relative error: ", area_err_node ,"\n")
print("1D FEM area prediction relative error: ", area_err_1d)

area_err_node_wave = norm(Atest_wave[:,6:end-5, :] - Atest_wave_org[:,6:end-5, :]) / norm(Atest_wave_org[:,6:end-5, :])
print("Neural ODE area prediction relative error - new waveform: ", area_err_node_wave ,"\n")

area_err_node_train = norm(Atrain[:,6:end-5, 70:80] - Atrain_org[:,6:end-5, 70:80]) / norm(Atrain_org[:,6:end-5, 70:80])
print("Neural ODE area prediction relative error - training: ", area_err_node_train ,"\n")

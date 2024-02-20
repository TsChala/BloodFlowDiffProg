# ------------------------------------------#
#        Expt 5: Perform a battery of negative inference tests from a single script and plot results, save metadata    
# ------------------------------------------#

Threads.nthreads()
begin
    using LinearAlgebra 
    using Plots
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
end 

# ------------------------------------------#
#        Simulation input parameters       #
# ------------------------------------------#

begin 
    global N = 256                      # number of cells 
    global L = 5π                       # total length of 1d sim
    global dt = 0.01                     # time step
    global T = 1.0                       # total time 
    global nt = T / dt                   # number of timesteps
    global dx = L/N                      # spatial step
    global x = 0.0 : dx : (L-dx)         # discretized spatial dimension 
    global xgrid = collect(x)            # grid 
    global tsteps = 0.0:dt:T             # discretized time dimension 
    global tspan = (0,T)                 # end points of time integration for ODEProbem

    global ω0 = 0.5                      # Coriolis factor 
    global c  = 2.5                      # Soliton velocity 
    global A  = 5.0                      # Amplitude of wave in IC
    global xS = 1.5                      # Soliton shift from origin 
    global γ  = 1                        # Less-resolved-timesteps-for-training factor ; dt_train = γ dt_true 
    global ckpt_save_freq = 10
    global nepochs = 3

    global train_maxiters = 5         # number of iterations of learning 
    global learning_rate  = 0.005         # learning rate , currently using PolyOpt default (0.1)
    # plot colors 
    pc = "black" ;       # black for true gKdV solution 
    pc2 = "darkblue";   # blue for trained learn gKdV 
    pc3 = "darkorange" ; # orange for untrained learn gKdV 
    pc4 = "blue3";       # 
    ls  = :dash;     
    y_min = -7.5 ;
    y_max = 15.5 ; 
end 


path_to_working_directory="/Users/arvindm/work/RESOURCES/juliaDiffProgExample/"

include("./src/numerical_derivatives.jl");
include("./src/train_utils.jl");

########################################################################
######### FUNCTION DEFINITIONS
########################################################################

function generateTrainData(N,dx)
    # derivative operators, 2nd order central-finite-difference
    ∂x1 = f1_secondOrder_central(N,dx)
    ∂x3 = f3_secondOrder_central(N,dx)

    function gKdV_ic_sinusoid(x,A,c,ω0)
        return -A*sin.(x./c .+ pi)
    end
    u0 = gKdV_ic_sinusoid(x,A,c,ω0) ;

    # true gKdV pde
    function true_gKdV(u,p,t)
        #ω0 = 0.075*randn(N);
        return ω0 .* (∂x1 * u) - 3.0/2.0 .* u .* (∂x1 *u) - 1.0 ./ 6.0 .* (∂x3 * u)
    end

    # learnable parameter p , not used in training data generation, initialized with zeros.
    # save only initial condition and last timestep
    p = zeros(N) ;
    true_gKdV_prob(u0,tspan) = ODEProblem(true_gKdV, u0, tspan,p)
    sol_true_gKdV = Array(solve(true_gKdV_prob(u0,tspan),Tsit5(),alg_hints=[:stiff], dt=dt, saveat=dt, reltol = Float32(1e-6)));
    println(sum(sol_true_gKdV.^2))
    return sol_true_gKdV
end

function train_gKdV(working_dir, ytrain, optimizer_choice1, ode_solver, train_maxiters, learning_rate, path_checkpoint=nothing, optimizer_choice2=nothing)
    # process user choices
    if !isdir(working_dir)
        mkdir(working_dir)
    end
    output_dir = working_dir*"/output/"
    cd(working_dir) #switch to working directory
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    println("optimizer 1 is $optimizer_choice1")
    if !isnothing(optimizer_choice2)
        println("optimizer 2 is $optimizer_choice2 optimizer")
    end
    
    println("ODE Time integrator selected:", ode_solver)
    
    # train on less resolved time-steps than the training data, i.e. : dt_train = γ dt_true
    ytrain2 = ytrain[:,1:γ:end] ;
    dt2 = γ * dt ; 
    nt2 = T/dt2 ;
    tsteps2 = 0.0:dt2:T; 
    println("Number of time-steps to train on: $nt2")
   
    # numerical schemes for learned PDE
    ∂x1 = f1_secondOrder_central(N,dx)
    ∂x3 = f3_secondOrder_central(N,dx)

    #initial condition
    function gKdV_ic_sinusoid(x,A,c,ω0)
        return -A*sin.(x./c .+ pi)
    end
    u0 = gKdV_ic_sinusoid(x,A,c,ω0) ;

    # NN embedded in PDE for Differential programming
    ann = Chain(
                Dense(N,10,tanh),
                Dense(10,10,tanh),
                Dense(10,10,tanh),
                Dense(10,N));

    # flatten parameters in NN for learning.                
    p, re = Flux.destructure(ann);
    ps = deepcopy(p)
    p_size = size(p,1);
    println("Number of parameters in neural network: $p_size"); 

    function learn_gKdV(u,p,t)
        Φ = re(p) # restructure flattened parameter vector into NN architecture.
        return  Φ(u) - 3.0/2.0 .* u .* (∂x1 *u) - 1.0 ./ 6.0 .* (∂x3 * u) # ϕ(u) is the NN taking u and predicting the missing first term
    end 

    #define learning problem.
    learn_gKdV_prob(u0,tspan) = ODEProblem(learn_gKdV, u0, tspan, p)
    
    # calculate and save the untrained solution 
    begin 
        learn_gKdV_prob_untrained(u0,tspan) = ODEProblem(learn_gKdV, u0, tspan,p)     # p changes after training   
        if ode_solver == "Tsit5"
            sol_learn_gKdV_untrained = Array(solve(learn_gKdV_prob_untrained(u0,tspan),Tsit5(),alg_hints=[:stiff], dt=dt2, saveat=dt2, reltol=1e-20)); # save on intervals for which we will train on, dt2 
        elseif ode_solver == "RK4"
            sol_learn_gKdV_untrained = Array(solve(learn_gKdV_prob_untrained(u0,tspan),RK4(),alg_hints=[:stiff], dt=dt2, saveat=dt2, reltol=1e-20)); # save on intervals for which we will train on, dt2 
        elseif ode_solver == "Rosenbrock23"
            sol_learn_gKdV_untrained = Array(solve(learn_gKdV_prob_untrained(u0,tspan),Rosenbrock23(),alg_hints=[:stiff], dt=dt2, saveat=dt2, reltol=1e-20)); # save on intervals for which we will train on, dt2 
        end
        save_object(output_dir*"/sol_learn_gKdV_untrained.jld2",sol_learn_gKdV_untrained)
        println("saved untrained solution")
    end 
    
    
    prob = ODEProblem(learn_gKdV, u0, tspan, p);

    function predict(θ)
        if ode_solver == "Tsit5"
            Array(solve(prob,Tsit5(),p=θ,dt=dt2,saveat=dt2))
        elseif ode_solver == "RK4"
            Array(solve(prob,RK4(),p=θ,dt=dt2,saveat=dt2))
        elseif ode_solver == "Rosenbrock23"
            Array(solve(prob,Rosenbrock23(),p=θ,dt=dt2,saveat=dt2))
        end
    end 

    function loss(θ)
        pred = predict(θ)
        l = predict(θ) - ytrain2
        return sum(abs2,l), pred
    end 

    l , pred = loss(ps)

    # training optimizer definition
    adtype = Optimization.AutoZygote() ;
    optf = Optimization.OptimizationFunction((x,p)->loss(x),adtype) ;

    # restart training from previous checkpoint if it exists, else start 
    # fresh training.
    if !isnothing(path_checkpoint)
        checkpoint_exists = isfile(path_checkpoint)
        if checkpoint_exists
            println("Checkpoint exists!")
            ckpt_restart = jldopen(path_checkpoint,"r")
            res1 = ckpt_restart["ckpt"]
            uinit = copy(res1.u)
        else
            println("ckpt path given, but no file found!")
        end
    else
        uinit = deepcopy(ps)
        println("Fresh training initialized")
    end

    println("Start training")
    println("Using $optimizer_choice1 optimizer")
    uinit = train_loop(uinit,adtype,optf,train_maxiters,learning_rate,optimizer_choice1)
    if !isnothing(optimizer_choice2)
        println("Switching to $optimizer_choice2 optimizer")
        uinit = train_loop(uinit,adtype,optf,train_maxiters,learning_rate,optimizer_choice2)
    else
        println("Completed training")
    end
end

########################################################################
######### EXECUTION
########################################################################

# generate training data
sol_true_gKdV = generateTrainData(N,dx);

# run experiments
println("Running Expt 1")
working_dir = "/Users/arvindm/work/RESOURCES/juliaDiffProgExample/Expt1"
train_gKdV(working_dir, sol_true_gKdV, "ADAM", "Tsit5", train_maxiters, 5e-03, nothing, "ADAM");

println("Running Expt 2")
working_dir = "/Users/arvindm/work/RESOURCES/juliaDiffProgExample/Expt2"
train_gKdV(working_dir, sol_true_gKdV, "ADADelta", "RK4", train_maxiters, 5e-03, nothing, "BFGS");

# println("Running Expt 3")
# working_dir = "/Users/arvindm/work/RESOURCES/juliaDiffProgExample/Expt3"
# train_gKdV(working_dir, sol_true_gKdV, "Nesterov", "Rosenbrock23", train_maxiters, 5e-03, nothing, "BFGS");

# println("Running Expt 4")
# working_dir = "/Users/arvindm/work/RESOURCES/juliaDiffProgExample/Expt4"
# train_gKdV(working_dir, sol_true_gKdV, "Nesterov", "Tsit5", train_maxiters, 5e-03, nothing, "BFGS");


using Plots
using OrdinaryDiffEq
using BlockArrays
using LinearAlgebra
using SparseArrays

#Define variables
ndata = 500 #number of ensembles for training
n = 64
L = 2.0
dt = 0.01
T = 0.5
dx = L/n #create grid
xgrid = collect(0:dx:L-dx)
tgrid = collect(0:dt:T)
∂x1 = zeros(n,n)
∂x2 = zeros(n,n)

function get_fd_operators(n, dx)
    # upwind scheme
    ∂x1 = (diagm(0 => ones(n), -1 => -ones(n-1)))./dx
    ∂x2 = (diagm(0 => -ones(n), 1 => ones(n-1)))./dx
    # boundary conditions
    ∂x1[1,end] = -1/dx
    ∂x2[end,1] = 1/dx
    return ∂x1, ∂x2
end

function eqn_inviscidBurgers(u,du,t)
    - 0.5 .* (u+ abs.(u)) .* (∂x1*u) - 0.5 .* (u - abs.(u)) .* (∂x2*u)
end


#get tridiagonal system for each operator
∂x1, ∂x2 = get_fd_operators(n, dx);

#compute training dataset
ntimesteps = Int64(T/dt + 1)
datasets = Array{Float32,3}(undef,ndata,n,ntimesteps);
for i = 1:ndata
    u0 = sin.(π*xgrid) + 0.05*randn(n); # different noisy init condition each instance.
    prob = ODEProblem(eqn_inviscidBurgers, u0, (0.0, T), saveat = dt);
    datasets[i,:,:] = Array(solve(prob, AutoTsit5(Rosenbrock23())));
end

using JLD2: @save
@save "inviscidburgersdata.jld2" datasets
println("Saved datasets to inviscidburgersdata.jld2")
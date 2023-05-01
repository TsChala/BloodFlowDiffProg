using SeqData, Distributions




A = rand(10, 400)
#(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Real, N_valid::Real; verbose=false, supervised=false)

# test unsupervised
train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true)

@assert train[1][:,:,1] == A[:,1:64]
train[end]
@assert train[2][:,:,1] == A[:,1+24:24+64]


# test supervised

train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true, supervised=true)

@assert train[1][1][:,:,1] == A[:,1:64]
@assert train[1][2][:,:,1] == A[:,2:65]
@assert train[2][1][:,:,1] == A[:,1+24:24+64]


B = rand(10,10, 400)

train, valid, test = SequentialData(B, 24, 64, 0.5, 0.25, verbose=true)

@assert train[1][:,:,:,1] == B[:,:,1:64]
train[end]
@assert train[2][:,:,:,1] == B[:,:,1+24:24+64]


# test supervised

train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true, supervised=true)

@assert train[1][1][:,:,1] == A[:,1:64]
@assert train[1][2][:,:,1] == A[:,2:65]
@assert train[2][1][:,:,1] == A[:,1+24:24+64]





train, valid, test = SequentialData(A, 0, 64, 0.5, 0.25, verbose=true)

@assert train[1] == A[:,1:64]
train[end]
@assert train[2] == A[:,2:65]



noise = Uniform(0.0, 0.01)
train, valid, test = SequentialData(A, 0, 1, 0.5, 0.25, verbose=true, supervised=true, stabilization_noise=noise)

for i in 1:length(train)
    @assert A[:,i] < train[i][1] < (A[:,i] .+ 0.01)
end

train, valid, test = SequentialData(A, 0, 1, 0.5, 0.25, verbose=true, supervised=false, stabilization_noise=noise)

for i in 1:length(train)
    @assert A[:,i] < train[i] < (A[:,i] .+ 0.01)
end


train, valid, test = SequentialData(A, 0, 1, 100, 200, verbose=true, supervised=true)

@assert length(train) == 100
@assert length(valid) == 200
@assert length(test) == 100

@assert train[end][2] == valid[1][1]

train, valid, test = SequentialData(A, 0, 1, 100, 200, verbose=true, supervised=true, overlap=false)

@assert train[end][2] != valid[1][1]
@assert train[end][2] == A[:,100]
@assert valid[1][1] == A[:,101]




C = rand(10,10,10,400)

train, valid, test = SequentialData(C, 0 , 1, 0.5, 0.25, verbose=true, supervised=true)

@assert train[1][1] == C[:,:,:,1]
@assert train[1][2] == C[:,:,:,2]
@assert train[2][1] == C[:,:,:,2]

true

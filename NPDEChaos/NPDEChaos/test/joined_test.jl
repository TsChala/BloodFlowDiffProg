

using SeqData


A = rand(10, 400)
#(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Real, N_valid::Real; verbose=false, supervised=false)

# test unsupervised
trainA, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true)

@assert trainA[1][:,:,1] == A[:,1:64]
trainA[end]
@assert trainA[2][:,:,1] == A[:,1+24:24+64]




B = rand(10, 400)
#(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Real, N_valid::Real; verbose=false, supervised=false)

# test unsupervised
trainB, valid, test = SequentialData(B, 24, 64, 0.5, 0.25, verbose=true)

C = JoinedSeqData([trainA,trainB], length(trainA)+length(trainB),[length(trainA),length(trainA)+length(trainB)])

@assert C[length(trainA)] == trainA[end]
@assert C[length(trainA)+1] == trainB[1]
@assert C[end] == trainB[end]

C = cat(trainA,trainB)

@assert C[length(trainA)] == trainA[end]
@assert C[length(trainA)+1] == trainB[1]
@assert C[end] == trainB[end]

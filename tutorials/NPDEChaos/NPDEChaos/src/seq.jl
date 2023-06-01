using Flux, Distributions

abstract type AbstractSeqData{T,N} end # make it a subtype of abstractarray?

"""
    SequentialData

Struct with initialization for preparing sequence data for a learning problem. It saves a lot of memory because it uses Julia's iterator interface and does not save the same data multiple times as a naive implementation would do. The default initialization splits the data into a train, valid and test set.

`SequentialData` can be used for `for`-loops with `for i in data` and also indexed with data[i]. Each element is a 2-element array containing the input and output for the learning problem.

It is possible to output batches of data.

# Fields of the struct

* `data::AbstractArray`: Raw Data
* `N_batch::Int`: Batch Size
* `N_length::Int`: Length of each sample
* `N::Int`: Total number of individual train/valid/test data pairs available
* `N_t::Int`: Number of time steps per trajectory

# Initialization

    SequentialData(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Int, N_valid::Int; supervised::Bool=false)

## Input

* `input_data::AbstractArray`: Raw data in either a ``N_dim \\times N_t``, ``N_x \\times N_y \\times N_t`` or ``N_x \\times N_y \\times N_z \\times N_t``
* `N_batch::Int`: Batch size, If `N_batch==0` the output will be 2D without any batching
* `N_length::Int`: Length of each sample
* `N_train::Real`: Relative Number of input samples for training
* `N_valid::Real`: Relative Number of input samples for validation. All remaining time steps will form the test set
* `supervised::Bool`: If true every indexing will return a pair of arrays where the second array is shifted by 1 from the first array.


## Output

Returns _three_ instances of `SequentialData` in order `(train_set, valid_set, test_set)`.

"""
struct SequentialData{T,N} <: AbstractSeqData{T,N}
    data::AbstractArray{T,N}
    N_batch::Int
    N_length::Int
    N::Int
    N_t::Int
    N_dims::Int
    noise_dist::Union{Sampleable, Nothing}
    _batches::Bool
    _supervised::Bool
    _stabilization_noise::Bool
    _GPU::Bool
end

# initializes all data
function SequentialData(input_data::AbstractArray, N_batch::Int, N_length::Int, N_train::Real, N_valid::Real; verbose=false, supervised=false, stabilization_noise::Union{Sampleable, Nothing}=nothing, overlap::Bool=true, GPU::Bool=true)


    _batches = ((N_batch == 0) | (N_batch == 1)) ? false : true
    _stabilization_noise = (stabilization_noise==nothing) ? false : true

    N_batch = N_batch == 0 ? 1 : N_batch
    #offset = supervised ? 0 : 1
    # convert data to   N_dim x N_i x N_t
    N_dims = ndims(input_data)
    if N_dims==2
        N_x, N_t = size(input_data)
        N_y = 1
        N_z = 1
    elseif N_dims==3
        N_x, N_y, N_t = size(input_data)
        N_z = 1
    elseif N_dims==4
        N_x, N_y, N_z, N_t = size(input_data)
    else
        error("input_data has to be 2- or 3 or 4-dimensional in (N_x x N_y x N_z x N_t) format")
    end


    input_data = reshape(input_data, (N_x, N_y, N_z, N_t))


    if (0. <= N_train <= 1.)
        @assert 0. <= N_valid <= 1.
        @assert N_train + N_valid <= 1.
        @assert 0 <= N_batch

        N_train = Int(ceil(N_t * N_train))
        N_valid = Int(ceil(N_t * N_valid))
    else
        @assert (N_train + N_valid) < N_t
        N_train = Int(N_train)
        N_valid = Int(N_valid)
    end

    if overlap
        N_Ntrain = Int(ceil((N_train)/N_batch)) #- N_batch ??
        N_Nvalid = Int(ceil((N_valid)/N_batch))
        N_Ntest = Int(ceil(((N_t - N_train - N_valid))/N_batch))
        overlap_N = N_length
    else
        N_Ntrain = Int(ceil((N_train - N_length)/N_batch)) #- N_batch ??
        N_Nvalid = Int(ceil((N_valid - N_length)/N_batch))
        N_Ntest = Int(ceil(((N_t - N_train - N_valid) - N_length)/N_batch))
        overlap_N = 0
    end

    if (N_valid - N_length) < 0
        @warn "Valid set may be empty"
    end

    if (N_t - N_train - N_valid - N_length) < 0
        @warn "Test set may be empty"
    end

    if verbose
        println("Train set length = ",N_Ntrain)
        println("Valid set length = ",N_Nvalid)
        println("Test set length = ",N_Ntest)
    end

    SequentialData(input_data[:,:,:,1:N_train+overlap_N], N_batch, N_length, N_Ntrain, N_train+overlap_N, N_dims, stabilization_noise, _batches, supervised, _stabilization_noise, GPU), SequentialData(input_data[:,:,:,N_train+1:N_train+N_valid+overlap_N], N_batch, N_length, N_Nvalid, N_valid+overlap_N, N_dims, stabilization_noise, _batches, supervised, _stabilization_noise, GPU), SequentialData(input_data[:,:,:,N_train+N_valid+1:end], N_batch, N_length, N_Ntest, N_t - N_train - N_valid, N_dims, stabilization_noise, _batches, supervised, _stabilization_noise, GPU)
end


function Base.iterate(iter::AbstractSeqData, state=1)
    if state>iter.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::AbstractSeqData) = iter.N
Base.eltype(iter::AbstractSeqData) = Array{typeof(iter.data),1}

function Base.getindex(iter::SequentialData{T}, i::Int) where T<:Number
    @assert 1 <= i <= iter.N

    N_batch = _batch_size(iter, i)

    dropds = []
    if iter._batches==false
        push!(dropds, 5)
    end

    # do something with N here
    if iter.N_dims==2
        push!(dropds, 2)
        push!(dropds, 3)
    elseif iter.N_dims==3
        push!(dropds, 3)
    end

    if iter.N_length == 1
        push!(dropds, 4)
    end

    dropds = Tuple(dropds)

    if !iter._supervised
        data = zeros(T, size(iter.data,1), size(iter.data, 2), size(iter.data,3), iter.N_length, N_batch)

        if iter._GPU
            data = togpu(data)
        end

        for (iib, ib) in enumerate(_batch_iterate_range(iter, i))
            data[:,:,:,:,iib] = iter.data[:,:,:,ib:ib-1+iter.N_length]
        end

        output_data = dropdims(data, dims=dropds)

        if iter._stabilization_noise
            output_data += rand(iter.noise_dist, size(output_data)...)
        end
    else
        data1 = zeros(T, size(iter.data,1), size(iter.data,2), size(iter.data,3), iter.N_length, N_batch)
        data2 = zeros(T, size(iter.data,1), size(iter.data,2), size(iter.data,3), iter.N_length, N_batch)

        if iter._GPU
            data1 = togpu(data1)
            data2 = togpu(data2)
        end

        for (iib, ib) in enumerate(_batch_iterate_range(iter, i))
            data1[:,:,:,:,iib] = iter.data[:,:,:,ib:ib-1+iter.N_length]
            data2[:,:,:,:,iib] = iter.data[:,:,:,ib+1:ib+iter.N_length]
        end

        if iter._stabilization_noise
            data1 += rand(iter.noise_dist, size(data1)...)
        end

        output_data = (dropdims(data1, dims=dropds), dropdims(data2, dims=dropds))
    end

    return output_data
end


"""
    _batch_size(data::AbstractSeqData, i::Int)

Outputs the batch size of the current batch. This is either the regular batch size that was chosen or for the last batch of each trajectory all possibly remaining pairs.

Input `i` is the index on the _current trajectory_.
"""
function _batch_size(data::AbstractSeqData, i::Int)
    @assert 1 <= i <= data.N
    if i != data.N
        N_batch = data.N_batch
    else
        N_batch = data.N_t - (i-1)*data.N_batch - data.N_length
        @assert 1 <= N_batch <= data.N_batch
    end
    return N_batch
end

"""
    _batch_iterate_range(data::AbstractSeqData, i::Int)

Outputs the range over all parts of a batch. This is either the regular batch size that was chosen or for the last batch of each trajectory all possibly remaining pairs.

Input `i` is the index on the _current trajectory_.
"""
function _batch_iterate_range(data::AbstractSeqData, i::Int)
    if i != data.N# all but the last batch per trajectory
        iterate_range = (data.N_batch*(i-1))+1:i*data.N_batch
    else  # last batch per trajectory might not have N_batch individuals
        iterate_range = (data.N_batch*(i-1))+1:(data.N_t - data.N_length)
    end
end

Base.firstindex(iter::AbstractSeqData) = 1
Base.lastindex(iter::AbstractSeqData) = iter.N

module SeqData

global const cuda_used = Ref(false)

using CUDA

function __init__() # automatically called at runtime to set cuda_used
    cuda_used[] = CUDA.functional()
end
togpu(x::AbstractArray) = cuda_used[] ? CuArray(x) : x

include("seq.jl")
include("masked_seq.jl")
include("joinedseqs.jl")

export SequentialData, AbstractSeqData, MaskedSequentialData, JoinedSeqData, cat

end # module

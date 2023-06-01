import Base.cat


struct JoinedSeqData{T,S} <: AbstractSeqData{T,S}
    data::Array{SequentialData{T,S},1}
    N
    cumN
end

Base.length(j::JoinedSeqData) = j.N
Base.eltype(j::JoinedSeqData{T,S}) where {T,S} = T

function Base.getindex(iter::JoinedSeqData, i::Int)
    @assert 1 <= i <= iter.N
    ind1 = 0
    ind2 = 0
    cumN = [0;iter.cumN]
    for j in 1:(length(cumN)-1)
        if i <= cumN[j+1]
            ind1 = j
            ind2 = i - cumN[j]
            break
        end
    end

    return iter.data[ind1][ind2]
end

function Base.cat(A::AbstractSeqData...)

    lengths = [length(A[i]) for i in eachindex(A)]
    N = sum(lengths)
    cumN = cumsum(lengths)

    return JoinedSeqData([A[i] for i in eachindex(A)],N,cumN)
end

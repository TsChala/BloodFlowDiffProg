using Flux, Distributions

"""
    MaskedSequentialData

Struct with initialization for preparing sequence data for a learning problem. Similar to ['SequentialData'], but with the additional possibily of setting masks so that the input/output of the learning problem is only part of the complete data.

# Initialiazer

`MaskedSequentialData{T}(data::SequentialData, mask_in, mask_out)``

`mask_in` has to be an array that can index the underlying array of `data` via `getindex(data.data, mask_in...)`. Eg. `[1:10, :]` is possible for 2D data. If an unsupervised problem is desired `mask_out` can be left `nothing`, only `mask_in` will be used.

"""
struct MaskedSequentialData{T,S} <: AbstractSeqData{T,S}
    data::SequentialData{T,S}
    mask_in
    fill_in
    mask_out
    fill_out

    MaskedSequentialData(data, mask_in, fill_in=nothing, mask_out=nothing, fill_out=nothing) where T<:Number = new{T,S}(data, mask_in, fill_in, mask_out, fill_out)
end


function Base.iterate(iter::MaskedSequentialData, state=1)
    if state>iter.data.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::MaskedSequentialData) = iter.data.N
Base.eltype(iter::MaskedSequentialData) = Array{typeof(iter.data.data),1}

function Base.getindex(iter::MaskedSequentialData{T}, i::Int) where T<:Number
    dat_i = getindex(iter.data, i)
    mask_in = iter.mask_in
    mask_out = iter.mask_out

    if !iter.data._supervised
        if iter.fill_in == nothing
            out = getindex(dat_i, mask_in...)
        else
            out = similar(dat_i)
            fill!(out, iter.fill_in)
            out_masked = getindex(dat_i, mask_in...)

            setindex!(out, out_masked, mask_in...)
        end
        return out
    else
        if iter.fill_in == nothing
            out_in = getindex(dat_i[1], mask_in...)
        else
            out_in = similar(dat_i[1])
            fill!(out_in, iter.fill_in)
            out_masked = getindex(dat_i[1], mask_in...)

            setindex!(out_in, out_masked, mask_in...)
        end

        if iter.fill_out == nothing
            out_out = getindex(dat_i[2], mask_out...)
        else
            out_out = similar(dat_i[2])
            fill!(out_out, iter.fill_out)
            out_masked = getindex(dat_i[2], mask_out...)

            setindex!(out_out, out_masked, mask_out)

        end

        return (out_in, out_out)
    end
end

Base.firstindex(iter::MaskedSequentialData) = 1
Base.lastindex(iter::MaskedSequentialData) = iter.data.N

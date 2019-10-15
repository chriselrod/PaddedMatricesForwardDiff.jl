module PaddedMatricesForwardDiff

using PaddedMatrices, ForwardDiff

abstract type AbstractDualWrappedArray{S,T,N,X,L,D,G} <: AbstractFixedSizeArray{S,ForwardDiff.Dual{G,T,D},N,X,L} end

struct DualWrappedArray{S,T,N,X,L,D,G,A<:PaddedMatrices.AbstractFixedSizeArray{S,T,N,X,L}} <: AbstractDualWrappedArray{S,T,N,X,L,D,G}
    data::A
end
struct PtrDualWrappedArray{S,T,N,X,L,D,G} <: AbstractDualWrappedArray{S,T,N,X,L,D,G}
    data::PtrArray{S,T,N,X,L,false}
end
struct ConstDualWrappedArray{S,T,N,X,L,D,G,A<:PaddedMatrices.AbstractFixedSizeArray{S,T,N,X,L}} <: AbstractDualWrappedArray{S,T,N,X,L,D,G}
    data::ConstantFixedSizeArray{S,T,N,X,L}
end

function DualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    DualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(A)
end
function DualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    DualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(A)
end
function PtrDualWrappedArray(A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(pointer(A))
end
function PtrDualWrappedArray(A::AbstractMutableFixedSizeArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    PtrDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(pointer(A))
end
function ConstantnDualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    ConstantDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(A)
end
function ConstantDualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    ConstantDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(A)
end


@inline function Base.getindex(A::AbstractDualWrappedArray{S,T,N,X,L,D,G}, I...) where {S,T,N,X,L,D,G}
    ind = PaddedMatrices.sub2ind(A, I...) + 1
    ForwardDiff.Dual{G,T,D}(
        a.data[ind], ForwardDiff.Partials(Base.setindex(ntuple(_ -> zero(T), Val(D)), one(T), ind))
    )
end




end # module

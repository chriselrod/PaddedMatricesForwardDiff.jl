abstract type AbstractDualWrappedArray{S,T,N,X,L,D,G} <: AbstractFixedSizeArray{S,ForwardDiff.Dual{G,T,D},N,X,L} end

struct DualWrappedArray{S,T,N,X,L,D,G,A<:AbstractFixedSizeArray{S,T,N,X,L}} <: AbstractDualWrappedArray{S,T,N,X,L,D,G}
    data::A
end
struct PtrDualWrappedArray{S,T,N,X,L,D,G} <: AbstractDualWrappedArray{S,T,N,X,L,D,G}
    data::PtrArray{S,T,N,X,L,false}
end
struct ConstDualWrappedArray{S,T,N,X,L,D,G} <: AbstractDualWrappedArray{S,T,N,X,L,D,G}
    data::ConstantFixedSizeArray{S,T,N,X,L}
end

function DualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    DualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(A)
end
function DualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    DualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(A)
end
function PtrDualWrappedArray(A::AbstractMutableFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(PtrArray(A))
end
function PtrDualWrappedArray(A::AbstractMutableFixedSizeArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    PtrDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(PtrArray(A))
end
function PtrDualWrappedArray(A::PtrArray{S,T,N,X,L}) where {S,T,N,X,L}
    PtrDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(A)
end
function PtrDualWrappedArray(A::PtrArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    PtrDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(A)
end
function ConstantDualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    ConstDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),nothing}(A)
end
function ConstantDualWrappedArray(A::AbstractFixedSizeArray{S,T,N,X,L},::Val{G}) where {S,T,N,X,L,G}
    ConstDualWrappedArray{S,T,N,X,L,PaddedMatrices.type_length(A),G}(A)
end


@inline function Base.getindex(A::AbstractDualWrappedArray{S,T,N,X,L,D,G}, I...) where {S,T,N,X,L,D,G}
    ind = PaddedMatrices.sub2ind(A, I...) + 1
    @boundscheck ind > L && throw(BoundsError())
    @inbounds ForwardDiff.Dual{G,T,D}(
        A.data[ind], ForwardDiff.Partials(ntuple(d -> d == ind ? one(T) : zero(T), Val(D)))
        # A.data[ind], ForwardDiff.Partials(Base.setindex(ntuple(_ -> zero(T), Val(D)), one(T), ind))
    )
end
# @generated function Base.getindex(A::AbstractDualWrappedArray{S,T,N,X,L,D,G}, I...) where {S,T,N,X,L,D,G}
    # quote
        # $(Expr(:meta,:inline))
        # ind = PaddedMatrices.sub2ind(A, I...) + 1
        # @boundscheck ind > $L && throw(BoundsError())
        # @inbounds ForwardDiff.Dual{G,T,D}(
        # A.data[ind], ForwardDiff.Partials(Base.setindex($(Expr(:tuple,[zero(T) for d in 1:D]...)), one(T), ind))
        # )
    # end
# end

function value_gradient(f, A::AbstractMutableFixedSizeArray{S,T}) where {S,T}
    GC.@preserve A begin
        dA = PtrDualWrappedArray(A)
        d = f(dA)
        g = ConstantFixedSizeVector{PaddedMatrices.type_length(A),T,PaddedMatrices.type_length(A)}(d.partials.values)
    end
    d.value, g
end
function value_gradient(f, A::AbstractConstantFixedSizeArray{S,T}) where {S,T}
    dA = ConstantDualWrappedArray(A)
    d = f(dA)
    d.value, ConstantFixedSizeVector{PaddedMatrices.type_length(A),T,PaddedMatrices.type_length(A)}(d.partials.values)
end
ForwardDiff.gradient(f, A::AbstractFixedSizeArray) = last(value_gradient(f, A))

function value_gradient(sp::StackPointer, f, A::AbstractMutableFixedSizeArray{S,T}) where {S,T}
    GC.@preserve A begin
        dA = PtrDualWrappedArray(A)
        sp, d = f(sp, dA)
        g = ConstantFixedSizeVector{PaddedMatrices.type_length(A),T,PaddedMatrices.type_length(A)}(d.partials.values)
    end
    sp, (d.value, g)
end
function value_gradient(sp::StackPointer, f, A::AbstractConstantFixedSizeArray{S,T}) where {S,T}
    dA = ConstantDualWrappedArray(A)
    sp, d = f(sp, dA)
    sp, (d.value, ConstantFixedSizeVector{PaddedMatrices.type_length(A),T,PaddedMatrices.type_length(A)}(d.partials.values))
end
ForwardDiff.gradient(sp::StackPointer, f, A::AbstractFixedSizeArray) = (sp, vg = value_gradient(f, A); (sp, last(vg)))



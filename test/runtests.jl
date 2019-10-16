using PaddedMatricesForwardDiff
using Test

@testset "PaddedMatricesForwardDiff.jl" begin
    # Write your own tests here.
end


using PaddedMatricesForwardDiff, PaddedMatrices, ForwardDiff, StaticArrays
using BenchmarkTools

x = @Mutable randn(16);
sx = SVector{16}(x);

function foo(x::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(x)
        s += x[i] * x[i]
    end
    s
end

gx = ForwardDiff.gradient(foo, x);
gsx = ForwardDiff.gradient(foo, sx);
@test all(
 i -> gx[i] ≈ gsx[i], eachindex(gx)
)

@benchmark ForwardDiff.gradient(foo, $x)
@benchmark ForwardDiff.gradient(foo, $sx)


function bar(x::AbstractVector{T}) where {T}
    s = zero(T)
    for i ∈ 3:length(x)-2
        s += exp(x[i-1]) * sin(x[i+1]) * (x[i] + 3) / (x[i-2] * x[+2])
    end
    s
end

typeof(sx)
gx = ForwardDiff.gradient(bar, x);
gsx = ForwardDiff.gradient(bar, sx);
@test all(
 i -> gx[i] ≈ gsx[i], eachindex(gx)
)

@benchmark ForwardDiff.gradient(bar, $x)
@benchmark ForwardDiff.gradient(bar, $sx)


const cS = (@Constant randn(16, 24)) |> x -> x * x';
const sS = SMatrix{16,16}(cS);

baz(x::PaddedMatrices.AbstractPaddedArray) = -0.5 * (x' * (cS * x))
baz(x::AbstractArray) = -0.5 * (x' * (sS * x))

baz(x)
baz(sx)

gx = ForwardDiff.gradient(baz, x);
gsx = ForwardDiff.gradient(baz, sx);
@test all(
 i -> gx[i] ≈ gsx[i], eachindex(gx)
)

@benchmark ForwardDiff.gradient(baz, $x) ## Allocates!!! Need to define appropriate methods?
@benchmark ForwardDiff.gradient(baz, $sx)

@benchmark baz($x)
@benchmark baz($sx)
# Static array is much faster!?!
# Need to fix.



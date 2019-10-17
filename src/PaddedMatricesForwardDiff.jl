module PaddedMatricesForwardDiff

using PaddedMatrices, ForwardDiff, StackPointers
using PaddedMatrices: AbstractFixedSizeArray, AbstractMutableFixedSizeArray, AbstractConstantFixedSizeArray

export value_gradient


include("dual_wrapped_array.jl")
include("dual_struct_of_arrays.jl")
include("dual_array_of_structs.jl")

end # module

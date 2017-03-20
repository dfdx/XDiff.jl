
"""
Find all types that may be considered ansestors of this type.
For number types it is the same as Julia type hierarchy.
For array types it includes unparametrized versions of types, i.e.:

    type_ansestors(Vector{Float64})
    # ==> [Array{Float64,1}, Array{T,1}, DenseArray{T,1},
           AbstractArray{T,1}, AbstractArray{T,N}, Any]
"""
function type_ansestors{T<:Number}(t::Type{T})
    types = Type[]
    while t != Any
        push!(types, t)
        t = supertype(t)
    end
    push!(types, Any)
    return types
end

type_ansestors{T}(t::Type{Vector{T}}) =
    [t, Vector, DenseVector, AbstractVector, AbstractArray, Any]
type_ansestors{T}(t::Type{Matrix{T}}) =
    [t, Matrix, DenseMatrix, AbstractMatrix, AbstractArray, Any]



function bcast_to_call(pex::Expr)
    @assert pex.head == :(.)
    return Expr(:call, pex.args[1], pex.args[2].args...)
end

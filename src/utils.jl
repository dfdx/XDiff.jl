
"""
Find all types that may be considered ansestors of this type.
For number types it is the same as Julia type hierarchy.
For array types it includes unparametrized versions of types, i.e.:

    type_ansestors(Vector{Float64})
    # ==> [Array{Float64,1}, Array{T,1}, DenseArray{T,1},
           AbstractArray{T,1}, AbstractArray{T,N}, Any]
"""
function type_ansestors(t::Type{T}) where T<:Number
    types = Type[]
    while t != Any
        push!(types, t)
        t = supertype(t)
    end
    push!(types, Any)
    return types
end

type_ansestors(t::Type{Vector{T}}) where {T} =
    [t, Vector, DenseVector, AbstractVector, AbstractArray, Any]
type_ansestors(t::Type{Matrix{T}}) where {T} =
    [t, Matrix, DenseMatrix, AbstractMatrix, AbstractArray, Any]



function bcast_to_call(pex::Expr)
    @assert pex.head == :(.)
    return Expr(:call, pex.args[1], pex.args[2].args...)
end


deriv_name(z::Symbol, x::Symbol) = Symbol("d$(z)!d$(x)")
split_deriv_name(vname) = Symbol.(split(String(vname), "!"))


function replace_node(g::AbstractExGraph, vname::Symbol, nds::Vector{ExNode})
    i = indexof(g, vname)
    delete!(g, vname)
    insert!(g, nds)
end


function find_related(g::AbstractExGraph, dydx_v::Symbol)
    subderivs = Symbol[]
    i = 1
    name = Symbol("$(dydx_v)__$(i)")
    while haskey(g, name)
        push!(subderivs, name)
        i += 1
        name = Symbol("$(dydx_v)__$(i)")
    end
    return subderivs
end


# (symbolic) derivative size propagation

const DERIV_NAME_PATTERN = r"(d.+)!(d.+)"

function propagate_deriv_size!(g::AbstractExGraph, dd_name::Symbol)
    sizes = @get_or_create(g.ctx, :sizes, Dict())
    rg = match(DERIV_NAME_PATTERN, String(dd_name))
    @assert length(rg.captures) == 2
    str_dnames = rg.captures
    zname = Symbol(str_dnames[1][2:end])
    xname = Symbol(split(str_dnames[2][2:end], "__")[1]) # cut down `__$(i)` part if any
    zsize, xsize = (sizes[zname], sizes[xname])
    if zsize == :(())
        # output var is constant
        sizes[dd_name] = xsize
    else
        sizes[dd_name] = :(($zsize..., $xsize...)) |> simplify
    end
end


function propagate_deriv_size!(g::AbstractExGraph)
    for nd in g.tape
        vname = varname(nd)
        if match(DERIV_NAME_PATTERN, String(vname)) != nothing
            propagate_deriv_size!(g, vname)
        end
    end
end


# (numeric) derivative size propagation

function infer_deriv_size!(g::AbstractExGraph, dd_name::Symbol)    
    rg = match(DERIV_NAME_PATTERN, String(dd_name))
    @assert length(rg.captures) == 2
    str_dnames = rg.captures
    zname = Symbol(str_dnames[1][2:end])
    xname = Symbol(split(str_dnames[2][2:end], "__")[1]) # cut down `__$(i)` part if any
    # in case z or x haven't been evaluated and their size isn't known yet
    evaluate!(g, zname)
    evaluate!(g, xname)
    sizes = @get_or_create(g.ctx, :rsizes, Dict())
    zsize, xsize = (sizes[zname], sizes[xname])
    sizes[dd_name] = (zsize..., xsize...)
end


function infer_deriv_size!(g::AbstractExGraph)
    for nd in g.tape
        vname = varname(nd)
        if match(DERIV_NAME_PATTERN, String(vname)) != nothing
            infer_deriv_size!(g, vname)
        end
    end
end


# top type

"The top type describing given data"
top_type(x::AbstractArray{T,N}) where {T,N} = AbstractArray{T,N}
top_type(x::Number) = Number

top_type(::Type{AT}) where {AT <: AbstractArray{T,N}} where {T,N} = AbstractArray{T,N}
top_type(::Type{T}) where {T <: Number} = Number

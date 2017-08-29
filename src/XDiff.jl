
__precompile__()

module XDiff

export
    _xdiff,
    xdiff,
    @scalardiff,
    @tensordiff,
    @specialdiff,
    VectorCodeGen,
    EinCodeGen,
    BlasCodeGen,
    # reexport from Espresso
    @get_or_create,
    # reexport from Einsum
    @einsum
    

include("core.jl")

end # module

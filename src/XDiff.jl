
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
    @get_or_create
    

include("core.jl")

end # module

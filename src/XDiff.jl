
__precompile__()

module XDiff

export
    _xdiff,
    xdiff,
    @scalardiff,
    @tensordiff,
    @specialdiff,
    # reexport from Espresso
    @get_or_create,
    sum_1,
    sum_2,
    squeeze_sum,
    squeeze_sum_1,
    squeeze_sum_2,
    # reexport from Einsum
    @einsum
    

include("core.jl")

end # module

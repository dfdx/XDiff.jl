

const SPECIAL_RULES = Dict{Symbol, Any}(
    :foo => :(y = foo(x)) => :(dzdy = foo_grad(x, y))
    
)

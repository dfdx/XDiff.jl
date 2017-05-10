
# codegen.jl - utils to generate code in different formats from ExGraph and EinGraph

# type for generating vectorized code 
struct VectorCodeGen
end


function generate_code(::VectorCodeGen, g::ExGraph, out_vars::Vector{Symbol})
    ex = to_expr(g)
    # push!(ex.args, Expr(:tuple, out_vars...))
    # ex = remove_unused(ex; output_vars=out_vars)
    return ex
end


function generate_code(::VectorCodeGen, g::EinGraph, out_vars::Vector{Symbol})
    ex = from_einstein(g)
    # push!(ex.args, Expr(:tuple, out_vars...))
    # ex = remove_unused(ex; output_vars=out_vars)
    return ex
end


struct EinCodeGen
end


function generate_code(::EinCodeGen, g::ExGraph, out_vars::Vector{Symbol})
    return to_einstein(g)
end

function generate_code(::EinCodeGen, g::EinGraph, out_vars::Vector{Symbol})
    return to_expr(g)
end

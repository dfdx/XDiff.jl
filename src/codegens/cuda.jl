
struct CuCodeGen
    buf_var_name::Symbol
end


const CUDA_NATIVE_RULES = [
    :(log.(x)) => :(CUDAnative.log.(x)),
    :(exp.(x)) => :(CUDAnative.exp.(x)),
    :(sqrt.(x)) => :(CUDAnative.sqrt.(x)),
    :(x .^ n) => :(CUDAnative.pow.(x, Float32(n)))
]


function cuda_buffer_expr(var, buffer_var, sz_ex)
    gpu_sz_ex = matchingex(:(zeros(_...)), sz_ex) ? :(GPUArray($sz_ex)) : sz_ex
    return :($var = @get_or_create $buffer_var $(Expr(:quote, var)) $gpu_sz_ex)
end



function generate_code(codegen::CuCodeGen, g::EinGraph)
    g = eliminate_common(g)
    ex = to_buffered(g)
    ex = rewrite_all(ex, CUDA_NATIVE_RULES; phs=[:x, :y, :z, :n])
    buffer_var = codegen.buf_var_name
    init_exs = [cuda_buffer_expr(var, buffer_var, sz_ex) for (var, sz_ex) in g.ctx[:buff_exprs]
                if haskey(g, var) && getcategory(g[var]) != :input]
    res = Expr(:block, init_exs..., ex.args...)
    return res
end

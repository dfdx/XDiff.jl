
# codegen.jl - utils to generate code in different formats from ExGraph and EinGraph

# type for generating vectorized code 
struct VectorCodeGen
end


function generate_code(::VectorCodeGen, g::ExGraph)
    ex = to_expr(g)
    return ex
end


function generate_code(::VectorCodeGen, g::EinGraph)
    ex = from_einstein(g)
    return ex
end


struct EinCodeGen
end


function generate_code(::EinCodeGen, g::ExGraph)
    return to_einstein(g)
end

function generate_code(::EinCodeGen, g::EinGraph)
    return to_expr(g)
end


struct BlasCodeGen
    ctx_var_name::Symbol
end



buffer_expr(var, buffer_var, sz_ex) =
    :($var = @get_or_create $buffer_var $(Expr(:quote, var)) $sz_ex)


function generate_code(codegen::BlasCodeGen, g::EinGraph)
    g = eliminate_common(g)
    ex = to_blas(g)
    buffer_var = codegen.ctx_var_name
    init_exs = [buffer_expr(var, buffer_var, sz_ex) for (var, sz_ex) in g.ctx[:buff_exprs]
                if haskey(g, var) && getcategory(g[var]) != :input]
    res = Expr(:block, init_exs..., ex.args...)
    return res
end





# function foo(w1, w2, w3, x1, mem=Dict())
#     dtmp763_dx3 = @get_or_create(mem, :dtmp763_dx3, zeros(Float64, (10,)))
#     dtmp763_dw1 = @get_or_create(mem, :dtmp763_dw1, zeros(Float64, (10, 10)))
#     tmp760 = @get_or_create(mem, :tmp760, zeros(Float64, (1,)))
#     dtmp763_dw3 = @get_or_create(mem, :dtmp763_dw3, zeros(Float64, (1, 10)))
#     tmp748 = @get_or_create(mem, :tmp748, zeros(Float64, (10,)))
#     xx2 = @get_or_create(mem, :xx2, zeros(Float64, (10,)))
#     x2 = @get_or_create(mem, :x2, zeros(Float64, (10,)))
#     tmp800 = @get_or_create(mem, :tmp800, zeros(Float64, (10,)))
#     tmp754 = @get_or_create(mem, :tmp754, zeros(Float64, (10,)))
#     tmp763 = @get_or_create(mem, :tmp763, zero(Float64))
#     x3 = @get_or_create(mem, :x3, zeros(Float64, (10,)))
#     tmp782 = @get_or_create(mem, :tmp782, zero(Float64))
#     dtmp763_dtmp748 = @get_or_create(mem, :dtmp763_dtmp748, zeros(Float64, (10,)))
#     dtmp763_dxx2 = @get_or_create(mem, :dtmp763_dxx2, zeros(Float64, (10,)))
#     tmp793 = @get_or_create(mem, :tmp793, zeros(Float64, (10,)))
#     x4 = @get_or_create(mem, :x4, zeros(Float64, (1,)))
#     tmp761 = @get_or_create(mem, :tmp761, zeros(Float64, (1,)))
#     tmp759 = @get_or_create(mem, :tmp759, zeros(Float64, (1,)))
#     tmp783 = @get_or_create(mem, :tmp783, zero(Float64))
#     tmp784 = @get_or_create(mem, :tmp784, zeros(Float64, (1,)))
#     dtmp763_dtmp760 = @get_or_create(mem, :dtmp763_dtmp760, zeros(Float64, (1,)))
#     tmp747 = @get_or_create(mem, :tmp747, zero(Float64))
#     tmp785 = @get_or_create(mem, :tmp785, zeros(Float64, (1,)))
#     dtmp763_dtmp759 = @get_or_create(mem, :dtmp763_dtmp759, zeros(Float64, (1,)))
#     tmp749 = @get_or_create(mem, :tmp749, zeros(Float64, (10,)))
#     dtmp763_dxx3 = @get_or_create(mem, :dtmp763_dxx3, zeros(Float64, (10,)))
#     dtmp763_dw2 = @get_or_create(mem, :dtmp763_dw2, zeros(Float64, (10, 10)))
#     dtmp763_dx4 = @get_or_create(mem, :dtmp763_dx4, zeros(Float64, (1,)))
#     dtmp763_dx1 = @get_or_create(mem, :dtmp763_dx1, zeros(Float64, (10,)))
#     tmp753 = @get_or_create(mem, :tmp753, zeros(Float64, (10,)))
#     dtmp763_dx2 = @get_or_create(mem, :dtmp763_dx2, zeros(Float64, (10,)))
#     xx3 = @get_or_create(mem, :xx3, zeros(Float64, (10,)))
#     dtmp763_dtmp753 = @get_or_create(mem, :dtmp763_dtmp753, zeros(Float64, (10,)))
#     A_mul_B!(xx2, w1, x1)
#     tmp747 = 1.0
#     x2 .= log.(tmp747 .+ exp.(xx2))
#     A_mul_B!(xx3, w2, x2)
#     x3 .= log.(tmp747 .+ exp.(xx3))
#     A_mul_B!(x4, w3, x3)
#     tmp761 .= tmp747 .+ exp.((.-)(x4))
#     tmp763 = sum(1.0 ./ tmp761)
#     tmp783 = -2.0
#     dtmp763_dx4 .= (.-)((tmp747 .* ((.-)(tmp747) .* (tmp747 .+ exp.((.-)(x4))) .^ tmp783)) .* exp.((.-)(x4)))
#     A_mul_Bt!(dtmp763_dw3, dtmp763_dx4, x3)
#     At_mul_B!(dtmp763_dx3, w3, dtmp763_dx4)
#     dtmp763_dxx3 .= (dtmp763_dx3 .* (tmp747 ./ (tmp747 .+ exp.(xx3)))) .* exp.(xx3)
#     A_mul_Bt!(dtmp763_dw2, dtmp763_dxx3, x2)
#     At_mul_B!(dtmp763_dx2, w2, dtmp763_dxx3)
#     dtmp763_dxx2 .= (dtmp763_dx2 .* (tmp747 ./ (tmp747 .+ exp.(xx2)))) .* exp.(xx2)
#     A_mul_Bt!(dtmp763_dw1, dtmp763_dxx2, x1)
#     At_mul_B!(dtmp763_dx1, w1, dtmp763_dxx2)
#     tmp806 = (tmp763, dtmp763_dw1, dtmp763_dw2, dtmp763_dw3, dtmp763_dx1)
# end

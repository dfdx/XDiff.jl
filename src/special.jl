

const SPECIAL_PHS = Set{Symbol}([:x, :x1, :x2, :x3, :x4, :y, :z, :dz!dx, :dz!dy])

const SPECIAL_RULES = Dict{Any, Any}(
    # :(Main.foo) => [:(y = Main.foo(x)) => :(dz!dx = foo_grad(dz!dy, x))],

)


function special_derivative(g::EinGraph, iex::Expr, dzdy_vname::Symbol, z::Symbol, x::Symbol)    
    ex = from_einstein(iex)
    op = ex.args[2].args[1]
    # in Einstein notation node may additionally contain sum(), unpack if so
    is_sum = false
    if op == :sum
        ex = :($(ex.args[1]) = $(ex.args[2].args[2]))
        op = ex.args[2].args[1]
        is_sum = true
    end
    rules = SPECIAL_RULES[op]
    dex = nothing
    for (pat, rpat) in rules
        wrt = split_deriv_name(rpat.args[1])[2] |> undname
        mt = matchex(pat, ex; phs=SPECIAL_PHS)
        if x == wrt && !isnull(mt)
            st = get(mt)
            st[:dz!dy] = dzdy_vname
            st[:dz!dx] = deriv_name(z, x)
            dex = subs(rpat, st)
            break
        end
    end
    # how to apply additional sum() here? should we include it into foo_grad!() itself?
    if dex != nothing
        lhs, rhs = dex.args
        @assert(getvalue(g[x]) != nothing, "$x is not evaluated in the source graph")
        lhs_w_idxs = with_indices(lhs, ndims(getvalue(g[x])))
        vnames = find_var_names(rhs)
        vars = [:($v[:]) for v in vnames]
        rhs_w_idxs = subs(rhs, Dict(vname => var for (vname, var) in zip(vnames, vars)))        
        return TensorDeriv(g, :($lhs_w_idxs = $rhs_w_idxs))
    else
        error("Can't find special differentiation rule matching $iex")
    end
end



function _specialdiff(f_ex, df_ex)
    op = f_ex.args[2].args[1]
    dop = df_ex.args[2].args[1]
    push!(Espresso.SPECIAL_FUNCS, op, dop)
    if !haskey(SPECIAL_RULES, op)
        SPECIAL_RULES[op] = []
    end
    push!(SPECIAL_RULES[op], f_ex => df_ex)
end


macro specialdiff(f_ex, df_ex)
    _specialdiff(f_ex, df_ex)
    nothing
end

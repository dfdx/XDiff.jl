
function permute_indices{Idx<:ExIndex}(lhs_idxs::Vector{Idx},
                                       rhs_idxs::Vector{Idx})
    diff_idxs = union(setdiff(Set(lhs_idxs), Set(rhs_idxs)),
                      setdiff(Set(rhs_idxs), Set(lhs_idxs)))
    f_rhs_idxs = [idx for idx in rhs_idxs if !in(idx, diff_idxs)]
    f_lhs_idxs = [idx for idx in lhs_idxs if !in(idx, diff_idxs)]
    st = Dict(zip(f_rhs_idxs, f_lhs_idxs))
    return [get(st, idx, idx) for idx in rhs_idxs]
end

function rev_step!(g::ExGraph, nd::ExNode{:(=)}, adj::Dict{Symbol, TensorDeriv})
    y = nd.var
    x = dependencies(nd)[1]
    dzdx = deepcopy(adj[y])
    wrtidxs = permute_indices(nd.idxs[1], nd.idxs[2])
    # Q: should we subs indices in dzdx.ex? (assuming not)
    dzdx.wrt.args = [dname(x), wrtidxs...]
    adj[x] = dzdx
end

function rev_step!(g::ExGraph, nd::ExNode{:constant},
                   adj::Dict{Symbol, TensorDeriv})
    # adj[nd.var] = TensorDeriv(0.)
    # do nothing
end

function rev_step!(g::ExGraph, nd::ExNode{:input},
                   adj::Dict{Symbol,TensorDeriv})
    # do nothing
end


function rev_step!(g::ExGraph, nd::ExNode{:call}, adj::Dict{Symbol, TensorDeriv})
    y = nd.var
    iex = to_iexpr(nd)
    dzdy = adj[y]
    sizes = g.ctx[:sizes]
    for x in dependencies(nd)
        dydx = tderivative(iex, x)
        dzdx = dzdy ⊗ dydx        
        if haskey(adj, x)
            adj[x] = adj[x] ⊕ dzdx
        else
            adj[x] = dzdx
        end
        if x != :I
            dzdx_name = single_var(dzdx).args[1]
            sizes[dzdx_name] = deriv_size(sizes[g.ctx[:z_var]], sizes[x])
        end
    end
end


function deriv_size(z_size::Expr, x_size::Expr)
    if z_size == :(())
        return x_size
    else
        # TODO: to fix it, just find nice form of (z_size..., x_size...)
        error("Differentiation of non-scalar output variable is not supported yet")
    end
end

# other utils

function to_expanded_expr(g::ExGraph, td::TensorDeriv)
    ex = to_expr(td).args[2]
    depv = dep_vars(g, ex)
    dep_exs = Expr[]
    for nd in g.tape
        if !isa(nd, ExNode{:input}) && nd.var in depv
            # TODO: use expand_const_1(g, nd) ?
            push!(dep_exs, to_iexpr(nd))
        end
    end
    result_var = single_var(td)
    result_ex = :($result_var = $ex)
    return to_block(dep_exs..., result_ex)
end


function expand_adjoints(g::ExGraph, adj::Dict{Symbol, TensorDeriv})
    return Dict([(var, to_expanded_expr(g, td)) for (var, td) in adj])
end


function tdiff(ex::Expr; ctx=Dict(), inputs...)
    ctx = to_context(ctx)
    tex = to_einstein(ex; inputs...)
    g, adj = _rdiff(tex; ctx=ctx, inputs...)
    vars = Set([var for (var, val) in inputs])
    dexs = Dict([(var, dex) for (var, dex) in adj if in(var, vars)])
    return dexs
end

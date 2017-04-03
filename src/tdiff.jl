
function permute_indices{T}(lhs_idxs::Vector{T},
                            rhs_idxs::Vector{T})
    diff_idxs = union(setdiff(Set(lhs_idxs), Set(rhs_idxs)),
                      setdiff(Set(rhs_idxs), Set(lhs_idxs)))
    f_rhs_idxs = [idx for idx in rhs_idxs if !in(idx, diff_idxs)]
    f_lhs_idxs = [idx for idx in lhs_idxs if !in(idx, diff_idxs)]
    st = Dict(zip(f_rhs_idxs, f_lhs_idxs))
    return [get(st, idx, idx) for idx in rhs_idxs]
end

function rev_step!(g::ExGraph, nd::ExNode{:(=)}, adj::Dict{Symbol, TensorDeriv})
    y = varname(nd)
    x = dependencies(nd)[1]
    dzdy = adj[y]
    new_wrt_idxs = permute_indices(varidxs(nd), get_indices(expr(nd))[1])
    new_wrt = make_indexed(dname(x), new_wrt_idxs)
    new_last_tde = copy(last_tde(dzdy); wrt=new_wrt)
    dzdx = deepcopy(dzdy)
    dzdx.wrt = new_wrt
    last_chain(dzdx)[end] = new_last_tde
    # dzdx = deepcopy(adj[y])
    # wrtidxs = permute_indices(varidxs(nd), get_indices(expr(nd))[1])
    # Q: should we subs indices in dzdx.ex? (assuming not)
    # dzdx.wrt.args = [dname(x), wrtidxs...]
    adj[x] = dzdx
end

function rev_step!(g::ExGraph, nd::ExNode{:constant},
                   adj::Dict{Symbol, TensorDeriv})
    # do nothing
end

function rev_step!(g::ExGraph, nd::ExNode{:input},
                   adj::Dict{Symbol,TensorDeriv})
    # do nothing
end


function rev_step!(g::ExGraph, nd::ExNode{:call}, adj::Dict{Symbol, TensorDeriv})
    y = varname(nd)
    iex = to_expr(nd)
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
        # TODO: there should be a better representation of (z_size..., x_size...)
        return :($(z_size)..., $(x_size)...)

    end
end

# other utils

function to_expanded_expr(g::ExGraph, td::TensorDeriv)
    ex = to_expr(td)
    depv = collect_deps(g, ex)
    dep_exs = Expr[]
    for nd in g.tape
        if !isa(nd, ExNode{:input}) && varname(nd) in depv
            # should we use expand_const_1(g, nd) to expand constants in-place?
            push!(dep_exs, to_expr(nd))
        end
    end
    # result_var = single_var(td)
    # result_ex = :($result_var = $ex)
    return to_block(dep_exs..., ex)
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

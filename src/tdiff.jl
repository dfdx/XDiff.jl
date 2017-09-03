
## utils

function extend_deriv!(g::EinGraph, dg::EinGraph, dzdx::TensorDeriv)
    vname, idxs = split_indexed(single_var(dzdx))
    old_idxs = varidxs(dg[vname])
    var = make_indexed(vname, old_idxs)
    # substitution table to reindex new dzdx with the old indices
    # st = Dict(zip(idxs, old_idxs))
    subderivs = find_related(dg, vname)
    pos = indexof(dg, vname)
    if isempty(subderivs)

        # if wrtname(dzdx) == :dmu
        #     global _g = g
        #     global _dg = dg
        #     global _dzdx = dzdx
        #     error("here")
        # end
            
        # first split
        old_dzdx = TensorDeriv(g, to_expr(dg[vname]))
        dzdx = reindex_siblings_to_match(old_dzdx, dzdx)[2]
        ex_1 = getexpr(old_dzdx)
        ex_2 = getexpr(dzdx)
        vname_1 = Symbol("$(vname)__1")
        var_1 = make_indexed(vname_1, old_idxs)
        vname_2 = Symbol("$(vname)__2")
        var_2 = make_indexed(vname_2, old_idxs)
        sub_dg = EinGraph()
        parse!(sub_dg, :($var_1 = $ex_1))
        parse!(sub_dg, :($var_2 = $ex_2))
        parse!(sub_dg, :($var = $var_1 .+ $var_2))
        sub_dg = fuse_assigned(sub_dg)
        new_nodes = sub_dg.tape
    else        
        # dg already contains subderivatives for dzdx_v
        last_idx = parse(Int, split(subderivs[end] |> String, "__")[end])
        new_var = make_indexed(Symbol("$(vname)__$(last_idx + 1)"), old_idxs)
        # TODO: should use reindex_sinlings_to_match instead
        # new_ex = subs(getexpr(dzdx), st)
        old_dzdx = TensorDeriv(g, to_expr(dg[vname]))
        dzdx = reindex_siblings_to_match(old_dzdx, dzdx)[2]              
        new_ex = getexpr(dzdx)
        prev_ex = getexpr(dg[vname])
        sub_dg = EinGraph()
        parse!(sub_dg, :($new_var = $new_ex))
        parse!(sub_dg, :($var = $prev_ex .+ $new_var))
        sub_dg = fuse_assigned(sub_dg)
        new_nodes = sub_dg.tape
    end
    delete!(dg, pos)
    insert!(dg, pos, new_nodes)
    return dg
end


function permute_indices(lhs_idxs::Vector{T},
                         rhs_idxs::Vector{T}) where T
    diff_idxs = union(setdiff(Set(lhs_idxs), Set(rhs_idxs)),
                      setdiff(Set(rhs_idxs), Set(lhs_idxs)))
    f_rhs_idxs = [idx for idx in rhs_idxs if !in(idx, diff_idxs)]
    f_lhs_idxs = [idx for idx in lhs_idxs if !in(idx, diff_idxs)]
    st = Dict(zip(f_rhs_idxs, f_lhs_idxs))
    return [get(st, idx, idx) for idx in rhs_idxs]
end


function deriv_size(z_size::Expr, x_size::Expr)
    if z_size == :(())
        return x_size
    else
        # TODO: there should be a better representation of (z_size..., x_size...)
        return :($(z_size)..., $(x_size)...)

    end
end



## reverse pass


function rev_step!(g::EinGraph, dg::EinGraph, nd::ExNode{:(=)})
    z = g.ctx[:z_var]
    y = varname(nd)
    x = dependencies(nd)[1]
    dzdy_var = getvar(dg[deriv_name(z, y)])
    dzdx_vname = deriv_name(z, x)
    dzdx_idxs = varidxs(g[x])
    dzdx_var = make_indexed(dzdx_vname, dzdx_idxs)
    dzdx_full_ex = :($dzdx_var = $dzdy_var)
    parse!(dg, dzdx_full_ex)
end


function rev_step!(g::EinGraph, dg::EinGraph, nd::ExNode{:constant})
    # do nothing
end

function rev_step!(g::EinGraph, dg::EinGraph, nd::ExNode{:input})
    # do nothing
end


function rev_step!(g::EinGraph, dg::EinGraph, nd::ExNode{:call})
    y = varname(nd)
    iex = to_expr(nd)
    z = g.ctx[:z_var]
    cg = cat(g, dg)
    dzdy = TensorDeriv(g, dg[deriv_name(z, y)] |> to_expr)
    # sizes = g.ctx[:sizes]
    for x in dependencies(nd)
        if isa(g[x], ExNode{:constant})
            # don't clog dg with unnesessary derivs
            continue
        end
        # can find derivatives of special functions if `z` is a scalar
        if isspecial(getexpr(nd).args[1]) && isa(z, Symbol)
            dzdy_vname = deriv_name(z, y)
            dzdx = special_derivative(g, iex, dzdy_vname, z, x)
        else
            dydx = tderivative(iex, x)
            dzdx = dzdy ⊗ dydx
        end
        dzdx = expand_const(cg, dzdx) |> simplify
        dzdx_vname = split_indexed(single_var(dzdx))[1]
        if haskey(dg, dzdx_vname)
            extend_deriv!(g, dg, dzdx)
        else
            parse!(dg, to_expr(dzdx))
        end
        # sizes[dzdx_vname] = deriv_size(sizes[z], sizes[x])
    end
end


function make_dzdz_expr(z::Symbol, z_idxs::Vector)
    vname = deriv_name(z, z)
    wrt_idxs = next_indices(Set(z_idxs), 1, length(z_idxs))
    vidxs = vcat(z_idxs, wrt_idxs)
    var = make_indexed(vname, vidxs)
    guards = Expr[:($i == $j) for (i,j) in zip(z_idxs, wrt_idxs)]
    ex = with_guards(1.0, guards)
    return :($var = $ex)
end


function reverse_pass!(g::EinGraph)
    z_var = haskey(g.ctx, :cost) ? getvar(g[g.ctx[:cost]]) : getvar(g[end])
    z, z_idxs = split_indexed(z_var)
    g.ctx[:z_var] = z
    dzdz_ex = make_dzdz_expr(z, z_idxs)
    dg = EinGraph(dzdz_ex)   
    cost_deps = collect_deps(g, z_var)
    for nd in reverse(g.tape)
        if varname(nd) in cost_deps
            rev_step!(g, dg, nd)
        end
    end
    outvars = [deriv_name(z, varname(nd)) for nd in g.tape if isa(nd, ExNode{:input})]
    return fuse_assigned(dg; outvars=outvars)
end

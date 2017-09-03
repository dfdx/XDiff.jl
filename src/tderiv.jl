
# tensor_deriv.jl - tensor derivative utils (using Einstein notation)

const TDIFF_PHS = [:A, :B, :C, :X, :Y, :V, :W, :Z,
                   :i, :j, :k, :m, :n, :p, :q, :r, :s, :t, :l]

const TDIFF_VAR_NAMES = [:V, :W, :X, :Y]


## TensorDerivExpr - an atom of TensorDeriv

"""Helper struct to simplify calculations on tensor derivative expressions"""
mutable struct TensorDeriv
    var::Union{Symbol,Expr}
    wrt::Union{Symbol,Expr}
    ex::Any
    guards::Vector{Expr}
end


"""
Create a TensorDeriv from an expression. E.g. in:

     dy!dx[i,j] = W[i,j]

`x` and `y` will be looked up in `g` to find their sizes and make a derivative, e.g.:

    dy[i] / dx[j] = W[i,j]
"""
function TensorDeriv(g::EinGraph, dex::Expr; guards=nothing)
    full_vname, idxs = split_indexed(dex.args[1])
    vname, wrtname_ = split_deriv_name(full_vname)
    var_idx_len = g[undname(vname)].val |> size |> length
    vidxs, wrtidxs_ = isempty(idxs) ? ([],  []) : (idxs[1:var_idx_len], idxs[var_idx_len+1:end])
    var, wrt = make_indexed(vname, vidxs), make_indexed(wrtname_, wrtidxs_)
    ex = without_guards(dex.args[2])
    guards = guards != nothing ? guards : find_guards(dex.args[2])
    return TensorDeriv(var, wrt, ex, guards)
end


"""
Create a TensorDeriv from a pretty-printed expression. E.g.:

     dy[i] / dx[j] = W[i,j]

`y[i]` will be parsed as var and `z[j]` as w.r.t. var.
"""
function TensorDeriv(dex::Expr; guards=nothing)
    var, wrt = dex.args[1].args[2:3]
    ex = without_guards(dex.args[2])
    guards = guards != nothing ? guards : find_guards(dex.args[2])
    return TensorDeriv(var, wrt, ex, guards)
end


function Base.copy(td::TensorDeriv; var=getvar(td), wrt=getwrtvar(td),
                   ex=getexpr(td), guards=getguards(td))
    return TensorDeriv(var, wrt, ex, guards)
end


Espresso.getvar(td::TensorDeriv) = td.var
Espresso.varname(td::TensorDeriv) = split_indexed(getvar(td))[1]
Espresso.varidxs(td::TensorDeriv) = split_indexed(getvar(td))[2]

getwrtvar(td::TensorDeriv) = td.wrt
wrtname(td::TensorDeriv) = split_indexed(getwrtvar(td))[1]
wrtidxs(td::TensorDeriv) = split_indexed(getwrtvar(td))[2]

lhs_indices(td::TensorDeriv) = vcat(varidxs(td), wrtidxs(td))
rhs_indices(td::TensorDeriv) = convert(Vector{Any}, unique(flatten(find_indices(getexpr(td)))))
free_indices(td::TensorDeriv) = convert(Vector{Any}, setdiff(rhs_indices(td), lhs_indices(td)))

Espresso.getexpr(td::TensorDeriv) = td.ex
Espresso.getguards(td::TensorDeriv) = td.guards


# # var_indices(td::TensorDeriv) = convert(Vector{Any}, td.dvar.args[2:end])
# # wrt_indices(td::TensorDeriv) = convert(Vector{Any}, td.wrt.args[2:end])
# # deriv_indices(td::TensorDeriv) = vcat(var_indices(td), wrt_indices(td))
# # all_indices(td::TensorDeriv) = union(deriv_indices(td),
# #                                     flatten(Any, get_indices(expr(td))))

function all_indices(td::TensorDeriv)
    return union(varidxs(td),
                 wrtidxs(td),
                 flatten(Any, get_indices(getexpr(td))))
end


function single_var(td::TensorDeriv)
    new_name = Symbol("$(varname(td))!$(wrtname(td))")
    new_idxs = vcat(varidxs(td), wrtidxs(td))
    return make_indexed(new_name, new_idxs)
end


function to_expr(td::TensorDeriv)
    lhs = single_var(td)
    lhs_idxs = get_indices(lhs)
    guards = getguards(td)
    rhs = getexpr(td)
    rhs_idxs = get_indices(rhs)
    # TODO: check if it works for blocks too
    return Espresso.apply_guards(:($lhs = $rhs),  guards)
end


function to_expr_pp(td::TensorDeriv)
    lhs = :($(getvar(td)) / $(getwrtvar(td)))
    grds = getguards(td)
    rhs = !isempty(grds) > 0 ? Expr(:call, :*, getexpr(td), grds...) : getexpr(td)
    return :($lhs = $rhs)
end


function Base.show(io::IO, td::TensorDeriv)
    print(io, to_expr_pp(td))
end



# TODO:  moved to Esresso, import from there

"""
Given a set of existing indices and current position of iterator,
find the next index not in the set.
"""
function next_index(existing::Set{T}, pos::Int) where T
    while pos <= length(IDX_NAMES) && in(IDX_NAMES[pos], existing)
        pos += 1
    end
    if pos <= length(IDX_NAMES)
        return IDX_NAMES[pos], pos + 1
    else
        throw(BoundsError("IDX_NAMES"))
    end
end


function next_indices(existing::Set{T}, pos::Int, count::Int) where T
    new_indices = Array{Symbol}(0)
    for i=1:count
        new_idx, pos = next_index(existing, pos)
        push!(new_indices, new_idx)
    end
    return new_indices
end


"""
Given a set of existing indicies and possible duplicates, find for each duplicate
a replacement - index from IDX_NAMES that is not used yet.
"""
function index_replacements(existing::Set{T}, maybedups::Vector{T}) where T
    repls = Dict{Symbol,Symbol}()
    pos = 1
    for idx in maybedups
        # maybedups should also be included in existing index set
        all_existing = union(existing, Set(maybedups), Set(keys(repls)))
        if in(idx, existing) && !in(idx, keys(repls))
            repls[idx], pos = next_index(all_existing, pos)
        end
    end
    return repls
end


function reindex_with_guards(full_ex, guards::Vector{Expr})
    @assert full_ex.head == :(=) && full_ex.args[1].args[1] == :/
    def, ex = full_ex.args
    anchors = Set(get_indices(def) |> flatten)
    pairs = Tuple{Any,Any}[(grd.args[2], grd.args[3]) for grd in guards]
    st, new_pairs = reduce_equalities(pairs, anchors)
    new_guards = [:($i1 == $i2) for (i1, i2) in new_pairs]
    new_ex = subs(ex, st)
    return :($def = $new_ex), new_guards
end


"""
Reindex second tensor derivative (dydx) so that:

    * dydx's var indices match dzdy's w.r.t. indices
    * no other indices in dydx equal any indices in dzdy
"""
function reindex_to_match(dzdy::TensorDeriv, dydx::TensorDeriv)
    common_idxs_st = Dict(zip(varidxs(dydx), wrtidxs(dzdy)))
    other_idxs_st = index_replacements(Set(all_indices(dzdy)), all_indices(dydx))
    st = merge(other_idxs_st, common_idxs_st)
    new_dydx_ex = subs(to_expr_pp(dydx), st) |> sanitize
    new_dydx = TensorDeriv(new_dydx_ex)
    return dzdy, new_dydx
end


"""
Reindex sibling derivatives so that:

    * td1's var indices match td2's w.r.t. indices
    * no other indices in td1 equal any indices in td2
"""
function reindex_siblings_to_match(td1::TensorDeriv, td2::TensorDeriv)
    common_idxs_st = Dict(zip(lhs_indices(td2), lhs_indices(td1)))
    # replace free indices from td2 with anything not in td1
    all_idxs1 = Set{Any}(all_indices(td1)) 
    free_idxs2_st = index_replacements(all_idxs1, collect(free_indices(td2)))        
    st = merge(free_idxs2_st, common_idxs_st)
    new_td2_ex = subs(to_expr_pp(td2), st) |> sanitize
    new_td2 = TensorDeriv(new_td2_ex)
    return td1, new_td2
end


# """
# Elementwise multuplication of tensor derivatives.
# Example:

#     dzdx = dzdy ⊗ dydx

# which may expand to:

#     dz/dy[i] = v[i]
#     dy[i]/dx[j] = w[i,j]
#     dz/dx[j] = v[i] .* w[i,j]
# """
function ⊗(dzdy::TensorDeriv, dydx::TensorDeriv)
    # can only multiply related derivatives, e.g. dz/dy * dy/dx
    @assert wrtname(dzdy) == varname(dydx)
    dzdy, dydx = reindex_to_match(dzdy, dydx)
    guards = vcat(getguards(dzdy), getguards(dydx))
    new_td_lhs = :($(getvar(dzdy)) / $(getwrtvar(dydx)))
    new_td_rhs = :($(single_var(dzdy)) .* $(getexpr(dydx))) |> simplify
    new_td_ex_no_guards = :($new_td_lhs = $new_td_rhs)
    new_td_ex, new_guards = reindex_with_guards(new_td_ex_no_guards, guards)
    new_td = TensorDeriv(new_td_ex; guards=new_guards)
    dzdx = TensorDeriv(new_td)
    return dzdx
end


# presumably, we need to use extend_deriv instead
function ⊕(td1::TensorDeriv, td2::TensorDeriv)
    @assert varname(td1) == varname(td2)
    @assert wrtname(td1) == wrtname(td2)
    dvar_idxs_st = Dict(zip(varidxs(td2), varidxs(td1)))
    wrt_idxs_st = Dict(zip(wrtidxs(td2), wrtidxs(td1)))
    st = merge(dvar_idxs_st, wrt_idxs_st)
    free_idxs = free_indices(td2)
    # TODO: should we also inclue all indicies of expr(td1)?
    all_existing_idxs = Set{Symbol}(vcat(keys(st)..., values(st)..., free_idxs))
    # TODO: use `index_replacements()` instead
    next_idx_pos = 1
    for idx in free_idxs
        if in(idx, values(st))
            st[idx], next_idx_pos = next_index(all_existing_idxs, next_idx_pos)
        end
    end
    wrt2_reindexed = subs(getwrtvar(td2), st)
    ex2_reindexed = subs(getexpr(td2), st)
    guards2_reindexed = Expr[subs(g, st) for g in find_guards(td2)]
    new_ex = simplify(getexpr(td1) ⊕ ex2_reindexed)
    new_guards = vcat(td1.guards, guards2_reindexed)
    new_td = TensorDeriv(getvar(td1), wrt2_reindexed, new_ex, new_guards)
    return reindex_with_guards(new_td)
end




## tensor differentiation rules

struct TensorDiffRule <: AbstractDiffRule
    pat::Expr             # pattern of expression to differentiate
    deriv::TensorDeriv    # pattern of differentiation expression
end

function Base.show(io::IO, rule::TensorDiffRule)
    print(io, "TensorDiffRule($(rule.pat) ==> $(to_expr(rule.deriv)))")
end


"""
Convert scalar diff rule to a tensor diff rule.

 * ew_rule   - elementwise (scalar) rule
 * orig_idxs - indices of full tensor expression,
               e.g. for `z[i] = X[i,j] * y[j]` it's [[:i], [:i, :j], [:j]]
 * idx       - index of input parameter to differentiate w.r.t. it
"""
function to_tensor_rule(ew_rule::DiffRule, orig_idxs::Vector{Vector{T}}, idx::Int) where T
    ew_pat = ew_rule.pat
    op = ew_pat.args[1]
    ew_ex = ew_rule.dpat
    # tensor var names and indices
    tvar_names = TDIFF_VAR_NAMES[1:length(ew_pat.args)-1]
    # tvar_idxs = IDX_NAMES[1:length(orig_idxs[1])]
    tvars = [make_indexed(name, IX) for (name, IX) in zip(tvar_names, orig_idxs[2:end])]
    tvar_idxs = IDX_NAMES[1:length(orig_idxs[1])]
    # tvars = [Expr(:ref, tvar, tvar_idxs...) for tvar in tvar_names]
    # dvar variable
    var_name = :Z
    dvar_name = dname(var_name)
    var = make_indexed(var_name, orig_idxs[1])
    dvar = make_indexed(dvar_name, orig_idxs[1])
    # w.r.t. variable
    wrt_name = tvar_names[idx]
    dwrt_name = dname(wrt_name)
    wrt_idxs = next_indices(Set(flatten(orig_idxs)), 1, length(orig_idxs[idx + 1]))
    dwrt = make_indexed(dwrt_name, wrt_idxs)
    # new pattern
    tpat = Expr(:call, op, tvars...)
    full_tpat = :($var = $tpat)
    # new tensor derivative expression
    tex = rewrite(tpat, ew_pat, ew_ex; phs=DIFF_PHS)
    # constructing tensor derivative
    if length(orig_idxs[idx + 1]) > 0
        # old and new w.r.t. indices must be equal
        tguards = Expr[:($i1 == $i2) for (i1, i2) in zip(orig_idxs[idx+1], wrt_idxs)]
    else
        tguards = Expr[]  # TODO: this should be covered by previous definition too
    end
    tderiv = TensorDeriv(dvar, dwrt, tex, tguards)
    return TensorDiffRule(full_tpat, tderiv)
end



const TENSOR_DIFF_RULES = Dict{Tuple{OpName, Int}, Vector{TensorDiffRule}}()


function push_tensordiff!(op::OpName, deriv_idx::Int, rule::TensorDiffRule)
    if !haskey(TENSOR_DIFF_RULES, (op, deriv_idx))
        TENSOR_DIFF_RULES[(op, deriv_idx)] = TensorDiffRule[]
    end
    push!(TENSOR_DIFF_RULES[(op, deriv_idx)], rule)
end

function _tensordiff(ex, dex)
    ex = sanitize(ex)
    dex = sanitize(dex)
    op = canonical(current_module(), ex.args[2].args[1])
    idxs = get_indices(ex.args[2])
    dvar = dex.args[1].args[2]
    wrt = dex.args[1].args[3]
    deriv_ex = without_guards(sanitize(dex.args[2]))
    grds = find_guards(dex)
    deriv = TensorDeriv(dvar, wrt, deriv_ex, grds)
    diff_var_name = Symbol(string(wrt.args[1])[2:end])
    var_names = [iex.args[1] for iex in ex.args[2].args[2:end]]
    deriv_idx = find(var_names .== diff_var_name)[1]
    rule = TensorDiffRule(ex, deriv)
    push_tensordiff!(op, deriv_idx, rule)
end


macro tensordiff(ex, dex)
    _tensordiff(ex, dex)
    nothing
end


function tfind_rule(fullex::Expr, idx::Int)
    @assert fullex.head == :(=) && fullex.args[2].head == :call
    op = fullex.args[2].args[1]
    haskey(TENSOR_DIFF_RULES, (op, idx)) || return Nullable{TensorDiffRule}()
    rules = TENSOR_DIFF_RULES[(op, idx)]
    matches = pat -> !isnull(matchex(pat, fullex; phs=TDIFF_PHS, allow_ex=false))
    matching = findfirst(matches,
                         [r.pat for r in rules])
    matching != 0 || return Nullable{TensorDiffRule}()
    return Nullable{TensorDiffRule}(rules[matching])
end

dname(var::Symbol) = Symbol("d$var")
undname(dvar::Symbol) = Symbol(string(dvar)[2:end])


"""dZ[i]/dX[j] = ... ==> Z[i]/X[i] = ..."""
function unpack_deriv(ex::Expr)
    @assert ex.head == :(=)
    @assert ex.args[1].head == :call && ex.args[1].args[1] == :/
    dvar, dwrt = [split_indexed(dv)[1] for dv in ex.args[1].args[2:3]]
    var, wrt = undname(dvar), undname(dwrt)
    return subs(ex, Dict(dvar => var, dwrt => wrt))
end

"""Z[i]/X[j] = ... ==> dZ[i]/dX[i] = ..."""
function pack_deriv(ex::Expr)
    @assert ex.head == :(=)
    @assert ex.args[1].head == :call && ex.args[1].args[1] == :/
    var, wrt = [split_indexed(v)[1] for v in ex.args[1].args[2:3]]
    dvar, dwrt = dname(var), dname(wrt)
    lhs = subs(ex.args[1], Dict(var => dvar, wrt => dwrt))
    rhs = ex.args[2]
    return Expr(:(=), lhs, rhs)
end


"""
Derivative of a primitive expression in Einstein notation.
"""
function tderivative(fullex::Expr, idx::Int)
    maybe_rule = tfind_rule(fullex, idx)
    if !isnull(maybe_rule)
        rule = get(maybe_rule)
        unpacked_rule_dex = unpack_deriv(to_expr_pp(rule.deriv))
        unpacked_dex = rewrite(fullex, rule.pat, unpacked_rule_dex; phs=TDIFF_PHS)
        dex = pack_deriv(unpacked_dex)
        return TensorDeriv(dex)
    else
        idxs = get_indices(fullex)
        # elementwise or broadcasting function
        op = opname(current_module(), fullex.args[2].args[1])
        types = [Float64 for i=1:length(fullex.args[2].args)-1]
        ew_maybe_rule = find_rule(op, types, idx)
        ew_rule = (!isnull(ew_maybe_rule) ? get(ew_maybe_rule) :
                   register_rule(op, types, idx))
        trule = to_tensor_rule(ew_rule, idxs, idx)
        push_tensordiff!(op, idx, trule)
        # now rule is registered, recursively call itself
        return tderivative(fullex, idx)
    end
end

function tderivative(fullex::Expr, var::Symbol)
    @assert fullex.head == :(=)
    @assert fullex.args[2].head == :call
    ivars = [var for var in fullex.args[2].args[2:end]]
    vars = [isa(ivar, Expr) ? ivar.args[1] : ivar for ivar in ivars]
    matching = findfirst(vars .== var)
    matching != 0 || error("Variable `$var` isn't present " *
                           "in expression `$fullex`")
    return tderivative(fullex, matching[1])
end

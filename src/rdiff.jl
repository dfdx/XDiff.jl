
# rdiff.jl - differentiate an expression or a function w.r.t. its arguments
#
# An approach taken in this file falls into a category of hybric method of
# computer-aided differentiation. General architecture closely follows
# reverse-mode automatic differentiation including building a "tape" and
# propagating derivative from an output to an input variables, but unlike AD
# it allows to output symbolic expressions for calculating partial derivatives
# in question.
#
# Advantages of this approach include efficient computation of derivatives
# (which may not always be a case for symbolic differentiation), outputting
# symbolic expression which only needs to be computed once at runtime
# (unlike full reverse-mode AD that requires 2 passes) and may be used to
# generate code for other systems like GPU (which is not possible with AD at all).
# The main disadvantage compared to AD is inability to handle conditions and
# loops since they create discontinuity and thus are very hard to analyze
# and produce symbolic expression.
#
#
# Architecture
# ------------
#
# Just as reverse-mode AD, we build derivative using 2 passes -
# forward and reverse.
#
# 1. During forward pass we parse expression and build a graph (`ExGraph`) -
#    "linearized" version of expression. Each node in this graph represents:
#
#     * function call: `ExNode{:call}`
#     * broadcasting: `ExNode{:bcast}`
#     * assignment: `ExNode{:(=)}`
#     * input variable: `ExNode{:input}`
#     * constant value: `ExNode{:constant}`
#     * tuple: `ExNode{:tuple}`
#
#    Nodes are created and put onto a "tape" (list of nodes) using `parse!()`
#    function in a topological order, so no node may refer to a dependency
#    that isn't defined yet. After the graph is built, all nodes are evaluated
#    using input variables and constants to provide "example values" for each
#    node (used for type inference and other stuff).
#
# 2. Reverse pass starts with an empty dict of "adjoints" - derivatives of
#    an output variable w.r.t. to current variable - and populating it
#    from the end to the beginning (or, more precisely, from dependent variables
#    to their dependencies).
#
#    The backbone of this process is `rev_step!` function, and the most
#    intersting method is the one that handles function call. Given a node
#    and a dict of all dependent adjoints, this method does the following
#    for each of its dependencies (here `z` stands for an output variable,
#    `y` - for current variable and `x` - for current dependency of `y`):
#
#     1) finds a differentiation rule for `dy/dx`, i.e. derivative of current
#        node w.r.t. current dependency
#     2) symbolically multiplies it by `dz/dy`, i.e. by derivative of output
#        variable w.r.t. current node, to obtain `dz/dx`, i.e. derivative of
#        output w.r.t. curent dependency
#     3) adds this new derivative to the adjoint dict
#
#
# Example
# -------
#
# Here's an example of this process (and some tips on how to debug it).
#
# Suppose we have expression:
#
#     ex = :(z = x1*x2 + sin(x1))
#
# We can build an `ExGraph` and perform forward pass on it like this (which is
# not necessary for high-level usage, but helpful for debugging):
#
#     g = ExGraph(ex; x1=1, x2=1)  # give it an example of possible values
#     forward_pass(g)
#
# After this our graph looks like:
#
#   ExGraph
#     ExNode{:input}(:x1,:($(Expr(:input, :x1))),1)
#     ExNode{:input}(:x2,:($(Expr(:input, :x2))),1)
#     ExNode{:call}(:tmp1,:(x1 * x2),1)
#     ExNode{:call}(:tmp2,:(sin(x1)),0.8414709848078965)
#     ExNode{:call}(:tmp3,:(tmp1 + tmp2),1.8414709848078965)
#     ExNode{:(=)}(:z,:(z = tmp3),1.8414709848078965)
#
# Let's take a node with a first call for example:
#
#   ExNode{:call}(:tmp1,:(x1 * x2),1)
#
# Here `ExNode` is parametrized by :call symbol to enable Julia's fancy method
# dispatching, :tmp1 is a name of a new variable which is a product of variables
# :x1 and :x2 and has example value of 1.
#
# Now let's run reverse pass to obtain derivatives of `z` w.r.t. all other vars:
#
#     adj = reverse_pass(g, :z)
#
# which results in a dict like this:
#
#     Dict{Symbol,Any} with 6 entries:
#        :tmp2 => 1.0
#        :x1   => :(x2 + cos(x1))
#        :z    => 1.0
#        :tmp1 => 1.0
#        :x2   => :x1
#        :tmp3 => 1.0
#
# This means that:
#
#     dz/dz == adj[:z] == 1.0
#     dz/dtmp3 == adj[:tmp3] == 1.0
#     dz/dx1 == adj[:x1] == :(x2 + cos(x1))
#     ...
#
# To see how it works, consider finding derivative `dz/dx1` at intermediate
# node :tmp1. Note, that in our example `tmp1 = x1 * x2`.
#
# 1) by the time of computing this derivative we already know that
#    `dz/dtmp1 == 1.0`
# 2) from primitive derivative rules for product of numbers we also infer that
#    `dtmp1/dx1 == x2`
# 3) using chain rule we obtain (part of*) `dz/dx1` as a symbolic product
#    `dz/dmp1 * dtmp1/dx1 == 1.0 * x2 == x2`.
#
# I say "part of" because :z depends on :x1 not only through :tmp1, but also
# through :tmp2, for which derivative `dz/dx1` turns to be `cos(x1)`.
# To combine 2 these "parts" we simply add them up and obtain final result:
#
#     dz/dx1 == x2 + cos(x1)
#


## utils

function extend_deriv!(dg::ExGraph, dzdx_v::Symbol, dzdx::Any)
    subderivs = find_related(dg, dzdx_v)
    pos = indexof(dg, dzdx_v)
    if isempty(subderivs)
        # first split
        dzdx_ex_1 = getexpr(dg[dzdx_v])
        dzdx_ex_2 = dzdx
        dzdx_v_1 = Symbol("$(dzdx_v)__1")
        dzdx_v_2 = Symbol("$(dzdx_v)__2")
        sub_dg = typeof(dg)()
        parse!(sub_dg, :($dzdx_v_1 = $dzdx_ex_1))
        parse!(sub_dg, :($dzdx_v_2 = $dzdx_ex_2))
        parse!(sub_dg, :($dzdx_v = $dzdx_v_1 .+ $dzdx_v_2))
        sub_dg = fuse_assigned(sub_dg)
        new_nodes = sub_dg.tape
    else
        # dg already contains subderivatives for dzdx_v
        last_idx = parse(Int, split(subderivs[end] |> String, "__")[end])
        dzdx_v_last = Symbol("$(dzdx_v)__$(last_idx + 1)")
        prev_dzdx_ex = getexpr(dg[dzdx_v])
        sub_dg = typeof(dg)()
        parse!(sub_dg, :($dzdx_v_last = $dzdx))
        parse!(sub_dg, :($dzdx_v = $prev_dzdx_ex .+ $dzdx_v_last))
        sub_dg = fuse_assigned(sub_dg)
        new_nodes = sub_dg.tape
    end
    delete!(dg, pos)
    insert!(dg, pos, new_nodes)
    return dg
end


function example_val{T}(::Type{T})
    if T <: Number
        return one(T)
    elseif T <: Array
        return ones(eltype(T), [1 for i=1:ndims(T)]...)
    else
        error("Don't know how to create an example value for type $T")
    end
end



## forward pass

"""Forward pass of differentiation"""
function forward_pass!(g::AbstractExGraph)
    evaluate!(g, varname(g.tape[end]))
    # propagate_size!(g)
    return g
end


## reverse step

"""
Perform one step of reverse pass. Add derivatives of output variable w.r.t.
node's dependenices to adjoint dictionary.
"""
function rev_step!(g::ExGraph, dg::ExGraph, nd::ExNode{:(=)})
    y = varname(nd)
    x = dependencies(nd)[1]
    adj[x] = adj[y]
end


function rev_step!(g::ExGraph, dg::ExGraph, nd::ExNode{:constant})
    # do nothing
end


function rev_step!(g::ExGraph, dg::ExGraph, nd::ExNode{:input})
    # do nothing
end


function rev_step!(g::ExGraph, dg::ExGraph, nd::ExNode{:call})
    y = varname(nd)
    z = g.ctx[:z_var]
    dzdy_v = deriv_name(z, y)
    cg = cat(g, dg)
    types = [typeof(getvalue(g[x])) for x in dependencies(nd)]
    for (i, x) in enumerate(dependencies(nd))
        xnd = g[x]
        if isa(xnd, ExNode{:constant})
            # don't clog dg with unnesessary derivs
            continue
        end
        dydx = derivative(getexpr(nd), types, i, mod=g.ctx[:mod])
        dzdx = dzdy_v ⊗ dydx
        dzdx = expand_const(cg, dzdx) |> simplify
        dzdx_v = deriv_name(z, x)
        if haskey(dg, dzdx_v)
            extend_deriv!(dg, dzdx_v, dzdx)
        else
            parse!(dg, :($dzdx_v = $dzdx))
        end
    end
end


function rev_step!(g::ExGraph, dg::ExGraph, nd::ExNode{:bcast})
    y = varname(nd)
    z = g.ctx[:z_var]
    dzdy_v = deriv_name(z, y)
    cg = cat(g, dg)
    types = [typeof(getvalue(g[x])) for x in dependencies(nd)]
    for (i, x) in enumerate(dependencies(nd))
        xnd = g[x]
        if isa(xnd, ExNode{:constant})
            # don't clog dg with unnesessary derivs
            continue
        end
        dydx = derivative(getexpr(nd) |> bcast_to_call, types, i, mod=g.ctx[:mod])
        dzdx = dzdy_v ⊗ dydx
        dzdx = expand_const(cg, dzdx) |> simplify
        dzdx_v = deriv_name(z, x)
        if haskey(dg, dzdx_v)
            extend_deriv!(dg, dzdx_v, dzdx)
        else
            parse!(dg, :($dzdx_v = $dzdx))
        end
    end
end


function reverse_pass!(g::ExGraph)
    z = varname(g.tape[end])
    g.ctx[:z_var] = z
    dzdz_var = deriv_name(z, z)
    dg = ExGraph(:($dzdz_var = 1.0))
    for nd in reverse(g.tape)
        rev_step!(g, dg, nd)
    end
    outvars = [deriv_name(z, varname(nd)) for nd in g.tape if isa(nd, ExNode{:input})]
    return fuse_assigned(dg; outvars=outvars)
end


function _xdiff(g::AbstractExGraph)
    forward_pass!(g)
    dg = reverse_pass!(g)
    return dg
end


# function _xdiff(ex::Expr; ctx=Dict(), inputs...)
#     g = isindexed(ex) ? EinGraph(ex; ctx=ctx, inputs...) : ExGraph(ex; ctx=ctx, inputs...)
#     return g, _xdiff(g)
# end



"""
xdiff(ex::Expr; ctx=Dict(), inputs...)

Differentiate expression `ex` w.r.t. variables `inputs`. `inputs` should be a list
of key-value pairs with keys representing variables in expression and values
representing 'example values' (used e.g. for type inference). Returns an expression
that calculates original value and derivatives of all inputs. Example:

    xdiff(:(x^n); x=1, n=1)
    # quote
    #   tmp704 = 1
    #   tmp708 = log(x)
    #   tmp705 = n - tmp704
    #   tmp706 = x .^ tmp705
    #   tmp702 = x ^ n
    #   dtmp702!dx = n * tmp706
    #   tmp709 = x .^ n
    #   dtmp702!dn = tmp708 * tmp709
    #   tmp711 = (tmp702, dtmp702!dx, dtmp702!dn)
    # end

"""
function xdiff(ex::Expr; ctx=Dict(), inputs...)
    ctx = to_context(ctx)
    # determine format: if any of arguments is a tensor, use Einstein notation
    # otherwise use simple scalar differentiation
    if isindexed(ex)
        g = EinGraph(ex; ctx=ctx, inputs...)
    elseif any(x -> !isa(x[2], Number), inputs)
        iex = to_einstein(ex; ctx=ctx, inputs...)
        g = EinGraph(iex; ctx=ctx, inputs...)
    else
        g = ExGraph(ex; ctx=ctx, inputs...)
    end
    dg = _xdiff(g)
    rg = cat(g, dg)
    outvars = unshift!([deriv_name(g.ctx[:z_var], var) for (var, _) in inputs], varname(g[end]))
    push!(rg, :tuple, Espresso.genname(), Expr(:tuple, outvars...))
    rg = topsort(rg)
    infer_deriv_size!(rg)  # need to know size to evaluate things like `dz!dx[i] = 1.0`
    evaluate!(rg)
    codegen = @get(ctx, :codegen, BufCodeGen(:mem))
    return generate_code(codegen, rg)
end


"""
fdiff(f::Function; ctx=Dict(), xs...)

Differentiate function `f` w.r.t. its arguments and return tuple of derivative
function, one per argument.

See also `xdiff(ex::Expr; ctx=Dict(), xs...)`.
"""
function xdiff(f::Function; ctx=Dict(), inputs...)
    types = ([typeof(val) for (name, val) in inputs]...)
    args, ex = funexpr(f, types)
    ex = sanitize(ex)
    dex = xdiff(ex; ctx=ctx, inputs...)
    mod = get(ctx, :mod, current_module())  # check why it's not filled in RNN benchmark
    typed_args = [Expr(:(::), x, t) for (x, t) in zip(args, types)]
    mem_ex = :($(Expr(:parameters, Expr(:kw, :mem, :(Dict())))))
    header = Expr(:tuple, mem_ex, typed_args...)
    fn_ex = Expr(:->, header, dex)
    fn = eval(mod, fn_ex)
    return fn
end



# function xdiff{N}(f::Function, types::NTuple{N,DataType}; ctx=Dict())
#     args, ex = funexpr(f, types)
#     ex = sanitize(ex)
#     inputs = [arg => example_val(t) for (arg, t) in zip(args, types)]
#     dex = xdiff(ex; ctx=ctx, inputs...)
#     mod = ctx[:mod]
#     typed_args = [Expr(:(::), x, t) for (x, t) in zip(args, types)]
#     mem_ex = :($(Expr(:parameters, :(mem=Dict()))))
#     header = Expr(:tuple, mem_ex, typed_args...)
#     fn_ex = Expr(:->, header, dex)
#     fn = eval(mod, fn_ex)
#     return fn
# end

# XDiff.jl - eXpression Differentiation package

[![Build Status](https://travis-ci.org/dfdx/XDiff.jl.svg?branch=master)](https://travis-ci.org/dfdx/XDiff.jl)

> This package is unique in that it can differentiate vector-valued expressions in Einstein notation. However, if you only need gradients of scalar-valued functions (which is typicial in machine learning), please use [XGrad.jl](https://github.com/dfdx/XGrad.jl) instead. XGrad.jl is re-thought and stabilized version of this package, adding many useful featues in place of (not frequently used) derivatives of vector-valued functions. If nevertheless you want to continue using XDiff.jl, please pin Espresso.jl to version `v3.0.0`, which is the last supporting Einstein notation. 

**XDiff.jl** is an expression differentiation package, supporting fully
symbolic approach to finding tensor derivatives.
Unlike automatic differentiation packages, XDiff.jl can output not only ready-to-use
derivative functions, but also their symbolic expressions suitable for
further optimization and code generation.

### Expression differentiation

`xdiff` takes an expression and a set of "example values" and returns another expression
that calculates the value together with derivatives of an output variable w.r.t each
argument. Example values are anything similar to expected data, i.e. with the same data type
and size.
In the example below we want `w` and `x` to be vectors of size `(3,)` while `b` to be a scalar. 

```julia
# expressions after a semicolon are "example values" - something that looks like expected data
xdiff(:(y = sum(w .* x) + b); w=rand(3), x=rand(3), b=rand())
# quote 
#     dy!dx = @get_or_create(mem, :dy!dx, zeros(Float64, (3,)))
#     dy!dw = @get_or_create(mem, :dy!dw, zeros(Float64, (3,)))
#     y = @get_or_create(mem, :y, zero(Float64))
#     tmp658 = @get_or_create(mem, :tmp658, zero(Float64))
#     dy!dtmp658 = @get_or_create(mem, :dy!dtmp658, zero(Float64))
#     tmp658 = sum(w .* x)
#     y = tmp658 .+ b
#     dy!dtmp658 = 1.0
#     dy!dw .= x
#     dy!dx .= w
#     tmp677 = (y, dy!dw, dy!dx, dy!dtmp658)
# end
```

By default, `xdiff` generates a highly-optimized code that uses a set of buffers stored in
a dictionary `mem`. You may also generate slower, but more readable expression using `VectorCodeGen`:

```julia
ctx = Dict(:codegen => VectorCodeGen())
xdiff(:(y = sum(w .* x) + b); ctx=ctx, w=rand(3), x=rand(3), b=rand())
# quote 
#     tmp691 = w' * x
#     y = tmp691 + b
#     dy!dtmp691 = 1.0
#     dy!db = 1.0
#     dy!dw = x
#     dy!dx = w
#     tmp698 = (y, dy!dw, dy!dx, dy!db)
# end
```

or in [Einstein indexing notation](https://en.wikipedia.org/wiki/Einstein_notation) using `EinCodeGen`:

```julia
ctx = Dict(:codegen => EinCodeGen())
xdiff(:(y = sum(w .* x) + b); ctx=ctx, w=rand(3), x=rand(3), b=rand())
# quote
#     tmp700 = w[i] .* x[i]
#     y = tmp700 + b
#     dy!dtmp700 = 1.0
#     dy!db = 1.0
#     dy!dw[j] = dy!dtmp700 .* x[j]
#     dy!dx[j] = dy!dtmp700 .* w[j]
#     tmp707 = (y, dy!dw, dy!dx, dy!db)
# end
```

### Function differentiation

`xdiff` also provides a convenient interface for generating function derivatives:

```julia
# evaluate using `include("file_with_function.jl")` 
f(w, x, b) = sum(w .* x) .+ b

df = xdiff(f; w=rand(3), x=rand(3), b=rand())
df(rand(3), rand(3), rand())
# (0.8922305671741435, [0.936149, 0.80665, 0.189789], [0.735201, 0.000282879, 0.605989], 1.0)
```
Note, that `xdiff` will _try_ to extract function body as it was written, but it doesn't always
work smoothly. One сommon case when function body isn't available is when function is defined
in REPL, so for better experience load functions using `include(<filename>)` or `using <module>`.



### Limitations

 * loops are not supported
 * conditional branching is not supported

Loops and conditional operators may introduce discontinuity points, potentially resulting in
very complex and heavy piecewise expressions, and thus are not supported.
However, many such expressions may be rewritten into analytical form. For example, many loops
may be rewritten into some aggregation function like `sum()` (already supported), and
many conditions may be expressed as multiplication like `f(x) * (x > 0) + g(x) * (x <= 0)`
(planned). Please, submit an issue if you are interested in supporting some specific feature.


### How it works

On the high level, scalar expressions are differentiated as follows:

1. Expression is parsed into an `ExGraph` - a set of primitive expressions,
mostly assignments and single function calls.
2. Resulting `ExGraph` is evaluated using example values to determine types and
shape of all variables (forward pass).
3. Similar to reverse-mode automatic differentiation, derivatives are propagated
backward from output to input variables. Unlike AD, however, derivatives aren't
represented as values, but instead as symbolic exprssions.

Tensor expressions exploit very similar pipeline, but act in Einstein notation.

1. Tensor expression is transformed into Einstein notation.
2. Expression in Einstein notation is parsed into an `Einraph` (indexed variant of `ExGraph`).
3. Resulting `EinGraph` is evaluated.
4. Partial derivatives are computed using tensor or element-wise rules for each element
of each tensor, then propagated from output to input variables.
5. Optionally, derivative expressions are converted back to vectorized notation. 



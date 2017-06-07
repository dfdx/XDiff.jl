# XDiff.jl - eXpression Differentiation package

**XDiff.jl** is an expression differentiation package, supporting fully
symbolic approach to finding tensor derivatives.
Unlike automatic differentiation packages, XDiff.jl can output not only ready-to-use
derivative functions, but also their symbolic expressions suitable for
further optimization and code generation.

### Expression differentiation

`xdiff` takes an expression and a set of "example values" and returns another expression
that calculates the value together with derivatives of an output variable w.r.t each
argument. Example values are anything similar to expected data, i.e. with the same data type
and number of dimensions, though not necessarily the same number of elements (for tensors).
In the example below we want `w` and `x` to be vectors while `b` to be a scalar. 

```julia
# expressions after a semicolon are "example values" - something that looks like expected data
xdiff(:(y = sum(w .* x) + b); w=rand(3), x=rand(3), b=rand())
# quote 
#     dy_dtmp862 = 1.0
#     dy_dw = dy_dtmp862 .* x
#     dy_db = 1.0
#     dy_dx = dy_dtmp862 .* w
#     tmp862 = w' * x
#     y = tmp862 .+ b
#     tmp869 = (y, dy_dw, dy_dx, dy_db)
# end
```

`xdiff` can also generate expressions for vector-valued functions (i.e. R^n -> R^m)
using a variant of [Einstein indexing notation](https://en.wikipedia.org/wiki/Einstein_notation):


```julia
xdiff(:(y = w .* x + b); ctx=Dict(:codegen => EinCodeGen()), w=rand(3), x=rand(3), b=rand())
# quote
#     tmp685 = 1.0
#     dy_dy[i, j] = tmp685 * (i == j)
#     dy_dtmp677[i, m] = dy_dy[i, i] * (i == m)
#     tmp686[i, i] = dy_dtmp677[i, i] .* x[i]
#     dy_dw[i, j] = tmp686[i, i] * (i == j)
#     tmp687[i, i] = dy_dtmp677[i, i] .* w[i]
#     dy_dx[i, j] = tmp687[i, i] * (i == j)
#     tmp677[i] = w[i] .* x[i]
#     dy_db[i] = dy_dy[i, i]
#     y[i] = tmp677[i] + b
#     tmp688 = (y, dy_dw, dy_dx, dy_db)
# end
```

### Function differentiation

`fdiff` provides a convenient interface for generating function derivatives:

```julia
# evaluate using `include("file_with_function.jl")` 
f(w, x, b) = sum(w .* x) .+ b
types = (Vector{Float64}, Vector{Float64}, Float64)

df = fdiff(f, types)
df(rand(3), rand(3), rand())
# (1.228714504221751, [0.444729, 0.238441, 0.741301], [0.666098, 0.302282, 0.517627], 1.0)
```
Note, that `fdiff` will _try_ to extract function body as it was written, but it doesn't always
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



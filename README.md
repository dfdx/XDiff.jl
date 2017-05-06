# XDiff.jl - eXpression Differentiation package

**XDiff.jl** is an expression differentiation package, supporting fully
symbolic approach to finding tensor derivatives.
Unlike automatic differentiation packages, XDiff.jl can output not only ready-to-use
derivative functions, but also their symbolic expressions suitable for
further optimization and code generation. Here's an example:

```julia
function ann(w1, w2, w3, x1)
    _x2 = w1 * x1
    x2 = log(1. + exp(_x2))   # soft RELU unit
    _x3 = w2 * x2
    x3 = log(1. + exp(_x3))   # soft RELU unit
    x4 = sum(w3 * x3)
    return 1. ./ (1. + exp(-x4))  # sigmoid output
end

# ANN input parameter types
types = (Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64})

# generate example input
w1, w2, w3, x1 = randn(10,10), randn(10,10), randn(1,10), randn(10)

# create a dict of symbolic derivatives
dexs = rdiff(ann, types)
dexs[:w1]   # ==> quote ... end

# create derivative functions
dw1, dw2, dw3, _ = fdiff(ann, types)
dw1(randn(100,100), randn(100,100), randn(1,100), randn(100))
```

Another unique feature of XDiff.jl is that it can generate expressions not only for functions R^n -> R,
but also functions R^n -> R^m using Einstein indexing notation:

```julia
# by default, rdiff tries to generate vectorized output
# we can make it return expressions in Einstein notation using :outfmt option
ctx = [:outfmt => :ein]

# when differtiating an expression, we need to provide "example values",
# i.e. anything that has the same type and number of dimensions as we expect
# from real values
dexs = rdiff(:(z = W*x + b); ctx=ctx, W=rand(3,4), x=rand(4), b=rand(3))
dexs[:W]   # ==> :(dz_dW[i,m,n] = x[n] * (i == m))
```

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

1. Expression is parsed into an `ExGraph` - a set of primitive expressions, mostly assginments and single function calls.
2. Resulting `ExGraph` is evaluated using example values to determine types and shape of all variables (forward pass).
3. Similar to reverse-mode automatic differentiation, derivatives are propagated backward from output to input variables. Unlike AD, however, derivatives aren't represented as values, but instead as symbolic exprssions.

Tensor expressions exploit very similar pipeline, but act in Einstein notation.

1. Tensor expression is transformed into an Einstein notation.
2. Expression in Einstein notation is parsed into an `ExGraph`.
3. Resulting `ExGraph` is evaluated.
4. Partial derivatives are computed using tensor or element-wise rules for each element of each tensor, then propagated from output to input variables.
5. Optionally, derivative expressions are converted back to vectorized notation. 



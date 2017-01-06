# XDiff

### Symbolic tensor differentiation

`rdiff` - find derivatives of an expression w.r.t. input params. Input parameters
should be initialized with example values so that algorithm could infer their type
and other metadata.

```
rdiff(:(x1*x2 + sin(x1)), x1=1., x2=1.)
# ==> [:(x2 + cos(x1)), :x1]
```

This is a hybrid algorithm in sense that it uses techniques from
automatic differentiation (AD), but produces symbolic expression for each input.

Differentiation algorithm is heavily inspired by [ReverseDiffSource.jl][1],
but has a number of differences in implementation and capabilities. In particular, 
one can differentiate user defined functions, e.g.:

```
# create new function that doesn't have any registered differentiation rule
logistic(x) = 1 / (1 + exp(-x))

rdiff(logistic, x=1)
# ==> :(exp(-x) * (1 + exp(-x)) ^ -2)

rdiff(:(x + logistic(x)), x=1)
# ==> [:(1.0 + exp(-x) * (1 + exp(-x)) ^ -2)]
```


[1]: https://github.com/JuliaDiff/ReverseDiffSource.jl
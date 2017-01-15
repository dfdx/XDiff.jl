
using XDiff

ex = quote
    y[i] = W[i,k] * x[k]
    v[i] = y[i] - x[i]
    z = v[i] * I[i]
end
inputs = [:W => rand(2, 2), :x => rand(2)]
ctx = Dict()
@time dexs = rdiff(ex; ctx=ctx, inputs...)

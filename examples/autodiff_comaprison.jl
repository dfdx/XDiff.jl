
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile_gradient

f(a, b) = sum(a' * b + a * b')

const ∇f! = compile_gradient(f, (rand(100, 100), rand(100, 100)))

a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
cfg = GradientConfig(inputs)

∇f!(results, inputs)


using Espresso
using XDiff

types = [typeof(a), typeof(b)]
da, db = fdiff(f, types)
xresults = da(a, b), db(a, b)

# manual
ex = :(sum(a' * b + a * b'))
inputs = [:a=>rand(3,3), :b=>rand(3,3)]
dexs = rdiff(ex, inputs...)

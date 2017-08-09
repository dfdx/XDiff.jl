
function linreg_loss(w, b, x, y)
    yhat = sum(w .* x) + b
    loss = sum(abs2(yhat - y))
end

inputs = [:w => [1.0, 1.0], :b => 2.0, :x => [3.0, 3.0], :y => 4.0]
vals = ([v for (k,v) in inputs]...)
dlinreg_loss = xdiff(linreg_loss; inputs...)
dvals = dlinreg_loss(vals...)
@test dvals == (16.0, [96.0, 96.0], 32.0, [32.0, 32.0], -32.0)


# ReverseDiff doesn't support scalars yet
# See: https://github.com/JuliaDiff/ReverseDiff.jl/issues/40
# linreg_loss2(W, b, x, y) = sum(abs2.(y - (W * x .+ b)))
# test_compare(linreg_loss2; W=randn(1,11), b = rand(), x=randn(11,1000), y=randn(1,1000))


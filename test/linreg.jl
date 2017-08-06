
function linreg_loss(w, b, x, y)
    yhat = sum(w .* x) + b
    loss = sum(abs2(yhat - y))
end

inputs = [:w => [1.0, 1.0], :b => 2.0, :x => [3.0, 3.0], :y => 4.0]
vals = ([v for (k,v) in inputs]...)
dlinreg_loss = xdiff(linreg_loss; inputs...)
dvals = dlinreg_loss(vals...)
@test dvals == (16.0, [96.0, 96.0], 32.0, [32.0, 32.0], -32.0)

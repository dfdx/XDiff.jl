
function linreg_loss(w, b, x, y)
    yhat = sum(w .* x) + b
    loss = sum(abs2(yhat - y))
end

dlinreg_loss = fdiff(linreg_loss, (Vector{Float64}, Float64, Vector{Float64}, Float64))
dvals = dlinreg_loss([1.0, 1.0], 2.0, [3.0, 3.0], 4.0)
@test dvals == (16.0, [96.0, 96.0], 32.0, [32.0, 32.0], -32.0)


logistic(x) = 1 ./ (1 + exp.(-x))
@diff_rule logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))


function ann1(w1, w2, w3, x1)
    xx2 = w1 * x1
    x2 = log.(1. + exp.(xx2))
    xx3 = w2 * x2
    x3 = log.(1. + exp.(xx3))
    x4 = w3 * x3
    sum(1. ./ (1. + exp.(-x4)))
end


function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput .- x) .^ 2.0)
    return cost
end

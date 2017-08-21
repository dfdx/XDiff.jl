
logistic(x) = 1 ./ (1 + exp.(-x))
@scalardiff logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))


function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput .- x) .^ 2.0)
    return cost
end


function mlp1(w1, w2, w3, x1)
    xx2 = w1 * x1
    x2 = log.(1. + exp.(xx2))
    xx3 = w2 * x2
    x3 = log.(1. + exp.(xx3))
    x4 = w3 * x3
    sum(1. ./ (1. + exp.(-x4)))
end


function mlp2(w1, w2, w3, b1, b2,  b3, x1)
    x2 = logistic(w1 * x1 .+ b1)
    x3 = logistic(w2 * x2 .+ b2)
    x4 = logistic(w3 * x3 .+ b3)
    sum(x4)
end



function rnn(Wxh, Whh, Wy, hprev, x, y)
    h = tanh.(Whh * hprev + Wxh * x)
    yhat = Why * h
    cost = sum((yhat .- y) .^ 2.0)
    return h, cost
end


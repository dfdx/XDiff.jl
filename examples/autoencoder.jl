
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile_gradient

logistic(x) = 1 ./ (1 + exp(-x))

# for reference, not used in this example
function autoencoder(We1, We2, Wd, b1, b2, input)
    firstLayer = logistic(We1 * input + b1)
    encodedInput = logistic(We2 * firstLayer + b2)
    reconstructedInput = logistic(Wd * encodedInput)
    return reconstructedInput
end

function autoencoder_cost(We1, We2, Wd, b1, b2, input)
    firstLayer = logistic(We1 * input + b1)
    encodedInput = logistic(We2 * firstLayer + b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput - input) .^ 2)
    return cost
end


function main()
    input = rand(5)
    We1 = rand(4, 5)
    b1 = rand(4)
    We2 = rand(3, 4)
    b2 = rand(3)
    Wd = rand(5, 3)
    inputs = (We1, We2, Wd, b1, b2, input)
    types = [typeof(input) for input in inputs]
    dWe1, dWe2, dWd, db1, db2, _ = fdiff(autoencoder_cost, types)

    grad = gradient(autoencoder_cost, inputs)
end


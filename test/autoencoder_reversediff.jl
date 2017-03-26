
using XDiff
using ReverseDiff: compile_gradient

logistic(x) = 1 ./ (1 + exp.(-x))

# for reference, not used in this example
function autoencoder(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x + b1)
    encodedInput = logistic(We2 * firstLayer + b2)
    reconstructedInput = logistic(Wd * encodedInput)
    return reconstructedInput
end

# autoencoder cost: sum of squared error
function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x + b1)
    encodedInput = logistic(We2 * firstLayer + b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput - x) .^ 2.0)
    return cost
end

# ReverseDiff requires equivalent, but syntactically slightly different function
function autoencoder_cost_reversediff(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x + b1)
    encodedInput = logistic(We2 * firstLayer + b2)
    reconstructedInput = logistic(Wd * encodedInput)
    squared_error = broadcast(^, (reconstructedInput - x), 2)
    cost = sum(squared_error)
    return cost
end


# generate example input
x = rand(5)
We1 = rand(4, 5)
b1 = rand(4)
We2 = rand(3, 4)
b2 = rand(3)
Wd = rand(5, 3)

input_tuple = (We1, We2, Wd, b1, b2, x)
inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x =>  x]
types = map(typeof, input_tuple)


ex = quote
    firstLayer = logistic(We1 * x + b1)
    encodedInput = logistic(We2 * firstLayer + b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput - x) .^ 2.0)
end


dex = rdiff(ex; inputs...)
dvals = eval(dex)

# XDiff: generate a dict of derivative expressions
# dex = rdiff(autoencoder_cost, types)
#  dvals = eval(dex)

# XDiff: generate derivative functions and calculate value at the same point
# dcost = fdiff(autoencoder_cost, types)
# dvals = dcost(We1, We2, Wd, b1, b2, x)

# ReverseDiff: generate and apply derivative functions
const ∇f! = compile_gradient(autoencoder_cost_reversediff, input_tuple)
results = map(similar, input_tuple)
∇f!(results, input_tuple)

# compare results
@test isapprox(dvals[1], results[1])



# TODO:
# 1) find why :tmp995 is missing
# 2) optimize common subexpressions

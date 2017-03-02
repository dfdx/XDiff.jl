
logistic(x) = 1 ./ (1 + exp.(-x))

# # autoencoder cost: sum of squared error
# function autoencoder_cost(We1, We2, Wd, b1, b2, x)
#     firstLayer = logistic(We1 * x + b1)
#     encodedInput = logistic(We2 * firstLayer + b2)
#     reconstructedInput = logistic(Wd * encodedInput)
#     cost = sum((reconstructedInput - x) .^ 2.0)
#     return cost
# end

# autoencoder cost: sum of squared error
function autoencoder_cost(We, Wd, x)
    reconstructedInput = Wd * (We * x)
    cost = sum((reconstructedInput - x) .^ 2)
    return cost
end


# foo(x, y) = sum((y - x) .^ 2)



# generate example input
x = rand(5)
We = rand(3, 5)
Wd = rand(5, 3)

input_tuple = (We, Wd, x)
types = map(typeof, input_tuple)

# XDiff: generate a dict of derivative expressions
dexs = rdiff(autoencoder_cost, types; ctx=Dict(:outfmt => :vec))

# XDiff: generate derivative functions and calculate value at the same point
dcost = fdiff(autoencoder_cost, types)
@time dvals = dcost(We, Wd, x)




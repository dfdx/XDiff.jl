
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
dexs = rdiff(autoencoder_cost, types)

# XDiff: generate derivative functions and calculate value at the same point
dcost = fdiff(autoencoder_cost, types)
# @time dWe1, dWe2, dWd, db1, db2, dx = dcost(We1, We2, Wd, b1, b2, x)




quote  # /home/slipslop/.julia/v0.6/Espresso/src/from_einstein.jl, line 98:
    tmp1 = We * x
    reconstructedInput = Wd * tmp1
    tmp3 = -(reconstructedInput, x)
    tmp4 = 2
    tmp36 = 1
    tmp37 = -.(tmp4, tmp36)
    tmp38 = .^(tmp3, tmp37)
    tmp39 = .*(tmp4, tmp38)
    tmp40 = -.(tmp39)
    tmp41 = 1
    tmp42 = -(tmp4, tmp41)
    tmp43 = .^(tmp3, tmp42)
    tmp44 = .*(tmp4, tmp43)
    tmp45 = tmp44 .* Wd
    tmp46 = (sum(tmp45, 1))'
    tmp47 = tmp46 .* We
    tmp48 = tmp40 .+ tmp47
    dcost_dx = sum(tmp48, 2)
end

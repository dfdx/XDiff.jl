
logistic(x) = 1 ./ (1 + exp.(-x))
@diff_rule logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))

# autoencoder cost: sum of squared error
function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput .- x) .^ 2.0)
    return cost
end

# @rdcmp autoencoder_cost We1=rand(4,5) We2=rand(3,4) Wd=rand(5,3) b1=rand(4) b2=rand(3) x=rand(5)
test_compare(autoencoder_cost; We1=rand(4,5), We2=rand(3,4), Wd=rand(5,3),
             b1=rand(4), b2=rand(3), x=rand(5))

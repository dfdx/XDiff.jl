

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



# slightly different version that uncovered a bug in previous version
function autoencoder_cost2(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum(reconstructedInput .- x .^ 2.0)
    return cost
end

# @rdcmp autoencoder_cost We1=rand(4,5) We2=rand(3,4) Wd=rand(5,3) b1=rand(4) b2=rand(3) x=rand(5)
test_compare(autoencoder_cost2; We1=rand(4,5), We2=rand(3,4), Wd=rand(5,3),
             b1=rand(4), b2=rand(3), x=rand(5))


# and the most simple autoencoder
function autoencoder_cost3(We, Wd, x)
    y = We * x
    xd = Wd * y
    cost = sum(xd .- x)
    return cost
end
test_compare(autoencoder_cost3; We=rand(3,4), Wd=rand(4,3), x=rand(4,2))

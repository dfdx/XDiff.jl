
logistic(x) = 1 ./ (1 + exp.(-x))


# autoencoder cost: sum of squared errors
function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput - x) .^ 2.0)
    return cost
end


function main()
    types = (Matrix{Float64}, Matrix{Float64}, Matrix{Float64},
             Vector{Float64}, Vector{Float64}, Matrix{Float64})
    names = [:We1, :We2, :Wd, :b1, :b2, :x]
    vals = (rand(4,5), rand(3,4), rand(5,3),
            rand(4), rand(3), rand(5, 10))
    We1, We2, Wd, b1, b2, x = vals
    eval()
    inputs = [n => v for (n, v) in zip(names, vals)]
    cost = autoencoder_cost
    _, ex = funexpr(cost, types)
    ctx = Dict(:outfmt => :vec)
    dexs = rdiff(ex; ctx=ctx, inputs...)

    
    dcost = fdiff(cost, types)
    @time dWe1 = dcost(vals...)[1]
end


include("core.jl")

logistic(x) = 1 ./ (1 + exp.(-x))


# autoencoder cost: sum of squared errors
# function autoencoder_cost(We1, We2, Wd, b1, b2, x)
#     firstLayer = logistic(We1 * x .+ b1)
#     encodedInput = logistic(We2 * firstLayer .+ b2)
#     reconstructedInput = logistic(Wd * encodedInput)
#     cost = sum((reconstructedInput - x) .^ 2.0)
#     return cost
# end


# autoencoder cost: sum of squared errors
function autoencoder_cost(We1, We2, Wd, x)
    firstLayer = We1 * x
    encodedInput = We2 * firstLayer
    reconstructedInput = Wd * encodedInput
    cost = sum(reconstructedInput - x)
    return cost
end


function main()
    types = (Matrix{Float64}, Matrix{Float64}, Matrix{Float64},
             Matrix{Float64})
    names = [:We1, :We2, :Wd, :x]
    vals = (rand(4,5), rand(3,4), rand(5,3),
            rand(5, 10))
    We1, We2, Wd, x = vals
    inputs = [n => v for (n, v) in zip(names, vals)]
    cost = autoencoder_cost
    _, ex = funexpr(cost, types)
    ctx = Dict(:outfmt => :vec)
    dexs = rdiff(ex; ctx=ctx, inputs...)

    ictx = Dict(:outfmt => :ein)
    idexs = rdiff(ex; ctx=ictx, inputs...)

    
    dcost = fdiff(cost, types)
    @time dWe1 = dcost(vals...)[1]
end


# ExGraph(idexs[:We1])
# ExGraph
#   ExNode{call}(t1[m] = Wd[k, m] * I[k] | nothing)
#   ExNode{call}(t2[m, n] = t1[m] .* We2[m, n] | nothing)
#   ExNode{call}(t3[n] = t2[m, n] * I[m] | nothing)
#   ExNode{call}(t4[n, j] = t3[n] .* x[n, j] | nothing)
#   ExNode{=}(dcost_dWe1[n, n] = t4[n, j] | nothing)



quote
    t1 = squeeze(sum(Wd, 1), 1)
    t2 = t1 .* We2
    t3 = squeeze(sum(t2, 1), 1)
    t4 = t3 .* x
    dcost_dWe1 = repmat((squeeze(sum(t4, 2), 2))', size(We1, 1))

end

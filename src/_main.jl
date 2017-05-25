
include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end

    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end


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
    # cost = autoencoder_cost
    # _, ex = funexpr(cost, types)
    ex = quote
        firstLayer = We1 * x
        encodedInput = We2 * firstLayer
        reconstructedInput = Wd * encodedInput
        cost = sum(reconstructedInput - x)
    end

    
    ctx = Dict(:codegen => VectorCodeGen())
    dex = xdiff(ex; ctx=ctx, inputs...)

    ctx = Dict(:codegen => EinCodeGen())
    idexs = xdiff(ex; ctx=ictx, inputs...)

    
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






function ann(w1, w2, w3, x1)
    _x2 = w1 * x1
    x2 = log.(1. + exp.(_x2))   # soft RELU unit
    _x3 = w2 * x2
    x3 = log.(1. + exp.(_x3))   # soft RELU unit
    x4 = sum.(w3 * x3)
    return 1. ./ (1. + exp.(-x4))  # sigmoid output
end


function main_100()
    ex = quote
        tx2 = w1 * x1
        x2 = log.(1. + exp.(tx2))
        tx3 = w2 * x2
        x3 = log.(1. + exp.(tx3))
        x4 = sum.(w3 * x3)
        1. ./ (1. + exp.(-x4))
    end

    # generate example input
    w1, w2, w3 = randn(10,10), randn(10,10), randn(1,10)
    x1 = randn(10)    
    inputs = [:w1 => w1, :w2 => w2, :w3 => w3, :x1 => x1]
    dex = xdiff(ex; ctx=Dict(:codegen => EinCodeGen()), inputs...)
    
    
    # ANN input parameter types
    types = (Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64})
    

    
    dann = ReverseDiffSource.xdiff(ann, types)
    _, rds_dw1_val, _, _, _ = dann(w1, w2, w3, x1)
    
    dw1, _, _, _ = fdiff(ann, types)
    dw1_val = dw1(w1, w2, w3, x1)
    
    if isapprox(dw1_val, rds_dw1_val)
        println("Equal!")
    else
        error("Not equal!")
    end

end


quote 
    dcost_dcost = 1.0
    dcost_dreconstructedInput = ones((size(Wd, 1),))
    dcost_db2 = dcost_dreconstructedInput .* Wd
    dcost_dtmp713 = dcost_dreconstructedInput .* Wd
    dcost_dtmp711 = dcost_dtmp713 .* We2
    dcost_dx__1 = -.(dcost_dcost)
    dcost_db1 = dcost_dtmp713 .* We2
    dcost_dx__2 = We1' * dcost_dtmp711
    dcost_dx = dcost_dx__1 .+ dcost_dx__2
    dcost_dWe1 = dcost_dtmp711 * x'
    tmp711 = We1 * x
    firstLayer = +.(tmp711, b1)
    dcost_dWe2 = dcost_dtmp713 * firstLayer'
    tmp713 = We2 * firstLayer
    encodedInput = +.(tmp713, b2)
    reconstructedInput = Wd * encodedInput
    cost = sum(-.(reconstructedInput, x))
    dcost_dWd = dcost_dreconstructedInput * encodedInput'
    tmp733 = (cost, dcost_dWe1, dcost_dWe2, dcost_dWd, dcost_db1, dcost_db2, dcost_dx)
end


function main_cw()
    
    ex = :(z = myexp.(x) + 1)
    ctx = Dict(:codegen => EinCodeGen())
    inputs = [:x => rand(2)]
    dex = xdiff(ex; ctx=ctx, inputs...)
end



include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end
    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end

logistic(x) = 1 ./ (1 + exp.(-x))
@scalardiff logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))


function find_bad(g)
    for i=1:length(g.tape)
        println("Evaliating $(i)th node $(g[i])")
        evaluate!(g, g[i])
    end
end


function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput .- x) .^ 2.0)
    return cost
end


using GPUArrays


function main_873()
    f = autoencoder_cost
    We1 = rand(2000, 10_000); b1 = rand(2000); We2 = rand(1000, 2000); b2 = rand(1000);
    Wd = rand(10_000, 1000); x = rand(10_000, 100);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];
    d_inputs = [k => GPUArray(v) for (k, v) in inputs]

    ctx = Dict()    
    df = xdiff(f; ctx=Dict(:codegen=>VectorCodeGen()), inputs...)


    ex = quote
        firstLayer = logistic(We1 * x .+ b1)
        encodedInput = logistic(We2 * firstLayer .+ b2)
        reconstructedInput = logistic(Wd * encodedInput)
        cost = sum((reconstructedInput .- x) .^ 2.0)        
    end
    dex = xdiff(ex; ctx=ctx, inputs...)
    

    # TODO:
    # 1. add warning that special functions may not be followed by sum
    # 2. create an issue for this and think out preprocessing that addes `1.0 .*`
    #    to all special functions followed by sum (act on graph level?)

    # 2. create conventional VAE to test the current state of the package
end

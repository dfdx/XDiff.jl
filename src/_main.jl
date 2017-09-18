

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

#########################################################

using GPUArrays


logistic(x) = 1 ./ (1 + exp.(-x))
@scalardiff logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))


function gpu_test()
    ex = quote
        firstLayer = logistic(We1 * x .+ b1)
        encodedInput = logistic(We2 * firstLayer .+ b2)
        reconstructedInput = logistic(Wd * encodedInput)
        cost = sum((reconstructedInput .- x) .^ 2.0)
    end

    FT = Float64
    We1 = rand(FT, 200, 1000); b1 = rand(FT, 200);
    We2 = rand(FT, 100, 200); b2 = rand(FT, 100);
    Wd = rand(FT, 1000, 100); x = rand(FT, 1000, 100);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];

    ctx = Dict()
    dex = xdiff(ex; ctx=ctx, inputs...)
    eval(dex)

    ctx = Dict(:codegen => CuCodeGen(:mem))
    dex = xdiff(ex; ctx=ctx, inputs...)

    mem = Dict()
    We1 = GPUArray(We1);  b1 = GPUArray(b1)
    We2 = GPUArray(We2); b2 = GPUArray(b2)
    Wd = GPUArray(Wd); x = GPUArray(x)
    eval(dex)
end



# -------------------------------------------------





function main_0193()
    ex = quote 
        xx2 = w1 * x1               # 200 x 10
        x2 = log.(1.0 + exp.(xx2))  # 200 x 10
        xx3 = w2 * x2               # 100 x 10
        x3 = log.(1.0 + exp.(xx3))  # 100 x 10
        x4 = w3 * x3                # 100 x 10
        sum(1.0 ./ (1.0 + exp.(-x4)))  # 1
    end
    w1=rand(200, 1000); w2=rand(100, 200); w3=rand(100, 100); x1=rand(1000, 10);
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1];
    ctx = Dict()

    dex = xdiff(ex; ctx=ctx, inputs...)
end


# using GPUArrays



function main_1923()
    x = GPUArray(rand(784, 100))
    y = GPUArray(rand(784, 100))
    z = GPUArray(zeros(784, 100))
    z .= log.(y)
end








using Distributions

function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end




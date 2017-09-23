

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


function predict(W, b, x)
    return W * x .+ b
end

function predict_cost(W, b, x, y)
    sum(predict(W, x, b) .- y)
end


function main_1642()
    W = rand(30, 64); b = rand(30); x = rand(64, 10)
    inputs = [:W => W, :b => b, :x => x]
    ctx = Dict(:codegen => EinCodeGen())
    ex = :(W * x .+ b)
    
    dex = xdiff(ex; ctx=ctx, inputs...)
end

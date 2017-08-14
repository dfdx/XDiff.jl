

include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end
    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end

logistic(x) = 1 ./ (1 + exp.(-x))
@diff_rule logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))


function find_bad(g)
    for i=1:length(g.tape)
        println("Evaliating $(i)th node $(g[i])")
        evaluate!(g, g[i])
    end
end


function main_873()    
    Wxh = randn(3, 4); Whh = randn(3,3); Why = randn(5,3)
    hprev = randn(3); h = rand(3)
    x = randn(4); y = rand(5)
    ctx = Dict(:cost => :cost)
    inputs = [:Wxh => Wxh, :Whh => Whh, :Why => Why, :hprev => hprev, :h => h, :x => x, :y => y]
        
    ex = quote    
        h = tanh.(Whh * hprev + Wxh * x)
        yhat = Why * h
        cost = sum((yhat .- y) .^ 2.0)
        h, cost
    end
    
    
    dex = xdiff(ex; inputs...)

end

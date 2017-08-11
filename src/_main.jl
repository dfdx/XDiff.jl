

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
    We1 = rand(20, 100); b1 = rand(20); We2 = rand(10, 20); b2 = rand(10);
    Wd = rand(100, 10); x = rand(100, 10);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];
    ctx = Dict()
    
    ex = quote
        firstLayer = logistic(We1 * x .+ b1)
        encodedInput = logistic(We2 * firstLayer .+ b2)
        reconstructedInput = logistic(Wd * encodedInput)
        cost = sum(reconstructedInput .- x .^ 2.0)
    end

    We = rand(3,4); Wd = rand(4,3); x = rand(4,2)
    inputs = [:We => We, :Wd => Wd, :x => x]
    ex = quote
        y = We * x
        xd = Wd * y
        cost = sum(xd .- x)
    end

    # TODO: g.ctx[:rsizes][:dcost!dxd] is ()
    # but should be (4, 2)
    
    ctx = Dict()
    dex = xdiff(ex; inputs...)

end

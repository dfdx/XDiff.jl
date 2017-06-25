
include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end
    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end


function main_873()
    ex = :(sum(x + 1))
    ctx = Dict()
    inputs = [:x => rand(2)]

    
    ex = quote
        Y = conv2(X, W)
        Z = sum(Y)
    end
    X = rand(5,5)
    W = rand(3,3)
    inputs = [:X => X, :W => X]


    dex = xdiff(ex; inputs...)
end

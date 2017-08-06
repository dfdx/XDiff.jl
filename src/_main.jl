

include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end
    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end


function main_873()
    ex = :(log(sum(x)))
    
    x = rand(2)
    inputs = [:x=>x]
    ctx = Dict()

    dex = xdiff(ex; inputs...)
    
end


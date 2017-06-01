
include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end

    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end


function main_873()
    ex = :(sum(x * n))
    x = [1.0, 1.0]
    n = 2.0
    inputs = [:x => x, :n => n]
    ctx = Dict()
    
    dex = xdiff(ex; inputs...)
end

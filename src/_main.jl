

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


function main_873()
    u = randn(3, 3)
    v = randn(3, 3)
    inputs = [:u => u, :v => v]
    # ctx = Dict(:codegen => EinCodeGen())
    ctx = Dict()
    ex = quote
        x = u .+ v
        y = foo(x)
        z = sum(1.0 .* y)
    end



    # TODO:
    # 1. add warning that special functions may not be followed by sum
    # 2. create an issue for this and think out preprocessing that addes `1.0 .*`
    #    to all special functions followed by sum (act on graph level?)

    # TODO:
    # 1. define @scalardiff, @tensordiff and @specialdiff
    # 2. finally try to create conventional VAE
    
    dex = xdiff(ex; ctx=ctx, inputs...)

end

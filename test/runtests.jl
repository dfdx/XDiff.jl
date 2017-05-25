using XDiff
using Espresso
using ReverseDiff
using Base.Test


macro rdcmp(f, params...)    
    inputs = [param.args[1] => eval(param.args[2]) for param in params]
    # println(inputs)
    ret = Expr(:let, Expr(:block))    
    body = quote
        # inputs = $([inp.args[1] => inp.args[2] for inp in _inputs])
        vals = [inp[2] for inp in $inputs]
        types = ([typeof(inp[2]) for inp in $inputs]...)        
        df = fdiff($f, types)
        dvals = df(vals...)

        rd_dvals = ReverseDiff.gradient($f, (vals...,))
        @test dvals[2:end] == rd_dvals
    end
    for arg in body.args
        push!(ret.args[1].args, arg)
    end
    return ret
end



# include("rdiff_test.jl")
# include("autoencoder_reversediff.jl")

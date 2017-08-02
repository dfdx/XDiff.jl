using XDiff
using Espresso: @get_or_create
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Base.Test
using BenchmarkTools

macro rdcmp(f, params...)    
    inputs = [param.args[1] => eval(param.args[2]) for param in params]
    ret = Expr(:let, Expr(:block))    
    body = quote
        # inputs = $([inp.args[1] => inp.args[2] for inp in _inputs])
        vals = [inp[2] for inp in $inputs]        
        df = xdiff($f, vals)
        dvals = df(vals...)
        dvals_a = [dvals...]

        rd_dvals = ReverseDiff.gradient($f, (vals...,))
        rd_dvals_a = [rd_dvals...]
        @test isapprox(dvals_a[2:end], rd_dvals_a)
    end
    for arg in body.args
        push!(ret.args[1].args, arg)
    end
    return ret
end


function perf_test(f; inputs...)
    vals = ([val for (name, val) in inputs]...)
    println("Compiling derivatives using XDiff")
    @time df = xdiff(f; inputs...)
    mem = Dict()
    println("Testing XDiff...")
    @benchmark df(vals...; mem=$mem)
    
    f_tape = GradientTape(f, vals)
    compiled_f_tape = compile(f_tape)    
    cfg = GradientConfig(vals)
    results = map(similar, vals)
    println("Testing ReverseDiff...")
    # @benchmark gradient!(results, compiled_f_tape, vals)
    @benchmark gradient!(results, f_tape, vals)
end


# include("linreg.jl")
include("ann.jl")
# include("autoencoder.jl")
# include("others.jl")

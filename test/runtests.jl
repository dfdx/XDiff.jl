using XDiff
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Base.Test


function test_compare(f; inputs...)
    vals = ([val for (name, val) in inputs]...)
    df = xdiff(f; inputs...)
    dvals = df(vals...)
    dvals_a = [dvals...]

    f_tape = GradientTape(f, vals)
    compiled_f_tape = compile(f_tape)
    cfg = GradientConfig(vals)
    results = map(similar, vals)
    gradient!(results, compiled_f_tape, vals)
    results_a = [results...]
    @test isapprox(results_a,  dvals_a[2:end])
end


# include("linreg.jl")
include("ann.jl")
# include("autoencoder.jl")
# include("others.jl")

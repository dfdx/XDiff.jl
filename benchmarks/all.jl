
using XDiff
using Espresso: @get_or_create
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using BenchmarkTools

include("functions.jl")


function perf_test(f; compile_tape=true, inputs...)
    vals = ([val for (name, val) in inputs]...)
    println("Compiling derivatives using XDiff")
    @time df = xdiff(f; inputs...)
    mem = Dict()
    println("Testing XDiff...")
    @benchmark $df($vals...; mem=$mem)
    
    f_tape = GradientTape(f, vals)
    if compile_tape
        compiled_f_tape = compile(f_tape)
    end
    cfg = GradientConfig(vals)
    results = map(similar, vals)
    println("Testing ReverseDiff...")
    if compile_tape
        @benchmark gradient!($results, $compiled_f_tape, $vals)
    else
        @benchmark gradient!($results, $f_tape, $vals)
    end
end




function benchmark_autoencoder()
    f = autoencoder_cost
    We1 = rand(2000, 10_000); b1 = rand(2000); We2 = rand(1000, 2000); b2 = rand(1000);
    Wd = rand(10_000, 1000); x = rand(10_000, 100);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];
    perf_test(f; inputs...)

    We1 = rand(200, 1000); b1 = rand(200); We2 = rand(100, 200); b2 = rand(100);
    Wd = rand(1000, 100); x = rand(1000, 100);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];    
    perf_test(f; compile_tape=false, inputs...)
end


function benchmark_ann1()
    f = ann1
    w1=rand(2000, 10000); w2=rand(1000, 2000); w3=rand(1000, 1000); x1=rand(10000, 500);
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1];    
    perf_test(f; inputs...)

    w1=rand(200, 1000); w2=rand(100, 200); w3=rand(100, 100); x1=rand(1000, 10);
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1];    
    perf_test(f; inputs...)
end

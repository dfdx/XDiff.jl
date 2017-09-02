

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
    We1 = randn(500, 784); be1 = randn(500);
    We2 = randn(500, 500); be2 = randn(500);
    We3 = randn(20, 500); be3 = randn(20);
    We4 = randn(20, 500); be4 = randn(20);
    Wd1 = randn(500, 20); bd1 = randn(500);
    Wd2 = randn(500, 500); bd2 = randn(500);
    Wd3 = randn(784, 500); bd3 = randn(784);
    x = rand(784, 100); eps = rand(Normal(0, 1),  20)
    inputs = [:We1 => We1, :We2 => We2, :We3 => We3, :We4 => We4,
              :Wd1 => Wd1, :Wd2 => Wd2, :Wd3 => Wd3, :eps => eps, :x => x,
              :be1 => be1, :be2 => be2, :be3 => be3, :be4 => be4,
              :bd1 => bd1, :bd2 => bd2, :bd3 => bd3]
    # ctx = Dict(:codegen => EinCodeGen())
    ctx = Dict()
    
    ex = quote
        he1 = tanh.(We1 * x) .+ be1
        he2 = tanh.(We2 * he1) .+ be2
        mu = We3 * he2 .+ be3
        log_sigma2 = We4 * he2 .+ be4
        z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
        # decoder
        hd1 = tanh.(Wd1 * z .+ bd1)
        hd2 = tanh.(Wd2 * hd1 .+ bd2)
        x_rec = logistic.(Wd3 * hd2 .+ bd3)
        # loss
        rec_loss = sum(x .* log.(1e-10 + x_rec) + (1 - x) .* log.(1e-10 + 1 - x_rec), 1)
        latent_loss = -0.5 * sum(1 + log_sigma2 .- mu .^ 2 - exp.(log_sigma2), 1)
        cost = sum(rec_loss .+ latent_loss)
    end
   
    dex = xdiff(ex; ctx=ctx, inputs...)
    eval(dex)    
    

end


using Distributions

function main_2w51()
    We1 = randn(500, 784); be1 = randn(500);
    We2 = randn(500, 500); be2 = randn(500);
    We3 = randn(20, 500); be3 = randn(20);
    We4 = randn(20, 500); be4 = randn(20);
    Wd1 = randn(500, 20); bd1 = randn(500);
    Wd2 = randn(500, 500); bd2 = randn(500);
    Wd3 = randn(784, 500); bd3 = randn(784);
    x = randn(784, 100); eps = rand(Normal(0, 1),  20)

    inputs = [:We1 => We1, :be1 => be1, :We2 => We2, :be2 => be2, :We3 => We3, :be3 => be3,
              :We4 => We4, :be4 => be4,
              :Wd1 => Wd1, :bd1 => bd1, :Wd2 => Wd2, :bd2 => bd2, :Wd3 => Wd3, :bd3 => bd3,
              :eps => eps, :x => x]
    vals = [inp[2] for inp in inputs]
    

    ex = quote
        # encoder
        he1 = tanh.(We1 * x .+ be1)
        he2 = tanh.(We2 * he1 .+ be2)
        mu = We3 * he2 .+ be3
        log_sigma2 = We4 * he2 .+ be4
        z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
        # decoder
        hd1 = tanh.(Wd1 * z .+ bd1)
        hd2 = tanh.(Wd2 * hd1 .+ bd2)
        x_rec = logistic.(Wd3 * hd2 .+ bd3)
        # loss
        rec_loss = sum(x .* log.(1e-10 + x_rec) + (1 - x) .* log.(1e-10 + 1 - x_rec))
        latent_loss = -0.5 * sum(1 + log_sigma2 .- mu .^ 2 - exp.(log_sigma2))
        cost = sum(rec_loss + latent_loss)
    end
    dex = xdiff(ex; inputs...)
end



quote
    tmp846 = @get_or_create(mem, :tmp846, zero(Int64))
    tmp849 = @get_or_create(mem, :tmp849, zeros(Float64, (3,)))
    tmp852 = @get_or_create(mem, :tmp852, zero(Float64))
    tmp853 = @get_or_create(mem, :tmp853, zero(Float64))
    tmp856 = @get_or_create(mem, :tmp856, zero(Float64))
    tmp854 = @get_or_create(mem, :tmp854, zeros(Float64, (3,)))
    dz!dx = @get_or_create(mem, :dz!dx, zeros(Float64, (3,)))
    tmp851 = @get_or_create(mem, :tmp851, zeros(Float64, (3,)))
    dz!dx__2 = @get_or_create(mem, :dz!dx__2, zeros(Float64, (3,)))
    dz!dx__1 = @get_or_create(mem, :dz!dx__1, zeros(Float64, (3,)))
    dz!dz = @get_or_create(mem, :dz!dz, zero(Float64))
    z = @get_or_create(mem, :z, zero(Float64))
    dz!dtmp846 = @get_or_create(mem, :dz!dtmp846, zero(Float64))
    dz!dz = 1.0
    tmp856 = 0.0
    tmp852 = -2.0
    tmp846 = length(x)
    tmp854 .= (.-)(x) .* tmp846 .^ tmp852
    dz!dtmp846 = sum(tmp854)
    dz!dx .= dz!dz .* (x ./ tmp846) .+ dz!dtmp846 .* tmp856
    z = sum(x ./ tmp846)
    tmp859 = (z, dz!dx)
end

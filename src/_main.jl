

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

#########################################################



function main_0193()
    ex = quote 
        xx2 = w1 * x1               # 200 x 10
        x2 = log.(1.0 + exp.(xx2))  # 200 x 10
        xx3 = w2 * x2               # 100 x 10
        x3 = log.(1.0 + exp.(xx3))  # 100 x 10
        x4 = w3 * x3                # 100 x 10
        sum(1.0 ./ (1.0 + exp.(-x4)))  # 1
    end
    w1=rand(200, 1000); w2=rand(100, 200); w3=rand(100, 100); x1=rand(1000, 10);
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1];
    ctx = Dict()

    dex = xdiff(ex; ctx=ctx, inputs...)
end


# using GPUArrays



function main_1923()
    x = GPUArray(rand(784, 100))
    y = GPUArray(rand(784, 100))
    z = GPUArray(zeros(784, 100))
    z .= log.(y)
end








using Distributions

function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


function ae(We, Wd, x)
    z = We * x
    x_rec = exp.(Wd * z)
    rec_loss_mat = x .* log.(x_rec)
    rec_loss = sum(rec_loss_mat, 1)
    latent_loss = sum(z, 1)
    cost = mean(latent_loss .+ rec_loss)
end


function main_3445()
    # works
    ex = quote
        z = We * x
        x_rec = exp.(Wd * z)
        rec_loss_mat = x .* log.(x_rec)
        rec_loss = sum(rec_loss_mat, 1)
        latent_loss = sum(z, 1)
        cost = mean(latent_loss .+ rec_loss)
    end
    # doesn't work
        z = We * x
        a = Wd * z
        x_rec = exp.(a)
        b = log.(x_rec)
        rec_loss_mat = x .* b
        rec_loss = sum(rec_loss_mat, 1)
        c = sum(z, 1)
        latent_loss = -0.5 * c
        d = latent_loss .+ rec_loss
        cost = mean(d)
    end


    We = xavier_init(20, 784)
    Wd = xavier_init(784, 20);
    x = rand(784, 100)    
    We = GPUArray(We); Wd = GPUArray(Wd); x = GPUArray(x)
    mem = Dict()
    
    
    inputs = [:We => We, :Wd => Wd, :x => x]

    ctx = Dict(:codegen => CuCodeGen(:mem))
    dex = xdiff(ex; ctx=ctx, inputs...)
    

    
    rd_vals = rd_diff(f; inputs...)

    

    # tmp1310 = 1
    # tmp1298 = 1
    # tmp1282 = -0.5
    # dcost!dlatent_loss[i,j] = 1.0
end




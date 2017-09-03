

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


function main_1923()
    ex = quote
        a = W * x
        b = sum(a, 1)
        c = 2b
        z = sum(c)
    end
    x = rand(2,3); W = rand(5, 2)
    inputs = [:W => W, :x => x]

    g = proper_graph(ex; inputs...)
    dg = _xdiff(g)
end








using Distributions

function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end

function main_2w51()
    We1 = xavier_init(500, 784); be1 = randn(500);
    We2 = xavier_init(500, 500); be2 = randn(500);
    We3 = xavier_init(20, 500); be3 = randn(20);
    We4 = xavier_init(20, 500); be4 = randn(20);
    Wd1 = xavier_init(500, 20); bd1 = randn(500);
    Wd2 = xavier_init(500, 500); bd2 = randn(500);
    Wd3 = xavier_init(784, 500); bd3 = randn(784);
    x = rand(784, 100); eps = rand(Normal(0, 1),  20)
    inputs = [:We1 => We1, :be1 => be1, :We2 => We2, :be2 => be2,
              :We3 => We3, :be3 => be3, :We4 => We4, :be4 => be4,
              :Wd1 => Wd1, :bd1 => bd1, :Wd2 => Wd2, :bd2 => bd2,
              :Wd3 => Wd3, :bd3 => bd3, :eps => eps, :x => x]
    vals = [inp[2] for inp in inputs]


    ex = quote
        dummy = 42.0
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
        rec_loss = sum(x .* log.(1e-10 + x_rec) + (1 - x) .* log.(1e-10 + 1 - x_rec), 1)
        latent_loss = -0.5 * sum(1 + log_sigma2 .- mu .^ 2 - exp.(log_sigma2), 1)
        cost = mean(rec_loss .+ latent_loss)
    end
    
    dex = xdiff(ex; inputs...)
    mem = Dict()
    eval(dex)
end


using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile


function rd_diff(f; inputs...)
    vals = ([val for (name, val) in inputs]...)
    f_tape = GradientTape(f, vals)
    compiled_f_tape = compile(f_tape)
    cfg = GradientConfig(vals)
    results = map(similar, vals)
    gradient!(results, compiled_f_tape, vals)
    results
end


f(We, Wd, x) = begin
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


    We = xavier_init(20, 784);
    Wd = xavier_init(784, 20);
    x = rand(784, 100)
    inputs = [:We => We, :Wd => Wd, :x => x]

    rd_vals = rd_diff(f; inputs...)

    g = proper_graph(ex; inputs...)
    dg = _xdiff(g)
    rg = cat(g, dg)
    outvars = unshift!([deriv_name(g.ctx[:z_var], var) for (var, _) in inputs], varname(g[end]))
    push!(rg, :tuple, Espresso.genname(), Expr(:tuple, outvars...))
    rg = topsort(rg)
    infer_deriv_size!(rg)  # need to know size to evaluate things like `dz!dx[i] = 1.0`
    evaluate!(rg)


    # tmp1310 = 1
    # tmp1298 = 1
    # tmp1282 = -0.5
    # dcost!dlatent_loss[i,j] = 1.0
end

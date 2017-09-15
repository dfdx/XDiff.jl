
using Distributions

function vae_cost(We1, be1, We2, be2, We3, be3, We4, be4, Wd1, bd1, Wd2, bd2, Wd3, bd3, eps, x)
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

function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


let
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

    test_compare(vae_cost; inputs...)
end

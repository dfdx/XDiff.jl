

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

using GPUArrays



function ann(W1, W2, b1, b2, x, y)
    l1 = W1 * x .+ b1
    l2 = W2 * l1 .+ b2
    cost = sum((l2 .- y) .^ 2)
end


function main_873()
    x = randn(Float32, 10_000, 1000);
    W1 = randn(Float32, 2000, 10_000); b1 = randn(Float32, 2000, 1000)
    W2 = randn(Float32, 2000, 2000); b2 = randn(Float32, 2000, 1000)
    y = randn(Float32, 2000, 1000)

    inputs = [:x => x, :W1 => W1, :b1 => b1, :W2 => W2, :b2 => b2, :y => y]
    
    ctx = Dict()
    df = xdiff(ann; ctx=ctx, inputs...)


    ex = quote
        l1 = W1 * x .+ b1
        l2 = W2 * l1 .+ b2
        cost = sum((l2 .- y) .^ 2)
    end
    dex = xdiff(ex; ctx=ctx, inputs...)


    hW1, hW2, hb1, hb2, hx, hy = W1, W2, b1, b2, x, y
    dW1, dW2, db1, db2, dx, dy = map(GPUArray, (W1, W2, b1, b2, x, y))

    W1, W2, b1, b2, x, y = dW1, dW2, db1, db2, dx, dy

    # TODO:
    # 1. add warning that special functions may not be followed by sum
    # 2. create an issue for this and think out preprocessing that addes `1.0 .*`
    #    to all special functions followed by sum (act on graph level?)

    # 2. create conventional VAE to test the current state of the package
end


function dann(W1, W2, b1, b2, x, y; mem=Dict())
    tmp945 = @get_or_create(mem, :tmp945, zeros(Float32, (2000, 1000)))
    dcost!dy = @get_or_create(mem, :dcost!dy, zeros(Float32, (2000, 1000)))
    tmp929 = @get_or_create(mem, :tmp929, zeros(Float32, (2000, 1000)))
    l2 = @get_or_create(mem, :l2, zeros(Float32, (2000, 1000)))
    dcost!db1 = @get_or_create(mem, :dcost!db1, zeros(Float32, (2000, 1000)))
    dcost!db2 = @get_or_create(mem, :dcost!db2, zeros(Float32, (2000, 1000)))
    tmp925 = @get_or_create(mem, :tmp925, zeros(Float32, (2000, 1000)))
    l1 = @get_or_create(mem, :l1, zeros(Float32, (2000, 1000)))
    tmp927 = @get_or_create(mem, :tmp927, zeros(Float32, (2000, 1000)))
    tmp946 = @get_or_create(mem, :tmp946, zeros(Float32, (2000, 1000)))
    cost = @get_or_create(mem, :cost, zero(Float32))
    dcost!dW1 = @get_or_create(mem, :dcost!dW1, zeros(Float32, (2000, 10000)))
    tmp944 = @get_or_create(mem, :tmp944, zero(Int64))
    tmp930 = @get_or_create(mem, :tmp930, zero(Int64))
    dcost!dx = @get_or_create(mem, :dcost!dx, zeros(Float32, (10000, 1000)))
    tmp943 = @get_or_create(mem, :tmp943, zero(Int64))
    dcost!dW2 = @get_or_create(mem, :dcost!dW2, zeros(Float32, (2000, 2000)))
    tmp943 = 1
    tmp930 = 2
    A_mul_B!(tmp925, W1, x)
    l1 .= tmp925 .+ b1
    A_mul_B!(tmp927, W2, l1)
    tmp929 .= (tmp927 .+ b2) .- y
    cost = sum(tmp929 .^ 2)
    dcost!db2 .= tmp943 .* (tmp930 .* ((tmp927 .+ b2) .- y) .^ (tmp930 .- tmp943))
    At_mul_B!(dcost!db1, W2, dcost!db2)
    At_mul_B!(dcost!dx, W1, dcost!db1)
    A_mul_Bt!(dcost!dW1, dcost!db1, x)
    A_mul_Bt!(dcost!dW2, dcost!db2, l1)
    dcost!dy .= (.-)(tmp943 .* (tmp930 .* ((tmp927 .+ b2) .- y) .^ (tmp930 .- tmp943)))
    tmp953 = (cost, dcost!dx, dcost!dW1, dcost!db1, dcost!dW2, dcost!db2, dcost!dy)
end


function dann_gpu(W1, W2, b1, b2, x, y; mem=Dict())
    tmp945 = @get_or_create(mem, :tmp945, GPUArray(zeros(Float32, (2000, 1000))))
    dcost!dy = @get_or_create(mem, :dcost!dy, GPUArray(zeros(Float32, (2000, 1000))))
    tmp929 = @get_or_create(mem, :tmp929, GPUArray(zeros(Float32, (2000, 1000))))
    l2 = @get_or_create(mem, :l2, GPUArray(zeros(Float32, (2000, 1000))))
    dcost!db1 = @get_or_create(mem, :dcost!db1, GPUArray(zeros(Float32, (2000, 1000))))
    dcost!db2 = @get_or_create(mem, :dcost!db2, GPUArray(zeros(Float32, (2000, 1000))))
    tmp925 = @get_or_create(mem, :tmp925, GPUArray(zeros(Float32, (2000, 1000))))
    l1 = @get_or_create(mem, :l1, GPUArray(zeros(Float32, (2000, 1000))))
    tmp927 = @get_or_create(mem, :tmp927, GPUArray(zeros(Float32, (2000, 1000))))
    tmp946 = @get_or_create(mem, :tmp946, GPUArray(zeros(Float32, (2000, 1000))))
    cost = @get_or_create(mem, :cost, zero(Float32))
    dcost!dW1 = @get_or_create(mem, :dcost!dW1, GPUArray(zeros(Float32, (2000, 10000))))
    tmp944 = @get_or_create(mem, :tmp944, zero(Int64))
    tmp930 = @get_or_create(mem, :tmp930, zero(Int64))
    dcost!dx = @get_or_create(mem, :dcost!dx, GPUArray(zeros(Float32, (10000, 1000))))
    tmp943 = @get_or_create(mem, :tmp943, zero(Int64))
    dcost!dW2 = @get_or_create(mem, :dcost!dW2, GPUArray(zeros(Float32, (2000, 2000))))
    tmp943 = 1
    tmp930 = 2
    A_mul_B!(tmp925, W1, x)
    l1 .= tmp925 .+ b1
    A_mul_B!(tmp927, W2, l1)
    tmp929 .= (tmp927 .+ b2) .- y
    cost = sum(tmp929 .^ 2)
    dcost!db2 .= tmp943 .* (tmp930 .* ((tmp927 .+ b2) .- y))
    At_mul_B!(dcost!db1, W2, dcost!db2)
    At_mul_B!(dcost!dx, W1, dcost!db1)
    A_mul_Bt!(dcost!dW1, dcost!db1, x)
    A_mul_Bt!(dcost!dW2, dcost!db2, l1)
    dcost!dy .= (.-)(tmp943 .* (tmp930 .* ((tmp927 .+ b2) .- y)))
    # GPUArrays.synchronize(cost)
    tmp953 = (cost, dcost!dx, dcost!dW1, dcost!db1, dcost!dW2, dcost!db2, dcost!dy)    
end

    

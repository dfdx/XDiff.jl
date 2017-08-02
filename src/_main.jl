

include("core.jl")

function load_espresso()
    for n in Base.names(Espresso, true) @eval import Espresso: $n end
    for n in Base.names(XDiff, true) @eval import XDiff: $n end
end


function main_873()
    ex = quote
        xx2 = w1 * x1
        x2 = log.(1. + exp.(xx2))   # soft RELU unit
        xx3 = w2 * x2
        x3 = log.(1. + exp.(xx3))   # soft RELU unit
        x4 = w3 * x3
        sum(1. ./ (1. + exp.(-x4)))  # sigmoid output
    end

    w1=rand(2000, 10000)
    w2=rand(1000, 2000)
    w3=rand(1000, 1000)
    x1=rand(10000)
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1]
    ctx = Dict()

    dex = xdiff(ex; inputs...)
    b_dex = blassify(dex; inputs...)
    
end




function ann(w1, w2, w3, x1)    
    xx2 = w1 * x1
    x2 = log.(1. + exp.(xx2))   # soft RELU unit
    xx3 = w2 * x2
    x3 = log.(1. + exp.(xx3))   # soft RELU unit
    x4 = w3 * x3
    sum(1. ./ (1. + exp.(-x4)))  # sigmoid output
end


function perf_test_890()
    xx2 = @get_or_create(mem, :xx2, zeros(Float64, (2000,)))
    tmp1071 = @get_or_create(mem, :tmp1071, zeros(Float64, (2000,)))
    dtmp1034_dx3 = @get_or_create(mem, :dtmp1034_dx3, zeros(Float64, (1000,)))
    tmp1031 = @get_or_create(mem, :tmp1031, zeros(Float64, (1000,)))
    dtmp1034_dtmp1031 = @get_or_create(mem, :dtmp1034_dtmp1031, zeros(Float64, (1000,)))
    x2 = @get_or_create(mem, :x2, zeros(Float64, (2000,)))
    dtmp1034_dx1 = @get_or_create(mem, :dtmp1034_dx1, zeros(Float64, (10000,)))
    tmp1030 = @get_or_create(mem, :tmp1030, zeros(Float64, (1000,)))
    dtmp1034_dxx3 = @get_or_create(mem, :dtmp1034_dxx3, zeros(Float64, (1000,)))
    tmp1064 = @get_or_create(mem, :tmp1064, zeros(Float64, (1000,)))
    tmp1054 = @get_or_create(mem, :tmp1054, zero(Float64))
    x3 = @get_or_create(mem, :x3, zeros(Float64, (1000,)))
    dtmp1034_dtmp1019 = @get_or_create(mem, :dtmp1034_dtmp1019, zeros(Float64, (2000,)))
    x4 = @get_or_create(mem, :x4, zeros(Float64, (1000,)))
    dtmp1034_dtmp1030 = @get_or_create(mem, :dtmp1034_dtmp1030, zeros(Float64, (1000,)))
    dtmp1034_dw1 = @get_or_create(mem, :dtmp1034_dw1, zeros(Float64, (2000, 10000)))
    tmp1019 = @get_or_create(mem, :tmp1019, zeros(Float64, (2000,)))
    dtmp1034_dtmp1024 = @get_or_create(mem, :dtmp1034_dtmp1024, zeros(Float64, (1000,)))
    dtmp1034_dw3 = @get_or_create(mem, :dtmp1034_dw3, zeros(Float64, (1000, 1000)))
    dtmp1034_dx2 = @get_or_create(mem, :dtmp1034_dx2, zeros(Float64, (2000,)))
    dtmp1034_dxx2 = @get_or_create(mem, :dtmp1034_dxx2, zeros(Float64, (2000,)))
    tmp1056 = @get_or_create(mem, :tmp1056, zeros(Float64, (1000,)))
    tmp1034 = @get_or_create(mem, :tmp1034, zero(Float64))
    tmp1024 = @get_or_create(mem, :tmp1024, zeros(Float64, (1000,)))
    tmp1055 = @get_or_create(mem, :tmp1055, zeros(Float64, (1000,)))
    tmp1025 = @get_or_create(mem, :tmp1025, zeros(Float64, (1000,)))
    tmp1018 = @get_or_create(mem, :tmp1018, zero(Float64))
    dtmp1034_dw2 = @get_or_create(mem, :dtmp1034_dw2, zeros(Float64, (1000, 2000)))
    tmp1020 = @get_or_create(mem, :tmp1020, zeros(Float64, (2000,)))
    tmp1053 = @get_or_create(mem, :tmp1053, zero(Float64))
    tmp1032 = @get_or_create(mem, :tmp1032, zeros(Float64, (1000,)))
    xx3 = @get_or_create(mem, :xx3, zeros(Float64, (1000,)))
    dtmp1034_dx4 = @get_or_create(mem, :dtmp1034_dx4, zeros(Float64, (1000,)))
    A_mul_B!(xx2, w1, x1)
    tmp1018 = 1.0
    x2 .= log.(tmp1018 .+ exp.(xx2))
    A_mul_B!(xx3, w2, x2)
    x3 .= log.(tmp1018 .+ exp.(xx3))
    A_mul_B!(x4, w3, x3)
    tmp1032 .= tmp1018 .+ exp.((.-)(x4))
    tmp1034 = sum(1.0 ./ tmp1032)
    tmp1054 = -2.0
    dtmp1034_dx4 .= (.-)((tmp1018 .* ((.-)(tmp1018) .* (tmp1018 .+ exp.((.-)(x4))) .^ tmp1054)) .* exp.((.-)(x4)))
    A_mul_Bt!(dtmp1034_dw3, dtmp1034_dx4, x3)
    At_mul_B!(dtmp1034_dx3, w3, dtmp1034_dx4)
    dtmp1034_dxx3 .= (dtmp1034_dx3 .* (tmp1018 ./ (tmp1018 .+ exp.(xx3)))) .* exp.(xx3)
    A_mul_Bt!(dtmp1034_dw2, dtmp1034_dxx3, x2)
    At_mul_B!(dtmp1034_dx2, w2, dtmp1034_dxx3)
    dtmp1034_dxx2 .= (dtmp1034_dx2 .* (tmp1018 ./ (tmp1018 .+ exp.(xx2)))) .* exp.(xx2)
    A_mul_Bt!(dtmp1034_dw1, dtmp1034_dxx2, x1)
    At_mul_B!(dtmp1034_dx1, w1, dtmp1034_dxx2)
    tmp1077 = (tmp1034, dtmp1034_dw1, dtmp1034_dw2, dtmp1034_dw3, dtmp1034_dx1)
end

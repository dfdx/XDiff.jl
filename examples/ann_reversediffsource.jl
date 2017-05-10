
using XDiff
import ReverseDiffSource

function ann(w1, w2, w3, x1)
    x2 = w1 * x1
    # x2 = log.(1. + exp.(_x2))   # soft RELU unit
    x3 = w2 * x2
    # x3 = log.(1. + exp.(_x3))   # soft RELU unit
    x4 = sum.(w3 * x3)
    # return 1. ./ (1. + exp.(-x4))  # sigmoid output
    return x4
end

# function ann(w1, w2, w3, x1)
#     _x2 = w1 * x1
#     x2 = log.(1. + exp.(_x2))   # soft RELU unit
#     _x3 = w2 * x2
#     x3 = log.(1. + exp.(_x3))   # soft RELU unit
#     x4 = sum.(w3 * x3)
#     return 1. ./ (1. + exp.(-x4))  # sigmoid output
# end


# ANN input parameter types
types = (Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64})

# generate example input
w1, w2, w3 = randn(10,10), randn(10,10), randn(1,10)
x1 = randn(10)

dann = ReverseDiffSource.rdiff(ann, types)
_, rds_dw1_val, _, _, _ = dann(w1, w2, w3, x1)

dw1, _, _, _ = fdiff(ann, types)
dw1_val = dw1(w1, w2, w3, x1)

if isapprox(dw1_val, rds_dw1_val)
    println("Equal!")
else
    error("Not equal!")
end

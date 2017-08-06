
function ann(w1, w2, w3, x1)
    xx2 = w1 * x1
    x2 = log.(1. + exp.(xx2))   # soft RELU unit
    xx3 = w2 * x2
    x3 = log.(1. + exp.(xx3))   # soft RELU unit
    x4 = w3 * x3
    sum(1. ./ (1. + exp.(-x4)))  # sigmoid output
end

test_compare(ann; w1=rand(10,10), w2=rand(10,10), w3=rand(1,10), x1=rand(10))

f = ann
inputs = [:w1=>rand(10,10), :w2=>rand(10,10), :w3=>rand(1,10), :x1=>rand(10)]

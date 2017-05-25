
logistic(x) = 1 ./ (1 + exp.(-x))
@diff_rule logistic(x::Number) 1 (logistic(x)*(1 - logistic(x)))

# autoencoder cost: sum of squared error
# function autoencoder_cost(We1, We2, Wd, b1, b2, x)
#     firstLayer = logistic(We1 * x + b1)
#     encodedInput = logistic(We2 * firstLayer + b2)
#     reconstructedInput = logistic(Wd * encodedInput)
#     cost = sum((reconstructedInput - x) .^ 2.0)
#     return cost
# end

function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = relu.(We1 * x + b1)
    encodedInput = We2 * firstLayer + b2
    reconstructedInput = Wd * encodedInput
    squared = (reconstructedInput - x) .^ 2
    cost = sum(squared)
    return cost
end


@rdcmp autoencoder_cost We1=rand(4,5) We2=rand(3,4) Wd=rand(5,3) b1=rand(4) b2=rand(3) x=rand(5)



##----------------------------------------------------------------------------------------------


function main_sdfsdf()
    # generate example input
    x = rand(5)
    We1 = rand(4, 5)
    b1 = rand(4)
    We2 = rand(3, 4)
    b2 = rand(3)
    Wd = rand(5, 3)
    
    input_tuple = (We1, We2, Wd, b1, b2, x)
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x =>  x]
    types = map(typeof, input_tuple)
    
    
    ex = quote
        firstLayer = We1 * x + b1
        encodedInput = We2 * firstLayer + b2
        reconstructedInput = Wd * encodedInput
        cost = sum(reconstructedInput - x)
    end

    dex = xdiff(ex; inputs...)


    df = fdiff(autoencoder_cost, types)
    dvals = df(input_tuple...)

    rd_dvals = ReverseDiff.gradient(autoencoder_cost, input_tuple)

    dvals[2:end] .== rd_dvals
end



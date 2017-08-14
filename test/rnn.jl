

function rnn(Wxh, Whh, Wy, hprev, x, y)
    h = tanh.(Whh * hprev + Wxh * x)
    yhat = Why * h
    cost = sum((yhat .- y) .^ 2.0)
    return h, cost
end


let
    Wxh = randn(4096, 4096); Whh = randn(4096, 4096); Why = randn(128, 4096);
    hprev = randn(4096); x = randn(4096); y = randn(128);    
    inputs = [:Wxh=>Wxh, :Whh=>Whh, :Why=>Why, :hprev => hprev, :x => x, :y=>y];
    # at least shouldn't throw an error
    df = xdiff(rnn; ctx=Dict(:cost => :cost), inputs...)
    vals = [val for (name, val) in inputs]
    dvals = df(vals...)
    @test size(dvals[1][1]) == 4096
end

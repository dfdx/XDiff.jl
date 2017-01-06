
ex = quote
    y = exp(W * x)
    sum((x - y).^2)
end
inputs = [:W => rand(5, 5), :x => rand(5)]
ctx = Dict()
@time ds = rdiff(ex; ctx=ctx, inputs...)

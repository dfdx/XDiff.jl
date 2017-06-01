
f(a, b) = sum(a' * b + a * b')

@rdcmp f a=rand(3,3) b=rand(3,3)



#-----------------

function main_1093()
    ex = :(sum(a' * b + a * b'))
    a = rand(3,3)
    b = rand(3,3)
    inputs = [:a => a, :b => b]
    ctx = Dict()
    dex = xdiff(ex; ctx=ctx, inputs...)
end

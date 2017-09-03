
myfunc(x) = log(sum(x))
test_compare(myfunc; x=rand(2))


quote 
    tmp685 = @get_or_create(mem, :tmp685, zero(Float64))
    tmp686 = @get_or_create(mem, :tmp686, zero(Float64))
    dtmp686!dx = @get_or_create(mem, :dtmp686!dx, zeros(Float64, (2,)))
    dtmp686!dtmp686 = @get_or_create(mem, :dtmp686!dtmp686, zero(Float64))
    tmp694 = @get_or_create(mem, :tmp694, zero(Float64))
    dtmp686!dtmp686 = 1.0
    tmp685 = sum(x)
    dtmp686!dx .= dtmp686!dtmp686 .* (dtmp686!dtmp686 ./ tmp685)
    tmp686 = log(tmp685)
    tmp696 = (tmp686, dtmp686!dx)
end

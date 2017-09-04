
function test_mean(x)
    z = mean(x)
end

function test_mean2(x)
    z = mean(x, 1)
end

test_compare(test_mean; x=rand(3))
test_compare(test_mean; x=rand(3,4))


function test_sum_1(x)
    y = sum(x, 1)
    y2 = y .+ 2
    z = sum(y2)
end

function test_sum_2(x)
    y = sum(x, 2)
    y2 = y .+ 2
    z = sum(y2)
end

test_compare(test_sum_1; x=rand(3,4))
test_compare(test_sum_2; x=rand(3,4))

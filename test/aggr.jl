
function test_mean(x)
    z = mean(x)
end

function test_mean2(x)
    z = mean(x, 1)
end

test_compare(test_mean; x=rand(3))
test_compare(test_mean; x=rand(3,4))

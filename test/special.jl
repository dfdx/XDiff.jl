
# sample function, its gradient & derivative definintion

foo(x) = sum(x, 2)
foo_grad(dzdy, x) = repmat(dzdy, 1, size(x, 2))
@specialdiff (y = foo(x)) (dz!dx = foo_grad(dz!dy, x))

function test_special(u, v)
    x = u .+ v
    y = foo(x)
    z = sum(1.0 .* y) # note: don't remove 1.0
end

test_compare(test_special; u=randn(3, 3), v=randn(3, 3))

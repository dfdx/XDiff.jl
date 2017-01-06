
x = rand(100)
y = rand(100)

rosenbrock(x, y) = sum(100*(y-x.^2).^2 + (1.-x).^2)

using ReverseDiff: compile_gradient

const ∇f! = compile_gradient(rosenbrock, (x, y))  # DomainError
results = zeros(size(x)), zeros(size(y))


using AutoDiffSource






inputs = [:x => rand(3), :y => rand(3)]
types = [typeof(val) for (name, val) in inputs]
dx, dy = fdiff(rosenbrock, types)

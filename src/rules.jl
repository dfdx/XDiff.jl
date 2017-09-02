
# rules.jl - differentiation rules for basic operations.

## basic rules

@scalardiff (-x::Number) 1 -1

# product

@scalardiff *(x::Number, y::Number) 1 y
@scalardiff *(x::Number, y::Number) 2 x

# elementwise product

@scalardiff .*(x::Number, y::Number) 1 y
@scalardiff .*(x::Number, y::Number) 2 x

# other arithmetic operations

@scalardiff (x::Number ^ n::Number) 1 (n * x.^(n-1))
@scalardiff (a::Number ^ x::Number) 2 (log(a) * a.^x)
@scalardiff (x::Number .^ n::Number) 1 (n * x.^(n-1))
@scalardiff (a::Number .^ x::Number) 2 (log(a) * a.^x)


@scalardiff (x::Number / y::Number) 1 (1 / y)
@scalardiff (n::Number / x::Real) 2 (-n * x .^ -2.0)
@scalardiff (x::Number ./ y::Number) 1 (1 / y)
@scalardiff (n::Number ./ x::Real) 2 (-n * x .^ -2.0)

@scalardiff (x::Any + y::Any) 1 1
@scalardiff (x::Any + y::Any) 2 1
@scalardiff (x::Any + y::Any + z::Any) 1 1
@scalardiff (x::Any + y::Any + z::Any) 2 1
@scalardiff (x::Any + y::Any + z::Any) 3 1
@scalardiff (w::Any + x::Any + y::Any + z::Any) 1 1
@scalardiff (w::Any + x::Any + y::Any + z::Any) 2 1
@scalardiff (w::Any + x::Any + y::Any + z::Any) 3 1
@scalardiff (w::Any + x::Any + y::Any + z::Any) 4 1

@scalardiff (x::Any .+ y::Any) 1 1
@scalardiff (x::Any .+ y::Any) 2 1

@scalardiff (x::Any - y::Any) 1 1
@scalardiff (x::Any - y::Any) 2 -1

@scalardiff (x::Any .- y::Any) 1 1
@scalardiff (x::Any .- y::Any) 2 -1

@scalardiff sum(x::Number) 1 1

# trigonomeric functions

@scalardiff sin(x::Number) 1 cos(x)
@scalardiff cos(x::Number) 1 -sin(x)
@scalardiff tan(x::Number) 1 (1. + tan(x)  * tan(x))
@scalardiff sinh(x::Number) 1 cosh(x)
@scalardiff cosh(x::Number) 1 sinh(x)
@scalardiff tanh(x::Number) 1 (1.0 .- tanh(x) .* tanh(x))
@scalardiff asin(x::Number) 1 (1 ./ sqrt(1 .- x .* x))
@scalardiff acos(x::Number) 1 (1 ./ sqrt(1 .- x .* x))
@scalardiff atan(x::Number) 1 (1 ./ (1 .+ x .* x))

# sqrt

@scalardiff sqrt(x::Number) 1 (0.5 * x .^ (-0.5))

# exp, log

@scalardiff exp(x::Number) 1 exp(x)
@scalardiff log(x::Number) 1 (1 ./ x)

# abs

@scalardiff abs(x::Number) 1 (sign(x) .* x)
@scalardiff abs2(x::Number) 1 (2 .* abs(x) .* sign(x) .* x)

# min, max

@scalardiff max(x::Number, y::Number) 1 (x > y)
@scalardiff max(x::Number, y::Number) 2 (y > x)

@scalardiff min(x::Number, y::Number) 1 (x < y)
@scalardiff min(x::Number, y::Number) 2 (y < x)

@scalardiff sign(x::Any) 1 0.

# transpose

@scalardiff transpose(x::Number) 1 1

@scalardiff size(x::Any) 1 0.0
@scalardiff size(x::Any, y::Any) 1 0.0
@scalardiff size(x::Any, y::Any) 2 0.0

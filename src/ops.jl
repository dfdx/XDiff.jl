

⊕(ex::Symbolic, v::Numeric) = :($ex .+ $v)
⊕(v::Numeric, ex::Symbolic) = :($v .+ $ex)
⊕(ex1::Symbolic, ex2::Symbolic) = :($ex1 .+ $ex2)
⊕(x, y) = x .+ y

# (.-)(ex::Symbolic, v::Numeric) = :($ex .- $v)
# (.-)(v::Numeric, ex::Symbolic) = :($v .- $ex)
# (.-)(ex1::Symbolic, ex2::Symbolic) = :($ex1 .- $ex2)

⊗(ex::Symbolic, v::Numeric) = :($ex .* $v)
⊗(v::Numeric, ex::Symbolic) = :($v .* $ex)
⊗(ex1::Symbolic, ex2::Symbolic) = :($ex1 .* $ex2)
⊗(x, y) = x .* y

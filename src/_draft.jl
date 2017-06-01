
julia> dzdy = TensorDeriv(:(dx4[]), :(dx3[j]), :(w3[i,j]), [])
dx4[]/dx3[j] = w3[i,j] 

julia> dydx = TensorDeriv(:(dx3[j]), :(dw2[m,n]), :(x2[n]), [:(j == m)])
dx3[j]/dw2[m,n] = x2[n]  * (j == m)

julia> dzdy
dx4[]/dx3[j] = w3[i,j] 

julia> dydx
dx3[j]/dw2[m,n] = x2[n]  * (j == m)

julia> dzdy.ex ⊗ dydx.ex
:(w3[i,j] .* x2[n])

julia> dzdy ⊗ dydx
dx4[]/dw2[m,n] = (w3[i,m] * I[i]) .* x2[n] 




julia> fullex = iex
:(tmp23[i] = w3[i,k] * x3[k])

julia> 

julia> 

julia> idx = 2
2

julia> 

julia> 

julia>     maybe_rule = tfind_rule(fullex, idx)
Nullable{TensorDiffRule}(TensorDiffRule(Z[i] = V[i,k] * W[k] ==> dZ[i]/dW[j] = V[i,k]  * (k == j)))

julia> get(maybe_rule)
TensorDiffRule(Z[i] = V[i,k] * W[k] ==> dZ[i]/dW[j] = V[i,k]  * (k == j))

julia> get(maybe_rule)

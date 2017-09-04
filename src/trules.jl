
# sum_n
@tensordiff (Z[i,j] = Espresso.sum_1(X[:,j])) (dZ[i,j]/dX[m,n] = 1.0 * (j == n))
@tensordiff (Z[i,j] = Espresso.sum_2(X[i,:])) (dZ[i,j]/dX[m,n] = 1.0 * (i == m))

# sum_n + sum
@tensordiff (Z = Espresso.sum_1(X[:,j])) (dZ[i,j]/dX[m,n] = 1.0)
@tensordiff (Z = Espresso.sum_2(X[i,:])) (dZ[i,j]/dX[m,n] = 1.0)


# matrix-by-matrix product
@tensordiff (Z[i,j] = X[i,k] * Y[k,j]) (dZ[i,j]/dX[m,n] = Y[n,j] * (i == m))
@tensordiff (Z[i,j] = X[i,k] * Y[k,j]) (dZ[i,j]/dY[m,n] = X[i,m] * (n == j))

# matrix-by-vector product
@tensordiff (Z[i] = X[i,k] * Y[k]) (dZ[i]/dX[m,n] = Y[n] * (i == m))
@tensordiff (Z[i] = X[i,k] * Y[k]) (dZ[i]/dY[m] = X[i,m])

# inner product of 2 vectors
@tensordiff (Z = X[i] * Y[i]) (dZ/dX[i] = Y[i])
@tensordiff (Z = X[i] * Y[i]) (dZ/dY[i] = X[i])

# outer product of 2 vectors
@tensordiff (Z[i,j] = X[i] * Y[j]) (dZ[i,j]/dX[m] = Y[j] * (i == m))
@tensordiff (Z[i,j] = X[i] * Y[j]) (dZ[i,j]/dY[m] = X[i] * (j == m))

# inner product of 2 matrices
@tensordiff (Z = X[i,j] * Y[i,j]) (dZ/dX[i,j] = Y[i,j])
@tensordiff (Z = X[i,j] * Y[i,j]) (dZ/dY[i,j] = X[i,j])

# index permutation (broken)
# @tensordiff (Z[i,j] = X[j,i]) (dZ[i,j]/dX[m,n] = 1 * (i == n) * (j == m))

# length
@tensordiff (Z = length(X[:])) (dZ/dX[i] = 0.0)

# convolution
@tensordiff (Z[i,j] = X[i+m-1, j+n-1] * W[m,n]) (dZ[i,j] / dW[m,n] = X[i+m-1, j+n-1])
@tensordiff (Z[i,j] = X[i+m-1, j+n-1] * W[m,n]) (dZ[i,j] / dX[p,q] = W[p-i+1, q-j+1])

@tensordiff (Z = X[i+m-1, j+n-1] * W[m,n]) (dZ / dW[m,n] = X[i+m-1, j+n-1])
@tensordiff (Z = X[i+m-1, j+n-1] * W[m,n]) (dZ / dX[p,q] = W[p-i+1, q-j+1])

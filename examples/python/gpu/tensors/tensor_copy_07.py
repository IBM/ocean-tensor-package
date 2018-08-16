import ocean

A = ocean.arange([15],ocean.float,ocean.gpu[0]).reshape([3,5])
print(A)
print(A.T)
print(A.T.strides)

X = ocean.cpu(A.T)
print(X)
print(X.strides)

B = A.flipAxis(0)
print(B)
print(B.T)

B = A.swapAxes(0,1)
print(B)
print(B.T)

print("\n========= Copy B internally =========");

print(B)
print(B.strides)

C = ocean.zeros(B.size, B.dtype, B.device)
print(C.strides)

C.copy(B)

print(C)
print(C.T)


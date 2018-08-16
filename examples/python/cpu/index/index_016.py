## Get index of byteswapped tensors
import pyOcean_cpu as ocean

A = ocean.arange(30).reshape([6,5])
A.byteswap()
A.readonly = True

print(A)
print(A[1,3])
print(A[2,:])

print(A[...,2])

print(A[2])

print(A[1,None,:,None,None,None])

print(A[0::3,::-1])
print(A[0:3])


print(A[[1,3,1]])

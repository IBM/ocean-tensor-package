import pyOcean_cpu as ocean

print("------ Original matrix ------")
A = ocean.tensor([1,4,1,3])
A.copy(range(A.nelem))

print("\n------ Squeeze all axes ------")
print(A.squeeze())

print("\n------ Squeeze only axis 0 ------")
print(A.squeeze(0))

print("\n------ In-place squeezing of axis 2 ------")
A.squeeze(2,True)
print(A)

print("\n------ In-place squeezing of all axes ------")
A.squeeze(True)
print(A)


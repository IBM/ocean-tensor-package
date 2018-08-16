import pyOcean_cpu as ocean

a = ocean.asTensor([[1,2,ocean.nan],[3,4,-5]],'r')
print(a)

print("\n========= normNaN-global =========")
print(ocean.normNaN(a,0))
print(ocean.normNaN(a,1))
print(ocean.normNaN(a,2))
print(ocean.normNaN(a,3))
print(ocean.normNaN(a,ocean.inf))

print("\n========= norm-axis =========")
print(ocean.norm(a,0,[0]))
print(ocean.norm(a,1,[0]))
print(ocean.norm(a,2,[0]))
print(ocean.norm(a,3,[0]))
print(ocean.norm(a,ocean.inf,[0]))

print("\n========= normNaN-axis =========")
print(ocean.normNaN(a,0,[0]))
print(ocean.normNaN(a,1,[0]))
print(ocean.normNaN(a,2,[0]))
print(ocean.normNaN(a,3,[0]))
print(ocean.normNaN(a,ocean.inf,[0]))


print(ocean.nnz(a))
print(ocean.nnzNaN(a))


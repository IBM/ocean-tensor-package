import pyOcean_cpu as ocean

# Create a misaligned double tensor
s = ocean.full(99,[10],ocean.int8).storage
d = ocean.tensor(s,1,[1],[8],1,ocean.double)
ocean.add(1,-2,d)
print(d)


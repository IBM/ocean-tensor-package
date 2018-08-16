import pyOcean_cpu as ocean

a = ocean.asTensor([-0.98,0.98, 3., -30.] + [-100 for i in range(20)],ocean.float)
print(a)

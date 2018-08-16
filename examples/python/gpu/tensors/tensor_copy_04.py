## Copying of tensor from GPU to CPU in compressed form
import ocean

a = ocean.arange(4,ocean.float,ocean.gpu[0]).broadcastTo([4,3])
print(a)

b = ocean.tensor([12],ocean.half,ocean.cpu)           
b.byteswapped = True

b.copy(a)
print(b.reshape([4,3]))


# ---------------------------------------------------------------

b = ocean.tensor([12],ocean.double,ocean.cpu)
b.byteswapped = True

b.copy(a)
print(b.reshape([4,3]))

# ---------------------------------------------------------------

b = ocean.tensor([12],ocean.int8,ocean.cpu)
b.copy(a)
print(b.reshape([4,3]))

# ---------------------------------------------------------------

b = ocean.tensor([12],ocean.double,ocean.cpu)
b.copy(a)
print(b.reshape([4,3]))
print(b.reshape([3,4]))
print(b.reshape([2,6]))
print(b.reshape([6,2]))

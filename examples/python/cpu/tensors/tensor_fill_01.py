## tensor.fill - Filling
import pyOcean_cpu as ocean

a = ocean.tensor([3,6], ocean.int16, ocean.cpu);

a.fill(1);
a.byteswap()
print(a)


import ocean
import numpy
import ocean_numpy

a = ocean.arange(64*65,ocean.double,ocean.gpu[0]).reshape(64,65).T;
b = a.convertTo('numpy')
print(a)

print(ocean.norm(a,2,0))
print(ocean.asTensor([float(numpy.linalg.norm(b[:,i],2)) for i in range(b.shape[1])]))

print(ocean.norm(a,2,1))
print(ocean.asTensor([float(numpy.linalg.norm(b[i,:],2)) for i in range(b.shape[0])]))

print(ocean.norm(a,2,[0,1]));
print(numpy.linalg.norm(range(64*65),2))

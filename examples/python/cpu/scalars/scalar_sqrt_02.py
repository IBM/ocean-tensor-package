import pyOcean_cpu as ocean
import pyOceanNumpy
import numpy as np

# 0 + 0j
a = np.asarray([0+0j],np.complex64)
a = a[0]
print("Input: %s" % a) 
print(np.sqrt(a))
print(ocean.sqrt(a))


# 0 + NaNj
a = np.asarray([0+0j],np.complex64)
a.imag[0] = np.nan
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))

# NaN + 2j
a = np.asarray([0+2j],np.complex64)
a.real[0] = np.nan
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))


# 1 + inf*j
a = np.asarray([1+0j])
a.imag[0] = np.inf
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))


# 1 - inf*j
a = np.asarray([1+0j])
a.imag[0] = -np.inf
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))


# inf + 1j
a = np.asarray([0+1j])
a.real[0] = np.inf
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))


# inf - 1j
a = np.asarray([0-1j])
a.real[0] = np.inf
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))



# -inf + 1j
a = np.asarray([0+1j])
a.real[0] = -np.inf
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))


# -inf - 1j
a = np.asarray([0-1j])
a.real[0] = -np.inf
a = a[0]
print("\nInput: %s" % a)
print(np.sqrt(a))
print(ocean.sqrt(a))


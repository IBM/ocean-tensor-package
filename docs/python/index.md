# The Python Ocean interface

1. [Importing Ocean](#importing-ocean)
2. [Available modules](#available-modules)
3. [Devices](#devices)
4. [Data types](#data-types)
5. [Scalars](#scalars)
6. [Storage](#storage)
7. [Streams](#streams)
8. [Tensors](#tensors)
9. [Indexing](#indexing)
10. [Casting with data types and devices](#casting-with-data-types-and-devices)
11. [Warning configuration](#warning-configuration)
12. [Math modes](#math-modes)
13. [Automatic type conversion](#automatic-type-conversion)
14. [Broadcasting of dimensions](#broadcasting-of-dimensions)
15. [Tensor reference counting](#tensor-reference-counting)
16. [Formatting](#formatting)
17. [Numpy and other tensor types](#numpy-and-other-tensor-types)
18. [CTypes](#ctypes)

## Importing Ocean

The Ocean package consists of various modules that define different operations or provide support for different device types. The generic package can be imported using

```python
import ocean
```

The `ocean.py` file simply imports some of the common modules and looks something like this:

```python
from pyOcean_cpu import *    # Core module for CPU
from pyOcean_gpu import *    # Core module for GPU
```

If GPU support is not needed it is also possible to use `import pyOcean_cpu as ocean`.

## Available modules

The following modules are currently available (click on the links to check the functions provided by the module). The core module is loaded automatically when importing ocean.

| Module | Interface | CPU | GPU | Description |
| ------ | --------- | --- | --- | ----------- |
| [Core](module_core.md)   | (pyOcean_core) | pyOcean_cpu / pyOcean_core | pyOcean_gpu | The core module defining devices, storage, and tensors |
| [Numpy](module_numpy.md)  | pyOceanNumpy | -- | -- | Conversion between Numpy arrays and Ocean Tensors |


## Devices

| Interface | Module | Description |
| --------- | ------ | ----------- |
| device    | pyOcean_cpu | Device base type |
| deviceCPU | pyOcean_cpu | Derived device type for CPU |
| cpu       | pyOcean_cpu | Instance of a CPU device |
| devices() | pyOcean_cpu | Function that returns a list of all available devices |
| deviceGPU | pyOcean_gpu | Derived device type for GPU |
| gpu       | pyOcean_gpu | List of GPU device instances |

**Device types and instances**

```python
>>> import ocean
>>> ocean.devices()
(<device 'cpu'>, <device 'gpu0'>)
>>> ocean.cpu
<device 'cpu'>
>>> type(ocean.cpu)
<type 'ocean.deviceCPU'>
>>> isinstance(ocean.cpu, ocean.deviceCPU)
True
>>> isinstance(ocean.cpu, ocean.device)
True
>>> len(ocean.gpu)
1
>>> ocean.gpu[0]
<device 'gpu0'>
>>> type(ocean.gpu[0])
<type 'ocean.deviceGPU'>
>>> isinstance(ocean.gpu[0], ocean.deviceGPU)
True
>>> isinstance(ocean.gpu[0], ocean.deviceCPU)
False
```

**Device properties**

The base Device type provides a number of read-only properties:

```python
>>> # Name of the device type
>>> ocean.cpu.type
'CPU'
>>> ocean.gpu[0].type
'GPU'
>>> # Name of the device instance
>>> ocean.gpu[0].name
'gpu0'
>>> # Device index (for example 2 for gpu[2], when available)
>>> ocean.gpu[0].index
0
>>> # Byteswapping
>>> ocean.cpu.supportsByteswap
True
>>> # Alignment of data
>>> ocean.gpu[0].requiresAlignedData
True
>>> # List of modules available for the device type
>>> ocean.gpu[0].modules
['core']
>>> # Reference count to the underlying device object (debugging)
>>> ocean.cpu.refcount
3
```

GPU device instances have the following additional properties (see `cudaGetDeviceProperties` for more information);

```python
>>> ocean.gpu[0].deviceName
'Tesla K40m'
>>> ocean.gpu[0].totalGlobalMem
12079136768L
>>> ocean.gpu[0].freeMem   # Free global memory
12079028372L
>>> ocean.gpu[0].sharedMemPerBlock
49152L
>>> ocean.gpu[0].regsPerBlock
65536L
>>> ocean.gpu[0].warpSize
32L
>>> ocean.gpu[0].memPitch
2147483647L
>>> ocean.gpu[0].maxThreadsPerBlock
1024L
>>> ocean.gpu[0].maxThreadsDim
[1024L, 1024L, 64L]
>>> ocean.gpu[0].maxGridSize
[2147483647L, 65535L, 65535L]
>>> ocean.gpu[0].totalConstMem
65536L
>>> ocean.gpu[0].major
3L
>>> ocean.gpu[0].minor
5L
>>> ocean.gpu[0].version
'3.5'
>>> ocean.gpu[0].clockRate  # Clock rate in megahertz
745.0
>>> ocean.gpu[0].deviceOverlap
True
>>> ocean.gpu[0].multiProcessorCount
15L
>>> ocean.gpu[0].kernelExecTimeoutEnabled
False
>>> ocean.gpu[0].integrated
False
>>> ocean.gpu[0].canMapHostMemory
True
>>> ocean.gpu[0].computeMode
'Compute-exclusive-process mode'
>>> ocean.gpu[0].concurrentKernels
True
>>> ocean.gpu[0].ECCEnabled
True
>>> ocean.gpu[0].pciBusID
27L
>>> ocean.gpu[0].pciDeviceID
0L
>>> ocean.gpu[0].tccDriver
False
```

Detailed device information can be obtained through the `info` property:

```
>>> print(ocean.gpu[0].info)
Device gpu0 - Tesla K80
--------------------------------------------------------------------------------------------
Clockrate           : 823.5MHz                Multi-processors    : 13
Memory clockrate    : 2505.0MHz               Compute capability  : 3.7
Total global memory : 11440Mb (11285Mb)       Maximum grid size   : 2147483647, 65535, 65535
Total const memory  : 64kb                    Maximum block size  : 1024, 1024, 64
L2 Cache size       : 1.5Mb                   Threads per block   : 1024
Global caching L1   : Enabled                 Warp size           : 32
Local caching L1    : Enabled                 Max resident threads: 2048
Can map host memory : Yes                     Max resident blocks : 16
Max. copy mempitch  : 2147483647              Max resident warps  : 64
Memory bus width    : 384                     Shared mem per mproc: 112kb
Unified addressing  : Yes                     Shared mem per block: 48kb
Managed memory      : ---                     Registers per block : 65536
Pageable memory     : ---                     Registers per mproc : 131072
ECC enabled         : Yes                     Compute mode        : 3 
TCC driver          : No                      Async. engine count : 2     
Integrated          : No                      Concurrent kernels  : Yes
Multi-GPU board     : Yes                     Kernel time-out     : Disabled
Multi-GPU group ID  : 0                       Stream priorities   : Yes
Concurrent managed  : ---                     PCI                 : 0000:001F:00
Host native atomic  : ---                     Single-double perf. : 0     

Peer devices: gpu1
```

**Device instantiation**

Devices are instantiated by the device-specific implementation of the core module, and can be exposed by importing the module. In case the device or required drivers are not available on the machine, or in case no access to them is needed, the user can simply choose not to import any of the modules for that device. The following example illustrates how devices are made available in each module as well as in the device list in the Ocean core module:

```python
>>> import pyOcean_core as ocean
>>> ocean.devices()
(<device 'cpu'>,)
>>> ocean.cpu
<device 'cpu'>
>>> import pyOcean_gpu as ocean_gpu
>>> ocean.devices()
(<device 'cpu'>, <device 'gpu0'>)
>>> ocean_gpu.gpu
(<device 'gpu0'>,)
```

**Default device**

The user can specify and retrieve the default device:

```python
>>> ocean.getDefaultDevice()
<device 'cpu'>
>>> ocean.setDefaultDevice(ocean.gpu[0])
>>> ocean.getDefaultDevice()
<device 'gpu0'>
>>> ocean.gpu[1].setDefault()
>>> ocean.getDefaultDevice()
<device 'gpu1'>
>>> ocean.setDefaultDevice(None)
>>> ocean.getDefaultDevice()
>>>
```

**Advanced operations**

It is possible to create new streams corresponding to a device, and set them as the default for future instantiations of storage (either directly, or through creation of a tensor):

```python
>>> import ocean
>>> s = ocean.cpu.createStream()
>>> ocean.cpu.defaultStream = s
>>> t = ocean.tensor([])
>>> print(t.storage.stream == s)
True
```

Each device has a number of temporary storage buffers that are used internally, for example when copying tensors with different memory layout across device boundaries, or when casting the data type or byte order during operations. When no temporary buffers are available Ocean needs to allocate these on-the-fly and delete them after the operation is done, thereby introducing a synchronization point. Each temporary buffer takes up space and it may therefore be important in advanced configurations to control the number and size of the buffers.

```python
>>> # Get the current number of buffers
>>> ocean.cpu.bufferCount
3
>>> # Allow up to five temporary buffers
>>> ocean.cpu.bufferCount = 5
>>> # Get the maximum size of each buffer (in bytes, zero denotes infinity)
>>> ocean.cpu.maxBufferSize
0
>>> # Set the maximum size of each buffer
>>> ocean.cpu.maxBufferSize = 1024 * 1024
```

Operations are also provided to control the buffer size and count globally:

```python
>>> ocean.getBufferCount(ocean.gpu[0])
3
>>> ocean.setBufferCount(ocean.gpu[0],5)      # Single device
>>> ocean.setBufferCount("GPU", 5)            # All GPU devices
>>> ocean.setBufferCount(5)                   # All devices currently instantiated
>>> ocean.getMaxBufferSize(ocean.cpu)
0
>>> ocean.setMaxBufferSize(ocean.gpu[1],1024) # Single device
>>> ocean.setMaxBufferSize("GPU", 1024)       # All GPU devices
>>> ocean.setMaxBufferSize(1024)              # All devices currently instantiated
```

## Data types

The `pyOcean` module defines all available data types:

| Type | Description |
| ---- | ----------- |
| dtype | Data-type type |
| bool  | Boolean (True / False) |
| int8  | Signed 8-bit integer |
| int16 | Signed 16-bit integer |
| int32 | Signed 32-bit integer |
| int64 | Signed 64-bit integer |
| uint8 | Unsigned 8-bit integer |
| uint16 | Unsigned 16-bit integer |
| uint32 | Unsigned 32-bit integer |
| uint64 | Unsigned 64-bit integer |
| half   | Half-precision floating point (16-bit) |
| float  | Standard floating point (32-bit) |
| double | Double-precision floating point (64-bit) |
| chalf | Complex half-precision floating point |
| cfloat | Complex standard floating point |
| cdouble | Complex double-precision floating point |

```python
>>> ocean.dtype
<type 'ocean.dtype'>
>>> type(ocean.bool)
<type 'ocean.dtype'>
>>> ocean.bool
<dtype 'bool'>
>>> ocean.int8
<dtype 'int8'>
>>> ocean.double
<dtype 'double'>
```

Data types have properties that can be queried

```python
>>> # Name of the data type
>>> ocean.double.name
'double'
>>> # Size of the data type in bytes
>>> ocean.double.size
8
>>> ocean.uint16.size
2
>>> # Size of the data type in number of bits
>>> ocean.bool.nbits
1
>>> ocean.int16.nbits
16
>>> # Minimum, maximum, and epsilon
>>> ocean.int8.min
-128
>>> ocean.float.min
-3.40282e+38
>>> ocean.int8.max
127
>>> ocean.double.eps
2.22045e-16
>>> # Flag indicating whether the data type is a number
>>> ocean.bool.isnumber
False
>>> ocean.int8.isnumber
True
>>> # Flag indicating whether the data type is signed
>>> ocean.float.issigned
True
>>> ocean.uint32.issigned
False
>>> # Flag indicating whether the data type is floating point
>>> ocean.half.isfloat
True
>>> ocean.int64.isfloat
False
>>> # Flag indicating whether the data type is complex
>>> ocean.double.iscomplex
False
>>> ocean.cdouble.iscomplex
True
```

**Default data type**

The user can specify and retrieve the default data type:

```python
>>> ocean.getDefaultDType()
<dtype 'float'>
>>> ocean.setDefaultDType(ocean.double)
>>> ocean.getDefaultDType()
<dtype 'double'>
>>> ocean.int8.setDefault()
>>> ocean.getDefaultDType()
<dtype 'int8'>
>>> ocean.setDefaultDType(None)
>>> ocean.getDefaultDType()
>>>
```

## Scalars

Scalars can be instantiated by calling a data type:

```python
>>> s = ocean.int8(3)
>>> s
3
>>> s.dtype
<dtype 'int8'>
>>> t = ocean.chalf(1+2j)
>>> t
1 + 2j
>>> # Get the real part of a scalar (new scalar object)
>>> t.real
1.0
>>> # Get the imaginary part of a scalar (new scalar object)
>>> t.imag
2.0
>>> # Convert to a tensor, optionally with device
>>> t.asTensor(ocean.gpu[0])
1 + 2j
<scalar.complex-half on gpu0>
```
Other operations defined on scalars include the following:

```python
>>> # Conversion to float, long, complex, bool
>>> complex(ocean.int16(7))
(7+0j)
>>> # Conversion to Python object and Ocean tensor
>>> a = ocean.int8(3)
>>> a.asPython()
3
>>> a.asTensor()
3
<scalar.int8 on cpu>
```

Mathematical operations:

```python
>>> # Comparison operations (<, <=, ==, !=, >=, >)
>>> ocean.uint8(9) < 9.2
True
>>> ocean.uint8(9) == ocean.half(9)
True
>>> # Unary negation
>>> a = ocean.int8(6)
>>> print(-a)
-6
```

**Scalar operations and data type**

For scalar operations we allow flexible data types, and no great effort is taken to keep the data type to its minimum. Consider for example the following in-place addition:

```python
>>> a = ocean.int8(100)
>>> a += 100
>>> a
200
>>> a.dtype
<dtype 'int64'>

>>> a = ocean.int8(100)
>>> a += ocean.int8(100)
-56
>>> a.dtype
<dtype 'int8'>
```

As described in the [design considerations](design_choices.md), the scalar 100 in Python has type int64. Even though the addition is in-place and although 100 certainly fits int8 we nevertheless use the canonical int64 representation, thereby obtaining an int64 result. When it is desirable to maintain the data type, it may be needed to strongly type 100 before using it, as illustrated in the second half of the above example.

**Scalar configuration**

The level of automatic type casting for scalar operations is controlled using the `getScalarCastMode` and `setScalarCastMode` functions. The three supported modes are: 0 = do not apply any automatic type casting; 1 = upcast to integer or double if needed; 2 = same as mode 1 but allows conversion to complex if needed. The default value is 2:

```python
>>> print(ocean.getScalarCastMode())
2
>>> ocean.sqrt(-1)
0 + 1j
>>> ocean.setScalarCastMode(1)
>>> ocean.sqrt(-1)
__main__:1: RuntimeWarning: Input argument to square-root must be nonnegative
nan
>>> ocean.setScalarCastMode(0)
>>> ocean.sqrt(-1)
0
```
Another example applies to unsigned integers:
```python
>>> ocean.setScalarCastMode(1)
>>> print(-ocean.uint8(3))
-3
>>> ocean.setScalarCastMode(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Scalar negation is not supported for type uint8
```

## Storage

| Interface  | Module     | Description                  |
| ---------- | ---------- | ---------------------------- |
| storage    | pyOcean    | Storage class type           |

**Storage creation**

Storage can be created using the `storage` constructor, which takes as input the storage size, followed by an optional device and data type. When omitted, the default device or data type is used. Instantiating `storage` without any input arguments gives an empty storage. Note that the storage data are not initialized to zero.

```python
>>> s = ocean.storage(3) # Use the default device and data type
>>> s
<storage.float of size 3 on cpu>
>>> s = ocean.storage()
>>> s
<storage.float of size 0 on cpu>
>>> s = ocean.storage(4, ocean.gpu[0])
>>> s
<storage.float of size 4 on gpu0>
>>> s = ocean.storage(5, ocean.int8)
>>> s
<storage.int8 of size 5 on cpu>
>>> s = ocean.storage(6, ocean.gpu[0], ocean.double)
>>> s
<storage.double of size 6 on gpu0>
```

**Storage properties**

```python
>>> s = ocean.storage(6, ocean.double, ocean.gpu[0])
>>> s.size # Size in bytes
48
>>> s.nelem # Size in elements
6
>>> s.device
<device 'gpu0'>
>>> s.dtype
<dtype 'double'>
>>> s.capacity
6
>>> s.owner
True
>>> s.byteswapped
False
>>> s.readonly
False
>>> # Get the element size (same as s.dtype.size)
>>> s.elemsize
8
>>> # Reference count to the underlying storage object (debugging)
>>> s.refcount
1
>>> # Pointer to the Ocean storage object (as integer)
>>> s.obj
13130928
>>> # Pointer to the storage data (as integer)
>>> s.ptr
161218736
```

Other functions defined for scalars include 


**Storage functions**

The following member functions are defined on storage objects 's':

| Function | Description |
| -------- | ----------- |
| `s.__str__()`     | Returns a formatted string of the storage |
| `s.copy(source)`  | Copy the contents of the source storage to s |
| `s.clone(device)` | Returns a deep copy of s on the given device |
| `s.asTensor()`    | Returns a canonical tensor for the storage |
| `s.byteswap()`    | Performs an in-place byte swap of the data |
| `s.zero()`        | Sets the storage data to zero |
| `s.sync()`        | Forces synchronization of the storage data |
| `s.asPython()`    | Converts storage to a Python list |

```python
>>> s1 = ocean.storage(6, ocean.float, ocean.cpu)
>>> s2 = ocean.storage(6, ocean.float, ocean.gpu[0])
>>>
>>> # Set the storage data to zero
>>> s1.zero()
>>> s1.footer
<storage.float of size 6 on cpu>
>>> # Print the contents of the storage (the size displayed is in number of elements)
>>> s1
   0   0   0   0   0   0
<storage.float of size 6 on cpu>
>>> # Byte swap the data
>>> s1.byteswap()
>>> s1
   0   0   0   0   0   0
<storage.float of size 6 on cpu (byteswapped)>
>>> # Force synchronization of the data
>>> s2.sync()
>>>
>>> # Clone the data
>>> s1.clone()
   0   0   0   0   0   0
<storage.float of size 6 on cpu (byteswapped)>
>>> s1.clone(ocean.gpu[0])
   0   0   0   0   0   0
<storage.float of size 6 on gpu0>
>>> s1.asTensor()
   0   0   0   0   0   0
<tensor.float of size 6 on cpu>
>>> s1.asPython()
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

When cloning the data, an attempt is made to keep the data in the given endianness. When the device does not implement a byteswap routine we disallow creation of byte-swapped storage and therefore clone in native byte order. When no device is given, clone will copy to the same device as the input storage.

**Storage comparison**

Comparison operations apply to the memory location of the storage objects, not to their content. When a comparison includes a type other than a storage object, the comparison applies to the memory location of the Python objects. In order to compare storage contents, wrap the storage as tensors:

```python
>>> t1 = ocean.asTensor([1,2,3])
>>> t2 = ocean.asTensor([1,2,3])
>>> s1 = t1.storage
>>> s2 = t2.storage
>>> s1 == s2
False
>>> s1 != s2
True
>>> ocean.tensor(s1) == ocean.tensor(s2)
   True   True   True
<tensor.bool of size 3 on cpu>
```

**Setting the data type**

The storage data type is used to interpret the data, and can be used as the default data type when creating a tensor from the storage. The storage buffer is allocated in terms of number of bytes and the size is given by `storage.size`. We can update the data type to view the data as an alternative type. The number of elements in the storage is the storage size divided by the element size of the new data type, rounded down to the nearest integer. Changing the storage data type does not affect any existing tensors derived from it:

```
>>> s = ocean.storage(3, ocean.cfloat)
>>> s.asTensor().fill(1+2j)   
>>> s
   1 + 2j    1 + 2j    1 + 2j
<storage.complex-float of size 3 on cpu>
>>> t = ocean.tensor(s)
>>> t
   1 + 2j    1 + 2j    1 + 2j
<tensor.complex-float of size 3 on cpu>
>>> # Change the data type
>>> s.dtype = ocean.float
>>> s
   1   2   1   2   1   2
<storage.float of size 6 on cpu>
>>> t
   1 + 2j    1 + 2j    1 + 2j
<tensor.complex-float of size 3 on cpu>
```

It is also possible to set the data type to `None`, which means that the storage is interpreted as a list of raw bytes.
```
>>> s = ocean.storage(6, ocean.int8)
>>> s.asTensor().fill(10)
>>> s
   10   10   10   10   10   10
<storage.int8 of size 6 on cpu>
>>> s.dtype = None
>>> s
   0A   0A   0A   0A   0A   0A
<raw storage of size 6 on cpu>
>>> s.dtype = ocean.int16
>>> s
   2570   2570   2570
<storage.int16 of size 3 on cpu>
```

**Storage indexing**

Storage can be indexed similar to tensors (see below) and results in a tensor. In fact, `storage[<index>]` is merely syntactic sugaring for `ocean.tensor(storage)[<index>]`, that is indexing applied to the canonical transformation of a storage to a tensor. Indexing can be used to apply operations on storage:

```python
>>> s = ocean.storage(5)
>>> s[...] = 0
>>> s
   0   0   0   0   0
<storage.float of size 5 on cpu>
>>> s[2:] += [1,2,3]
>>> s
   0   0   1   2   3
<storage.float of size 5 on cpu>
```

**Byteswapping the data**

The `storage.byteswap()` function byte swaps the data in the storage according to the current data type, and updates the byteswap flag. In some cases it may be necessary to directly set or unset the byteswap flag without performing a byte swap. This can be done simply by setting the `storage.byteswapped` field to true or false.

NOTE: The byteswap status of tensors is relative to that of the storage. That means that byteswapping the storage data automatically byteswaps all tensors that use the storage. When tensors of different data types all use the same underlying data, the data order of those tensors may no longer make sense, and care therefore needs to be taken when byteswapping storage.


## Streams

Associated with each storage object and therefore, indirectly, with each tensor, is a stream.

```python
>>> # Create a new stream with a given or default device
>>> s = ocean.stream(ocean.cpu)
>>> print(s)
<stream 0x1977bd0 on cpu>
>>> s.device
<device 'cpu'>
>>> s2 = ocean.cpu.createStream()
>>> s == s2
False
>>> # Synchronize the stream
>>> s.sync()
>>> # Get the stream reference count
>>> print(s.refcount)
1
```

By default a new stream is created for each new storage object. By setting the `device.defaultStream` property it is possible to create storage objects that create the same stream. All tensors associated with a given storage use the same underlying stream to make sure that all operations remain consistent. Although streams are provided on the CPU device they are not currently used to store pending tasks (CPU operations are executed synchronously). The CPU default stream is therefore initialized to a stream created at start up, to avoid creating new streams for CPU tensors.

```python
>>> ocean.gpu[0].defaultStream = ocean.gpu[0].createStream()
>>> t1 = ocean.tensor([], ocean.gpu[0])
>>> t2 = ocean.tensor([], ocean.gpu[0])
>>> t1.storage.stream == t2.storage.stream
True
```


## Tensors

| Interface  | Module     | Description                  |
| ---------- | ---------- | ---------------------------- |
| tensor     | pyOcean    | Tensor class type            |

#### Tensor dimension ordering

Tensors in Ocean are multi-dimensional arrays starting from zero-dimensional scalars, one-dimensional vectors, two-dimensional matrices, up to the current maximum of eight-dimensional arrays. One of the design considerations is how to interpret the order of the dimensions. In Numpy and other packages dimensions start from the end such that a kxnxm tensor consists of k matrices of size nxm, and A[k,j,i] denotes the k-th matrix with column j and row i. In Matlab the dimensions start from the beginning such that the mxnxk tensor consists of k matrices of size mxn and A[i,j,k] denotes column i, row j, of matrix k. In Ocean we choose the latter dimension ordering, and accordingly order memory following the Fortran-style layout. The main reason for this is that a (one-dimensional) vector then automatically denotes a column vector as opposed to Numpy where it represents a row vector. Consider the following example that illustrates the use of broadcasting rules when adding a vector to a matrix:

```python
>>> import numpy as np
>>> A = np.zeros([3,3])
>>> b = np.asarray([1,2,3])
>>> A + b
array([[ 1.,  2.,  3.],
       [ 1.,  2.,  3.],
       [ 1.,  2.,  3.]])
```

In numpy dimensions are broadcast on the left, thereby replicating the vector as rows of a matrix. In Ocean broadcasting of dimensions is done on the right, thereby giving the following:

```python
>>> A = ocean.zeros([3,3])
>>> b = ocean.asTensor([1,2,3])
>>> A + b
(:,:)
   1   1   1
   2   2   2
   3   3   3
<tensor.double of size 3x3 on cpu>
```

In Ocean we use the convention that the transpose of an n-vector results in a 1-by-n matrix. Adding a row vector to the matrix can therefore be done as follows:

```python
>>> b.T
(:,:)
   1   2   3
<tensor.int64 of size 1x3 on cpu>
>>> A + b.T
(:,:)
   1   2   3
   1   2   3
   1   2   3
<tensor.double of size 3x3 on cpu>
```

Converting a nested list of scalars to a tensor from the outside works from the left-most to the right-most dimension, with a list corresponding to a column. This can be confusing when writing matrices:

```python
>>> ocean.asTensor([[1,2,3],[4,5,6]])
>>> ocean.asTensor([[1,2,3],[4,5,6]])
(:,:)
   1   4
   2   5
   3   6
<tensor.int64 of size 3x2 on cpu>
```

Specifying a row-ordering "R" in the `asTensor` function transposes the tensor after it is formed (with strides pre-adjusted such that the result has a column-major ordering) thereby allowing more intuitive specification of matrices:

```python
>>> ocean.asTensor([[1,2,3],[4,5,6]],"R")
(:,:)
   1   2   3
   4   5   6
<tensor.int64 of size 2x3 on cpu>
```

In fields such as machine learning, data often has a row-major ordering with rows representing feature vectors. When imported into Ocean such data will have the intended dimensions completely reversed. For example:

```python
>>> v1 = numpy.asTensory([1,2,3])
>>> v2 = numpy.asTensor([4,5,6])
>>> v3 = numpy.asTensor([7,8,9])
>>> X = numpy.asTensor([[v1,v2],[v2,v3]])
>>> X
array([[[1, 2, 3],
        [4, 5, 6]],

       [[4, 5, 6],
        [7, 8, 9]]])
>>> X.shape
(2, 2, 3)
```
Importing this data preserves dimensions thereby leading to a (2x2) x 3 tensor consisting of three matrices of size 2x2:

```python
>>> ocean.asTensor(X)
>>> ocean.asTensor(X)
(:,:,0)
   1   4
   4   7

(:,:,1)
   2   5
   5   8

(:,:,2)
   3   6
   6   9
<tensor.int64 of size 2x2x3 on cpu>
```

We can obtain the desired dimension ordering by reversing the axes (this merely creates a new view of the data):
```python
>>> A = ocean.asTensor(X).reverseAxes()
>>> A
(:,:,0)
   1   4
   2   5
   3   6

(:,:,1)
   4   7
   5   8
   6   9
<tensor.int64 of size 3x2x2 on cpu>
>>> A.strides
(8, 24, 48)
```

Note that the same results is obtained by creating a new tensor: `ocean.asTensor([[v1,v2],[v2,v3]])`.

The last two dimensions of Numpy multi-arrays are treated as matrices in operations such as matrix multiplication. Reversing all axes would therefore mean that we transpose the matrices. To avoid this we can use `reverseAxes2` which swaps all axes and then exchanges the first two (if the number of dimensions is at least two). Note that this operation can lead to memory layouts that are no longer purely row or column major:

```python
>>> X = np.arange(36).reshape([2,3,6])
>>> X
array([[[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17]],

       [[18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]]])
>>> A = ocean.asTensor(X).reverseAxes2()
>>> print(A)
(:,:,0)
    0    1    2    3    4    5
    6    7    8    9   10   11
   12   13   14   15   16   17

(:,:,1)
   18   19   20   21   22   23
   24   25   26   27   28   29
   30   31   32   33   34   35
<tensor.int64 of size 3x6x2 on cpu>
>>> A.strides
(48, 8, 144)
```

#### Tensor creation

The standard way to create a new tensors is through the `tensor` constructor. The first syntax is:

`tensor(size [, strides [, unitsize]] [, dtype] [, device] [, stream])`

The default size is zero, or `[]`. When strides are omitted the default row-major contiguous layout is chosen. The current default device or data type are used when the parameters are omitted. The unitsize parameter gives the value by which the strides are multiplied. By default it is set to the data type size. Setting it to 1 allows exact specification of the strides in bytes. Tensors contents are not initialized to zero, so instead of printing the tensor we print the footer in the examples below. The second syntax is:

`tensor(storage [, offset [, size [, strides [, unitsize]]]]] [, dtype])`

This creates a tensor from an existing storage object (which defines the device). The offset indicates the offset within the storage where the zeroth element of the tensor resides. The offset plus the extent of the tensor must fit the storage. The data type of the tensor is determined as follows. When `dtype` is given this value is used, otherwise the storage data type is used. If neither is available the default data type is used, or an error is given if the default is not set. When the size is omitted, a one-dimensional tensor matching the storage size is returned. The unitsize parameter gives the value by which the offset and strides are multiplied. By default it is set to the data type size. Setting it to 1 allows exact specification of the offset and strides in bytes.

Example uses of the first syntax:

```python
>>> # Create scalar tensors
>>> T = ocean.tensor([])
>>> T.footer
'<scalar.float on cpu>'
>>> T = ocean.tensor([], ocean.int8)
>>> T.footer
'<scalar.int8 on cpu>'
>>>
>>> # Create tensors
>>> T = ocean.tensor([3,4])
>>> T.footer
'<tensor.float of size 3x4 on cpu>'
>>> 
>>> T = ocean.tensor([3,5], ocean.int16, ocean.cpu)
>>> T.footer
'<tensor.int16 of size 3x5 on cpu>'
>>> T.strides
(1, 3)
>>> # Specify the strides
>>> T = ocean.tensor([3,5], [1,3], ocean.int16, ocean.cpu)
>>> T.footer
'<tensor.int16 of size 3x5 on cpu>'
>>> T.strides
(2, 6)
```

Example uses of the second syntax. We assume a storage of size 12 with values 0 though 11.

```python
>>> S
    0    1    2    3    4    5    6    7    8    9   10   11
<storage.int16 of size 12 on cpu>
>>> ocean.tensor(S)
    0    1    2    3    4    5    6    7    8    9   10   11
<tensor.int16 of size 12 on cpu>
>>>
>>> ocean.tensor(S,0,[3,4])
(:,:)
    0    3    6    9
    1    4    7   10
    2    5    8   11
<tensor.int16 of size 3x4 on cpu>
>>> 
>>> ocean.tensor(S,0,[3,4],[4,1])
(:,:)
    0    1    2    3
    4    5    6    7
    8    9   10   11
<tensor.int16 of size 3x4 on cpu>
>>> 
>>> ocean.tensor(S,2,[10])
    2    3    4    5    6    7    8    9   10   11
<tensor.int16 of size 10 on cpu>
>>> 
>>> ocean.tensor(S,2,[3,3])
(:,:)
    2    5    8
    3    6    9
    4    7   10
<tensor.int64 of size 3x3 on cpu>
>>> # Strides causing data overlaps are allowed
>>> ocean.tensor(S,0,[4,5],[1,2])
(:,:)
    0    2    4    6    8
    1    3    5    7    9
    2    4    6    8   10
    3    5    7    9   11
<tensor.int16 of size 4x5 on cpu>
>>> # The tensor needs not exactly cover the storage
>>> ocean.tensor(S,1,[4,6],[1,1])
(:,:)
   1   2   3   4   5   6
   2   3   4   5   6   7
   3   4   5   6   7   8
   4   5   6   7   8   9
<tensor.int16 of size 4x6 on cpu>
```

#### Tensor memory

By specifying the strides of a tensor it is possible to create tensors that are self overlapping in memory. For example, consider the construction of a Toeplitz matrix:

```python
>>> A = ocean.arange(8)
>>> B = ocean.tensor(A.storage,3,[4,5],[-1,1])
>>> A
   0   1   2   3   4   5   6   7
<tensor.int64 of size 8 on cpu>
>>> B
(:,:)
   3   4   5   6   7
   2   3   4   5   6
   1   2   3   4   5
   0   1   2   3   4
<tensor.int64 of size 4x5 on cpu>
>>> B.isSelfOverlapping()
True
```

Self-overlapping tensors cannot be modified or used as a destination to avoid undesirable behavior. For example, taking the elementwise square root of a self-overlapping tensor would cause the operation to be applied repeatedly on those memory location that are used multiple times. Trivial self-overlap (when tensors have zero strides in memory, for example due to broadcasting) is easier to deal with and some operations are permitted. For example, the `fill` operations, as well as copying from tensors with the same or more general trivial self-overlap pattern:

```python
>>> A = ocean.arange(3, ocean.double).broadcastTo([3,5])
>>> B = ocean.asTensor(2, ocean.double).broadcastTo([3,5])
>>> print(A)
(:,:)
   0   0   0   0   0
   1   1   1   1   1
   2   2   2   2   2
<tensor.double of size 3x5 on cpu>
>>> print(B)
(:,:)
   2   2   2   2   2
   2   2   2   2   2
   2   2   2   2   2
<tensor.double of size 3x5 on cpu>
>>> A.fill(3)
>>> print(A)
(:,:)
   3   3   3   3   3
   3   3   3   3   3
   3   3   3   3   3
<tensor.double of size 3x5 on cpu>
>>> A.copy(B)
>>> print(A)
(:,:)
   2   2   2   2   2
   2   2   2   2   2
   2   2   2   2   2
<tensor.double of size 3x5 on cpu>
>>> print(A.storage)
   2   2   2
<storage.double of size 3 on cpu>
>>> B.copy(A)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Zero-stride dimensions in destination tensor must be matched by the source tensor
```

In Ocean strides are represented in bytes, and it is therefore possible that strides are not integer multiples of the data type. When using `ocean.tensor` with strides, the default unit is the size of the data type. This can be overridden by specifying another unit, for example `ocean.int8.size` or equivalently `1`. When permitted by the device type, this allows the creation of tensors with data that is not aligned in memory:

```
>>> T = ocean.tensor([2],[3],ocean.int8.size,ocean.int16)
>>> T.storage.zero() 
>>> T.strides
(3,)
>>> T.isAligned()
False
>>> T.copy([10000,20000])
>>> T
   10000   20000
<tensor.int16 of size 2 on cpu>
>>> T.strides
(3,)
>>> T.storage
   10   27   00   20   4E
<raw storage of size 5 on cpu>
```

On GPU devices the data must be aligned and trying to create a tensor with unaligned memory gives an error:

```python
>>> T = ocean.tensor([2],[3],ocean.int8.size,ocean.int16,ocean.gpu[0])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Device gpu0 requires data to be aligned
```

#### Tensor properties

```python
>>> T = ocean.tensor([2,3],ocean.float)
>>> T.zero()
>>> T
(:,:)
   0   0   0
   0   0   0
<tensor.float of size 2x3 on cpu>
>>> T.footer
'<tensor.float of size 2x3 on cpu>'
>>> T.storage
   0   0   0   0   0   0
<storage.float of size 6 on cpu>
>>> 
>>> T.device
<device 'cpu'>
>>> T.dtype
<dtype 'float'>
>>> T.ndims
2
>>> T.size
(2, 3)
>>> T.strides
(3, 1)
>>> T.nelem
6
>>> # Offset within the storage data
>>> T.offset
0L
>>> # Get the element size (same as T.storage.dtype.size)
>>> T.elemsize
4
>>> # Reference count to the underlying tensor object (debugging)
>>> T.refcount
1
>>> # Pointer to the Ocean tensor object (as integer)
>>> T.obj
13130016
>>> # Pointer to the tensor data (as integer)
>>> T.ptr
161218736
>>> # Settable attribute for read-only tensors
>>> t.readonly
False
```

Additional tensor attributes are available through query functions:

```python
>>> T = ocean.tensor([2,3],ocean.cfloat)
>>> T.isReal()
False
>>> T.isComplex()
True
>>> 
>>> T.isScalar()
False
>>> T.isEmpty()
False
>>> 
>>> T.isContiguous()
True
>>> T.isLinear()
True
>>> T.isFortran()
True
>>> T.isAligned()
True
>>> T = ocean.tensor([2,3],'C',ocean.float)
>>> T.isFortran()
False
>>> T.isLinear()
False
>>> T.isContiguous()
True
>>> T.isSelfOverlapping()
False
```

#### <a name="tensor_operators">Tensor operators</a>

| Operator | Meaning |
| -------- | ------- |
| +, +=    | Elementwise addition |
| -, -=    | Elementwise subtraction |
| /, /=    | Elementwise division |
| \*       | Tensor multiplication (see [ocean.mtimes](module_core.md#tensor-operations)) |
| \*=      | Elementwise multiplication |
| @        | Elementwise multiplication (Python 3) |

In the above table note that the notational inconsistency between tensor multiplication (`*`) and in-place elementwise scaling (`*=`). The rationale for this is that `A * B` naturally reads as regular tensor multiplication rather than elementwise scaling. In-place tensor multiplication does not make too much sense, and for convenience we therefore use `*=` for that. In Python 3 there is a new operator (`@`) especially for matrix multiplication (see [PEP 465](http://legacy.python.org/dev/peps/pep-0465/) for a detailed discussion). Writing `A @ B` for tensor multiplication has the disadvantage of decreased readability as well as a limitation to use in Python 3. As a result, we use `A @ B` to denote elementwise multiplication in Python 3, rather than the intended matrix multiplication.

## Indexing

Tensors can be indexed with (combinations of) the following objects:
* Scalars - `A[3]`
* Indices (lists of iterators) - `A[[1,2,3]]`
* Ranges - `A[1:5]`, `A[:]`
* Boolean masks - `A[[True, False, True]]`
* Index objects - `A[index]`
* Ellipsis - `A[...,2]`
* None - `A[None,3]`

```python
>>> A = ocean.arange(24).reshape([4,6])
>>> A
(:,:)
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23
<tensor.int64 of size 4x6 on cpu>
>>> A[1,2]
9
>>> A[2,:]
    2    6   10   14   18   22
<tensor.int64 of size 6 on cpu>
>>> A[[0,0,2],0:4]
(:,:)
    0    4    8   12
    0    4    8   12
    2    6   10   14
<tensor.int64 of size 3x4 on cpu>
>>> A[:,[True,False,False,True,True,False]]
(:,:)
    0   12   16
    1   13   17
    2   14   18
    3   15   19
<tensor.int64 of size 4x3 on cpu>
>>> A[...,-1]
   20   21   22   23
<tensor.int64 of size 4 on cpu>
>>> A[None,3]
```

Negative scalar or index values are converted according to the Python convention. When scalar indices are given for all dimensions, an Ocean scalar object is returned. When only scalar and/or range indices are use the result is a view on the original tensor:

```python
>>> A = ocean.zeros([3,5])
>>> B = A[1,1:4]
>>> B.copy([1,2,3])
>>> B
   1   2   3
<tensor.float of size 3 on cpu>
>>> A
(:,:)
   0   0   0   0   0
   0   1   2   3   0
   0   0   0   0   0
<tensor.float of size 3x5 on cpu>
>>> # Indices cannot be used to create a view:
>>> B = A[1,[1,2,3]]
>>> B
   1   2   3
<tensor.float of size 3 on cpu>
>>> B.fill(3)
>>> B
   3   3   3
<tensor.float of size 3 on cpu>
>>> A
(:,:)
   0   0   0   0   0
   0   1   2   3   0
   0   0   0   0   0
<tensor.float of size 3x5 on cpu>
```

Boolean masks can span multiple dimensions, but the mask size must match that of the corresponding dimensions. When `None` is used in indexing a new dimension of size one is inserted in the output:

```python
>>> A[:,1]
   4   5   6   7
<tensor.int64 of size 4 on cpu>
>>> A[:,None,1]
(:,:)
   4
   5
   6
   7
<tensor.int64 of size 4x1 on cpu>
```

Multi-dimensional indices can be used to index individual elements along multiple dimensions:

```python
>>> A = ocean.zeros([3,5])
>>> A[[[0,0],[-1,0],[1,2],[0,4],[-1,-1]]] = [1,2,3,4,5]
>>> A
(:,:)
   1   0   0   0   4
   0   0   3   0   0
   2   0   0   0   5
<tensor.float of size 3x5 on cpu>
```

The ellipsis object (`...`) can appear as an index at most once, and indicates that as many `:` index elements are inserted as needed (greater than or equal than zero) to index all dimensions. When tensor dimensions are omitted the all-indices object `:` is implied:

```python
>>> A[2]
    2    6   10   14   18   22
<tensor.int64 of size 6 on cpu>
>>> A[[True,False,False,True]]
(:,:)
    0    4    8   12   16   20
    3    7   11   15   19   23
<tensor.int64 of size 2x6 on cpu>
```

Index assignment has similar modes as index extraction, except for `None`, which is not supported.

```python
>>> A = ocean.zeros([3,5])
>>> A
(:,:)
   0   0   0   0   0
   0   0   0   0   0
   0   0   0   0   0
<tensor.float of size 3x5 on cpu>
>>> A[1,2] = 6
>>> A
(:,:)
   0   0   0   0   0
   0   0   6   0   0
   0   0   0   0   0
<tensor.float of size 3x5 on cpu>
>>> A[:,-1] = [1,2,3]
>>> A
(:,:)
   0   0   0   0   1
   0   0   6   0   2
   0   0   0   0   3
<tensor.float of size 3x5 on cpu>
>>> A[A==0] = 10
>>> A
(:,:)
   10   10   10   10    1
   10   10    6   10    2
   10   10   10   10    3
<tensor.float of size 3x5 on cpu>
```

When assigning subtensors using indices, care needs to be taken to provide unique values. Currently there are no checks for repeated entries, and the behavior of the assignment is not defined (for example, the final value may not be the value corresponding to the last occurrence of the repeated index).

```python
>>> A = ocean.zeros(6)
>>> A[[1,2,-1]] = [1,2,3]
>>> A
   0   1   2   0   0   3
<tensor.float of size 6 on cpu>
>>> # Assigning with repeated indices works but the behavior is not defined!
>>> A[[0,0,0]] = [4,5,6]
>>> A
   6   1   2   0   0   3
<tensor.float of size 6 on cpu>
```

### In-place operations

In-place operations on indexed tensors are also supported:

```python
>>> A = ocean.arange(24).reshape([4,6])
>>> A[A >= 10] += 100
>>> A
(:,:)
     0     4     8   112   116   120
     1     5     9   113   117   121
     2     6   110   114   118   122
     3     7   111   115   119   123
<tensor.int64 of size 4x6 on cpu>
>>> A[A < 4] *= 10
(:,:)
     0     4     8   112   116   120
    10     5     9   113   117   121
    20     6   110   114   118   122
    30     7   111   115   119   123
<tensor.int64 of size 4x6 on cpu>
```

Note that in-place operations such as addition (`tensor[<index>] += X`) in Python are implemented as follows:

```python
__tensor = tensor[<index>]
__tensor += X
tensor[<index>] = __tensor
```

The above semantics also explain the possibly confusing result obtained when using in-place addition with multiple indices:

```python
>>> A = ocean.zeros(5)
>>> A[[0,0,0,0,1]] += 1
>>> A
   1   1   0   0   0
<tensor.float of size 5 on cpu>
```

### Index objects

In the implementation of Ocean, tensors of indices are first checked to ensure the values match the indexed tensor dimensions and are then converted to vectors of offsets (based on the strides of the indexed tensor). For boolean masks, the number of nonzero entries first needs to be determined, and the mask is then converted either to a tensor of indices, or directly to a vector of offsets. These conversion operations take time and are wasteful when repeatedly using the same boolean mask or tensor of indices. Index objects can be used to solve the aforementioned problem by using pre-computed indices.

Index objects are constructed by indexing the ocean.index object. The reason for using this syntax rather than the more conventional constructor is that certain index types (such as ranges `1:3`) cannot be used when calling a constructor. Any tensor objects, such as indices or a boolean mask are cloned to ensure that the values are not changed from outside the index.

```python
>>> A = ocean.arange(24).reshape(4,6)
>>> A   
(:,:)
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23
<tensor.int64 of size 4x6 on cpu>
>>> idx = ocean.index[1,:]
>>> idx
<tensor index [1,:]>
```

Once constructed, an index object can be used just like any other index:

```python
>>> A[idx]
    1    5    9   13   17   21
<tensor.int64 of size 6 on cpu>
>>> A[1,:]
    1    5    9   13   17   21
<tensor.int64 of size 6 on cpu>
>>> idx1 = ocean.index[1]
>>> idx2 = ocean.index[idx1,:]
>>> A[idx2]
    1    5    9   13   17   21
<tensor.int64 of size 6 on cpu>
>>> A[idx1,:]
    1    5    9   13   17   21
<tensor.int64 of size 6 on cpu>
```

Initially, index objects are not bound to any particular tensor size. For example, we may have

```python
>>> idx = ocean.index[...,-1]
>>> A = ocean.arange(5)
>>> A[idx]
4
>>> A = ocean.arange(8)
>>> A[idx]
7
>>> A = ocean.arange(24).reshape(4,6)
>>> A
(:,:)
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23
<tensor.int64 of size 4x6 on cpu>
>>> A[idx]
   20   21   22   23
<tensor.int64 of size 4 on cpu>
```

### Index properties and functions

The following member functions and properties are defined on index objects 'idx':

| Function          | Description |
| ----------------- | ----------- |
| `idx.clone()`       | Create a deep copy of the index object |
| `idx.clear()`       | Delete all elements of the index object |
| `idx.append(index)` | Appends an index object |
| `idx.bind(size [,strides] [,inplace=False])` | Bind the index dimensions and strides |
| `dix.setDevice(device [,inplace=False])` | Set the device for all tensors in the index |
| `idx.isScalar()`    | Returns `True` when the result is a scalar, and `False` otherwise |
| `idx.isView()`      | Returns `True` when the result is a view, and `False` otherwise |
| `idx.isBound()`     | Returns a tuple with two boolean values indicating whether the size and strides have been bound |

| Property   | Description |
| ---------- | ----------- |
| nelem      | Number of index elements in the index object |
| inputSize  | The input dimensions required by the index, `None` when the input dimension could not be determined |
| outputSize | The output dimensions when applying the index, `None` when the output dimensions could not be determined |
| strides    | The strides required by the tensor (after binding) |

Binding of an index to tensor dimensions, or to tensor dimensions and strides can be done using the `bind` function. When binding only the index dimensions, all negative indices are converted to the appropriate value, and boolean masks are converted to (multi-dimensional) tensor indices. When strides are provided as well boolean masks and indices are converted to offset vectors.

```python
>>> idx = ocean.index[...,-1]
>>> idx
<tensor index [...,-1]>
>>> idx.bind([3,6],True)
>>> idx
<tensor index [:,5]>
```

The following example illustrates the conversion of a boolean mask:

```python
>>> mask = [True,False,False,True]
>>> idx = ocean.index[mask]
>>> idx
<tensor index [Mask(1D)]>
>>> idx.bind(4)                   # New index object
<tensor index [Indices(1D)]>
>>> idx.bind(4,ocean.double.size) # New index object
<tensor index [Offsets(1D)]>
>>>
>>> # Provided that the sizes match it is also possible to specialize the index in stages:
>>> idx.bind(4,True)
>>> idx.bind(4,ocean.double.size,True)
>>> idx
<tensor index [Offsets(1D)]>
```

The values of the index properties can change as a result of binding:

``` python
>>> idx = ocean.index[-1,None,:,None]
>>> idx.inputSize  # None
>>> idx.outputSize # None
>>> idx.strides    # None
>>> idx.isBound()
(False, False)
>>>
>>> A = ocean.zeros([4,6])
>>> idx.bind(A.size, True)
>>> idx.inputSize
(4, 6)
>>> idx.outputSize
(1, 6, 1)
>>> idx.strides    # None
>>> idx.isBound()
(True, False)
>>> idx.isView()
True
>>> idx.isScalar()
False
>>>
>>> idx.bind(A.size, A.strides, True)
>>> idx.strides
(4, 16)
>>> idx.isBound()
(True, True)
```

Applying an index with bound size or strides to a tensor with mismatching size or strides gives an error:

```python
>>> idx = ocean.index[:]
>>> idx.bind(5,True)
>>> A = ocean.zeros(6)
>>> A[idx]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Mismatch in size at dimension 0 (expected 5 got 6)

>>> idx.bind(5,2,True)
>>> A = ocean.zeros(5,ocean.bool)
>>> A[idx]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Mismatch in strides at dimension 0 (expected 2 got 1)

>>> A = ocean.zeros(5,ocean.int16)
>>> A[idx]
   0   0   0   0   0
<tensor.int16 of size 5 on cpu>
```

Index objects can contain tensors for indices, boolean mask, or offsets. These tensors are all maintained on the device of the original tensor. In order to ensure that all tensors in the index are on a certain device, it is possible to use casting with a device:

```python
>>> idx1 = ocean.index([1,2,3],[True,False])
>>> idx2 = ocean.gpu[0](idx1)
>>> idx1.setDevice(ocean.gpu[0], True)
```


## Casting with data types and devices

Data types and devices can be called to cast the input argument to the desired data type or device. An example of this was seen in the instantiation of scalars, in which Python long or float values were converted to Ocean scalars. Scalars are available only on the CPU and device conversion (even to CPU) converts them to tensors.

```python
>>> # Conversion of numeric values and scalars
>>> s = ocean.float(5.6)
>>> t = ocean.int8(s)
>>> t
5
>>> t.dtype
<dtype 'int8'>
>>> ocean.cpu(1.2)
1.2
<scalar.double on cpu>
>>> ocean.cpu([1,2,3])
   1   2   3
<tensor.int64 of size 3 on cpu>
>>> ocean.gpu[0](s)
5.6
<scalar.double on gpu0>
```

When applied to storage we can change both the data type and the device. When the destination type or device match the current storage we simply return the object itself:
```python
>>> # Conversion of storage
>>> s = ocean.storage(5)
>>> s.asTensor().fill(1)
>>> s
   1   1   1   1   1
<storage.float of size 5 on cpu>
>>> t = ocean.chalf(s)
>>> t
   1 + 0j    1 + 0j    1 + 0j    1 + 0j    1 + 0j
<storage.complex-half of size 5 on cpu>
>>> ocean.gpu[0](t)
   1 + 0j    1 + 0j    1 + 0j    1 + 0j    1 + 0j
<storage.complex-half of size 5 on gpu0>
```

Device casting can also be applied to index objects to ensure that any tensors (indices, boolean mask, or offsets) are on the given device. For more information on casting with device and data type objects, see the core module documentation.


## Warning configuration

The warning level can be controlled and queried using the `ocean.setWarningMode(mode)` and `ocean.getWarningMode()` functions. The three modes are `0` to suppress all warning, `1` to output each warning only once, and `2` to output all warnings. Due to the way warnings are filtered in Python, however, even with `mode` set to `2`, each unique warning is displayed only once. By default the warning mode is set to `1`.

```python
>>> ocean.setWarningMode(0) 
>>> ocean.sqrt([-1],'w')
   nan
<tensor.double of size 1 on cpu>
>>> ocean.setWarningMode(1) 
>>> ocean.sqrt([-1],'w')
__main__:1: RuntimeWarning: Tensor elements must be nonnegative for square-root
   nan
<tensor.double of size 1 on cpu>
>>> ocean.sqrt([-1],'w')
   nan
<tensor.double of size 1 on cpu>
```


## Math modes

There are four math modes, which determine how domain constraints are dealt with in tensor functions such as `sqrt` (in this case for dealing with negative numbers). Each mode is indicated by a character string: The default `-` mode silently ignores domain issues (in fact, the domain is not checked prior to the operation, thereby slightly reducing the runtime and avoiding synchronization), mode 'w' checks the domain and raises a warning when a violation is detected, 'e' does the same but raises an error. Finally, mode 'c' casts the data type if needed (for the square-root example, it would switch to complex numbers). For scalars the input domain is always checked, and type conversions are done more freely. Note that automatic type casting modes (see below) do not affect the behavior for the `c` mode. The math mode can be specified explicitly with all relevant functions, and the default can be set and checked using the `ocean.setDefaultMathMode(mode)`, and `ocean.getDefaultMathMode()` functions. Note that explicit control in the relevant functions is useful especially in general-purpose libraries where it may not be desirable to have the behavior depend on global settings.

```python
>>> # --- Silent mode (default) ---
>>> ocean.setDefaultMathMode('-')
>>> ocean.sqrt([-1]) # ocean.sqrt([-1],'-')
   nan
<tensor.double of size 1 on cpu>
>>> # --- Warning mode ---
>>> ocean.setDefaultMathMode('w')
>>> ocean.sqrt([-1]) # ocean.sqrt([-1],'w')
__main__:1: RuntimeWarning: Tensor elements must be nonnegative for square-root
   nan
<tensor.double of size 1 on cpu>
>>> # --- Error mode ---
>>> ocean.setDefaultMathMode('e')
>>> ocean.sqrt([-1]) # ocean.sqrt([-1],'e')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor elements must be nonnegative for square-root
>>> # --- Cast mode ---
>>> ocean.setDefaultMathMode('c')
>>> ocean.sqrt([-1]) # ocean.sqrt([-1],'c')
   0 + 1j
<tensor.complex-double of size 1 on cpu>
```

## Automatic type conversion

By default automatic type and device conversion is enabled:

```python
>>> A = ocean.asTensor([1,2,3],ocean.double,ocean.cpu)
>>> B = ocean.asTensor([1,2,3],ocean.int8,ocean.cpu)
>>> C = ocean.asTensor([1,2,3],ocean.int8,ocean.gpu[0])
>>> A+C
   2   4   6
<tensor.double of size 3 on cpu>
>>> C+B
   2   4   6
<tensor.int8 of size 3 on gpu0>
>>> # --- Disable auto typecasting ---
>>> ocean.setAutoTypecast(False)
>>> A+B
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor data type mismatch (automatic type casting is disabled)
>>> B+C
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor device mismatch (automatic type casting is disabled)
```

The following examples illustrate the behavior of implicit scalar tensors:

```python
>>> A + 1
   2   3   4
<tensor.double of size 3 on cpu>
>>> A + 0.5 
   1.5   2.5   3.5
<tensor.double of size 3 on cpu>
>>> C + 1
   2   3   4
<tensor.int8 of size 3 on gpu0>
>>> C + 0.5
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor data type mismatch (automatic type casting is disabled)
```

The last example fails because adding double to integer gives a result of type double, which requires casting `c` from int8 to double. 
Note that adding a large integer to `c` also causes an error, because it requires `c` to be cast to a larger integer type:

```python
>>> C + 200
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor data type mismatch (automatic type casting is disabled)
```

When automatic typecasting is disabled implicit tensors must be of the correct data type (they too would otherwise have to be cast to another type):

```python
>>> B + [1]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor data type mismatch (automatic type casting is disabled)
>>> B + ocean.int8([1])
   2   3   4
<tensor.int8 of size 3 on cpu>
```

## Broadcasting of dimensions

Many operations on tensors support automatic broadcasting of the dimensions. Consider for example the copy of a tensor A of size 3 to a tensor B of size 3x4:

```python
>>> A = ocean.asTensor([1,2,3],ocean.double)
>>> print(A)
   0   1   2
<tensor.double of size 3 on cpu>
>>> B = ocean.zeros([3,4])
>>> print(B)
(:,:)
   0   0   0   0
   0   0   0   0
   0   0   0   0
<tensor.float of size 3x4 on cpu>
>>> B.copy(A)
>>> B
(:,:)
   1   1   1   1
   2   2   2   2
   3   3   3   3
<tensor.float of size 3x4 on cpu>
```

What happens here is that tensor A is automatically extended to a 3x4 tensor prior to the copy. Broadcasting of dimensions can generally be done whenever the dimensions can be made to match by adding new dimensions at the end and possibly by changing singleton dimensions to larger values matching the reference size. In the above example this means that a scalar tensor (zero dimensions), or a scalar vector (one dimension of size 1) would match. A 1x4 tensor or a 3x1 tensor would also match, but a 2x4 tensor would not.

```python
>>> A = ocean.asTensor([[1,2,3,4]],"R")
>>> A
(:,:)
   1   2   3   4
<tensor.int64 of size 1x4 on cpu>
>>> B = ocean.zeros([3,4])
>>> B.copy(A)
>>> B
(:,:)
   1   2   3   4
   1   2   3   4
   1   2   3   4
<tensor.float of size 3x4 on cpu>
```

Automatic broadcasting of dimensions can be controlled using the `ocean.getAutoBroadcast` and `ocean.setAutoBroadcast` functions:

```python
>>> B.copy(A)
>>> print(ocean.getAutoBroadcast())
True
>>> ocean.setAutoBroadcast(False)
>>> print(ocean.getAutoBroadcast())
False
>>> B.copy(A)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor size mismatch (automatic broadcasting of tensor dimensions is disabled)
```

The example below shows the behavior for operations such as addition. Note that scalars and scalar tensors continue to be broadcast even when automatic broadcasting is disabled.

```python
>>> # --- Enable automatic broadcasting ---
>>> ocean.setAutoBroadcast(True)
>>> A = ocean.asTensor([1,2,3],ocean.double)
>>> A + 1
   2   3   4
<tensor.double of size 3 on cpu>
>>> A + [1]
   2   3   4
<tensor.double of size 3 on cpu>
>>>
>>> # --- Disable automatic broadcasting ---
>>> ocean.setAutoBroadcast(False)
>>> A + 1
   2   3   4
<tensor.double of size 3 on cpu>
>>> A + [1]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Mismatch in tensor dimensions (automatic broadcasting of tensor dimensions is disabled)
```

Explicit broadcasting of tensors can be done using the `broadcastLike` function (see the [Core](module_core.md) module).

## Tensor reference counting

In Python it is important to understand the behavior of objects and references to them. Consider for example:

```python
>>> a = [1,2,3]
>>> b = a
>>> b.append(4)
>>> a
[1, 2, 3, 4]
```

In the assignment `b = a` a new variable `b` is introduced and the reference count to the object pointed to by `a` is incremented. That means that `a` and `b` represent the same object. The Python interface to Ocean is implemented by wrapping tensor and storage object pointers in Python objects (accessible through `tensor.obj` and `storage.obj`). Tensor objects are reference counted have pointers to storage objects (see `tensor.storage` and `tensor.storage.obj`), which themselves are reference counted. Sharing of objects in this way can reduce the memory footprint, runtime, and be useful in many other cases.

Let's start with a simple example of some of the sharing behavior:

```python
>>> t = ocean.tensor([2,3], ocean.float)
>>> print(t.size)
(2, 3)
>>> print(t.strides)
(4, 8)
>>> print(t.refcount)
1
>>> print(t.storage.refcount)
2
```

we construct a Python object `t` which references an Ocean tensor object. The tensor is only accessible through `t` and therefore has a reference count of 1. It may seem that the reference count to the storage also equals 1, but creation of the intermediate object `t.storage` temporarily increases the reference count to 2.

We now cast `t` to its own data type to create a new python object `s`. As can be seen from the code below, the Python objects `t` and `s` differ, but point to the same underlying tensor, which consequently has a reference count of 2. The reference count to storage remains the same.

```python
>>> s = t.dtype(t)
>>> print(s is t)
False
>>> print(s.obj ==t.obj)
True
>>> print(t.refcount)
2
>>> print(t.storage.refcount)
2
```

We now apply an axes swap to generate a new tensor `r`. Because the operation is not in-place we create a new underlying tensor, and the tensor objects no longer match. The storage referenced by the tensors does remain shared, and changing the elements in `r` therefore changes the elements in `s` and `t`.

```python
>>> r = t.swapAxes(0,1)
>>> print(t.size)
(2, 3)
>>> print(r.size)
(3, 2)
>>> print(t.obj == r.obj)
False
>>> print(t.refcount)
2
>>> print(r.refcount)
1
>>> print(t.storage.refcount)
3
>>> print(t.storage.obj == r.storage.obj)
True
```

To avoid obscure bugs, Ocean uses the convention that operations that reshape a tensor create a new underlying tensor object. When reshape operations are done in-place the Python tensor object remains the same, but the underlying tensor object is replaced:

```python
>>> t.flipAxis(0,True)
>>> print(t.size)
(2, 3)
>>> print(t.strides)
(-4, 8)
>>> print(t.obj == s.obj)
False
>>> print(s.strides)
(4, 8)
>>> print(r.size)
(3, 2)
>>> print(r.strides)
(8, 4)
```

**Detach operations**

In some cases it may be desirable to control the references. For example, suppose we want to make sure that a tensor is of type float.

```python
>>> s = ocean.float(t)
```
When `t` has a type other than float a new tensor object will be created, otherwise a shallow copy of `t` is returned. To make sure that the data is not shared we can use the `detach` function. As an example consider casting a tensor that already has the correct type (in this case `ocean.float(t,True)` would also achieve the same):

```python
>>> t = ocean.tensor([3],ocean.float)
>>> s = ocean.float(t)
>>> print(s.storage == t.storage)
True
>>> s.detach()
>>> print(s.storage == t.storage)
False
```

In other words, `detach` ensures that the reference count to the underlying storage is 1. Storage is also copied when the reference count is 1 but the data is not owned:

```python
>>> import numpy as np
>>> import pyOceanNumpy
>>> a = np.asarray([1,2,3])
>>> b = ocean.asTensor(a)
>>> b.storage.owner
False
>>> b.storage.refcount -1 # Exclude the reference from the b.storage object
1
>>> b.detach()
>>> b.storage.owner
True
```

**Tensor deallocation**

Automatic garbage collection in Python can delay the deletion of a tensor even though there are no more references to it. This can cause problems with available memory, especially when the to-be-deleted tensor is large. The `dealloc` tensor function can be used to decrement the reference count to the underlying Ocean tensor, which ensures that memory is freed when the Ocean tensor reference count reaches zero. The tensor object itself is not deleted, but the underlying tensor is replaced by an empty Ocean tensor, to ensure that any subsequent operation on the tensor does not cause any problems.

```python
>>> a = ocean.zeros([3,4])
>>> a
(:,:)
   0   0   0   0
   0   0   0   0
   0   0   0   0
<tensor.float of size 3x4 on cpu>
>>> a.dealloc()
>>> a
<empty tensor.int8 of size 0 on cpu (read-only)>
```

## Formatting

The formatting of tensors and storage can be controlled using a number of functions.

```python
>>> # Set the width of the display in characters
>>> ocean.setDisplayWidth(60)
```

## Numpy and other tensor types

Support for external tensor formats can be added by writing plug-ins that provide conversion routines to Ocean. One example of this is the plug-in for Numpy. Once loaded it is possible to use Numpy scalars and arrays as if they were the Ocean equivalent:

```python
>>> import ocean
>>> import numpy as np
>>> import pyOceanNumpy
>>> a = np.asarray([1,2,3])
>>> b = ocean.tensor([3])
>>> b.copy(a)
>>> print(b)
   1   2   3
<tensor.float of size 3 on cpu>
>>>
>>> b.fill(np.int16(5))
>>> print(b)
   5   5   5
<tensor.float of size 3 on cpu>
>>>
>>> c = ocean.asTensor([a,[4,5,6]], "R", ocean.float, ocean.gpu[0])
>>> print(c)
(:,:)
   1   2   3
   4   5   6
<tensor.float of size 2x3 on gpu0>
>>>
>>> d = c.convertTo('numpy')
>>> print(repr(d))
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)
```

For a discussion on the tensor dimension order and conversion between order types, see the "Tensor dimension order" section above.
The interpretation of the dimension order in Numpy differs from that in Ocean; for example a tensor of size (m,n,k) in Numpy represents m matrices of size n-by-k, whereas in Ocean it represents k matrices of size m-by-n. The `reverseAxes` and `reverseAxes2` functions in Ocean are provided to simplify data import. The `reverseAxes` function reverses all dimensions, which is useful, for example, when rows of the tensor represent data vectors. Applying this to the (k,n,m) tensor gives a tensor view of size (m,n,k). The `reverseAxes2` function reverses all dimensions and, provided that the tensors has dimension at least two, then switches the first and second dimension in the result. For the (k,m,n) tensor, this results in a tensor view of size (m,n,k). More information on the reverse axes functions can be found in the [core module](module_core.md) documentation.

## CTypes

The raw data for tensors and storage can be accessed though `tensor.ptr` and `storage.ptr`. Whenever the data resides on the cpu it could therefore be manipulated using ctypes although this is not recommended. In particular great care needs to be taken with respect to strides, as well as with asynchronous computations. Here is an example in which a floating point tensor is filled using ctypes:

```python
>>> import ocean
>>> import ctypes
>>>
>>> # Create a ctypes data type
>>> FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
>>>
>>> # Create a tensor and get a pointer to the data
>>> t   = ocean.tensor([3,4])
>>> ptr = ctypes.cast(t.ptr, FLOAT_PTR)
>>>
>>> # Fill the tensor
>>> for i in range(12) :
...    ptr[i] = i
>>>
>>> print(t)
(:,:)
    0    3    6    9
    1    4    7   10
    2    5    8   11
<tensor.float of size 3x4 on cpu>
```

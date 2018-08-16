# Module: Core

* [Tensor construction](#tensor-construction) (tensor, zeros, ones, full, tensorLike, zerosLike, onesLike, fullLike, asTensor, diag, eye, arange, linspace)
* [Tensor casting and conversion](#tensor-casting-and-conversion) (ensure, cast, byteswap, convertTo, asScalar, asPython, asContiguous)
* [Tensor copy functions](#tensor-copy-functions) (copy, clone, replicate, shallowCopy, sync)
* [Tensor info functions](#tensor-info-functions) (isEmpty, isScalar)
* [Tensor shape functions](#tensor-shape-functions) (transpose, ctranspose, reshape, broadcastTo, broadcastLike, squeeze, unsqueeze, flatten, flipAxis, fliplr, flipud, swapAxes, reverseAxes, reverseAxes2, permuteAxes)
* [Tensor extraction and combination](#tensor-extraction-and-combination) (real, imag, diag, slice, split, merge, find)
* [Tensor operations](#tensor-operations) (operators (+,-,\*,/ etc), fill, fillNaN, zero, multiply, scale, divide, add, subtract, negative, conj, fabs, absolute, sign, ceil, floor, trunc, round, mod, fmod, min, max, fmin, fmax, bitwiseAnd, bitwiseOr, bitwiseXor, bitwiseNot, bitshiftLeft, bitshiftRight, logicalAnd, logicalOr, logicalXor, logicalNot, reciprocal, power, sqrt, cbrt, square, sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, exp, exp2, exp10, expm1, log, log2, log10, log1p, isinf, isnan, isfinite, isposinf, isneginf)
* [Tensor reductions](#tensor-reductions) (any, all, allFinite, anyInf, anyNaN, allLT, allLE, allGT, allGE, allInRange, nnz, nnzNaN, sum, sumNaN, sumAbs, sumAbsNaN, prod, prodNaN, minimum, minimumAbs, maximum, maximumAbs, norm, norm1, norm2, normInf, normNaN)
* [Tensor saving and loading](#tensor-saving-and-loading)
* [Tensor reference management](#tensor-reference-management) (refcount, detach, dealloc)

## Storage functions

## Tensor construction
##### `ocean.tensor`

Create a new tensor. There are two parameter modes:
* `ocean.tensor(size [,strides [,unitsize]] [,dtype] [,device] [,stream])`
* `ocean.tensor(storage [,offset [,size [,strides [,unitsize]]]] [,dtype]])`

In the first mode we create a new tensor of given size with uninitialized data. The strides can be specified if needed, otherwise the default column-major ordering is used. Strides can be a list of integers matching the size and unitsize is a scalar by which the entries of the strides and offset are multiplied. By default it is set to the size of the data type. In case strides are given in bytes, the unitsize can be set to 1. The strides parameter can also be a character with 'C' denoting C-style row major, 'F' denoting Fortran-style column major, and 'R' denoting a mixed mode similar to column major, but with the stride order of the first two dimensions exchanged (such that, for two-dimensional tensors and higher, applying `transpose` or `swapAxes(0,1)` will give a column-major tensor). Additional optional parameters are the data type, device, and a stream. When a stream is given, the device type has to match the device type corresponding to the stream.

In the second mode we create a new tensor from existing storage. The offset is given in terms of the unitsize (which by default is set to the data type). When omitted the data type is set to the data type of the storage. In case the storage has no associated data type, the default data type is used.

```python
>>> t = ocean.tensor([3,4])
>>> print(t.footer)
<tensor.float of size 3x4 on cpu>
>>> print(t.elemsize)
4
>>> t.strides
(4, 12)
>>> t = ocean.tensor([3,4],'C')
>>> t.strides
(16, 4)
>>> t = ocean.tensor([3,4],[1,3])
>>> t.strides
(4, 12)
>>>
>>> t = ocean.asTensor(range(8), ocean.float)
>>> s = t.storage; print(s)
   0   1   2   3   4   5   6   7
<storage.float of size 8 on cpu>
>>> t = ocean.tensor(s); print(t)
   0   1   2   3   4   5   6   7
<tensor.float of size 8 on cpu>
>>> t = ocean.tensor(s,2); print(t)
   2   3   4   5   6   7
<tensor.float of size 6 on cpu>
>>> t = ocean.tensor(s,1,[2,2]); print(t)
(:,:)
   1   3
   2   4
<tensor.float of size 2x2 on cpu>
>>> t = ocean.tensor(s,0,[2,4],[1,2]); print(t)
(:,:)
   0   2   4   6
   1   3   5   7
<tensor.float of size 2x4 on cpu>
>>> t = ocean.tensor(s,0,[2,3],'F'); print(t)
(:,:)
   0   2   4
   1   3   5
<tensor.float of size 2x3 on cpu>
```

##### `ocean.zeros(size [,dtype] [,device])`, `ocean.ones(size [,dtype] [,device])`, `ocean.full(size, value [,dtype] [,device]`

Create a new tensor of given size filled with 0, 1, or the provided value. When provided the data type and device will be used. When omitted the default values will be used for `zeros` and `ones`. For `full` the data type is inferred from the value, unless the value is a standard Python type (int, long, complex). The data order is always column-major; if another order is needed, use `ocean.tensor` followed by `tensor.zero()` or `tensor.fill(value)`.

```python
>>> print(ocean.zeros([2,3]))
(:,:)
   0   0   0
   0   0   0
<tensor.float of size 2x3 on cpu>
>>> print(ocean.ones([6],ocean.half))
   1   1   1   1   1   1
<tensor.half of size 6 on cpu>
>>> print(ocean.full([3], 2, ocean.cpu)) # Default float
   2   2   2
<tensor.float of size 3 on cpu>
>>> print(ocean.full([3], ocean.int16(3))) # Strongly-typed value (int16)
   3   3   3
<tensor.int16 of size 3 on cpu>
```

##### `ocean.tensorLike(tensor)`, `ocean.zerosLike(tensor)`, `ocean.onesLike(tensor)`, `ocean.fullLike(tensor, value)`

Create a new tensor (uninitialized, zeros, ones, and filled with a constant) with size, data type, and device matching the input tensor.

##### `ocean.asTensor`

Create a new tensor from existing tensors or tensor-like data. There are two parameter modes:
* `ocean.asTensor(tensor [,dtype] [,device] [,deepcopy])`
* `ocean.asTensor(tensor-like [,scalar] [,order] [,dtype] [,device]])`

In the parameter parsing tensors are defined as any valid tensor object, which can be an Ocean tensor or any tensor type defined by extension modules such as the ocean_numpy module. Tensor-like objects are more general and include tensors, scalars, and nested lists or tuples of tensor-like objects. When no `dtype` omitted, the data type is automatically inferred to the minimum possible data type needed to store the values (see [design choices](/docs/design_choices.md#data-types) for details). The scalar padding value is not used in the determination of the data type. The default data type for empty tensors is Boolean.

When the tensor device is not specified it is determined as follows. When all tensors occurring in the data specification share the same device, this device will be used. Scalars are given a weak CPU device type, which means that they do not override any device information by the tensor components. When the tensor specification contains only scalars, or tensors with different devices, the device is set to CPU.

The first syntax is intended to make shallow or deep copies of existing tensors, especially non-Ocean tensors. The tensor is copied to the given data type and device, when provided. The deepcopy flag can be set to force a deep copy of the tensor. By default a shallow copy of the tensor is made, provided that the tensor type supports it and that the data type and device are omitted or otherwise match that of the tensor.

```python
import numpy as np
import ocean_numpy
A = np.asarray([[1,2,3],[4,5,6]])
B = ocean.asTensor(A)
print(B)
B.fill(3)
print(A)

A = np.asarray([[1,2,3],[4,5,6]])
B = ocean.asTensor(A, True)
B.fill(3)
print(A)
```

Note: there may be a difference in the interpretation of the tensor dimension order between Ocean and other packages. These differences can be corrected by reordering the dimensions. See for example the `reverseAxes`, `reverseAxes2`, and `permuteAxes` functions.

The second syntax is intended to construct custom tensors, for example from Python data. We start with a simple example:

```python
>>> A = ocean.asTensor([1,2,3],ocean.float)
>>> A
   1   2   3
<tensor.float of size 3 on cpu>
>>> ocean.asTensor([[1,2,3],[4,5,6]],ocean.gpu[0])
(:,:)
   1   4
   2   5
   3   6
<tensor.int64 of size 3x2 on gpu0>
```

Lists or tuples are processed from the outside to the inside, from the right-most to the left-most dimension. The innermost lists or tuples therefore correspond to columns of the tensor. Sometimes it is more convenient to specify the tensor by row rather than column. This is where the `order` parameter can be used to specify the ordering of the tensor elements: 'F', 'C', or 'R'. Suppose we have a nested list of depth three as input with sizes s3 for the outermost, s2 for the middle level, and s1 for the innermost level. For all modes we create and fill a tensor of size (s1,s2,s3). In the default Fortran style order 'F' we return this matrix. In the C-style mode 'C' the initial tensor has strides increasing from the last (rightmost) to the first dimension (leftmost). After copying the data, the axis order is reversed (`reverseAxes`) thus obtaining a Fortran-style tensor of size (s3,s2,s1). In the row-based mode "R" strides are in Fortran order except for the first two axes, which are swapped. After copying the data, the first two axes are again exchanged resulting in a Fortran order tensor of size (s2,s1,s3). Note that regardless of the order type, the result tensor always has a Fortran order.

```python
>>> ocean.asTensor([[[1,2],[3,4]],[[5,6],[7,8]]],"F",ocean.int8)
(:,:,0)
   1   3
   2   4

(:,:,1)
   5   7
   6   8
<tensor.int8 of size 2x2x2 on cpu>
>>> ocean.asTensor([[[1,2],[3,4]],[[5,6],[7,8]]],"C",ocean.int8)
(:,:,0)
   1   3
   5   7

(:,:,1)
   2   4
   6   8
<tensor.int8 of size 2x2x2 on cpu>
>>> ocean.asTensor([[[1,2],[3,4]],[[5,6],[7,8]]],"R",ocean.int8)
(:,:,0)
   1   2
   3   4

(:,:,1)
   5   6
   7   8
<tensor.int8 of size 2x2x2 on cpu>
```

Specifying the `scalar` parameter indicates that the tensor elements may differ in size (but not in number of dimensions), and is used as the default value when entries are missing:

```python
>>> ocean.asTensor([[1],[2,3],[4,5,6]],0)
(:,:)
   1   2   4
   0   3   5
   0   0   6
<tensor.int64 of size 3x3 on cpu>
>>> ocean.asTensor([ocean.full([3,1],1),ocean.full([2,5],2),ocean.full([2,3],3)],0)
(:,:,0)
   1   0   0   0   0
   1   0   0   0   0
   1   0   0   0   0

(:,:,1)
   2   2   2   2   2
   2   2   2   2   2
   0   0   0   0   0

(:,:,2)
   3   3   3   0   0
   3   3   3   0   0
   0   0   0   0   0
<tensor.int64 of size 3x5x3 on cpu>
```

Iterators are allowed as entries of vectors:

```python
>>> ocean.asTensor([range(5),xrange(3)],0,"R",ocean.int8)
(:,:)
   0   1   2   3   4
   0   1   2   0   0
<tensor.int8 of size 2x5 on cpu>
```


##### `ocean.diag(value [,index] [,dtype] [,device])`

Creates a two-dimensional diagonal matrix with the diagonal entries set to the entries in the value vector. The index of the diagonal can be selected, with negative values indicating an offset along the rows, and a positive value along the columns. The size is determined based on the index and the length of the value vector. When omitted, the data type and device are set to those of value vector.

```python
>>> ocean.diag([1,1,1], ocean.float)
(:,:)
   1   0   0
   0   1   0
   0   0   1
<tensor.float of size 3x3 on cpu>
>>> ocean.diag([1,2,3], 2)
(:,:)
   0   0   1   0   0
   0   0   0   2   0
   0   0   0   0   3
   0   0   0   0   0
   0   0   0   0   0
<tensor.int64 of size 5x5 on cpu>
```

##### `ocean.eye(rows [,columns=rows [,index=0]] [,dtype] [,device])`

Creates a two-dimensional tensor of size rows-by-columns with ones on the diagonal indicated by index (see `tensor.diag`). When omitted the default data type or device is used.

```python
>>> print(ocean.eye(3))
(:,:)
   1   0   0
   0   1   0
   0   0   1
<tensor.float of size 3x3 on cpu>
>>> print(ocean.eye(3,5,2,ocean.int16))
(:,:)
   0   0   1   0   0
   0   0   0   1   0
   0   0   0   0   1
<tensor.int16 of size 3x5 on cpu>
```

##### `ocean.arange([start,] stop [,step] [,dtype] [,device])`

Creates a vector with entries from start up to stop (not included) with step increments. By default start value is 0 and the default step size is 1. When omitted, the data size is inferred from the range parameters or the default value. An exception is raised if any of the values in the range falls outside the data type range

```python
>>> print(ocean.arange(6))
   0   1   2   3   4   5
<tensor.float of size 6 on cpu>
>>> print(ocean.arange(2,7))
   2   3   4   5   6
<tensor.float of size 5 on cpu>
>>> print(ocean.arange(0,4,0.5,ocean.int8,ocean.gpu[0]))
   0   0   1   1   2   2   3   3
<tensor.int8 of size 8 on gpu0>
>>> print(ocean.arange(200,280,20,ocean.uint8))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: The stop value (260) is out of range for data type uint8
```
Note that the behavior of `arange` differs from that in Numpy in two of the above examples:

```python
>>> import numpy as np
>>> np.arange(0,4,0.5,np.int8)
array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
>>> np.arange(200,280,20,np.uint8)
array([200, 220, 240,   4], dtype=uint8)
```
From the first example we see that Numpy implements the arange operation by first determining the number of elements in the range, and then generating the range with start and step values cast to the output data type (giving a step size of 0 in this case). In the second example we see that the range is modulo 256, as a result of default addition arithmetic on uint8. Another example where the behavior differs is when the increment is infinity:

```python
>>> np.arange(10,20,100)
array([10])
>>> np.arange(10,20,np.inf)
array([], dtype=float64)
>>> ocean.arange(10,20,100)
   10
<tensor.float of size 1 on cpu>
>>> ocean.arange(10,20,ocean.inf)
   10
<tensor.float of size 1 on cpu>
```

##### `ocean.linspace(start, stop [,num] [,endpoint [,spacing]] [,dtype] [,device])`

Creates a linear space from start to stop with `num` points. The `endpoint` flag indicates whether the stop value has to be included, and has a default value of True. When the spacing flag is set, the output will be a tuple containing the range and the step size; by default `spacing` is set to False. Like `ocean.arange` the `linspace` function checks if all values fit in the given data type; if not an exception will be raised.

```python
>>> print(ocean.linspace(0,10,5))
    0.0    2.5    5.0    7.5   10.0
<tensor.float of size 5 on cpu>
>>> print(ocean.linspace(0,10,5,False))
   0   2   4   6   8
<tensor.float of size 5 on cpu>
>>> print(ocean.linspace(1+2j,2+1j,5,ocean.chalf,ocean.gpu[1]))
   1.00 + 2.00j    1.25 + 1.75j    1.50 + 1.50j    1.75 + 1.25j    2.00 + 1.00j
<tensor.complex-half of size 5 on gpu1>
>>> a = ocean.linspace(0,2,5,True,True)
>>> a[0] # Range
   0.0   0.5   1.0   1.5   2.0
<tensor.float of size 5 on cpu>
>>> a[1] # Step size
0.5
>>> # Example of out-of-range values
>>> print(ocean.linspace(0,1+70000j,ocean.chalf))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: The stop.imag value (7.000000e+04) is out of range for data type complex-half
```


## Tensor casting and conversion

##### `ocean.ensure(obj [,dtype] [,device] [,inplace=False])`

The `ensure` function ensures that the object has the given data type and device. When the `inplace` flag is set, the object data type and device are updated in place, otherwise a copy of the object (shallow whenever possible) is returned.

The following rules apply for input objects:

* Tensor objects: In in-place mode, applies the data type and device to the tensor object. When `inplace` is `False` a shallow copy of the tensor when data type and device match (note that in this case only the storage is shared, the tensor object itself is copied), otherwise a copy of the tensor is returned.
* Storage objects: Similar semantics as for tensor objects, but for storage objects.
* Scalar objects: In in-place mode only data type conversions are allowed, when a device is given (even the CPU device) an error is returned. When `inplace` is `False` a copy of the scalar object is returned when no device is given, otherwise the scalar is converted to a tensor object.
* Tensor-like and scalar-like objects: In-place operations are not allowed. The semantics otherwise is similar to tensor and scalar objects, respectively.

```python
>>> A = ocean.asTensor([1,2,3])
>>> print(ocean.ensure(A, ocean.int8))
   1   2   3
<tensor.int8 of size 3 on cpu>
>>> ocean.ensure(A, ocean.float, ocean.gpu[0], True)
>>> print(A)
   1   2   3
<tensor.float of size 3 on gpu0>
>>>
>>> # Check shallow copy
>>> B = ocean.ensure(A,A.dtype)
>>> A.obj == B.obj
False
>>> A.storage.obj == B.storage.obj
True
```

Conversion of storage and tensor-like objects
```python
>>> A = ocean.ensure([1,2,3],ocean.chalf,ocean.gpu[1])
>>> print(A)
   1 + 0j    2 + 0j    3 + 0j
<tensor.complex-half of size 3 on gpu1>
>>> S = A.storage
>>> ocean.ensure(S,ocean.int16,ocean.cpu,True)
>>> print(S)
   1   2   3
<storage.int16 of size 3 on cpu>
>>> print(A)
   1 + 0j    2 + 0j    3 + 0j
<tensor.complex-half of size 3 on gpu1>
```

Scalars
```python
>>> a = ocean.uint8(9)
>>> print([a, a.dtype])
[9, <dtype 'int64'>]
>>> ocean.ensure(a, ocean.int8, True)
>>> print([a, a.dtype])
[9, <dtype 'int8'>]
>>> b = ocean.ensure(5, ocean.float)
>>> print([b, b.dtype])
[5, <dtype 'int64'>]
>>> # Prescribing a device for scalars gives a tensor
>>> ocean.ensure(b, ocean.cpu)
5
<scalar.int64 on cpu>
>>> # In-place specification of a device for scalars is not allowed
>>> ocean.ensure(b, ocean.cpu, True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: In-place ensure device does not apply to scalar objects
```

##### `device(obj [,inplace=False])`

The `device(obj [,inplace=False])` function is equivalent to `ocean.ensure(obj, device [,inplace=False])`.

```python
>>> A = ocean.asTensor([1,2,3])
>>> ocean.gpu[1](A)
   1   2   3
<tensor.int64 of size 3 on gpu1>
>>> ocean.gpu[1](A.storage)
   1   2   3
<storage.int64 of size 3 on gpu1>
>>> ocean.gpu[0]([[1,2,3],[4,5,6.]])
(:,:)
   1   4
   2   5
   3   6
<tensor.double of size 3x2 on gpu0>
>>> ocean.gpu[0](1.2)
1.2
<scalar.double on gpu0>
```

##### `dtype(obj [,inplace=False])`

The `dtype(obj [,inplace=False])` function is equivalent to `ocean.ensure(obj, dtype [,inplace=False])`.

```python
>>> A = ocean.asTensor([1,2,3], ocean.float, ocean.gpu[0])
>>> ocean.int8(A)
   1   2   3
<tensor.int8 of size 3 on gpu0>
>>> ocean.cdouble(A.storage)
   1 + 0j    2 + 0j    3 + 0j
<storage.complex-double of size 3 on gpu0>
>>> ocean.half([[1,2,3],[4,5,6.]],"R")
(:,:)
   1   2   3
   4   5   6
<tensor.half of size 2x3 on cpu>
>>> ocean.float(1.2)
1.2
<scalar.float on cpu>
```
###### `bool(tensor)`, `int(tensor)`, `long(tensor)`, `float(tensor)`, `complex(tensor)`

These type cast can be applied to scalar tensors to return the corresponding Python scalar. The Boolean conversion operation is also used implicitly when using the tensor as a logical value.

```python
>>> a = ocean.asTensor([[1+2j]],ocean.chalf)
>>> print(bool(a))
True
>>> print(int(a))
1
>>> print(long(a))
1
>>> print(float(a))
1.0
>>> print(complex(a))
(1+2j)
>>> if a :
...     print("OK")
... 
OK
>>>
```

###### `tensor.byteswap()`, `tensor.byteswapped` (property)

Swap the byte order of the tensors (when supported by the device) and update the byte-order status. To change the status without actually changing the byte order, set the `tensor.byteswapped` flag.

```python
>>> t = ocean.asTensor([1,2,3],ocean.int16)
>>> print(t)
   1   2   3
<tensor.int16 of size 3 on cpu>
>>> t.byteswap()
>>> print(t)
   1   2   3
<tensor.int16 of size 3 on cpu (byteswapped)>
>>> print(t.byteswapped)
True
```

##### `tensor.convertTo(typename [, deepcopy=False])`

Convert the tensor to the format indicated by the `typename`. The `deepcopy` parameter can be set to True to force a deep copy; by default a shallow copy is made (when supported by the given format). The typename must have been registered (see `ocean.getExportTypes()`).

```python
>>> import ocean
>>> import ocean_numpy
>>> ocean.getExportTypes()
['numpy']
>>> A = ocean.asTensor([1,2,3], ocean.float, ocean.cpu)
>>> B = A.convertTo('numpy')
>>> print(B)
[ 1.  2.  3.]
>>> B.fill(3)
>>> A
   3   3   3
<tensor.float of size 3 on cpu>
>>>
>>> # Convert from a tensor on GPU[1], forces deep copy
>>> A = ocean.asTensor([1,2,3], ocean.float, ocean.gpu[1])
>>> B = A.convertTo('numpy')
>>> print(B)
[ 1.  2.  3.]
>>> B.fill(3)
>>> print(A)
   1   2   3
<tensor.float of size 3 on gpu1>
```

##### `tensor.asScalar()`

Converts single-element tensors of any dimension to the corresponding scalar on the CPU with native byte order.

```python
>>> s = ocean.asTensor([3], ocean.half)
>>> s.byteswap()
>>> print(s)
   3
<tensor.half of size 1 on cpu (byteswapped)>
>>> print(s.asScalar())
3
>>> s = ocean.asTensor(3+2j, ocean.gpu[0])
>>> print(s.asScalar())
3 + 2j
```

##### `tensor.asPython([mode='F'])`

Converts the tensor to a hierarchical list structure.

```python
>>> s = ocean.asTensor([[1,2,3],[4,5,6]], ocean.float, ocean.gpu[0])
>>> print(s)
(:,:)
   1   4
   2   5
   3   6
<tensor.float of size 3x2 on gpu0>
>>> print(s.asPython())
[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
>>> print(s.asPython("R"))
[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
>>> s = ocean.asTensor(7, ocean.float, ocean.gpu[0])
>>> print(s.asPython())
7.0
```

##### `tensor.asContiguous([type='F'])`

Creates a tensor that is in either C or Fortran contiguous layout. When the tensor already had the desired layout a reference to the same tensor is returned. The type character can be 'c' or 'C' for C-style contiguous layout and 'f' or 'F' for Fortran-style contiguous layout. Note that the size of the tensor remains the same and no difference is seen when printing the tensor, only the underlying memory order changes.

```python
>>> s = ocean.asTensor([[1,2,3],[4,5,6]],"R",ocean.int8)
>>> print(s)
(:,:)
   1   2   3
   4   5   6
<tensor.int8 of size 2x3 on cpu>
>>> print(s.strides)
(1, 2)
>>> t = s.asContiguous('C')
>>> print(t)
(:,:)
   1   2   3
   4   5   6
<tensor.int8 of size 2x3 on cpu>
>>> print(t.strides)
(3, 1)
>>> print(s.obj == t.obj)
False
>>> t = s.asContiguous('F')
>>> print(t.strides)
(1, 2)
>>> print(s.obj == t.obj)
True
```

## Tensor copy functions

##### `tensor.copy(src)`
Copy the contents of the source tensor to the tensor. The source tensor can have a different data type, device, and byte order. If needed, and when enabled, the source tensor is automatically broadcast to the correct dimension.

```python
>>> A = ocean.asTensor([1,2,3])
>>> print(A)
   1   2   3
<tensor.int64 of size 3 on cpu>
>>> 
>>> B = ocean.tensor([3,2], ocean.double, ocean.gpu[0])
>>> B.copy(A)
>>> print(B)
(:,:)
   1   1
   2   2
   3   3
<tensor.double of size 3x2 on gpu0>
```

The source tensor is parsed as a generic tensor, which means that any object that can be converted to a tensor (see `ocean.asTensor`) will be accepted:

```python
>>> import ocean_numpy
>>> import numpy as np
>>> A = np.asarray([1,2,3])
>>> B = ocean.tensor([2,3], ocean.half, ocean.gpu[0])
>>> B.copy(A.T)
>>> print(B)
(:,:)
   1   2   3
   1   2   3
   1   2   3
<tensor.half of size 2x3 on gpu0>
```

##### `tensor.sync()`
Synchronizes the underlying tensor stream (see also `tensor.stream`).

##### `tensor.shallowCopy(src)`
Create a shallow copy of the source tensor: copy the tensor object but use the same storage.

```python
>>> a = ocean.tensor([3])
>>> b = a.shallowCopy()
>>> a.obj == b.obj
False
>>> a.storage.obj == b.storage.obj
True
```

##### `tensor.clone([device])`,  `tensor.replicate([device])`
Create a deep copy of the tensor on the same or given device. The two differences between `clone` and replicate are that: (1) `clone` maintains all dimensions with zero strides, whereas `replicate` expands them; (2) `replicate` always creates a new array using the device native byte order, `clone` only does so when the device changes and preserves the original byte order when the device remains the same.

```python
>>> t = ocean.asTensor([1,2,3],ocean.float)
>>> print(t.clone())
   1   2   3
<tensor.float of size 3 on cpu>
>>> print(t.clone(ocean.gpu[1]))
   1   2   3
<tensor.float of size 3 on gpu1>
```

We now create an example with zero strides to illustrate the difference
```python
>>> t = ocean.tensor([3,4],[ocean.float.size,0], ocean.float)
>>> t.copy([1,2,3])
>>> print(t)
(:,:)
   1   1   1   1
   2   2   2   2
   3   3   3   3
<tensor.float of size 3x4 on cpu>
>>> t.storage
   1   2   3
<storage.float of size 3 on cpu>
>>> t.strides
(4, 0)
>>>
>>> a = t.clone(ocean.gpu[0])
>>> a.storage
   1   2   3
<storage.float of size 3 on gpu0>
>>> a.strides
(4, 0)
>>> 
>>> b = t.replicate(ocean.gpu[0])
>>> b.storage
   1   2   3   1   2   3   1   2   3   1   2   3
<storage.float of size 12 on gpu0>
>>> b.strides
(4, 12)
```

## Tensor info functions

##### `tensor.isEmpty()`
Checks if the number of elements is equal to 0 (see also `tensor.nelem`).

```python
>>> t = ocean.tensor([2,3])
>>> t.isEmpty()
False
>>> t = ocean.tensor([2,0])
>>> t.isEmpty()
True
>>> t = ocean.tensor([]) # Scalar tensor
>>> t.isEmpty
False

```

##### `tensor.isScalar()`
Checks if the tensor is a scalar (zero dimensions and a single element).

```Python
>>> ocean.tensor([]).isScalar()
True
>>> ocean.tensor([2]).isScalar()
False
```

## Tensor shape functions

##### `tensor.transpose([inplace=False])`, `tensor.T`

Transpose of the tensor. Tensor dimensions are padded with ones whenever the dimension is less than two. After transposing tensors of dimension two, trailing dimensions of size 1 are removed. That is, transposing a vector of length n gives a 1xn matrix, and transposing matrices of size 1xm and 1x1 respectively gives a vector of length m, and a scalar value. Use `swapAxes(0,1)` when it is undesirable that trailing dimensions are removed. (See [design choices](/docs/design_choices.md#tensor-transpose) for more details.) When `inplace` is False a new tensor view is returned, otherwise the tensor object is updated with no return value. The `tensor.T` property is equivalent to `tensor.transpose()`. The result is a view on the original data.

```python
>>> a = ocean.asTensor([1,2,3])
>>> print(a)
   1   2   3
<tensor.int64 of size 3 on cpu>
>>> print(a.T)
(:,:)
   1   2   3
<tensor.int64 of size 1x3 on cpu>
>>>
>>> a = ocean.asTensor([[1+2j, 3+4j],[5+6j, 7+8j]],"R")
>>> print(a)
(:,:)
   1 + 2j    3 + 4j
   5 + 6j    7 + 8j
<tensor.complex-double of size 2x2 on cpu>
>>> b = a.T
>>> print(b)
(:,:)
   1 + 2j    5 + 6j
   3 + 4j    7 + 8j
<tensor.complex-double of size 2x2 on cpu>
>>> print(a.storage == b.storage)
True
>>> a.transpose(True)
>>> print(a)
(:,:)
   1 + 2j    5 + 6j
   3 + 4j    7 + 8j
<tensor.complex-double of size 2x2 on cpu>
```

##### `tensor.ctranspose([inplace=False])`, `tensor.H`

Conjugate or Hermitian transpose of the tensor. Tensor dimensions are padded with ones whenever the dimension is less than two. After transposing tensors of dimension two, trailing dimensions of size 1 are removed. That is, transposing a vector of length n gives a 1xn matrix, and transposing matrices of size 1xm and 1x1 respectively gives a vector of length m, and a scalar value. Use `swapAxes(0,1)` combined with `conj` when it is undesirable that trailing dimensions are removed. When `inplace` is False a new tensor view is returned, otherwise the tensor object is updated with no return value. The `tensor.H` property is equivalent to `tensor.ctranspose()`. When the tensor data type is not complex the result is equivalent to `tensor.transpose`. When the tensor data type is complex the tensor data is copied, conjugated, and transposed.

```python
>>> a = ocean.asTensor([[1+2j, 3+4j],[5+6j, 7+8j]],"R")
>>> print(a)
(:,:)
   1 + 2j    3 + 4j
   5 + 6j    7 + 8j
<tensor.complex-double of size 2x2 on cpu>
>>> b = a.H
>>> print(b)
(:,:)
   1 - 2j    5 - 6j
   3 - 4j    7 - 8j
<tensor.complex-double of size 2x2 on cpu>
>>> print(a.storage == b.storage)
False
>>> a.ctranspose(True)
>>> print(a)
(:,:)
   1 - 2j    5 - 6j
   3 - 4j    7 - 8j
<tensor.complex-double of size 2x2 on cpu>
```

##### `tensor.reshape(size [,inplace=False])`, `tensor.reshape(dim1 [,dim2 [,dim3 ...]] [,inplace=False])`

Reshapes the tensor to the given size (the number of elements have to match). A new tensor is returned when the `inplace` parameter is False, and the current tensor is reshaped otherwise. When the tensor memory layout matches the size a view is returned otherwise a copy of the data is made in the desired shape (in this case the tensor will be contiguous column-major).

```python
>>> a = ocean.arange(6, ocean.float)
>>> a
   0   1   2   3   4   5
<tensor.float of size 6 on cpu>
>>> a.reshape([2,3])
(:,:)
   0   2   4
   1   3   5
<tensor.float of size 2x3 on cpu>
>>> a
   0   1   2   3   4   5
<tensor.float of size 6 on cpu>
>>> a.reshape(3,2,True)
>>> a
(:,:)
   0   3
   1   4
   2   5
<tensor.float of size 3x2 on cpu>
```

##### `tensor.broadcastLike(src [,mode=0] [,inplace=False])`, `tensor.broadcastTo(size [,mode=0] [,inplace=False])`
Broadcast the tensor to the dimensions of the source tensor or the given size, when compatible. The mode determines whether the dimensions are padded on the right (mode=0, default) or on the left (mode=1). The `inplace` parameter indicates whether a new tensor should be created, or if the broadcast should apply in-place for the tensor.

```python
>>> A = ocean.asTensor([1,2,3])
>>> B = ocean.tensor([3,2,1], ocean.int8) # Only size information is used
>>> A.broadcastLike(B)
(:,:,0)
   1   1
   2   2
   3   3
<tensor.int64 of size 3x2x1 on cpu>
>>> A.broadcastLike(ocean.tensor([2,4]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor dimensions are incompatible with broadcast dimensions
>>>
>>> A.broadcastTo([3,2])
(:,:)
   1   1
   2   2
   3   3
<tensor.int64 of size 3x2 on cpu>
>>> A.broadcastTo([2,3],1)
(:,:)
   1   2   3
   1   2   3
<tensor.int64 of size 2x3 on cpu>
>>> # In-place broadcast changes the tensor
>>> A.broadcastTo([3,2,1], True)
>>> print(A)
(:,:,0)
   1   1
   2   2
   3   3
<tensor.int64 of size 3x2x1 on cpu>
```

##### `tensor.squeeze([axis] [,inplace=False])`

Creates a tensor with a given (axis) or all unit dimensions removed. When `inplace` is False a new tensor view is returned, otherwise the tensor object is replaced with no return value. An error is raised if an axis is specified outside the valid range or when the corresponding size is not one.

```python
>>> a = ocean.tensor([1,2,3,1,1])
>>> print(a.squeeze().size)
(2, 3)
>>> print(a.squeeze(0).size)
(2, 3, 1, 1)
>>> a.squeeze(True)
>>> a.size
(2, 3)
```
##### `tensor.unsqueeze(axis [,inplace=False])`

Inserts a unit dimension at the given axis. When `inplace` is False a new tensor view is returned, otherwise the tensor object is updated with no return value.

```python
>>> a = ocean.tensor([2,3])
>>> print(a.unsqueeze(2).size)
(2, 3, 1)
>>> a.unsqueeze(1,True)
>>> a.size
(2, 1, 3)
```

##### `tensor.flatten([type='F'] [inplace=False])`

Returns a vectorized version of the tensor. The entry order is determined by the `type` parameter, with 'C' indicating a C-style row-major order; 'F' a Fortran-style column-major order; 'A' indicating Fortran-style when the tensor has this style, and C-style otherwise; and 'K' indicating C-style when the tensor has this style, and Fortran-style otherwise.

```python
>>> a = ocean.arange(6,ocean.float).reshape([2,3])
>>> a
(:,:)
   0   2   4
   1   3   5
<tensor.float of size 2x3 on cpu>
>>> a.flatten('F')
   0   1   2   3   4   5
<tensor.float of size 6 on cpu>
>>> a.flatten('C')
   0   2   4   1   3   5
<tensor.float of size 6 on cpu>
>>> a.flatten('A',True)
>>> a
   0   1   2   3   4   5
<tensor.float of size 6 on cpu>
```

##### `tensor.swapAxes(axis1, axis2 [,inplace=False])`
Exchanges the given axes. The `inplace` parameter indicates whether a new tensor should be created, or if the broadcast should apply in-place for the tensor. Note that the underlying storage remains unaffected; the operation only affects the tensor size and strides.

```python
>>> A = ocean.arange(24, ocean.float).reshape([3,4,2])
>>> A
(:,:,0)
    0    3    6    9
    1    4    7   10
    2    5    8   11

(:,:,1)
   12   15   18   21
   13   16   19   22
   14   17   20   23
<tensor.float of size 3x4x2 on cpu>
>>> 
>>> A.swapAxes(0,1)
(:,:,0)
    0    1    2
    3    4    5
    6    7    8
    9   10   11

(:,:,1)
   12   13   14
   15   16   17
   18   19   20
   21   22   23
<tensor.float of size 4x3x2 on cpu>
>>> A.swapAxes(1,2,True)
>>> print(A)
(:,:,0)
    0   12
    1   13
    2   14

(:,:,1)
    3   15
    4   16
    5   17

(:,:,2)
    6   18
    7   19
    8   20

(:,:,3)
    9   21
   10   22
   11   23
<tensor.float of size 3x2x4 on cpu>
>>> A.size
(3, 2, 4)
>>> A.strides
(4, 48, 12)
```

##### `tensor.flipAxis(axis [,inplace=False])`, `tensor.fliplr([inplace=False])`, `tensor.flipud([inplace=False])`
The `flipAxis` function flips the direction of the given axis, `flipud` flips the along axis 0, and `fliplr` flips along axis 1 (or axis 0 if the input tensor is one-dimensional). The `inplace` parameter indicates whether a new tensor should be created, or if the broadcast should apply in-place for the tensor. Note that the underlying storage remains unaffected; the operation only affects the tensor strides and offset.

```python
>>> A = ocean.arange(10, ocean.float).reshape([2,5])
>>> print(A)
>>> A
(:,:)
   0   2   4   6   8
   1   3   5   7   9
<tensor.float of size 2x5 on cpu>
>>> A.strides
(4, 8)
>>> A.offset
0L
>>>
>>> # Flip different axes
>>> A.flipAxis(0)
(:,:)
   1   3   5   7   9
   0   2   4   6   8
<tensor.float of size 2x5 on cpu>
>>> A.flipAxis(1)
(:,:)
   8   6   4   2   0
   9   7   5   3   1
<tensor.float of size 2x5 on cpu>
>>> A.flipAxis(0).flipAxis(1)
(:,:)
   8   6   4   2   0
   9   7   5   3   1
<tensor.float of size 2x5 on cpu>
>>>
>>> # In-place reversal of the axis direction
>>> A = ocean.arange(10, ocean.float).reshape([2,5])
>>> A.flipAxis(1, True)
>>> print(A)
(:,:)
   8   6   4   2   0
   9   7   5   3   1
<tensor.float of size 2x5 on cpu>
>>> A.strides
(4, -8)
>>> A.offset
32L
```

##### `tensor.reverseAxes([inplace=False])`, `tensor.reverseAxes2([inplace=False])`

The `reverseAxes` function reverses the order of the tensor dimensions (sizes and strides); the storage and data are left untouched. The `reverseAxes2` function first reverses the order of the tensor dimensions and then swaps back the first and second dimension in the result (provided the tensor has at least two dimensions).

```python
>>> A = ocean.tensor([1,2,3,4],"C")
>>> print(A.size)
(1, 2, 3, 4)
>>> print(A.strides)
(96, 48, 16, 4)
>>>
>>> B = A.reverseAxes()
>>> print(B.size)
(4, 3, 2, 1)
>>> print(B.strides)
(4, 16, 48, 96)
>>>
>>> C = A.reverseAxes2()
>>> print(C.size)
(3, 4, 2, 1)
>>> print(C.strides)
(16, 4, 48, 96)
```

These functions are especially useful when importing data with alternative dimension orderings:

```python
>>> A = np.arange(16).reshape(2,2,4)
>>> A
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],

       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]]])
>>> ocean.asTensor(A)
(:,:,0)
    0    4
    8   12

(:,:,1)
    1    5
    9   13

(:,:,2)
    2    6
   10   14

(:,:,3)
    3    7
   11   15
<tensor.int64 of size 2x2x4 on cpu>
>>> ocean.asTensor(A).reverseAxes()
(:,:,0)
    0    4
    1    5
    2    6
    3    7

(:,:,1)
    8   12
    9   13
   10   14
   11   15
<tensor.int64 of size 4x2x2 on cpu>
>>> ocean.asTensor(A).reverseAxes2()
(:,:,0)
    0    1    2    3
    4    5    6    7

(:,:,1)
    8    9   10   11
   12   13   14   15
<tensor.int64 of size 2x4x2 on cpu>
```

##### `tensor.permuteAxes(order [, inplace=False])`

Permute the axes of the tensor according to the given order. The elements in order must be a permutation of [0,tensor.ndims-1].

```python
>>> a = ocean.tensor([2,3,4,5])
>>> print(a.size)
(2, 3, 4, 5)
>>> print(a.strides)
(4, 8, 24, 96)
>>> b = a.permuteAxes([0,3,2,1])
>>> print(b.size)
(2, 5, 4, 3)
>>> print(b.strides)
(4, 96, 24, 8)
>>> a.permuteAxes([1,2,3,0],True)
>>> print(a.size)
(3, 4, 5, 2)
>>> print(a.strides)
(8, 24, 96, 4)
```


## Tensor extraction and combination

##### `tensor.real` (property), `tensor.imag` (property)

The `real` and `imag` properties give views of the real and imaginary parts of the tensor data. The imaginary part of a real tensor is a zero tensor that is read-only to avoid the intuition that modifying tensor.imag would make the tensor complex.

```python
>>> A = ocean.asTensor([1+2j,3+4j,5+6j],ocean.chalf)
>>> print(A.real)
   1   3   5
<tensor.half of size 3 on cpu>
>>> print(A.imag)
   2   4   6
<tensor.half of size 3 on cpu>
>>> A.real.fill(1)
>>> A.imag.fill(2)
>>> print(A)
   1 + 2j    1 + 2j    1 + 2j
<tensor.complex-half of size 3 on cpu>
>>>
>>> A = ocean.asTensor([1,2,3])
>>> print(A.imag)
   0   0   0
<tensor.int64 of size 3 on cpu (read-only)>
```

##### `tensor.diag(tensor [,index])`

The diag function creates a view to the diagonal given by index (default 0).

```python
>>> A = ocean.arange(24).reshape([4,6])
>>> A
(:,:)
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23
<tensor.int64 of size 4x6 on cpu>
>>> print(A.diag())
    0    5   10   15
<tensor.int64 of size 4 on cpu>
>>> print(A.diag(2))
    8   13   18   23
<tensor.int64 of size 4 on cpu>
>>> A.diag(-1).fill(99)
>>> print(A)
(:,:)
    0    4    8   12   16   20
   99    5    9   13   17   21
    2   99   10   14   18   22
    3    7   99   15   19   23
<tensor.int64 of size 4x6 on cpu>
```

##### `tensor.slice(axis, offset [,size=1])`
Returns a slice of the input tensor along the given axis, starting at offset with given size. The result is a view into the tensor.

```python
>>> A = ocean.tensor([3,6])
>>> A.copy(range(A.nelem))
>>> print(A)
(:,:)
    0    3    6    9   12   15
    1    4    7   10   13   16
    2    5    8   11   14   17
<tensor.float of size 3x6 on cpu>
>>> print(A.slice(0,1,2))
(:,:)
    1    4    7   10   13   16
    2    5    8   11   14   17
<tensor.float of size 2x6 on cpu>
>>> print(A.slice(1,2,3))
(:,:)
    6    9   12
    7   10   13
    8   11   14
<tensor.float of size 3x3 on cpu>
```

##### `tensor.split`

There are three syntaxes for the tensor split 
* `tensor.split(axis, parts, [detach])`
* `tensor.split(axis, sizes, [detach])`
* `tensor.split(axis, devices [, sizes] [detach])`

The first syntax splits the given tensor along the given axis in the given number of parts. If the tensor size along the axis is not a multiple of `parts` it will be split as evenly as possible with the first set of tensors (dimension modulo parts) one larger than the others. The detach flag indicates whether a deep copy is required; by default it is set to False, therefore returning views into the base tensor.

The second syntax is similar to the first except that the sizes of each of the parts can be specified. The sizes must be nonnegative and add up to the size of the selected axis. The third syntax enables splitting of the tensor across different devices by specifying a list of devices (in this case `detach` only applies to tensors that stay on the same device as the base tensor). The splitting behavior itself is similar to the second syntax when sizes are provided, otherwise it is similar to the first syntax with the number of parts matching the size of the device list.

```python
>>> A = ocean.arange(30).reshape([5,6])
>>> print(A)
(:,:)
    0    5   10   15   20   25
    1    6   11   16   21   26
    2    7   12   17   22   27
    3    8   13   18   23   28
    4    9   14   19   24   29
<tensor.int64 of size 5x6 on cpu>
>>> 
>>> V = A.split(0,3)
>>> for tensor in V: print(tensor)
... 
(:,:)
    0    5   10   15   20   25
    1    6   11   16   21   26
<tensor.int64 of size 2x6 on cpu>
(:,:)
    2    7   12   17   22   27
    3    8   13   18   23   28
<tensor.int64 of size 2x6 on cpu>
(:,:)
    4    9   14   19   24   29
<tensor.int64 of size 1x6 on cpu>
>>> V = A.split(1,[2,1,3])
>>> for tensor in V: print(tensor)
... 
(:,:)
   0   5
   1   6
   2   7
   3   8
   4   9
<tensor.int64 of size 5x2 on cpu>
(:,:)
   10
   11
   12
   13
   14
<tensor.int64 of size 5x1 on cpu>
(:,:)
   15   20   25
   16   21   26
   17   22   27
   18   23   28
   19   24   29
<tensor.int64 of size 5x3 on cpu>
>>>
>>> V = A.split(0,ocean.gpu)
>>> for tensor in V: print(tensor)
... 
(:,:)
    0    5   10   15   20   25
    1    6   11   16   21   26
    2    7   12   17   22   27
<tensor.int64 of size 3x6 on gpu0>
(:,:)
    3    8   13   18   23   28
    4    9   14   19   24   29
<tensor.int64 of size 2x6 on gpu1>

```
Lists of devices can be created and manipulated as general Python lists. As an example we could write `ocean.gpu + [ocean.cpu]` to augment the GPU list with the list containing the CPU device.

##### `ocean.merge(tensors, axis [, result])`, `ocean.merge(tensors, axis [,dtype] [,device]`

The merge function concatenates a list of tensors along a given axis. When the results tensor is provided it must have the correct size (the device, data type, and byte order can differ) and the merge function will not have a return value. If no result tensor is specified a new tensor is created and returned. A new return tensor will have the same data type and/or device if a consistent value can be found from the list of input tensors. When provided the data type and device will be forced, and when one of the types could not be inferred the global default device or data type will be used.

```python
>>> # Merge a list of blocks
>>> V = [ocean.full([i,8],i) for i in range(4)]
>>> print(ocean.merge(V,0))
(:,:)
   1   1   1   1   1   1   1   1
   2   2   2   2   2   2   2   2
   2   2   2   2   2   2   2   2
   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3
<tensor.int64 of size 6x8 on cpu>
>>>
>>> A = ocean.arange(30).reshape([5,6])
>>> print(A)
(:,:)
    0    5   10   15   20   25
    1    6   11   16   21   26
    2    7   12   17   22   27
    3    8   13   18   23   28
    4    9   14   19   24   29
<tensor.int64 of size 5x6 on cpu>
>>>
>>> # Merge blocks in reverse order
V = A.split(1,3)
print(ocean.merge(V[::-1],1))
>>> V = A.split(1,3)
>>> print(ocean.merge(V[::-1],1))
(:,:)
   20   25   10   15    0    5
   21   26   11   16    1    6
   22   27   12   17    2    7
   23   28   13   18    3    8
   24   29   14   19    4    9
<tensor.int64 of size 5x6 on cpu>

>>> # Scatter to GPU (two devices), fill, merge
>>> A = ocean.tensor([4,8])
>>> V = A.split(0,ocean.gpu)
>>> for i in range(len(V)) : V[i].fill(i)
... 
>>> ocean.merge(V,0,A)
>>> print(A)
(:,:)
   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0
   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1
<tensor.float of size 4x8 on cpu>
```

##### `ocean.find(tensor)`

The find operation returns a 64-bit integer tensor with indices of the non-zero entries in the tensor. The dimension of the output is equal to tensor.ndims by the number of non-zero entries.

```python
>>> A = ocean.asTensor([[0,1,2,0],[3,4,0,5]],'r')
>>> print(A)
(:,:)
   0   1   2   0
   3   4   0   5
<tensor.int64 of size 2x4 on cpu>
>>> print(ocean.find(A))
(:,:)
   1   0   1   0   1
   0   1   1   2   3
<tensor.int64 of size 2x5 on cpu>
```

## Tensor operations

##### Tensor operators (+,-,\*,/)

| Operator | Meaning |
| -------- | ------- |
| +, +=    | Elementwise addition |
| -, -=    | Elementwise subtraction |
| /, /=    | Elementwise division |
| \*       | Tensor multiplication (see [ocean.multiply](module_core.md#multiply)) |
| \*=      | Elementwise multiplication (see [ocean.scale](module_core.md#scale)) |
| @        | Elementwise multiplication (Python 3) |
| &, &=    | Bitwise AND |
| \|, \|=    | Bitwise OR |
| ^, ^=      | Bitwise XOR |

See [here](index.md#tensor_operators) for a discussion of the notation.


##### `tensor.fill(value)`, `tensor.zero(value)`, `tensor.fillNaN(value)`

Fills the tensor with the given value or with zeros. The `fillNaN` function only updates the tensor elements that are NaN.

```python
>>> A = ocean.tensor([3,4])
>>> A.fill(3)
>>> print(A)
(:,:)
   3   3   3   3
   3   3   3   3
   3   3   3   3
<tensor.float of size 3x4 on cpu>
>>> A.zero()
>>> print(A)
(:,:)
   0   0   0   0
   0   0   0   0
   0   0   0   0
<tensor.float of size 3x4 on cpu>
```

```python
>>> a = ocean.asTensor([ocean.nan,3,4,ocean.nan], ocean.float)
>>> a.fillNaN(ocean.inf)
>>> print(a)
   inf     3     4   inf
<tensor.float of size 4 on cpu>
```

##### <a name="multiply">Tensor multiplication</a>: `A * B`, `ocean.multiply(A, [transA='N',] B [,transB='N'] [,C])`, `ocean.gemm(alpha, A, [transA='N',] B [,transB='N'] [, beta=0, C])`

The gemm function computes `C <- alpha * op(A) * op(B) + beta * C`. When `C` is omitted, a new tensor is created and returned. The `op(X)` function applies the transpose when the transX is 't' or 'T', the conjugate transpose when transX is 'c' or 'C', and does nothing when transX is 'n' or 'N'. The multiply function provides a simplified interface for gemm with alpha and beta equal to 1 and 0, respectively. The `A * B` syntax is equivalent to `ocean.multiply(A,B)`. Tensor multiplication in Ocean is based on the interpretation of tensors as collection of matrices. Tensor dimensions are padded with ones when needed. The output of the operation is a vector if C is a vector, or, when C is omitted, if B is a vector and alpha and beta are at most two dimensional.

```python
>>> a = ocean.asTensor([1,2,3],ocean.float)     # column vector
>>> b = ocean.asTensor([-1,0,1],ocean.float).T  # row vector
>>> a * a.T
(:,:)
   1   2   3
   2   4   6
   3   6   9
<tensor.float of size 3x3 on cpu>
>>> ocean.multiply(a,b)
(:,:)
   -1   0   1
   -2   0   2
   -3   0   3
<tensor.float of size 3x3 on cpu>
ocean.multiply(a,a,'T')
(:,:)
   1   2   3
   2   4   6
   3   6   9
<tensor.float of size 3x3 on cpu>
>>> ocean.gemm(1+2j,a,b)
(:,:)
   -1 - 2j    0 + 0j    1 + 2j
   -2 - 4j    0 + 0j    2 + 4j
   -3 - 6j    0 + 0j    3 + 6j
<tensor.complex-float of size 3x3 on cpu>
>>> 
>>> 
>>> c = ocean.ones([3,3],ocean.cfloat)
>>> ocean.gemm(1+2j,a,b,100,c)
>>> c
(:,:)
    99 - 2j    100 + 0j    101 + 2j
    98 - 4j    100 + 0j    102 + 4j
    97 - 6j    100 + 0j    103 + 6j
<tensor.complex-float of size 3x3 on cpu>
```

##### <a name="scale">Tensor scaling</a>: `tensor *= tensor`, `ocean.scale(A,B[,C]), operator `@` (Python 3)

The scale function and the in-place `\*=` and `@` operators perform elementwise scaling of the tensor elements. The scale operator scales the elements in A by B. The result is stored in destination tensor C when given or returned otherwise.

```python
>>> a = ocean.asTensor([1,2,4,6])
>>> b = ocean.asTensor([1,1,2,2])
>>> ocean.scale(a,b)
    1    2    8   12
<tensor.int64 of size 4 on cpu>
>>> # (Python 3)
>>> a @ a
    1    4   16   36
<tensor.int64 of size 4 on cpu>
>>> a *= [1,1,-1,-1]
>>> a
    1   2  -4  -6
<tensor.int64 of size 4 on cpu>
```

##### Tensor division: `tensor / tensor`, `tensor /= tensor`, `ocean.divide(A,B[,C])`

The divide function performs elementwise division of tensors. No explicit checks for division by zero are done at the moment. Integer division by zero gives 0, Boolean division by False returns False, and floating-point division by zero gives NaN.

```python
>>> c = ocean.asTensor([4.5, 5.5, 6.5], ocean.float)
>>> c / 2
   2.25   2.75   3.25
<tensor.float of size 3 on cpu>
>>> c /= 0.5
>>> c
    9   11   13
<tensor.float of size 3 on cpu>
>>> ocean.divide(c,2,c)
>>> c
   4.5   5.5   6.5
<tensor.float of size 3 on cpu>
```

##### Tensor addition: `tensor + tensor`, `tensor += tensor`, `ocean.add(A,B[,C])`

The add function performs elementwise addition of tensors.

```python
>>> a = ocean.arange(5,ocean.float)
>>> a
   0   1   2   3   4
<tensor.float of size 5 on cpu>
>>> a + 3
   3   4   5   6   7
<tensor.float of size 5 on cpu>
>>> a += 3.5
>>> a
   3.5   4.5   5.5   6.5   7.5
<tensor.float of size 5 on cpu>
>>> c = ocean.zeros([5],ocean.float)
>>> ocean.add(a,3,c)
>>> c
    6.5    7.5    8.5    9.5   10.5
<tensor.float of size 5 on cpu>
>>> c - 2.5
   4   5   6   7   8
<tensor.double of size 5 on cpu>
>>> ocean.subtract(a,3,c)
>>> c
   0.5   1.5   2.5   3.5   4.5
<tensor.float of size 5 on cpu>
```

##### Tensor subtraction: `tensor - tensor`, `tensor -= tensor`, `ocean.subtract(A,B,[C])`

The subtract function performs elementwise subtraction of tensors (see tensor addition for examples).

##### `ocean.negative(A[,B])`

The `negative` function multiplies all entries by -1; when the input tensor is Boolean logical negation is applied.

```python
>>> a = ocean.asTensor([True,False])
>>> a
    True   False
<tensor.bool of size 2 on cpu>
>>> ocean.negative(a)
   False    True
<tensor.bool of size 2 on cpu>
>>> ocean.negative(ocean.arange(6))
    0  -1  -2  -3  -4  -5
<tensor.int64 of size 6 on cpu>
>>> b = ocean.tensor(6, ocean.float)
>>> ocean.negative([1,-2,3,-4,5,-6],b)
>>> b
   -1   2  -3   4  -5   6
<tensor.float of size 6 on cpu>
```

##### `ocean.conj(A[,B])`, `A.conj()`

The `conj` function applies the complex conjugate. For convenience the function is also implemented as a tensor function (in this case no output tensor can be specified).

```python
>>> a = ocean.asTensor([1+2j, 3-4j])
>>> a
   1 + 2j    3 - 4j
<tensor.complex-double of size 2 on cpu>
>>> ocean.conj(a)
   1 - 2j    3 + 4j
<tensor.complex-double of size 2 on cpu>
>>> a.conj()
   1 - 2j    3 + 4j
<tensor.complex-double of size 2 on cpu>
>>> b = ocean.tensor([2],ocean.chalf)
>>> ocean.conj(a,b)
>>> b
   1 - 2j    3 + 4j
<tensor.complex-half of size 2 on cpu>
```

##### `ocean.fabs(A[,B])`, `ocean.absolute(A[,B])`

The `fabs` and `absolute` functions computes the elementwise absolute value of the input. For complex data types this amounts to the magnitude of the elements. When output tensor `B` is omitted, the result type of `fabs` will have the same data type as that of `A`, whereas that of `absolute` will match the input type for non-complex input and the element type for complex numbers:

```python
>>> a = ocean.asTensor([3-4j])
>>> ocean.fabs(a)
   5 + 0j
<tensor.complex-double of size 1 on cpu>
>>> ocean.absolute(a)
   5
<tensor.double of size 1 on cpu>
```

##### `ocean.sign(A[,B])`

The `sign` function returns the sign of each of the tensor elements. For real-valued tensors this gives -1 for negative values, 1 for positive values, 0 for zero values. Entries that are NaN give NaN as a result. For complex-valued elements `a+bj` return the sign of `a` whenever it is non-zero, and the sign of `b` otherwise; the imaginary part of the result is always zero.

##### `ocean.ceil(A[,B])`, `ocean.floor(A[,B])`, `ocean.trunc(A[,B])`, `ocean.round(A[,B])`

These are functions for rounding up (`ceil`) or down (`floor`) to the nearest integer, truncation of the fractional part (`trunc`), and rounding towards the nearest integer (`round`). For complex numbers the operation is applied on both the real and imaginary part.

```python
>>> a = ocean.cfloat([1.23+4.56j, -1.23-4.56j])
>>> a
    1.23 + 4.56j   -1.23 - 4.56j
<tensor.complex-float of size 2 on cpu>
>>> ocean.floor(a)
    1 + 4j   -2 - 5j
<tensor.complex-float of size 2 on cpu>
>>> ocean.ceil(a)
    2 + 5j   -1 - 4j
<tensor.complex-float of size 2 on cpu>
>>> ocean.trunc(a)
    1 + 4j   -1 - 4j
<tensor.complex-float of size 2 on cpu>
>>> ocean.round(a)
    1 + 5j   -1 - 5j
<tensor.complex-float of size 2 on cpu>
```

##### Tensor modulo: `tensor % tensor`, `tensor %= tensor`, `ocean.mod(A,B[,C])`, `ocean.fmod(A,B[,C])`

The modulo operator applies to boolean, integer and non-complex floating point types. The `mod` and `fmod` functions differ in the way they deal with sign differences: `mod` maintains the sign of tensor `B`, whereas `fmod` maintains the sign of tensor `A`.

```python
>>> a = ocean.double([-20,-10,0,10,20])
>>> a
   -20  -10    0   10   20
<tensor.double of size 5 on cpu>
>>> ocean.mod(a,3)
   1   2   0   1   2
<tensor.double of size 5 on cpu>
>>> ocean.fmod(a,3)
   -2  -1   0   1   2
<tensor.double of size 5 on cpu>
>>> ocean.mod(a,-3)
   -2  -1   0  -2  -1
<tensor.double of size 5 on cpu>
>>> ocean.fmod(a,-3)
   -2  -1   0   1   2
<tensor.double of size 5 on cpu>
>>> a % 3
   1   2   0   1   2
<tensor.double of size 5 on cpu>
>>> a %= 6
>>> a
   4   2   0   4   2
<tensor.double of size 5 on cpu>
```

##### `ocean.reciprocal(A[,B])`

The reciprocal function applies 1/x elementwise.

```python
>>> ocean.reciprocal([1,2,-3])
   1   0   0
<tensor.int64 of size 3 on cpu>
>>> ocean.reciprocal([1,2,-3.])
    1.00000   0.50000  -0.33333
<tensor.double of size 3 on cpu>
>>> ocean.reciprocal([1+2j])
   0.2 - 0.4j
<tensor.complex-double of size 1 on cpu>
```
##### Tensor power: `A ** B`, `A **=B`, `ocean.power(A,B[,mode][,C])`, `pow(A,B)`, `ocean.square(A[,C])`

The tensor power functions take the elementwise power of the entries in A with the entries in B. When A has integer type the entries in B are converted to signed 16-bit integer format, if needed. When either A or B is floating-point the common data type is used for computation and output (when none is specified). When A and B are scalar and both of integer type we convert A to a 64-bit integer preserving the signedness of the data type of A, and convert B to signed 16-bit integer. Otherwise, we take the common data type of A and B for the output. In `power` it is possible to specify the math mode, in particular the mode can be set to `c` to check for negative entries in `A` and values in `B` less than one to switch to complex numbers, if needed. For scalar power, the result tensor C cannot be specified. The optional third parameter for a modulo factor in the Python `pow` function is not yet supported. The `square` function computes the square of the elements in A.

```python
>>> ocean.power([2,2,4],[2,-1,0.5])
   4.0   0.5   2.0
<tensor.double of size 3 on cpu>
>>> ocean.power([1+2j],3-2j)
   -14.40586 + 101.33690j
>>> ocean.square([2,3j])
    4 + 0j   -9 + 0j
<tensor.complex-double of size 2 on cpu>
```

##### `ocean.sqrt(A[,mode][,B])`, `ocean.cbrt(A[,B])`

The `sqrt` and `cbrt` respectively compute the elementwise square root and cube root of A. The mode, when set, is used to specify that the A should be checked for negative values. When found, the mode indicates `w` for warning, `e` for error, and `c` for automatic casting to complex numbers (see the math mode section for more details).

```python
>>> ocean.sqrt([1,4,9])
   1   2   3
<tensor.double of size 3 on cpu>
>>> ocean.sqrt([1,-1])
     1   nan
<tensor.double of size 2 on cpu>
>>> ocean.sqrt([1,-1],'w')
__main__:1: RuntimeWarning: Tensor elements must be nonnegative for square-root
     1   nan
<tensor.double of size 2 on cpu>
>>> ocean.sqrt([1,-1],'e')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Tensor elements must be nonnegative for square-root
>>> ocean.sqrt([1,-1],'c')
   1 + 0j    0 + 1j
<tensor.complex-double of size 2 on cpu>
>>> b = ocean.tensor([2],ocean.float)
>>> ocean.sqrt([1,-1],'c',b)
__main__:1: RuntimeWarning: Casting complex intermediate value to real output discards imaginary part
>>> b
   1   0
<tensor.float of size 2 on cpu>
```

In the last example, note that complex output `0+1j` is converted to float, which discards the imaginary part and therefore gives `0`.


##### `ocean.min(A,B[,C])`, `ocean.max(A,B[,C])`, `ocean.fmin(A,B[,C])`, `ocean.fmax(A,B[,C])`

These functions apply the pairwise minimum or maximum of the elements in A and B. When given, the output of the operation is stored in tensor C. The difference between min/max and fmin/fmax is that the former propagates NaN values, whereas the latter chooses non-NaN values over NaN values.

```python
>>> a = ocean.asTensor([1,2,3,4,5])
>>> b = ocean.asTensor([5,4,3,2,1])
>>> ocean.min(a,b)
   1   2   3   2   1
<tensor.int64 of size 5 on cpu>
>>> c = ocean.tensorLike(a)
>>> ocean.max(a,b,c)
>>> print(c)
   5   4   3   4   5
<tensor.int64 of size 5 on cpu>
```

##### Tensor bitwise: `A & B`, `A &= B`, `ocean.bitwiseAnd(A,B[,C])`, `A | B`, `A |= B`, `ocean.bitwiseOr(A,B[,C])`, `A ^ B`, `A ^= B`, `ocean.bitwiseXor(A,B[,C])`

These functions apply bitwise AND, OR, and XOR to the elements of the input tensors. The type of the input tensors must be integer. The output tensor will have the same type as A when the number of bits in `A.dtype.nbits >= B.dtype.nbits` and that of B otherwise.

##### Tensor bitwise: `~A`, `ocean.bitwiseNot(A[,B])`

The ocean `bitwiseNot` function can be applied to Boolean and integer tensors and applies a bitwise-NOT to every element.

```python
>>> ocean.bitwiseNot(ocean.uint8(1))
254
>>> ocean.bitwiseNot(ocean.int8(2))
-3
>>> ocean.bitwiseNot(ocean.bool(True))
False
>>> ocean.bitwiseNot(ocean.bool(False))
True
```

##### Tensor bitshift: `A << B`, `A >> B`, `ocean.bitshiftLeft(A,B[,C])`, `ocean.bitshiftRight(A,B[,C])`

The tensor bitshift functions can be applied to integer tensors, shifting the bitwise representation left or right. The amount of bitshift is converted to int8 format. Negative shift values result in a shift in the opposite direction.


##### Tensor logical: `ocean.logicalAnd(A,B[,C])`, `ocean.logicalOr(A,B[,C])`, `ocean.logicalXor(A,B[,C])`

These functions apply logical AND, OR, and XOR to the elements of the input tensors, after converting them to Boolean type if needed.

```python
>>> v1 = [0,0,1,1]
>>> v2 = [0,1,0,1]
>>> ocean.logicalAnd(v1,v2)
   False   False   False    True
<tensor.bool of size 4 on cpu>
>>> ocean.logicalOr(v1,v2)
   False    True    True    True
<tensor.bool of size 4 on cpu>
>>> ocean.logicalXor(v1,v2)
   False    True    True   False
<tensor.bool of size 4 on cpu>
```


##### Trigonometric and hyperbolic functions

| Function | Description                | Domain restrictions               |
| -------- | -------------------------- | --------------------------------- |
| sin      | sine                       | -                                 |
| cos      | cosine                     | -                                 |
| tan      | tangent                    | -                                 |
| sinh     | hyperbolic sine            | -                                 |
| cosh     | hyperbolic cosine          | -                                 |
| tanh     | hyperbolic tangent         | -                                 |
| arcsin   | inverse sine               | [-1,1] for real, none for complex |
| arccos   | inverse cosine             | [-1,1] for real, none for complex |
| arctan   | inverse tangent            | -                                 |
| arcsinh  | inverse hyperbolic sine    | -                                 |
| arccosh  | inverse hyperbolic cosine  | [1,inf] for real, none for complex|
| arctanh  | inverse hyperbolic tangent | [-1,1] for real, none for complex |

Syntax: `ocean.cos(A[,B])`, `ocean.arccos(A[,mode][,B])`
The `mode` parameter applies when the function has domain restrictions. 

##### Exponential and logarithmic functions

| Function | Description | Domain restrictions                 |
| -------- | ----------- | ----------------------------------- |
| exp      | e^x         | -                                   |
| exp2     | 2^x         | -                                   |
| exp10    | 10^x        | -                                   |
| expm1    | e^x - 1     | -                                   |
| log      | ln(x)       | [0,Inf] for real, none for complex  |
| log2     | log_2(x)    | [0,Inf] for real, none for complex  |
| log10    | log_10(x)   | [0,Inf] for real, none for complex  |
| log1p    | ln(1+x)     | [-1,Inf] for real, none for complex |

Syntax: `ocean.exp(A[,B])`, `ocean.log(A[,mode][,B])`

The `mode` parameter applies when the function has domain restrictions. 

##### Floating-point value checks

| Function | Description                              | Non-float value | Complex |
| -------- | ---------------------------------------- | --------------- | ------- |
| isinf    | Checks if elements are infinite          | False           | OR      |
| isnan    | Checks if elements are NaN               | False           | OR      |
| isfinite | Checks if elements are finite            | True            | AND     |
| isposinf | Checks if elements are positive infinity | False           | -       |
| isneginf | Checks if elements are negative infinity | False           | -       |

All functions have a syntax of the form `ocean.isinf(A[,B])`, where A is the input tensor, and B is an optional output tensor. When B is omitted a new Boolean tensor containing the result is returned. For complex numbers, the function is applied on both the real and imaginary part and the outputs are combined as indicated in the table above (functions `isposinf` and `isneginf` are not available for complex input).

```python
>>> a = ocean.asTensor([1,ocean.inf,-ocean.inf,ocean.nan])
>>> ocean.isinf(a)
   False    True    True   False
<tensor.bool of size 4 on cpu>
>>> ocean.isfinite(a)
    True   False   False   False
<tensor.bool of size 4 on cpu>
>>> ocean.isnan(a)
   False   False   False    True
<tensor.bool of size 4 on cpu>
>>> ocean.isposinf(a)
   False    True   False   False
<tensor.bool of size 4 on cpu>
>>> ocean.isneginf(a)
   False   False    True   False
<tensor.bool of size 4 on cpu>
```

## Tensor reductions

Many of the tensor reductions have the form `ocean.reduce(A [, axes [,keepdims=False] [,B]])`. Input argument `A` is a tensor or a tensor-like object. When no additional arguments are provided, the reduction is done along all axes and the output will be a scalar (not a scalar tensor). Then one or more axes is provided, reduction is done only along these dimensions. Axes values can range from `-ndims` to `ndims-1`, where negative dimensions follow the Python convention with -1 being the last dimensions, -2 the one before that, etc; the set of indexed axes cannot included duplicates. When the keepdims flag is set to True, all indexed axes will have size 1 in the output, otherwise they do not appear in the output dimensions. The final (optional) parameter `B` can be used to specify the output tensor. When given, the dimension must match that of the operation. When the data type or device mismatch the output is generated though an intermediate array. The same applies when the output tensor `B` is byte-swapped or does not satisfy the memory alignment requirements for the device.

##### `ocean.any(A [, axes [,keepdims=False] [,B]])`, `ocean.all(A [, ...])`, `ocean.allFinite(A [, ...])`, `ocean.anyInf(A [, ...])`, `ocean.anyNaN(A [, ...])`

The `any` function returns True when at least one entry is non-zero and False otherwise. The `all` function returns True only when all elements in A are non-zero. The `allFinite` function returns True when all elements in A are finite (that is, it does not include any infinity or NaN values). The `anyInf` and `anyNaN` functions check if there are any infinity or NaN elements, respectively.

```python
>>> ocean.any([0,0,0])
False
>>> ocean.any([0,0,1])
True
>>> ocean.all([0,0,1])
False
>>> ocean.all([1,1,1])
True
>>> ocean.allFinite([1,2])
True
>>> ocean.allFinite([1,ocean.inf])
False
>>> ocean.allFinite([1,ocean.nan])
False
>>> ocean.anyInf([1,ocean.nan])
False
>>> ocean.anyInf([1,ocean.inf])
True
>>> ocean.anyNaN([1,ocean.nan])
True
>>> ocean.anyNaN([1,ocean.inf])
False
```

Some examples of logic reductions with axes:

```
>>> a = ocean.asTensor([[1,0,1,0],[0,0,1,1]],'r')
>>> print(a)
(:,:)
   1   0   1   0
   0   0   1   1
<tensor.int64 of size 2x4 on cpu>
>>> print(ocean.all(a,0))
   False   False    True   False
<tensor.bool of size 4 on cpu>
>>> 
>>> b = ocean.tensor([1,4],ocean.bool)
>>> ocean.any(a,0,True,b)
>>> print(b)
(:,:)
    True   False    True    True
<tensor.bool of size 1x4 on cpu>
```

##### Domain checks: `ocean.allLT(A, bound)`, `ocean.allLE(A, bound)`, `ocean.allGT(A, bound)`, `ocean.allGE(A, bound)`

These functions perform global reductions and check whether all elements in the tensors are strictly less than (LT), less than or equal (LE), strictly greater than (GT), or greater than or equal (GE) to the given bound. Any NaN values in the tensor are ignored.

```python
>>> ocean.allLE([1,2,3],3.2)
True
>>> ocean.allLT([1,2,3],3)
False
>>> ocean.allGT([1,2,3],1.1)
False
>>> ocean.allGE([1,2,3],1)
True
>>> ocean.allGE([1,2,3],1+2j)
False
>>> ocean.allGE([1,2,3],1-2j)
True
```

##### Domain check: `ocean.allInRange(A, lower [, lowerInclusive=True], upper [, upperInclusive])`

The all-in-range function checks whether all elements in A are between the given lower and upper bounds (with inclusiveness of the bounds indicated by the optional flags). The lower and upper bounds can be set to `None` to indicate trivial satisfaction of that bound (this is equivalent to setting the bound to infinity, as floating-point arithmetic comparison of infinity with equality results in true). Any NaN values in the tensor are ignored.

```python
>>> ocean.allInRange([1,2,3],1,3)
True
>>> ocean.allInRange([1,2,3],1,False,3)
False
>>> ocean.allInRange([1,ocean.inf],0,ocean.inf)
True
>>> ocean.allInRange([1,ocean.inf],0,None)
True
>>> ocean.allInRange([1,ocean.nan],0,2)
True
```

##### `ocean.nnz(A [, axes [,keepdims=False] [,B]])`, `ocean.nnzNaN(a, [,...])`

The `nnz` function determines the number of non-zero values in the tensor (this includes NaN values). The output type of nnz is always `uint64`. The `nnzNaN` function excludes any NaN entries from the count.

```python
>>> a = [0,0,1,2,ocean.inf, ocean.nan]
>>> ocean.nnz(a)
4
>>> ocean.nnzNaN(a)
3
>>> a = ocean.asTensor([[0,0,1,1],[1,1,1,0]],'r',ocean.float)
>>> a
(:,:)
   0   0   1   1
   1   1   1   0
<tensor.float of size 2x4 on cpu>
>>> ocean.nnz(a,0)
   1   1   2   1
<tensor.uint64 of size 4 on cpu>
>>> ocean.nnz(a,1)
   2   3
<tensor.uint64 of size 2 on cpu>
>>> ocean.nnz(a,[0,1])
5
<scalar.uint64 on cpu>
```

##### `ocean.sum(A [,axes [,keepdims=False] [,B]])`, `ocean.sumNaN(A, [,...])`, `ocean.sumAbs(A [,...])`, `ocean.sumAbsNaN(A [,...])`,  `ocean.prod(A [,axis [,B]])`, `ocean.prodNaN(A, [,axis [,B]])`

These functions compute the summation of the all elements, or the elements along the given axes. The `sumNaN` function includes NaN values, `subAbs` computes the sum of absolute values (magnitudes for complex numbers), and `sumAbsNaN` computes the sum of absolute values excluding any NaN values. Similarly, the prod function compute the product of the elements. The output types of these functions are as follows:

| Function | Bool, unsigned int | signed int | Real floating-point | Complex floating-point |
| -------- | ------------------ | ---------- | ------------------- | ---------------------- |
| sum      | uint64             | int64      | input type          | input type             |
| sumNaN   | uint64             | int64      | input type          | input type             |
| sumAbs   | uint64             | uint64     | input type          | base of input type     |
| sumAbsNaN| uint64             | uint64     | input type          | base of input type     |
| prod     | uint64             | int64      | input type          | input type             |
| prodNan  | uint64             | int64      | input type          | input type             |

Some examples:

```python
>>> a = ocean.asTensor([[1,2,ocean.nan],[3+4j,4,5]])
>>> a
(:,:)
     1 + 0j      3 + 4j
     2 + 0j      4 + 0j
   nan + 0j      5 + 0j
<tensor.complex-double of size 3x2 on cpu>
>>> 
>>> ocean.sum(a,0)
   nan + 0j     12 + 4j
<tensor.complex-double of size 2 on cpu>
>>> ocean.sumNaN(a,0)
    3 + 0j    12 + 4j
<tensor.complex-double of size 2 on cpu>
>>> ocean.sumAbs(a,0)
   nan    14
<tensor.double of size 2 on cpu>
>>> ocean.sumAbsNaN(a,0)
    3   14
<tensor.double of size 2 on cpu>
>>> ocean.prod(a,0)
   nan + nanj     60 +  80j
<tensor.complex-double of size 2 on cpu>
>>> ocean.prodNaN(a,0)
    2 +  0j    60 + 80j
<tensor.complex-double of size 2 on cpu>
```

Summation or products over empty slices result in tensors with elements set to the initial values of 0 and 1, respectively:

```
>>> a = ocean.tensor([0,3,4])
>>> ocean.prod(a,[0,-1])
   1   1   1
<tensor.float of size 3 on cpu>
>>> ocean.sum(a,0)
(:,:)
   0   0   0   0
   0   0   0   0
   0   0   0   0
<tensor.float of size 3x4 on cpu>
>>> ocean.sum(a,1)
<empty tensor.float of size 0x4 on cpu>
```

##### `ocean.minimum(A [,axes [,keepdims=False] [,B]])`, `ocean.minimumAbs(A, [,...])`, `ocean.maximum(A [,...])`, `ocean.maximumAbs(A, [,...])`

These functions can be used to determine the minimum and maximum (absolute) value of the elements in the tensor, or along a slice indicated by the axes. The number of elements in the reduction must be strictly greater than zero. For each of the operations, the result is initialized by the first element in the tensor or slice of elements. While accumulating the final result, elements with NaN components are ignored, and the result of the operation is therefore a valid number, unless all elements contain NaN, in which case the result is equal to the initial value. The result type of `minimum` and `maximum` is equal to the input type, and that of `minimumAbs` and `maximumAbs` is the same for boolean, unsigned integer, and float; is unsigned integer with the same number of bits for signed integers; and the base type for complex types. If NaN values are important a separate check can be made with the `ocean.anyNaN` function.

```python
>>> a = ocean.asTensor([[2,-3,ocean.nan],[3,-1,2]],'r')
>>> a
(:,:)
     2    -3   nan
     3    -1     2
<tensor.double of size 2x3 on cpu>
>>> ocean.minimum(a)
-3
>>> ocean.maximum(a)
3
>>> ocean.maximum(a,0)
    3  -1   2
<tensor.double of size 3 on cpu>
>>> ocean.maximumAbs(a,0)
   3   3   2
<tensor.double of size 3 on cpu>
>>> ocean.minimumAbs(a,1,True)
(:,:)
   2
   1
<tensor.double of size 2x1 on cpu>
>>>
>>> ocean.maximum([1+2j,3+4j])
3 + 4j
>>> ocean.maximumAbs([1+2j,3+4j])
5
```

The minimum and maximum functions do not apply to empty slices:

```python
>>> a = ocean.tensor([0,3,4])
>>> ocean.minimum(a)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: The tensor "minimum" function does not apply to empty tensors
>>> ocean.minimum(a,[0,1])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: The number of reduction elements in axisMinimum cannot be zero.
>>> ocean.minimum(a,2)
<empty tensor.float of size 0x3 on cpu>
```

##### `ocean.norm(A [,p=2 [,axes [,keepdims=False] [,B]]])`, `ocean.normNaN(A [,...])`
##### `ocean.norm1(A [,axes [,keepdims=False] [,B]])`, `ocean.norm2(A [,...])`, `ocean.normInf(A [,...])`

The `norm` function computes the p-norm (power 1/p of the summation of the absolute value of the entries raised to the power p) over all entries or over slices along given axes. The `nornNaN` function treats any NaN entries as 0. The function is defined for p >= 0, and is a proper norm only for p >= 1. Choosing p = 0 gives the number of non-zero entries and is therefore equivalent to `ocean.nnz` or `ocean.nnzNaN`. Other special cases include p = 1, which is equivalent to `ocean.sumAbs` and `ocean.sumAbsNaN`, respectively; and p equal to infinity, which is equivalent to `ocean.maximumAbs` (note that NaN values are always ignored in this case). The functions `norm1`, `norm2`, and `normInf` are equivalent to the `norm` function with `p=1`, `p=2`, and `p=Inf`, respectively.

```python
>>> a = ocean.asTensor([[1,2,ocean.nan],[3,4,-5]],'r')
>>> a
(:,:)
     1     2   nan
     3     4    -5
<tensor.double of size 2x3 on cpu>
>>> print(ocean.norm(a,1,[0]))
     4     6   nan
<tensor.double of size 3 on cpu>
>>> print(ocean.normNaN(a,1,[0]))
   4   6   5
<tensor.double of size 3 on cpu>
>>> print(ocean.normNaN(a,2))
7.4162
>>> print(ocean.norm(a,3,[0]))
   3.03659   4.16017       nan
<tensor.double of size 3 on cpu>
>>> print(ocean.norm(a,ocean.inf))
5
```

## Tensor saving and loading

Saving and loading functions have not yet been implemented. For the time being it is therefore recommended to convert between Ocean tensors to Numpy arrays and use the save and load functions provided by Numpy.

## Tensor reference management

##### `tensor.refcount` (property)

The `refcount` property gives the number of references to the underlying tensor object. In most cases this value will be one, as shallow copies of the tensors are typically made to avoid unexpected behavior when manipulating linked tensors.

##### `tensor.detach()`

The `detach` function ensures that the reference count to the underlying tensor object is one.

##### `tensor.dealloc()`

This function is provided to force a reference count decrease of the tensor object instead of relying on Python's garbage collection.
The Python tensor object is set to a read-only empty tensor to ensure that accidental use afterwards does not lead to segmentation faults. Deallocating a tensor a second time merely replaces the empty tensor by itself.

```python
>>> import ocean
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
>>> a.dealloc()
>>> a
<empty tensor.int8 of size 0 on cpu (read-only)>
```

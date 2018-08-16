# Design choices

## Tensor dimensions and data layout

Tensors are multi-dimensional arrays of scalar values. When using tensors each of the axes will have specific interpretations, for example a three-dimensional tensor may consist of a data volume with indices corresponding to the x, y, and z axes, or it may represent a set of two-dimensional matrices. Given a set of k matrices of size m-by-n the main questions are: what should be the tensor dimensions, and how should the elements be stored in memory? The first option, used by Numpy and Torch is to represent this as a tensor T of size (k,m,n) with strides (m\*n, m, 1). Extracting the first matrix is done by T[1,:,:] or simply T[1]. The main difficulty with this representation emerges when reducing the number of dimensions: the (k,m,n) tensor consist of k tensors of size (m,n). Each (m,n) matrix consists of m vectors of size n. This implies that vectors are row vectors, and indeed, when using broadcast rules, adding a vector v to a matrix means that the vector is interpreted as a row vector, which is then replicated m times before being added to the matrix. For example, consider the following Numpy example

```python
>>> M = numpy.zeros([3,3])
>>> M + [1,2,3]
array([[ 1.,  2.,  3.],
       [ 1.,  2.,  3.],
       [ 1.,  2.,  3.]])
```

When multiplying the same vector with a diagonal matrix, the vector is interpreted as a column vector:

```python
>>> D = numpy.diag([1,2,3])
>>> D * [1,2,3]
array([[1, 0, 0],
       [0, 4, 0],
       [0, 0, 9]])
```

The second option, used by Matlab, is to represent k matrices of size m-by-n as a (m,n,k) array with strides (1,m,m\*n). Extracting the first matrix is done using T[:,:,1]. Reducing the number of dimensions does become more consistent with conventional linear-algebra notation: the (m,n,k) array consists of k matrices of size (m,n), and (m,n) matrices consist of n vectors of size m, each of which is a column vector.

In order to ensure consistent notation, Ocean uses the second option (except that it does not have the convention used in Matlab that vectors and scalar values are two-dimensional). For linear-algebra purposes, tensors are seen as collections of matrices. For data order Fortran-style strides are used, starting with unitary strides for the first dimension, and contiguously going up for every subsequent dimension.

## Tensor transpose

The transpose of a tensor can be defined as exchanging the first two dimensions. For zero-dimensional tensors (scalar) the transpose is merely the tensor itself, and it therefore only remains to define the transpose of one-dimensional tensors (vectors). In Numpy and Torch this operation is defined as the identity operation, which means that `x` and `x.T` are identical. This is undesirable if, for readability, we want to compute the inner product `x.T * x`, the outer product `x * x.T`, or the scalar quantity `x.T * A * x`. The transpose of a vector v of size n should therefore be of size 1xn. The main question is what the transpose of 1xn should be: should it be nx1 or a vector of size n? That is, should `(v.T).T` be a matrix or a vector? There are several possible approaches:

1. Add a flag to vectors and matrices indicating that the tensor is inherently a vector. When applying the transpose to a matrix with the vector flag set we can return the n vector, otherwise we return an nx1 matrix. The advantage of this approach is that applying the transpose to a vector twice gives the same vector. The disadvantage is that determining the semantics of the flag is cumbersome.
2. Simply define the transpose of a 1xn matrix as the corresponding nx1 matrix.
3. Introduce None dimensions. These dimensions are of size 1, but are removed whenever they appear at the end of the dimensions. This allows us to differentiate between the (total number of) tensor dimensions and the intrinsic tensor dimensions (with all None dimensions omitted). For example, the transpose of a vector of length n would be of size None x n, which has dimension 2 and intrinsic dimension 1. This approach is quite elegant but requires addition flags to indicate which unit dimensions are of the None type.
4. Whenever the result of the transpose is two dimensional, remove all trailing unitary dimensions.

In the end the option 4 was selected, mostly for its simplicity. In case removing trailing unitary dimensions for two-dimensional tensors is undesirable, it is possible to use `swapAxes(0,1)` instead. The above discussion also applies to the conjugate transpose with appropriate changes to account for conjugacy.

## Tensor multiplication

As mentioned above, Ocean interprets tensors as collections of matrices for linear-algebra operations. That means, for example, that the multiplication of a (n1,n2,k) matrix with a (n2,n3,k) tensor is interpreted as k independent matrix-matrix multiplications of size (n1,n2) times (n2,n3), therefore resulting in a (n1,n3,k) tensor. For tensor multiplication, broadcasting is done only along the third and higher dimensions. Multiplying a (m,n) matrix and a (n) vector results in a vector of size (m). Multiplying a (m,n) matrix by a (1,1,k) tensor is interpreted as k scalar multiplications of the (m,n) matrix, therefore resulting in a (m,n,k) tensor.

## Data types

In Python integer and floating-point numbers are represented by Int64 and Double, respectively. This raises some issues when, for example, given a Float tensor T and adding a scalar:

```python
>>> T + 3.0
```

In the Python-C API this will call the addition operator associated with the tensor object with the double scalar as the second input argument. Directly converting the 3.0 to a tensor does not work for two reasons:

1. The result of adding a Float and a Double tensor results in a Double tensor, which is clearly undesirable in this situation;
2. When reversing the order: 3.0 + T, the implicit tensor corresponding to the scalar would be on CPU, and may force the entire expression to CPU when T resides on another device.

In Numpy scalars are treated differently than tensors and the above example would indeed result in a Float tensor. In general, scalars are converted to whichever type possible, based on the range of the number:

```python
>>> (np.int8([32]) + 3)       # array([35],  dtype=int8)
>>> (np.int8([32])  + 127)    # array([-97], dtype=int8)
>>> (np.uint8([32]) + 127)    # array([159], dtype=uint8)
>>> (np.int8([32])  + 128)    # array([160], dtype=int16)
>>> (np.uint8([32]) + 128)    # array([160], dtype=uint8)
>>> (np.float32([0]) + 1e10)  # array([  1.00000000e+10], dtype=float32)
>>> (np.float32([0]) + 1e100) # array([  1.00000000e+100])
```

Note that 127 is not directly converted to int8, since addition of int8 and uint8 should give a type of int16. When repeating the same experiment with array versions of the scalar: `[k]`, Numpy returns int64 results for integers and double results for floating-point scalars.

In Ocean there is the additional difficulty of determining the device. Adding a scalar 3 to a tensor on a given device should result in a tensor on the same device. One approach when parsing Python objects (or equivalent objects in other language bindings) is to label ad-hoc intermediate tensors as weakly typed, which then indicates that the data type or device is flexible and should be dealt with accordingly when determining the data type and device of the tensor operations. For example, consider the tensor-like object `[1, 128, 4096, 2.0, np.float32(3.0)]`. What would be the type of this tensor and how would be determine it?

### Weak data types

One way would be to consider the smallest possible data type that fits all data without losing precision (except, possibly, when combining 64-bit integers with double-precision floats). For integers we would additionally need to keep track of whether the data is or could be signed. A procedure could look as follows:

1. The integer 1 fits the int8 range and is unsigned (allowing it to be added to uint8 without changing the type);
2. We next parse 128, which has uint8 as its lowest type. Combining an unsigned int8 value with uint8 gives uint8;
3. It would seem that any unsigned integer type can automatically be strongly typed (adding uint8 to int8 necessarily gives int16, when the values are disregarded). The unsigned 1024 value fits in int16, so combined with uint8 we obtain an unsigned int16 value.
4. When parsing 2.0 we clearly need to switch to floating-point numbers, but how? In order to maintain the full precision we generally need to maintain the double type, but adding to a single-precision float should clearly maintain that type. The approach taken here is that flags are maintained whether the weakly-typed float lies in the range of 16 and 32 bit floats (\* see discussion below). At this point we should also go back to integers. Any int8 number can be exactly represented in 16, 32, and 64 bit floating point representation, but int16 generally cannot be represented exactly as a 16-bit float. When parsing 1024 we could maintain the floating-point flags to indicate that 1024 may be typed as int16, but that the current range still fits float 16 (and therefore float 32 and 64).
5. When parsing a strongly-typed scalar, such as `np.float32` we avoid inspection of the values (it could be a tensor on the GPU whose values are still being computed) and therefore assume that the number can be anywhere in the range of the data type. For integer objects such as `ocean.int32` we would similarly have to conclude that the value could be negative, and might not be represented exactly as a 32-bit float.

Note \*: one of the questions that may be asked here is how to define `ocean.float([1.]) + 1e100`: should this give a single-precision `Inf` value, or a double precision `1e100`? Likewise, the number `1e-100` fits in the range of 16-bit float, but cannot be represented by it, and would be cast to `0`. In that case, what would `ocean.float([1.]) / 1e-100` be? Division by zero, or `1e100` as a double. When treated as a double, would `ocean.float([1.]) + 1e-100` also be double precision? For for scalar operations, such as `ocean.float(1.) / 1e-100` we may simply cast float and double to double, since the value of the operation is always readily available for inspection on CPU. In the case of tensors, by contrast, there will generally be too much overhead in analyzing the range of the values. Moreover, such analysis would add an undesirable synchronization point in the compute pipeline, since values need to be available at the time of the decision.

We might want to force tensors to have a strong type and add `ocean.float([1,2,3])` to a tensor. In this case, the data type should clearly be fixed, but the device has not yet been specified. Should this be equivalent to `ocean.asTensor([1,2,3],ocean.float)`, and likewise `ocean.gpu[0]([1,2])` be equivalent to `ocean.asTensor([1,2,3],ocean.gpu[0])`? Should weakly typed tensors ever be exposed to the user? When the device is weak, this lead to surprises lated in the code when it is no longer obvious that the tensor device is weakly determined. For weak data types the problem is worse: the weakly typed double array `a = [100.,200.]` clearly fits in the range of 32-bit floats. However, when applying an in-place `ocean.exp` this is no longer the case, and adding checks on every operation and potentially force the type may be undesirable. To guarantee that weakly typed tensors never reach the user there are two options:

1. Add checks in all tensor operations to make sure that a weakly typed tensor is always strongly typed prior to return to the user;
2. Add checks in the wrapping of tensor objects and scalars.

### Weak data types versus language dependent solutions

One of the problems that arises from dealing with weak scalar and tensor types is the need for good semantics of the weak types. In the above there are many situations where the solution was clear, but the result of operations on weak types is not always clear. For example, suppose we type `ocean.add([100],[100])`. When using weak types both tensors would be recognized as weak nonnegative int8 tensors, and it would therefore make sense to define the result as int8. However, this means that the result would be `[-56]`. Calling `ocean.add([100],[128])` would combine weak int8 and weak uint8 tensors resulting in in16 `[228]`. Although this scenario is somewhat contrived, it does show that the use of weak types can result in unexpected results. After implementing a preliminary version of weak typing it was decided that it both complicates code and confuses results, and therefore to keep things simple it was decided not to use weakly-typed data, and instead use a language (Python) specific implementation.

#### Parsing of tensor-like objects and scalars

Just like in numpy we use the convention that scalar operations use the data type provided by Python. That is, all integer are int64, all floating-point numbers are double, and all complex numbers are cdouble. Operations are done in the minimum common data type:

```python
>>> (np.int8(0) + 0).dtype
dtype('int64')
>>> (np.float32(0) + 0.).dtype
dtype('float64')
>>> np.int8(120) + np.int8(8)
__main__:1: RuntimeWarning: overflow encountered in byte_scalars
-128
>>>
>>> (ocean.int8(0) + 0).dtype
<dtype 'int64'>
>>> (ocean.float(0) + 0).dtype
<dtype 'double'>
>>> (ocean.int8(120) + ocean.int8(8)).dtype
<dtype 'int8'>
```

When using weakly-typed scalars (Python objects) in tensor operations we convert the values to the appropriate data type based on the value and a reference data type. Weakly-typed scalars appearing in tensor-like objects will have the data type native to Python (int64 for integers, and (complex) double for (complex) floating-point numbers).

Earlier code implemented the following approach, which is now obsolete: when parsing tensor-like object we maintain the principle that the data type should be the minimum that guarantees lossless encoding (with the exception of long integers and doubles, which cannot be combined in a lossless manner; in this case we choose double). Note that the conversion of integer to floating-point types is not based on the range of the latter, but the ability to represent the integers exactly. For the half-precision format this means that integers between -2048 and 2048, inclusively, can be represented exactly, but 2049 cannot. Therefore [0,1,2048,ocean.half(3)] has type half, whereas replacing 2048 by 2049 would result in float type. Python floats have double-precision, and [ocean.float(3), 0.0] is stored as double to prevent losing accuracy of the Python scalar (no checks are done on the ability of float to exactly represent 0.0 as this would give very unpredictable types; 1./2 would give float and 1./3 would give double). Data types that are strongly typed, such as ocean or numpy scalars and tensors, are not inspected in the sense of checking all the values.  Instead it is assumed in these cases that the given data type is needed. Numpy differs in two aspects in that it (i) inspects scalar values and ignores their type, and (ii) does try to infer the minimum type needed when parsing tensor-like objects, instead it just combines the types givens (such as int64 for integer values):

#### Device determination

The device of a tensor-like object is determined by its components. Scalars have a weak CPU type and as such do not force the device type. Tensor objects have a strong device type and the tensor-like object inherits this strong type provided that all tensors appearing in it have the same device. When a mixture of devices appears, the resulting tensor-like object will have a weak-CPU device type.

## Operations with tensors on different devices or mixed data types

For convenience and completeness, Ocean allows operations on tensors that reside on different devices. Given an operation, such as `ocean.add(A,B,C)`, where A is on CPU, B on GPU\[0\], and C on GPU\[1\], where should we do the computation, and what would be the device for the result of `A + B`? Likewise, what data type should we use for the computation when A is of type double, B of type int8, and C of type float?

We use three main principles to determine to mode of operation:
1. Destination tensors are generally ignored when determining the data type and device used for the operation. When the destination tensors does not have the desired data type or device, we first perform the operation with an intermediate tensor as output and then copy the result. There are some exceptions, for example adding two complex tensors with a real output tensor creates views on the real part of the input and performs the addition only over the real part, instead of computing the complex addition, and then copying the real part. Note that such simplifications are operations-specific optimizations and do not otherwise affect the principle.
2. The device used to perform the operation is determined by the first tensor. In the addition example, `ocean.add(A,B,C)` and `A + B` are evaluated on CPU, and `B + A` is evaluated on GPU\[0\]. Note that implicit tensors such as `[1,2,3]` are on the CPU, unless all non-scalar components are consistently on another device.
3. Operations done generally in the smallest common data type

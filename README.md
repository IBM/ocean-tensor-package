# The Ocean Tensor Package

The Ocean Tensor Package, or Ocean for short, provides a comprehensive set of tensor functions.

1. [Introduction](#introduction)
2. [Installation](docs/installation.md)
3. [Python interface](docs/python/index.md) with modules [core](docs/python/module_core.md) and [numpy](docs/python/module_numpy.md)
4. [Future work](docs/future_work.md), [design choices](docs/design_choices.md), and [implementation](docs/implementation.md)

## Introduction

In recent years there has been a tremendous proliferation of hardware acceleration and computation devices such as GPUs and FPGAs to address the ever-increasing need for computational power. Deep learning, which requires the processing of large volumes of data through computationally intensive neural networks for training and inference, has both been enabled by and driven the development of advanced computational hardware. In conjunction, several tensor-computation packages specialized towards deep learning have been developed to leverage the compute capabilities of these advanced new devices. However, tensor computations arise in various fields outside of deep learning, including:

* Scientific computing
* Numerical optimization
* Image and signal processing
* General machine learning
* Data science

Current implementations of tensor operations, present in all deep-learning packages, still lack in one or more aspects, including: missing data types or too strong typing of tensors, lack of modularity, minimal or no flexibility in memory layout, inadequate usage of streams, and limited support for direct tensor operations. Consequently, there is a need for a comprehensive general-purpose tensor package. The Ocean Tensor Package, provides the foundation library provides low-level access to multiple device types through a high-level and easy-to-use interface needed in these areas. The modular design makes it easy to add new functionality, provide support for new and emerging device types, and install packages on a per-need basis. The Ocean Tensor Package, or Ocean for short, consists of three layers:

1. The Solid foundation library, which provides low-level functions that are independent of the higher-level tensor representation;
2. The Ocean tensor library, which implements the tensor and module infrastructure and provides the high-level tensor APIs;
3. A Python interface to the package that provides user-friendly access to all tensor functions, and provides interoperability with existing packages.

The Ocean Tensor Package provides support for various integer, floating-point, and complex [data types](docs/python/index.md#data-types) and supports non-aligned and byteswapped memory layouts. Automatic conversion between data types and devices, as well as dimension broadcasting is supported and can be configured to provide low-level control over all operations. On the GPU, high levels of asynchronicity are enabled by consistent usage of streams and the usage of special intermediate tensors. 

As an example of the flexible, but well-defined, usage of different devices and data types, consider the following implementation of the modified Gram-Schmidt algorithm for QR factorization, in which the byteswapped double-precision Q matrix is updated in-place on the CPU, and the single-precision R matrix is maintained on a GPU device

```Python
import ocean

# ------------------------------------------------------------------------
# Initialize the matrix
# ------------------------------------------------------------------------
A = ocean.arange(25, ocean.double).reshape(5,5)
d = A.diag()
d += 10
print(A)

# ------------------------------------------------------------------------
# Factorize A = QR using modified Gram-Schmidt: copy A to Q and factorize
# in place. Matrix R is stored on GPU device 0.
# ------------------------------------------------------------------------
Q = A.clone()
Q.byteswap()
R = ocean.zeros(A.size, ocean.float, ocean.gpu[0])
n = A.size[1]

for i in range(n) :
   q = Q[:,i]
   r = ocean.sqrt(q.T * q)
   q /= r
   R[i,i] = r
   for j in range(i+1,n) :
      r = q.T * Q[:,j]
      Q[:,j] -= q * r
      R[i,j] = r

# ------------------------------------------------------------------------
# Print the factorization along with the differences Q'*Q and A - QR
# ------------------------------------------------------------------------
print(Q)
print(R)
print(ocean.norm(Q.T * Q - ocean.eye(n)))
print(ocean.norm(Q*R - A))
````

with output

```Python
# Matrix A
(:,:)
   10    5   10   15   20
    1   16   11   16   21
    2    7   22   17   22
    3    8   13   28   23
    4    9   14   19   34
<tensor.double of size 5x5 on cpu>

# Matrix Q
(:,:)
    0.87706  -0.32040  -0.20835  -0.18629  -0.22361
    0.08771   0.82876  -0.43466  -0.25793  -0.22361
    0.17541   0.26913   0.85917  -0.32958  -0.22361
    0.26312   0.26486   0.14593   0.88844  -0.22361
    0.35082   0.26059   0.09052   0.02866   0.89443
<tensor.double of size 5x5 on cpu (byteswapped)>

# Matrix R
(:,:)
   11.40175   12.27881   21.92645   31.57409   41.22173
    0.00000   18.00641   18.92488   25.39694   31.86899
    0.00000    0.00000   15.20130   10.33189   12.04086
    0.00000    0.00000    0.00000   12.89668    5.01538
    0.00000    0.00000    0.00000    0.00000   11.18034
<tensor.float of size 5x5 on gpu0>

# ocean.norm(Q.T * Q - ocean.eye(n))
1.0561e-15

# ocean.norm(Q*R - A)
2.02686e-6
```

The Ocean Tensor Package runs on various platforms including MacOS and Linux running on Intel and Power machines with or without GPU devices.

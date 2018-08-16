# Future work

This page summarizes some of the parts of the code that could be improved, extended, or cleaned up.


## <a name="CMakeImprovements">Improvements to the cmake files</a>
* Support for `make install`
* Support for more CBlas implementations in `external/ocean-blas/cmake/FindOcBlas_cblas.cmake`
* CMake builds from external directories works, but automatically generated header files, intermediate files for the Python interface compilation, and all library files are still output in the source directory tree
* Compilation assumes sequential builds; using parallel compilation with several jobs (`make -j <#jobs>`) does not currently work because of missing declarations of dependencies in the cmake files (see also the library structure issue below).
* Use `find(PythonInterp)` in the Python interface scripts to ensure that Python include files are available on the system. Also ensure that there is no version mismatch between the Python interpreter and the Python include files.

## Library structure

The Ocean Tensor Package is designed to be modular to allow compilation with or without CPU, compilation of only the Solid library, or Solid with Ocean, etc. The downside of this is the large number of library files that are created, and an increased complexity of the make files. Should the libraries be compiled as static libraries, and how are they best grouped functionally? Should future modules be included as part of the package, or as separate repositories?


## Extensions

* Core module
   * Tensor and storage save/load routines. Should device be stored as well? Options to force load on same device or allow CPU?
   * Tensor resize, storage resize may be possible but only when the size grows.
   * Storage allocation: pinned memory, memory-mapped files, what should the interface be?
   * Library packages with initialization and (automatic) finalization with mechanisms to add backends and modules, allow listing of loaded modules and possibly versions. Module registration must be done in ocean, not ocean_base, maybe include ocean base in ocean?
    * Implement ocean.cumsum
* Add a random-number module or include in the Core module (this would add dependencies on possibly external libraries)
* Add a linear-algebra module
* Add a module for fast transformations (such as DCT, FFT, Wavelet, Hadamard)


## Indexing

* Implement conversion from Boolean mask to indices or offsets on GPU
* Support more data types for indexing (currently only int64)
* Detect repeating entries in indices, how to deal with this for index assign?
* Improved manipulation of index objects: allow indexing of index objects


## Performance improvements

* Several optimizations and improvements are possible for the GEMM routine:
  * When the underlying GEMM routine has support for more general strides, take advantage of this;
  * For GPU, we can use `cublasSetAtomicsMode()` for faster but non-reproducible multiplication. Should there be a variable in the configuration that controls its use, should this mode be used by default, or only when enabled by the user?
  * The current implementation of GEMM for booleans and integers should be considered only as a reference implementation, and could be optimized for both CPU and GPU.
  * Use batched GEMM when available whenever possible.

* Tensor simplification in `OcSolid_analyzeElemwise2b`: it may be possible to normalize the sizes and strides further by considering consecutive blocks of dimensions with matching total number of elements. The order of the elements may be flipped for all dimensions in a block, and the order of the blocks may be changed to improve memory access patterns (sort by smallest or largest stride within block). After this it may be possible to further merge dimensions.

* GPU code tuning, improvements, and load time
   * Tune GPU apply-elemwise3 kernels
   * Unary GPU operations should avoid the use of "if-then" when possible (see, e.g. sqrt)
   * Loading time: currently there may be too many function specializations, which slows down the compilation and loading of the libraries (more specifically, the initialization of Cuda). For example copy routines between all types with specializations for tensors that are [1,2,3,n] x [1,2,3,x]. It may suffice to only support (1,1), (2,2), (3,3), and (n,n).
   * Check usage of `cudaMemcpyPeerAsync` and possible usage of UVA direct memory access

* See all references to "Possible optimization" or "Possible improvement" in the code


## Floating-point errors and type conversion   

On the CPU it is possible to check the status of the floating-point register to catch certain numerical issues when converting data types or performing operations on them. For others, such as integer division by zero, explicit checks are needed. On the GPU there are no floating-point status registers and all checks would need to be explicit. Some functions may require conversion to the complex domain to give meaningful results. The current solution for functions such as square root is to add flags that can enable and disable explicit checks, and determine the behavior when checks fail: give a warning or raise and error. More checks may be needed, or a more structure approach may be needed. A default mode may also be useful, but to ensure consistent behavior of libraries, explicit flags may be needed.

* Improve comparison operations: (1) deal with the special case of comparing between int64, uint64, and double. Currently we cast such input to the least common data type, which is double; (2) when comparing integer tensors with floating-point scalars we can avoid casting the input tensor to float. For example when comparing `a <= 3.5` we can compare `a <= 3`, or `a >= 3.8` reduces to `a >= 4`. When the rounded floating-point value does not fit the integer type we can have a trivial result of all True or all False values. Additional care needs to be taken when dealing with scalar infinity, NaN, or complex scalars.
* Float to int copy: NaN and Inf entries are copied as 0 without any warning (floating-point status flags are not set)
* Improved range checks on inputs: For example adding `[1,2,ocean.inf]` to an `int8` tensor
* In-place division (`T /= s`) where `T` is integer and `s` may give unexpected results; maybe a warning or error should be added

## Miscellaneous tasks

* Improved algorithm for detection of tensor (self) overlap.
* In the python interface, the current list of devices can currently only be accessed through the `devices()` function. Attempts to add a `devices` variable to the module dictionary, and calling an update device list function from each module that adds devices does not work: during import of the module the dictionary is copied and updates in C are not reflected in Python, and vice versa. One possible solution would be to define a device list object that implements a length and query function (possibly with a string for device instance or device type, in which case a list would be returned).
* Better integration of GPU device capabilities. For example, should there be checks for device capability < 1.3 for double support? How about 5.3 for half-precision computation support?
* Should `OcSize` and `OcIndex` both be signed int64? `OcSize` was initially an unsigned integer (`size_t`), but this led to compilation errors/warnings at the Python interface. If `OcSize` is changed to an unsigned integer, the `OcShape_isValidSize` function becomes obsolete.
* Refactor `ocean.merge` into C code with array of tensors, similar for `ocean.split` with list of devices
* Create an example module that replaces or wraps an existing function
* Python interface: allow specification of the formatting of floating-point numbers
* Solid libraries
   * Solid interface: replace all `ptrdiff_t` by `solid_index`, `size_t` by `solid_size`
   * Use consistent naming of data type classes between Solid and Ocean: real, float, integer, integer with or without Boolean, signed integers, unsigned integers, etc.
   * Add checks on number of dimensions to ensure that ndims does not exceed `SOLID_MAX_TENSOR_DIMS`

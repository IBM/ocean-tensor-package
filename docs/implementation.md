# Implementation

Here we describe some of the basic implementation details of the Ocean Tensor Package. At the highest left, the package consists of three parts:

1. The Solid foundation, which provides a library of low-level tensor operations that are independent on the exact tensor representation: all tensors are represented by the number of dimensions along with the dimension sizes and strides (in bytes);
2. The Ocean library of functions that provides the organization of function modules, and provides the logic between tensor operations. This includes parameter checks, conversions between tensor types, copying data to intermediate tensors, as well as determining availability of functions on given devices, and within each module, finding and calling the appropriate function for the given data type;
3. The Python interface and possible other language bindings. These bindings are intended to be minimal and provide the interface between the given language and the Ocean library. In the Python interface this amounts to parsing of function arguments and Python objects, and includes a mechanism for providing plug-ins to enable interoperability with other tensor representations (such as Numpy).

Currently the package contains the following libraries. Some of these libraries may be merged in later releases to reduce the number of libraries and simplify linking from other code.

| Location                                   | description |
| ------------------------------------------ | ----------- |
| `solid/lib/<system>/libsolid_base.so`      | Error handling, floating-point exceptions, half-precision conversion |
| `solid/lib/<system>/libsolid_cpu.so`       | Abstraction of parallel functions to allow compilation with and without OpenMP |
| `solid/lib/<system>/libsolid_gpu.so`       | Abstraction of GPU Device initialization, finalization, and properties |
| `solid/lib/<system>/libsolid_core_cpu.so`  | Implementation of the core module tensor operations for CPU |
| `solid/lib/<system>/libsolid_core_gpu.so`  | Implementation of the core module tensor operations for GPU |
| `lib/<system>/libocean_base.so`            | Generic functions for device, module, and tensor management |
| `lib/<system>/libocean.so`                 | Implementation of the core module interface and CPU instantiation |
| `lib/<system>/libocean_gpu.so`             | Implementation of the core module GPU instantiation |
| `lib/<system>/libocean_solid.so`           | Interface to the Solid library |
| `lib/<system>/libocean_blas.so`            | Abstraction of BLAS functions (Blas, CBlas, and reference implementation) |
| `lib/<system>/libocean_dummy_itf.so`       | Example module interface |
| `lib/<system>/libocean_dummy_cpu.so`       | Example module instantiation for CPU devices |
| `lib/<system>/libocean_dummy_gpu.so`       | Example module instantiation for GPU devices |
| `interfaces/python/lib/<system>/pyOcean_cpu_v#.so` | Python interface to the core Ocean module |
| `interfaces/python/lib/<system>/pyOcean_gpu_v#.so` | Python interface to the GPU implementation of the core module |
| `interfaces/python/lib/<system>/pyOceanNumpy_v#.so` | Python plug-in for interoperability with Numpy |
| `interfaces/python/lib/<system>/pyOceanDummy_itf_v#.so` | Python interface to the example module interface |
| `interfaces/python/lib/<system>/pyOceanDummy_cpu_v#.so` | Python interface to the example module CPU implementation |
| `interfaces/python/lib/<system>/pyOceanDummy_gpu_v#.so` | Python interface to the example module GPU implementation |


## Modules

The Ocean Tensor Package implements a modular system for grouping tensor functionality. Each module provides an interface, which consists of several parts:

1. A function look-up table for each device type, which can be instantiated by device-specific implementations of the module;
2. A module context on each device instance to record module-specific information for each device, such as intermediate buffers or random number generator states;
3. Device-independent implementation of the tensor functions provided by the module. These functions take care of all generic parts of the operations including parameter checks, preparation of intermediate tensors, and data type casting. The interface also checks whether the given operation is supported on the given device by looking up the function in the device type look-up table. When found, the desired function is called after completing all generic preparations.

Implementations of an interface register all supported functions in the look-up table for the corresponding device type. The device specific implementation is common for all data types, and it is up to the device-level function to call the correct underlying function, when supported. In the core module the interface and CPU implementation are combined in a single library to avoid convoluted code. The dummy library (click [here](python/module_dummy.md) for more information) provides an reference implementation where the interface and device bindings are separated.

The main advantages of the modular structure is that it:

* Avoids huge monolithic compilation with a large number of external dependencies when only a small number of function is needed.
* Provides generic interfaces to functions without having to implement them on all possible device types
* Allows future expansion to new device types
* Allows modules to override certain functions (for example for performance evaluation, or replace certain functions with highly-optimized alternatives)
* Allows different implementations of the same interface (for example a generic implementation as well as a proprietary implementation)

As an example of the last example consider the following construction:

```python
# Use to choose between implementation
if standard :
   import pyOceanModule_cpu
else :
   import pyOceanModule_cpu_proprietary
```

### Module initialization and finalization

Modules should register a finalization function during initialization. This ensures that modules are deleted in the reverse order in which they were added. It also simplifies finalization in the C level in that only the `OcFinalize()` function needs to be called:

```C
int main(void)
{  int result;

   if ((result = OcInit()) != 0) goto final;
   if ((result = OcInitGPU()) != 0) goto final;

final:
   OcFinalize();
   return 0;
}
```

In languages such as Python it is possible that some of the Ocean objects remain in the garbage collection processing after the modules are finalized. This is properly dealt with and a final shutdown function, which is called when all device types reference counts reach zero.

### Adding new functions

When adding new functions to an existing module it is important to recompile the interface as well as all implementations of the interface. Module should provide a coherent set of tensor operations, and depending on the type of function to add it may be appropriate to add it to and existing module, or to create a new module. There are different levels of the code that are involved in the addition of a new function. As an example we consider the tensor `fill` function

#### Solid or library level

The fill function first appears in the header file template located in `solid/src/core/cpu/solid_core_cpu.src` file:

```
FUNCTIONS(cpu, int, fill, "int ndims, const size_t *size, const ptrdiff_t *strides, void *ptr, solid_scalar value", all)
```

The template header file is similar to C header files, except that it includes FUNCTIONS and FUNCTIONS2 commands that are expanded depending on the data types required (for example `all`, `integer`, `integer + bool`, `all - complex`). Each FUNCTION in the template results in a several parts in the resulting header file `solid/include/solid_core_cpu.h`:

1. The function type (parameter splitting across lines is done according to the specification in the FUNCTIONS command)
```
typedef int (*solid_funptr_cpu_fill)(int ndims, const size_t *size, const ptrdiff_t *strides, void *ptr, solid_scalar value);
```
2. A function look-up table:
```
extern solid_funptr_cpu_fill solid_cpu_fill[15];
```
3. Declaration of all relevant function instances:
```
/* Function declarations for solid_cpu_fill */
int solid_cpu_fill_bool   (int ndims, const size_t *size, const ptrdiff_t *strides, void *ptr, solid_scalar value);
int solid_cpu_fill_uint8  (int ndims, const size_t *size, const ptrdiff_t *strides, void *ptr, solid_scalar value);
:
:
int solid_cpu_fill_cdouble(int ndims, const size_t *size, const ptrdiff_t *strides, void *ptr, solid_scalar value);
```
4. A parameter structure type
```
/* Parameter structure for solid_cpu_fill */
typedef struct
{  int           ndims;
   size_t       *size;
   ptrdiff_t    *strides;
   void         *ptr;
   solid_scalar  value;
} solid_param_cpu_fill;
```
5. A macro for calling the function
```
#define solid_macro_cpu_fill(FUNPTR,PARAM)  FUNPTR((PARAM)->ndims, (PARAM)->size, (PARAM)->strides, (PARAM)->ptr, (PARAM)->value)
```

The implementation of the different `fill` functions is done in `template_fill.c`, using the `SOLID_APPLY_ELEMWISE1` macro. The Solid library is provided as an independent library of tensor function, which could be called directly on custom tensor types, or used as the foundation for other tensor packages. Although Ocean and Solid are closely linked it is entirely possible for Ocean to use fill functions that are provided by another library. Because the implementation of the operation is data-type specific it helps to have a library to provides easy access to the different versions; otherwise the device-specific implementation of a module function may have to rely on switch statements or other constructions to find the desired functions.

When using the `SOLID_APPLY_ELEMWISE1` or similar macros to generate code it is important to remember that the code in the macro call cannot contain commas. For example using the code `{int a, b; do something;}` confuses the compiler in determining the number of macro parameters and can lead to obscure error messages. To avoid such problems the above code should be written as `{ int a; int b; do something; }` instead.


#### Ocean device-specific code

The device-specific code for `fill` is located in `src/core/cpu/op/tensor_fill_cpu.c`. The main purpose of this layer is to convert any parameters to a format suitable for the underlying library (Solid in this case) and call the function appropriate for the given data type. This may include omitting tensor dimensions with zero strides, reordering the dimensions, and merging dimensions. If the data type is not supported an error is returned. Any new functions should also be added to the initialization part of the module, to ensure that the look-up tables are properly initialized.

In GPU based code the function also takes care of adding appropriate synchronization between the tensor streams.


#### Ocean interface code

The main `OcTensor_fill` function is found in `src/core/interface/tensor_itf.c`. Functions in the interface typically need to check the following:

* Availability of the desired function on the given device
* Checking compatibility of the tensor sizes
* Allocation of the output and intermediate tensor(s) if needed
* Type casting and broadcasting of the tensors
* Check whether destination tensors are read-only or self-overlapping
* Check whether input-output or output-output tensors overlap
* Checking byte order and byte-swapped data (in the fill operation we byteswap the scalar fill value if needed). Not all devices support byteswapping (see the `supportByteswap` field in the device type)

Special support for intermediate tensors is available to ensure that operations remain asynchronous (deleting a tensor storage requires synchronization of the underlying stream, thereby introducing a synchronization point).


### Ocean module

The module look-up table can be found in `include/ocean/core/interface/module_core.h`. Changes to this file require recompilation of the module and its implementations. Some rudimentary checks on the size of the look-up table are made when initializing a module to detect inconsistencies between the interface and implementations.


### Python interface

The Python wrapper for the `fill` function is located in `interfaces/python/pyOcean/cpu/pyOcean_tensor.c`. This function implements the parsing of the different parameter options, and calling the appropriate underlying Ocean function.

Parameter parsing is done using the functions defined in `pyOcean_args.c`. The typical structure is as follows:

```
PyOceanArgs param;
PyOceanArgs_Init(&param, args, "tensor.fill");
PyOceanArgs_GetScalarLike(&param, &scalar, 1);
PyOceanArgs_GetTensorLike(&param, &mask, 0);
if (!PyOceanArgs_Success(&param)) return NULL;

<operations>

PyOceanArgs_Finalize(&param);
```

After initialization of the parameters we check for the appropriate data types. The final boolean flag indicates whether the parameter is mandatory. The return value of the function is negative when an error occurred, 0 when no argument was parsed, and 1 if the argument was successfully parsed. When a function fails, all subsequent calls to parse the arguments automatically fail as well. Some functions, such as the `PyOceanArgs_GetTensorLike` can create new objects (for example Ocean wrappers to Numpy tensors, or Python lists converted to tensors), which need to be freed (see `pyOcean_args.h` for information on which functions use dynamic memory allocation). The `PyOceanArgs_Success` function check for failure and calls `PyOceanArgs_Finalize` in case an error occurred. When dynamic memory is allocated the finalize function should be called before leaving the functions. When all arguments are static, it is possible to replace the `PyOceanArgs_Success` call by `PyOceanArgs_Finalize` and omitting the latter at the end.

Returning of Ocean object to Python is done through the new and wrap functions, for example `PyOceanTensor_Wrap` and `PyOceanTensor_New`. The new function increases the reference counter to the object and returns the corresponding Python object. The wrap function steals a reference and then wraps the object as a Python object. In case the conversion fails, the reference count to the Ocean object will be lowered by wrap, and remains the same for new.

When adding a new function it should be determine whether the operation should be part of the PyOcean module, or the PyOcean tensor object, for example `ocean.zeros`, but `tensor.fill`. In some cases both might be appropriate.


## Miscellaneous

* Strides in Ocean and Solid are always given in bytes, not in multiples of the size of the data type
* The `OcDTypeNone` data type is intended for internal usage only to indicate the equivalent of a `NULL` pointer for a data type, typically to indicate that a default value for the data type should be used. The `OcDTypeNone` data type should never be used to instantiate a tensor or storage
* The `OcIncref<Type>` and `OcDecref<Type>` function are are not atomic. Ocean is expected to be single threaded, with the exception for the computational kernels, which can be multi-threaded
* Compilation of Ocean requires the C99 standard due to the inclusion of functions such as `isfinite` and `isnumber`


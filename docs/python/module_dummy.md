# Module: pyOceanDummy

The pyOceanDummy module provides the Python interface and device implementation mapping of the Dummy module in Ocean. Both modules serve as an example reference for implementing modules in C and their wrapper in Python. There are three parts of the pyOceanDummy module: the interface, the CPU implementation, and the GPU implementation.

The `pyOceanDummy_itf` interface module defines the `hello` function, which takes a singe Device object as input argument. After the necessary device-independent Python type checks it then calls the `HelloWorld` interface function provided in the Dummy module. The latter then uses a function look-up table to find the device-specific implementation of the function. When found, the function is called with the appropriate input arguments, otherwise an error is raised.

The `pyOceanDummy_cpu` and `pyOceanDummy_gpu` modules simply call the Dummy module registration for the given device types, which populates the look-up table. These modules can be loaded independently of the `pyOceanDummy_itf` module, however, since these modules do not provide any interface there is no way to actually call the functions.

For convenience we also implement a `pyOceanDummy` module, which simply imports the above three modules:


The following code illustrates the interface and implementation parts:

```python
>>> import ocean
>>> import pyOceanDummy_itf as dummy
>>> dummy.hello
<built-in function hello>
>>> dummy.hello(ocean.cpu)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Function OcModuleDummy_HelloWorld is not supported on device CPU
>>> dummy.hello(ocean.gpu[0])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Function OcModuleDummy_HelloWorld is not supported on device GPU
```

The actual calls to `hello` generates a runtime error because the look-up table is still empty. To fill it, we simply load the device implementation of the module:

```python
>>> # continuing from above ...
>>> import pyOceanDummy_cpu
>>> dummy.hello(ocean.cpu)
Hello World from the CPU device (cpu)
>>> dummy.hello(ocean.gpu[0])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Function OcModuleDummy_HelloWorld is not supported on device GPU
>>> import pyOceanDummy_gpu
>>> dummy.hello(ocean.gpu[0])
Hello World (1) from GPU device gpu0
>>> dummy.hello(ocean.gpu[0])
Hello World (2) from GPU device gpu0
```

Note that the GPU implementation of the function maintains a device instance specific context, which keeps track of the number of function calls to the HelloWorld function.

As mentioned above, the implementation modules merely register the available functions in the look-up table. It is therefore perfectly valid to load the implementation modules before loading the interface:

```python
>>> import ocean
>>> import pyOceanDummy_gpu
>>> # Nothing to call yet
>>> import pyOceanDummy_itf as dummy
>>> dummy.hello(ocean.gpu[0])
Hello World (1) from GPU device gpu0
>>> dummy.hello(ocean.cpu)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Function OcModuleDummy_HelloWorld is not supported on device CPU
>>> import pyOceanDummy_cpu
>>> dummy.hello(ocean.cpu)
Hello World from the CPU device (cpu)
```

There is quite a bit of flexibility in importing the modules. In the following example we initialize everything in reverse order. Note that all modules automatically load the `pyOcean_cpu`, or equivalently `pyOcean_core`, module.

```python
>>> import pyOceanDummy_gpu
>>> import pyOceanDummy_itf as dummy
>>> import ocean
>>> dummy.hello(ocean.gpu[0])
Hello World (1) from GPU device gpu0
```

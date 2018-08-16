/* ------------------------------------------------------------------------ */
/* Copyright 2018, IBM Corp.                                                */
/*                                                                          */
/* Licensed under the Apache License, Version 2.0 (the "License");          */
/* you may not use this file except in compliance with the License.         */
/* You may obtain a copy of the License at                                  */
/*                                                                          */
/*    http://www.apache.org/licenses/LICENSE-2.0                            */
/*                                                                          */
/* Unless required by applicable law or agreed to in writing, software      */
/* distributed under the License is distributed on an "AS IS" BASIS,        */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and      */
/* limitations under the License.                                           */
/* ------------------------------------------------------------------------ */

#include "pyOcean_device.h"
#include "pyOcean_compatibility.h"
#include "pyOcean/gpu/pyOcean_device_gpu.h"

#include "ocean/core/gpu/device_gpu.h"

#include <stdio.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

static PyObject *pyOcDeviceGPU_deviceName(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_totalGlobalMem(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_totalMem(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_freeMem(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_sharedMemPerBlock(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_regsPerBlock(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_warpSize(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_memPitch(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_maxThreadsPerBlock(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_maxThreadsDim(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_maxGridSize(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_totalConstMem(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_major(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_minor(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_version(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_clockRate(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_deviceOverlap(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_multiProcessorCount(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_kernelExecTimeoutEnabled(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_integrated(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_canMapHostMemory(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_computeMode(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_concurrentKernels(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_ECCEnabled(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_pciBusID(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_pciDeviceID(pyOcDevice *self, void *closure);
static PyObject *pyOcDeviceGPU_tccDriver(pyOcDevice *self, void *closure);



/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

#define PY_GET_ENTRY(name, comment) {#name, (getter)pyOcDeviceGPU_##name, NULL, comment, NULL}
struct PyGetSetDef py_oc_device_getseters[] = {
   PY_GET_ENTRY(deviceName,          "device name"),
   PY_GET_ENTRY(totalGlobalMem,      "total amount of global memory on the device in bytes"),
   PY_GET_ENTRY(totalMem,            "total amount of global memory on the device in bytes"),
   PY_GET_ENTRY(freeMem,             "free amount of global memory on the device in bytes"),
   PY_GET_ENTRY(sharedMemPerBlock,   "maximum amount of shared memory available to a thread block"),
   PY_GET_ENTRY(regsPerBlock,        "maximum number of 32-bit registers available to a thread block"),
   PY_GET_ENTRY(warpSize,            "warp size in threads"),
   PY_GET_ENTRY(memPitch,            "maximum pitch in bytes allows by the memory copy functions"),
   PY_GET_ENTRY(maxThreadsPerBlock,  "maximum number of threads per block"),
   PY_GET_ENTRY(maxThreadsDim,       "maximum size of each dimension of a block"),
   PY_GET_ENTRY(maxGridSize,         "maximum size of each dimension of a grid"),
   PY_GET_ENTRY(totalConstMem,       "total amount of constant memory available on the device in bytes"),
   PY_GET_ENTRY(major,               "major revision number"),
   PY_GET_ENTRY(minor,               "minor revision number"),
   PY_GET_ENTRY(version,             "revision number"),
   PY_GET_ENTRY(clockRate,           "clock frequency in megahertz"),
   PY_GET_ENTRY(deviceOverlap,       "True if the device can concurrently copy memory, False if not"),
   PY_GET_ENTRY(multiProcessorCount, "number of multiprocessors on the device"),
   PY_GET_ENTRY(kernelExecTimeoutEnabled, "True if there is a run time limit for kernels, False if not"),
   PY_GET_ENTRY(integrated,          "True if the device is integrated, False if not"),
   PY_GET_ENTRY(canMapHostMemory,    "True if the device can map host memory, False if not"),
   PY_GET_ENTRY(computeMode,         "compute mode that the device is currently in"),
   PY_GET_ENTRY(concurrentKernels,   "True if the device supports concurrent kernels, False if not"),
   PY_GET_ENTRY(ECCEnabled,          "True if the device has ECC support turned on, False if not"),
   PY_GET_ENTRY(pciBusID,            "PCI bus identifier of the device"),
   PY_GET_ENTRY(pciDeviceID,         "PCI device identifier of the device"),
   PY_GET_ENTRY(tccDriver,           "True if the device is using a TCC driver, False if not"),
   {NULL}  /* Sentinel */
};
#undef PY_GET_ENTRY



/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

PyTypeObject py_oc_device_gpu_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ocean.deviceGPU",          /* tp_name      */
    sizeof(pyOcDevice)          /* tp_basicsize */
};

PyTypeObject *PyOceanDeviceGPU;


/* -------------------------------------------------------------------- */
int pyOcDeviceGPU_Initialize(void)
/* -------------------------------------------------------------------- */
{
   /* Construct the device type object */
   PyOceanDeviceGPU = &py_oc_device_gpu_type;

   PyOceanDeviceGPU -> tp_flags       = Py_TPFLAGS_DEFAULT;
   PyOceanDeviceGPU -> tp_alloc       = PyType_GenericAlloc;
   PyOceanDeviceGPU -> tp_getset      = py_oc_device_getseters;
   PyOceanDeviceGPU -> tp_doc         = "Ocean device GPU";
   PyOceanDeviceGPU -> tp_base        = PyOceanDevice;

   if (PyType_Ready(PyOceanDeviceGPU) < 0) return -1;

   /* Register device type to make sure that OcDevice instances */
   /* of type GPU can be converted to GPU devices in Python.    */
   if (pyOcDevice_RegisterType("GPU", PyOceanDeviceGPU) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcDeviceGPU_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{  PyObject *deviceList;
   PyObject *device;
   int       result;
   int       i, n;

   Py_INCREF(PyOceanDeviceGPU); /* Static object - do not delete */
   PyModule_AddObject(module, "deviceGPU", (PyObject *)PyOceanDeviceGPU);

   /* Create a tuple of GPU devices */
   n = OcDeviceGPUCount();
   deviceList = PyTuple_New(n);
   if (deviceList == NULL) return -1;

   for (i = 0; i < n; i++)
   {  device = PyOceanDevice_New(OcDeviceGPUByIndex(i));
      if (device == NULL) { Py_DECREF(deviceList); return -1; }

      /* Add the device to the list of GPU devices */
      result = PyTuple_SetItem(deviceList, i, device);
      if (result != 0) { Py_DECREF(deviceList); return -1; }
   }

   /* Add the GPU device tuple to the module */
   PyModule_AddObject(module, "gpu", deviceList);

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcDeviceGPU_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Empty */
}



/* ===================================================================== */
/* Internal function definitions                                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_deviceName(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyString_FromString(((OcDeviceGPU *)(self -> device)) -> properties.name);
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_totalGlobalMem(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.totalGlobalMem));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_totalMem(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return pyOcDeviceGPU_totalGlobalMem(self, closure);
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_freeMem(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   if (OcDeviceGPU_updateProperties((OcDeviceGPU *)(self -> device)) != 0) return NULL;

   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.freeGlobalMem));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_sharedMemPerBlock(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.sharedMemPerBlock));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_regsPerBlock(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.regsPerBlock));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_warpSize(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.warpSize));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_memPitch(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.memPitch));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_maxThreadsPerBlock(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.maxThreadsPerBlock));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_maxThreadsDim(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *list, *element;
   int       result, i;

   /* Allocate a new list */
   list = PyList_New(0);
   if (list == NULL) return NULL;

   for (i = 0; i < 3; i++)
   {  /* Create a new element */
      element = PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.maxThreadsDim[i]));
      if (element == NULL) { Py_DECREF(list); return NULL; }

      /* Append the element */
      result = PyList_Append(list, element);
      Py_DECREF(element);
      if (result != 0) { Py_DECREF(list); return NULL; }
   }

   return list;
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_maxGridSize(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *list, *element;
   int       result, i;

   /* Allocate a new list */
   list = PyList_New(0);
   if (list == NULL) return NULL;

   for (i = 0; i < 3; i++)
   {  /* Create a new element */
      element = PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.maxGridSize[i]));
      if (element == NULL) { Py_DECREF(list); return NULL; }

      /* Append the element */
      result = PyList_Append(list, element);
      Py_DECREF(element);
      if (result != 0) { Py_DECREF(list); return NULL; }
   }

   return list;
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_totalConstMem(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.totalConstMem));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_major(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.major));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_minor(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.minor));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_version(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  char buffer[32];

   snprintf(buffer, 32, "%d.%d",
            (((OcDeviceGPU *)(self -> device)) -> properties.major),
            (((OcDeviceGPU *)(self -> device)) -> properties.minor));

   return PyString_FromString(buffer);
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_clockRate(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyFloat_FromDouble((double)(((OcDeviceGPU *)(self -> device)) -> properties.clockRate) / 1000.0);
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_deviceOverlap(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.deviceOverlap));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_multiProcessorCount(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.multiProcessorCount));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_kernelExecTimeoutEnabled(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.kernelExecTimeoutEnabled));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_integrated(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.integrated));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_canMapHostMemory(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.canMapHostMemory));
}


/* -------------------------------------------------------------------- */
const char *pyOcDeviceGPU_getComputeMode(pyOcDevice *self)
/* -------------------------------------------------------------------- */
{
   switch (((OcDeviceGPU *)(self -> device)) -> properties.computeMode)
   {  case cudaComputeModeDefault          : return "Default mode";
      case cudaComputeModeExclusive        : return "Compute-exclusive mode";
      case cudaComputeModeProhibited       : return "Compute-prohibited mode";
      case cudaComputeModeExclusiveProcess : return "Compute-exclusive-process mode";
      default :
         break ;
   }

   return "Unrecognized mode";
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_computeMode(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyString_FromString(pyOcDeviceGPU_getComputeMode(self));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_concurrentKernels(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.concurrentKernels));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_ECCEnabled(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.ECCEnabled));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_pciBusID(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.pciBusID));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_pciDeviceID(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.pciDeviceID));
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcDeviceGPU_tccDriver(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(((OcDeviceGPU *)(self -> device)) -> properties.tccDriver));
}

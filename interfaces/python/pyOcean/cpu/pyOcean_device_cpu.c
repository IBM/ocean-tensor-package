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
#include "pyOcean_device_cpu.h"


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

PyTypeObject py_oc_device_cpu_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ocean.deviceCPU",          /* tp_name      */
    sizeof(pyOcDevice)          /* tp_basicsize */
};

PyTypeObject *PyOceanDeviceCPU;


/* -------------------------------------------------------------------- */
int pyOcDeviceCPU_Initialize(void)
/* -------------------------------------------------------------------- */
{
   /* Construct the device type object */
   PyOceanDeviceCPU = &py_oc_device_cpu_type;

   PyOceanDeviceCPU -> tp_flags = Py_TPFLAGS_DEFAULT;
   PyOceanDeviceCPU -> tp_alloc = PyType_GenericAlloc;
   PyOceanDeviceCPU -> tp_doc   = "Ocean device CPU";
   PyOceanDeviceCPU -> tp_base  = PyOceanDevice;

   if (PyType_Ready(PyOceanDeviceCPU) < 0) return -1;

   /* Register device type to make sure that OcDevice instances */
   /* of type CPU can be converted to CPU devices in Python.    */
   if (pyOcDevice_RegisterType("CPU", PyOceanDeviceCPU) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcDeviceCPU_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
   Py_INCREF(PyOceanDeviceCPU); /* Static object - do not delete */
   PyModule_AddObject(module, "deviceCPU", (PyObject *)PyOceanDeviceCPU);
   PyModule_AddObject(module, "cpu", PyOceanDevice_New(OcCPU));

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcDeviceCPU_Finalize(void)
/* -------------------------------------------------------------------- */
{
}

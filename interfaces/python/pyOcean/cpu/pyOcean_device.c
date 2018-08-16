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

#include "pyOcean_args.h"
#include "pyOcean_stream.h"
#include "pyOcean_device.h"
#include "pyOcean_storage.h"
#include "pyOcean_tensor.h"
#include "pyOcean_core.h"

#include <stdio.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions */
static PyObject *pyOcDevice_call(PyObject *callable_object, PyObject *args, PyObject *kw);
static PyObject *pyOcDevice_richcompare(PyObject *o1, PyObject *o2, int opid);
static PyObject *pyOcDevice_str(pyOcDevice *self);

/* Get and set functions */
static void      pyOcDevice_dealloc         (pyOcDevice *self);
static PyObject *pyOcDevice_gettype         (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getname         (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getindex        (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getinfo         (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getbyteswap     (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getaligned      (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getstream       (pyOcDevice *self, void *closure);
static int       pyOcDevice_setstream       (pyOcDevice *self, PyObject *value, void *closure);
static PyObject *pyOcDevice_getbuffercount  (pyOcDevice *self, void *closure);
static int       pyOcDevice_setbuffercount  (pyOcDevice *self, PyObject *value, void *closure);
static PyObject *pyOcDevice_getmaxbuffersize(pyOcDevice *self, void *closure);
static int       pyOcDevice_setmaxbuffersize(pyOcDevice *self, PyObject *value, void *closure);
static PyObject *pyOcDevice_getrefcount     (pyOcDevice *self, void *closure);
static PyObject *pyOcDevice_getmodules      (pyOcDevice *self, void *closure);

/* Member functions */
static PyObject *pyOcDevice_setDefault    (pyOcDevice *self, PyObject *args);
static PyObject *pyOcDevice_createstream  (pyOcDevice *self, PyObject *args);


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

PyObject *pyoc_device_type_dict = NULL;



/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

struct PyGetSetDef py_oc_device_getseters[] = {
   {"type",               (getter)pyOcDevice_gettype,      NULL, "type",                             NULL},
   {"name",               (getter)pyOcDevice_getname,      NULL, "name",                             NULL},
   {"index",              (getter)pyOcDevice_getindex,     NULL, "device index",                     NULL},
   {"info",               (getter)pyOcDevice_getinfo,      NULL, "device info",                      NULL},
   {"supportsByteswap",   (getter)pyOcDevice_getbyteswap,  NULL, "device supports byteswapping",     NULL},
   {"requiresAlignedData",(getter)pyOcDevice_getaligned,   NULL, "device requires aligned data",     NULL},
   {"defaultStream",      (getter)pyOcDevice_getstream,
                          (setter)pyOcDevice_setstream,          "default stream",                   NULL},
   {"bufferCount",        (getter)pyOcDevice_getbuffercount,
                          (setter)pyOcDevice_setbuffercount,     "number of temporary buffers",      NULL},
   {"maxBufferSize",      (getter)pyOcDevice_getmaxbuffersize,
                          (setter)pyOcDevice_setmaxbuffersize,   "maximum temporary buffer size",    NULL},
   {"refcount",           (getter)pyOcDevice_getrefcount,  NULL, "reference count",                  NULL},
   {"modules",            (getter)pyOcDevice_getmodules,   NULL, "list of available modules",        NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef py_oc_device_methods[] = {
   {"setDefault",   (PyCFunction)pyOcDevice_setDefault,   METH_NOARGS, "Set the default device"},
   {"createStream", (PyCFunction)pyOcDevice_createstream, METH_NOARGS, "Create a new stream"},
   {NULL}  /* Sentinel */
};

PyTypeObject py_oc_device_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.device",             /* tp_name      */
   sizeof(pyOcDevice),         /* tp_basicsize */
};

PyTypeObject *PyOceanDevice;


/* -------------------------------------------------------------------- */
int pyOcDevice_Initialize(void)
/* -------------------------------------------------------------------- */
{
   /* Create a dictionary for device constructors */
   if ((pyoc_device_type_dict = PyDict_New()) == NULL) return -1;

   /* Construct the device type object */
   PyOceanDevice = &py_oc_device_type;

   PyOceanDevice -> tp_flags       = Py_TPFLAGS_DEFAULT;
   PyOceanDevice -> tp_alloc       = PyType_GenericAlloc;
   PyOceanDevice -> tp_dealloc     = (destructor)pyOcDevice_dealloc;
   PyOceanDevice -> tp_call        = (ternaryfunc)pyOcDevice_call;
   PyOceanDevice -> tp_str         = (reprfunc)pyOcDevice_str;
   PyOceanDevice -> tp_repr        = (reprfunc)pyOcDevice_str; /* [sic] */
   PyOceanDevice -> tp_richcompare = pyOcDevice_richcompare;
   PyOceanDevice -> tp_getset      = py_oc_device_getseters;
   PyOceanDevice -> tp_methods     = py_oc_device_methods;
   PyOceanDevice -> tp_doc         = "Ocean device";

   if (PyType_Ready(PyOceanDevice) < 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcDevice_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
    Py_INCREF(PyOceanDevice); /* Static object - do not delete */
    PyModule_AddObject(module, "device", (PyObject *)PyOceanDevice);

    return 0;
}


/* -------------------------------------------------------------------- */
void pyOcDevice_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Free the device constructor dictionary */
   Py_XDECREF(pyoc_device_type_dict);
   pyoc_device_type_dict = NULL;

   /* Free the default device */
   OcDevice_setDefault(NULL);
}



/* ===================================================================== */
/* Device constructor registration                                       */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int pyOcDevice_RegisterType(const char *deviceType, PyTypeObject *type)
/* -------------------------------------------------------------------- */
{
   if (pyoc_device_type_dict == NULL) return -1;

   return PyDict_SetItemString(pyoc_device_type_dict, deviceType, (PyObject *)type);
}


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanDevice_New(OcDevice *device)
/* -------------------------------------------------------------------- */
{  PyTypeObject *type = NULL;
   pyOcDevice   *obj;

   if (device == NULL) return NULL;

   /* Find the Python type object for the given device type */
   if (pyoc_device_type_dict != NULL)
      type = (PyTypeObject *)PyDict_GetItemString(pyoc_device_type_dict, device -> type -> name);

   /* Use the generic type object if none was found */
   if (type == NULL) type = PyOceanDevice;

   /* Construct the object */
   obj = (pyOcDevice *)PyOceanDevice -> tp_alloc(type,0);
   if (obj == NULL) return NULL;

   /* Set the device  */
   obj -> device = OcIncrefDevice(device);

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanDevice_Wrap(OcDevice *device)
/* -------------------------------------------------------------------- */
{  PyTypeObject *type = NULL;
   pyOcDevice   *obj;

   if (device == NULL) return NULL;

   /* Find the Python type object for the given device type */
   if (pyoc_device_type_dict != NULL)
      type = (PyTypeObject *)PyDict_GetItemString(pyoc_device_type_dict, device -> type -> name);

   /* Use the generic type object if none was found */
   if (type == NULL) type = PyOceanDevice;

   /* Construct the object */
   obj = (pyOcDevice *)PyOceanDevice -> tp_alloc(type,0);
   if (obj == NULL)
   {  /* Decrement the reference count */
      OcDecrefDevice(device);
      return NULL;
   }

   /* Set the device (do not increment the reference count) */
   obj -> device = device;

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanDevice_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanDevice);
}



/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_call(PyObject *callable_object, PyObject *args, PyObject *kw)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *obj, *result = NULL;
   OcDevice    *device;
   int          flagInplace = 0;

   /* ================================================ */
   /* Syntax: ocean.device(storage [, inplace=False])  */
   /* Syntax: ocean.device(tensor  [, inplace=False])  */
   /* Syntax: ocean.device(index   [, inplace=False])  */
   /* ================================================ */

   /* Parameter checks */
   if (kw != NULL) OcError(NULL, "Keyword arguments are not supported");

   /* Extract the device */
   device = PYOC_GET_DEVICE(callable_object);

   /* Parse the parameters */
   PyOceanArgs_Init(&param, args, "device.call");
   PyOceanArgs_GetPyObject(&param, &obj, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the internal ensure function */
   result = pyOceanCore_intrnl_ensure(obj, OcDTypeNone, device, flagInplace);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   return result;
}


/* ===================================================================== */
/* Internal function definitions                                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcDevice_dealloc(pyOcDevice *self)
/* -------------------------------------------------------------------- */
{
   if (self -> device) OcDecrefDevice(self -> device);

   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_gettype(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
  return PyString_FromString(self -> device -> type -> name);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getname(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyString_FromString(self -> device -> name);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getrefcount(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> device -> refcount);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getindex(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> device -> index);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getbyteswap(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong(OcDevice_supportsTensorByteswap(self -> device));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getaligned(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong(self -> device -> requiresAlignedData);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getinfo(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *result;
   char     *str;

   /* Format the information */
   if (OcDevice_formatInfo(self -> device, &str, NULL, NULL) != 0) return NULL;

   /* Create the result */
   result = PyString_FromString(str);

   /* Free the string */
   free(str);
   
   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getstream(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanStream_New(OcDevice_getDefaultStream(self -> device));
}


/* -------------------------------------------------------------------- */
static int pyOcDevice_setstream(pyOcDevice *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  OcStream *stream = NULL;

   if (value != Py_None)
   {  if (PyOceanStream_Check(value) == 0)
         OcError(1, "Default stream must be an Ocean stream");
      stream = PYOC_GET_STREAM(value);
   }

   return OcDevice_setDefaultStream(self -> device, stream);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getbuffercount(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  int count;

   if (OcDevice_getBufferCount(self -> device, &count) != 0) return NULL;
   return PyInt_FromLong((long)count);
}


/* -------------------------------------------------------------------- */
static int pyOcDevice_setbuffercount(pyOcDevice *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  long int count;

   if (PyInt_Check(value))
      count = PyInt_AsLong(value);
   else if (PyLong_Check(value))
      count = PyLong_AsLong(value);
   else
      OcError(1, "The buffer count must be an integer");

   return OcDevice_setBufferCount(self -> device, count);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getmaxbuffersize(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcSize size;

   if (OcDevice_getMaxBufferSize(self -> device, &size) != 0) return NULL;
   return PyLong_FromLong((long)size);
}


/* -------------------------------------------------------------------- */
static int pyOcDevice_setmaxbuffersize(pyOcDevice *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  long int size;

   if (PyInt_Check(value))
      size = PyInt_AsLong(value);
   else if (PyLong_Check(value))
      size = PyLong_AsLong(value);
   else
      OcError(1, "The maximum buffer size must be an integer");

   return OcDevice_setMaxBufferSize(self -> device, size);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_getmodules(pyOcDevice *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcDeviceModule *deviceModule;
   PyObject       *list;
   size_t          offset = 0;
   
   /* Create a list */
   list = PyList_New(0);
   if (list == NULL) return NULL;

   /* Add the names of all available modules */
   while ((deviceModule = OcDeviceTypeNextModule(self -> device -> type, &offset)))
   {  if (deviceModule -> available)
      {  if (PyList_Append(list, PyString_FromString(deviceModule -> module -> name)) != 0)
         {  Py_DECREF(list);
            return NULL;
         }
      }
   }

   /* Return the list */
   return list;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_str(pyOcDevice *self)
/* -------------------------------------------------------------------- */
{  char buffer[256];

   snprintf(buffer, 256, "<device '%s'>", self -> device -> name);

   return PyString_FromString(buffer);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  OcDevice *device1, *device2;

   /* Make sure the object is an Ocean device */
   if (!PyOceanDevice_Check(obj))
   {  if (opid == Py_EQ) Py_RETURN_FALSE;
      if (opid == Py_NE) Py_RETURN_TRUE;
   }
   else
   {  /* Get the devices */
      device1 = ((pyOcDevice *)self) -> device;
      device2 = ((pyOcDevice *)obj) -> device;

      /* Evaluate supported comparison operations */
      if (opid == Py_EQ) return PyBool_FromLong(device1 == device2);
      if (opid == Py_NE) return PyBool_FromLong(device1 != device2);
   }

   OcError(NULL, "The given comparison operation is not implemented");
}



/* ===================================================================== */
/* Class functions - member functions                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_setDefault(pyOcDevice *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   OcDevice_setDefault(self -> device);

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDevice_createstream(pyOcDevice *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   /* Create a new stream */
   return PyOceanStream_Wrap(OcDevice_createStream(self -> device));
}

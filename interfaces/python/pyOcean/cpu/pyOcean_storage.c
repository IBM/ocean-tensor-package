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
#include "pyOcean_dtype.h"
#include "pyOcean_device.h"
#include "pyOcean_stream.h"
#include "pyOcean_scalar.h"
#include "pyOcean_storage.h"
#include "pyOcean_tensor.h"
#include "pyOcean_opaque.h"
#include "pyOcean_module_core.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions */
static void      pyOcStorage_dealloc    (pyOcStorage *self);
static PyObject *pyOcStorage_new        (PyTypeObject *subtype, PyObject *args, PyObject *kwargs);
static PyObject *pyOcStorage_richcompare(PyObject *o1, PyObject *o2, int opid);
static PyObject *pyOcStorage_str        (pyOcStorage *self);

/* Mapping protocol functions */
static PyObject *pyOcStorage_mp_subscript    (pyOcStorage *self, PyObject *args);
static int       pyOcStorage_mp_ass_subscript(pyOcStorage *self, PyObject *index, PyObject *value);

/* Get and set functions */
static PyObject *pyOcStorage_getdevice     (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getstream     (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getobject     (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getptr        (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getdtype      (pyOcStorage *self, void *closure);
static int       pyOcStorage_setdtype      (pyOcStorage *self, PyObject *value, void *closure);
static PyObject *pyOcStorage_getsize       (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getcapacity   (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getnelem      (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getelemsize   (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getbyteswapped(pyOcStorage *self, void *closure);
static int       pyOcStorage_setbyteswapped(pyOcStorage *self, PyObject *value, void *closure);
static PyObject *pyOcStorage_getreadonly   (pyOcStorage *self, void *closure);
static int       pyOcStorage_setreadonly   (pyOcStorage *self, PyObject *value, void *closure);
static PyObject *pyOcStorage_getowner      (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getrefcount   (pyOcStorage *self, void *closure);
static PyObject *pyOcStorage_getfooter     (pyOcStorage *self, void *closure);

/* Member functions */
static PyObject *pyOcStorage_copy     (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_clone    (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_sync     (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_void     (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_byteswap (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_zero     (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_asTensor (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_asPython (pyOcStorage *self, PyObject *args);
static PyObject *pyOcStorage_isAligned(pyOcStorage *self, PyObject *args);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

struct PyGetSetDef py_oc_storage_getseters[] = {
   {"device",     (getter)pyOcStorage_getdevice,      NULL, "device type",                 NULL},
   {"stream",     (getter)pyOcStorage_getstream,      NULL, "stream",                      NULL},
   {"obj",        (getter)pyOcStorage_getobject,      NULL, "pointer to OcStorage object", NULL},
   {"ptr",        (getter)pyOcStorage_getptr,         NULL, "pointer to the data",         NULL},
   {"dtype",      (getter)pyOcStorage_getdtype,
                  (setter)pyOcStorage_setdtype,             "data type",                   NULL},
   {"size",       (getter)pyOcStorage_getsize,        NULL, "size",                        NULL},
   {"capacity",   (getter)pyOcStorage_getcapacity,    NULL, "capacity",                    NULL},
   {"nelem",      (getter)pyOcStorage_getnelem,       NULL, "number of elements",          NULL},
   {"elemsize",   (getter)pyOcStorage_getelemsize,    NULL, "element size",                NULL},
   {"byteswapped",(getter)pyOcStorage_getbyteswapped,
                  (setter)pyOcStorage_setbyteswapped,       "byteswapped",                 NULL},
   {"readonly",   (getter)pyOcStorage_getreadonly,
                  (setter)pyOcStorage_setreadonly,          "read-only",                   NULL},
   {"owner",      (getter)pyOcStorage_getowner,       NULL, "owner",                       NULL},
   {"refcount",   (getter)pyOcStorage_getrefcount,    NULL, "reference count",             NULL},
   {"footer",     (getter)pyOcStorage_getfooter,      NULL, "storage footer string",       NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef py_oc_storage_methods[] = {
   {"copy",      (PyCFunction)pyOcStorage_copy,      METH_VARARGS, "Copy the storage data"},
   {"clone",     (PyCFunction)pyOcStorage_clone,     METH_VARARGS, "Clone the storage data"},
   {"sync",      (PyCFunction)pyOcStorage_sync,      METH_NOARGS,  "Synchronizes the storage data"},
   {"dealloc",   (PyCFunction)pyOcStorage_void,      METH_NOARGS,  "Deallocates the storage data"},
   {"byteswap",  (PyCFunction)pyOcStorage_byteswap,  METH_NOARGS,  "Byteswap the storage data"},
   {"zero",      (PyCFunction)pyOcStorage_zero,      METH_NOARGS,  "Set the storage data to zero"},
   {"asTensor",  (PyCFunction)pyOcStorage_asTensor,  METH_NOARGS,  "Return a canonical tensor for the storage"},
   {"asPython",  (PyCFunction)pyOcStorage_asPython,  METH_NOARGS,  "Return a Python list with storage entries"},
   {"isAligned", (PyCFunction)pyOcStorage_isAligned, METH_NOARGS,  "Checks if the storage is memory aligned"},
   {NULL}  /* Sentinel */
};

static PyMappingMethods py_oc_storage_as_mapping = {0};

PyTypeObject py_oc_storage_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.storage",            /* tp_name      */
   sizeof(pyOcStorage),        /* tp_basicsize */
};

PyTypeObject     *PyOceanStorage;
static OcStorage *py_oc_void_storage = NULL;
static int        py_oc_storage_magic;


/* -------------------------------------------------------------------- */
int pyOcStorage_Initialize(void)
/* -------------------------------------------------------------------- */
{  PyMappingMethods *mp = &py_oc_storage_as_mapping;

   /* Construct the storage type object */
   PyOceanStorage = &py_oc_storage_type;

   #if PY_MAJOR_VERSION >= 3
   PyOceanStorage -> tp_flags       = Py_TPFLAGS_DEFAULT;
   #else
   PyOceanStorage -> tp_flags       = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES;
   #endif
   PyOceanStorage -> tp_alloc       = PyType_GenericAlloc;
   PyOceanStorage -> tp_dealloc     = (destructor)pyOcStorage_dealloc;
   PyOceanStorage -> tp_new         = (newfunc)pyOcStorage_new;
   PyOceanStorage -> tp_str         = (reprfunc)pyOcStorage_str;
   PyOceanStorage -> tp_repr        = (reprfunc)pyOcStorage_str;
   PyOceanStorage -> tp_richcompare = pyOcStorage_richcompare;
   PyOceanStorage -> tp_getset      = py_oc_storage_getseters;
   PyOceanStorage -> tp_methods     = py_oc_storage_methods;
   PyOceanStorage -> tp_as_mapping  = mp;
   PyOceanStorage -> tp_doc         = "Ocean storage";

   /* Mapping functions */
   mp -> mp_subscript     = (binaryfunc)pyOcStorage_mp_subscript;
   mp -> mp_ass_subscript = (objobjargproc)pyOcStorage_mp_ass_subscript;

   /* Finalize the type */
   if (PyType_Ready(PyOceanStorage) < 0) return -1;

   /* Get the magic number for storage types */
   py_oc_storage_magic = pyOcOpaque_CreateMagic();

   /* Create a read-only placeholder storage to deallocate objects */
   py_oc_void_storage = OcStorage_create(0, OcDTypeInt8, OcCPU);
   if (py_oc_void_storage == NULL) return -1;
   OcStorage_setReadOnly(py_oc_void_storage, 1);

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcStorage_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
   Py_INCREF(PyOceanStorage); /* Static object - do not delete */
   PyModule_AddObject(module, "storage", (PyObject *)PyOceanStorage);

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcStorage_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Decrement the void storage */
   OcXDecrefStorage(py_oc_void_storage);
}


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanStorage_New(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  pyOcStorage  *obj;

   if (storage == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcStorage *)PyOceanStorage -> tp_alloc(PyOceanStorage,0);
   if (obj == NULL) return NULL;

   /* Set the storage */
   obj -> storage = OcIncrefStorage(storage);

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanStorage_Wrap(OcStorage *storage)
/* -------------------------------------------------------------------- */
{  pyOcStorage  *obj;

   if (storage == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcStorage *)PyOceanStorage -> tp_alloc(PyOceanStorage,0);
   if (obj == NULL)
   {  /* Decrement the reference count */
      OcDecrefStorage(storage);
      return NULL; 
   }

   /* Set the storage (do not increment the reference count) */
   obj -> storage = storage;

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanStorage_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanStorage);
}


/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcStorage_dealloc(pyOcStorage *self)
/* -------------------------------------------------------------------- */
{
   if (self -> storage) OcDecrefStorage(self -> storage);

   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{  OcDevice   *device;
   OcStream   *stream;
   OcDType     dtype;
   PyOceanArgs param;
   long int    size;
   OcStorage  *storage;

   /* ======================================================= */
   /* Syntax: Storage(size [, dtype] [, device] [, stream])   */
   /* ======================================================= */

   /* Make sure there are no key-word arguments */
   if (kwargs != NULL) OcError(NULL, "The storage constructor does not take keyword arguments");

   /* Parse the parameters */
   PyOceanArgs_Init(&param, args, "ocean.storage");
   PyOceanArgs_GetScalarInt(&param, &size, 1);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   PyOceanArgs_GetOcStream(&param, &stream, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;
   
   /* Additional checks */
   if (size < 0) OcError(NULL, "The storage size cannot be negative");

   /* Create the Ocean storage object */
   if (stream == NULL)
        storage = OcStorage_create((size_t)size, dtype, device);
   else storage = OcStorage_createWithStream((size_t)size, dtype, stream);

   /* Return the result */
   return PyOceanStorage_Wrap(storage);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_str(pyOcStorage *self)
/* -------------------------------------------------------------------- */
{  PyObject *obj = NULL;
   char     *str = NULL, *footer = NULL;

   /* Format the footer */
   if ((OcStorage_formatFooter(self -> storage, &footer, "<",">")) != 0) return NULL;

   /* Format the content string */
   if (OcStorage_format(self -> storage, &str, NULL, footer) != 0) goto finish;

   /* Create the result string */
   obj = PyString_FromString(str);

finish :
   /* Free the formatted string */
   if (footer) free(footer);
   if (str) free(str);

   return obj;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  void *ptr1, *ptr2;
   int   result;

   /* Make sure the object is an Ocean storage */
   if (!PyOceanStorage_Check(self) || !PyOceanStorage_Check(obj))
   {  ptr1 = (void *)self;
      ptr2 = (void *)obj;
   }
   else
   {  /* Get the storage objects */
      ptr1 = (void *)(((pyOcStorage *)self) -> storage);
      ptr2 = (void *)(((pyOcStorage *)obj)  -> storage);
   }

   /* Evaluate supported comparison operations */
   switch(opid)
   {  case Py_LT : result = (ptr1 <  ptr2); break;
      case Py_LE : result = (ptr1 <= ptr2); break;
      case Py_EQ : result = (ptr1 == ptr2); break;
      case Py_NE : result = (ptr1 != ptr2); break;
      case Py_GE : result = (ptr1 >= ptr2); break;
      case Py_GT : result = (ptr1 >  ptr2); break;
      default    : OcError(NULL, "The given comparison operation is not implemented");
   }

   /* Return the result */
   return PyBool_FromLong(result);
}


/* ===================================================================== */
/* Class functions - mapping protocol functions                          */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_mp_subscript(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcStorage *storage = self -> storage;
   OcTensor  *tensor;
   PyObject  *obj1, *obj2;

   /* Create the tensor */
   tensor = OcTensor_createFromStorage(storage, -1, NULL, NULL, 0, storage -> dtype);
   if ((obj1 = PyOceanTensor_Wrap(tensor)) == NULL) return NULL;

   /* Call the tensor mp_subscript function */
   obj2 = PyOceanTensor -> tp_as_mapping -> mp_subscript(obj1, args);

   /* Delete the intermediate tensor */
   Py_DECREF(obj1);

   /* Return the result */
   return obj2;
}


/* -------------------------------------------------------------------- */
static int pyOcStorage_mp_ass_subscript(pyOcStorage *self, PyObject *args, PyObject *value)
/* -------------------------------------------------------------------- */
{  OcStorage *storage = self -> storage;
   OcTensor  *tensor;
   PyObject  *obj;
   int        result;

   /* Create the tensor */
   tensor = OcTensor_createFromStorage(storage, -1, NULL, NULL, 0, storage -> dtype);
   if ((obj = PyOceanTensor_Wrap(tensor)) == NULL) return -1;

   /* Call the tensor mp_ass_subscript function */
   result = PyOceanTensor -> tp_as_mapping -> mp_ass_subscript(obj, args, value);

   /* Delete the intermediate tensor */
   Py_DECREF(obj);

   /* Return the result */
   return result;
}



/* ===================================================================== */
/* Class functions - get and set functions                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getdevice(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
  return PyOceanDevice_New(OcStorage_device(self -> storage));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getstream(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
  return PyOceanStream_New(OcStorage_stream(self -> storage));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getobject(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanOpaque_New((void *)(self -> storage), py_oc_storage_magic);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getptr(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{  return PyLong_FromVoidPtr((void *)(self -> storage -> data));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getdtype(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   if (OcStorage_isRaw(self -> storage)) Py_RETURN_NONE;
   
   return PyOceanDType_New(self -> storage -> dtype);
}


/* -------------------------------------------------------------------- */
static int pyOcStorage_setdtype(pyOcStorage *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{
   /* Check and process the parameter */
   if (value == Py_None)
   {  OcStorage_setDTypeRaw(self -> storage);
   }
   else if (PyOceanDType_Check(value))
   {  OcStorage_setDType(self -> storage, PYOC_GET_DTYPE(value));
   }
   else
   {  OcError(-1, "Invalid value for storage.dtype");
   }

   return 0;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getsize(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> storage -> size);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getcapacity(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> storage -> capacity);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getnelem(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> storage -> nelem);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getelemsize(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> storage -> elemsize);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getbyteswapped(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)OcStorage_isByteswapped(self -> storage));
}


/* -------------------------------------------------------------------- */
static int pyOcStorage_setbyteswapped(pyOcStorage *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  OcDevice *device = OcStorage_device(self -> storage);
   int flag;

   if (PyBool_Check(value) == 0)
      OcError(1, "Byte-swap value must be Boolean");

   /* Determine the byteswap flag */
   flag = (value == Py_True) ? 1 : 0;
   if (flag && !OcDevice_supportsTensorByteswap(device))
      OcError(1, "Byteswapped storage data is not supported on device %s", device -> name);

   OcStorage_setByteswapped(self -> storage, flag);

   return 0;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getreadonly(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcStorage_isReadOnly(self -> storage)));
}


/* -------------------------------------------------------------------- */
static int pyOcStorage_setreadonly(pyOcStorage *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{
   if (PyBool_Check(value) == 0)
      OcError(1, "Read-only value must be Boolean");

   OcStorage_setReadOnly(self -> storage, (value == Py_True) ? 1 : 0);

   return 0;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getrefcount(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long int)(self -> storage -> refcount));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getowner(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)OcStorage_isOwner(self -> storage));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_getfooter(pyOcStorage *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *obj = NULL;
   char     *footer = NULL;

   /* Format the footer */
   if ((OcStorage_formatFooter(self -> storage, &footer, "<",">")) != 0) return NULL;

   /* Create the result string */
   obj = PyString_FromString(footer);

   /* Free the formatted string */
   if (footer) free(footer);

   return obj;
}



/* ===================================================================== */
/* Class functions - member functions                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_copy(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcStorage   *storage;
   int          result;

   /* ============================== */
   /* Syntax: storage.copy(storage)  */
   /* ============================== */

   PyOceanArgs_Init(&param, args, "storage.copy");
   PyOceanArgs_GetOcStorage(&param, &storage, 1);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Copy the data */
   result = OcStorage_copy(storage, self -> storage);

   /* Success */
   if (result != 0) return NULL; else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_clone(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcDevice    *device = NULL;
   OcStorage   *result;

   /* ================================ */
   /* Syntax: storage.clone([device])  */
   /* ================================ */

   PyOceanArgs_Init(&param, args, "storage.clone");
   if (PyOceanArgs_GetOcDevice(&param, &device, 0) == 0)
   {  device = self -> storage -> stream -> device;
   }
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Clone the storage */
   result = OcStorage_cloneTo(self -> storage, device);
   
   /* Return as a Python object */
   return PyOceanStorage_Wrap(result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_sync(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   if (OcStorage_synchronize(self -> storage) != 0) return NULL;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_void(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   /* ================================ */
   /* Syntax: storage.dealloc()        */
   /* ================================ */
   OcIncrefStorage(py_oc_void_storage);
   OcDecrefStorage(self -> storage);
   self -> storage = py_oc_void_storage;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_byteswap(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   if (OcStorage_byteswap(self -> storage) != 0) return NULL;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_zero(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   if (OcStorage_zero(self -> storage) != 0) return NULL;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_asTensor(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcTensor  *tensor;
   OcStorage *storage = self -> storage;

   /* Create the tensor */
   tensor = OcTensor_createFromStorage(storage, -1, NULL, NULL, 0, storage -> dtype);

   /* Return the result */
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_asPython(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyObject       *list, *obj;
   Py_int_ssize_t  nelem;
   Py_int_ssize_t  i;
   
   OcScalar        scalar;
   OcStorage      *storage;
   OcIndex         elemsize;
   char           *ptr;
   int             byteswapped;
   int             success = 1;

   /* Cast storage to CPU */
   storage = OcStorage_castDevice(self -> storage, OcCPU);
   if (storage == NULL) return NULL;
   elemsize    = storage -> elemsize;
   nelem       = storage -> nelem;
   byteswapped = OcStorage_hasHostByteOrder(storage) ? 0 : 1;
   
   /* Synchronize the storage */
   OcStorage_synchronize(storage);

   /* Initialize the data pointer */
   ptr = (char *)(storage -> data);
   scalar.dtype = storage -> dtype;
      
   /* Create the list */
   list = PyList_New(nelem);
   if (list == NULL) { success = 0; goto final; }

   /* Boolean scalars */
   if (storage -> dtype == OcDTypeBool)
   {  for (i = 0; i < nelem; i++)
      {  obj = (*((OcBool *)ptr) == 0) ? Py_False : Py_True;
         Py_INCREF(obj);
         PyList_SetItem(list, i, obj);
         ptr += elemsize;
      }
      goto final;
   }

   /* Integer scalars */
   if (!OcDType_isFloat(storage -> dtype))
   {  for (i = 0; i < nelem; i++)
      {  OcScalar_importData(&scalar, ptr, byteswapped);
         obj = PyInt_FromLong(OcScalar_asInt64(&scalar));
         if (obj == NULL) { success = 0; goto final; }
         PyList_SetItem(list, i, obj);
         ptr += elemsize;
      }
      goto final;
   }

   /* Float scalars */
   if (!OcDType_isComplex(storage -> dtype))
   {  for (i = 0; i < nelem; i++)
      {  OcScalar_importData(&scalar, ptr, byteswapped);
         obj = PyFloat_FromDouble(OcScalar_asDouble(&scalar));
         if (obj == NULL) { success = 0; goto final; }
         PyList_SetItem(list, i, obj);
         ptr += elemsize;
      }
      goto final;
   }

   /* Complex scalars */
   {  OcCDouble value;

      for (i = 0; i < nelem; i++)
      {  OcScalar_importData(&scalar, ptr, byteswapped);
         value = OcScalar_asCDouble(&scalar);
         obj = PyComplex_FromDoubles(value.real, value.imag);
         if (obj == NULL) { success = 0; goto final; }
         PyList_SetItem(list, i, obj);
         ptr += elemsize;
      }
      goto final;
   }

final: ;
   OcDecrefStorage(storage);
   if (!success) { Py_XDECREF(list); return NULL; }
   return list;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcStorage_isAligned(pyOcStorage *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcStorage_isAligned(self -> storage)));
}

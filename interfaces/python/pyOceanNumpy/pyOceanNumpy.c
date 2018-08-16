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

#include <Python.h>
#include "pyOcean.h"
#include "ocean.h"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Module finalization */
#if PY_MAJOR_VERSION < 3
static void   pyOceanNumpy_dealloc(PyObject *obj);
static void (*pyOceanNumpy_original_dealloc)(PyObject *) = NULL;
#endif

static void   pyOceanNumpy_free(void *data);

/* Import functions */
static int pyOceanNumpy_isScalar       (PyObject *obj);
static int pyOceanNumpy_isTensor       (PyObject *obj);
static int pyOceanNumpy_isScalarTensor (PyObject *obj);
static int pyOceanNumpy_getScalar      (PyObject *obj, OcScalar *scalar);
static int pyOceanNumpy_getScalarTensor(PyObject *obj, OcTensor **tensor);
static int pyOceanNumpy_getTensor      (PyObject *obj, OcTensor **tensor);
static int pyOceanNumpy_getScalarType  (PyObject *obj, OcDType *dtype, int verbose);
static int pyOceanNumpy_getTensorLayout(PyObject *obj, int *ndims, OcSize *size, OcIndex *strides,
                                        OcDType *dtype, OcDevice **device, int verbose);
static int pyOceanNumpy_exportTensor   (OcTensor *tensor, PyObject **obj, int deepcopy);

/* Conversion type */
static pyOceanConvert py_ocean_numpy_format =
   {  pyOceanNumpy_isScalar,
      pyOceanNumpy_isTensor,
      pyOceanNumpy_isScalarTensor,
      pyOceanNumpy_getScalar,
      pyOceanNumpy_getScalarTensor,
      pyOceanNumpy_getTensor,
      pyOceanNumpy_getScalarType,
      pyOceanNumpy_getTensorLayout,
      pyOceanNumpy_exportTensor,
      "numpy",
      NULL
   };


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

static PyMethodDef py_ocean_numpy_methods[] = {
   {NULL}  /* Sentinel */
};

static PyObject *module = NULL;

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef py_ocean_numpy_module = {
        PyModuleDef_HEAD_INIT,
        PyMODULE_NAME_STR("pyOceanNumpy"), /* Module name               (m_base) */
        NULL,                              /* Module docstring           (m_doc) */
        -1,                                /* Module state size         (m_size) */
        py_ocean_numpy_methods,            /* Module methods         (m_methods) */
        NULL,                              /* Reload func      (m_reload, < 3.5) */
                                           /* Slot definitions (m_slots, >= 3.5) */
        NULL,                              /* Traverse func         (m_traverse) */
        NULL,                              /* Clear func               (m_clear) */
        pyOceanNumpy_free                  /* Free function             (m_free) */
};
#endif


/* -------------------------------------------------------------------- */
PyMODULE_INIT(pyOceanNumpy)                  /* Module name must match! */
/* -------------------------------------------------------------------- */
{  PyObject *py_oc_module_ocean = NULL;
   int status = 0;

   /* Load the pyOcean module */
   py_oc_module_ocean = PyImport_ImportModule("pyOcean_cpu");
   if (py_oc_module_ocean == NULL)
   {  PyErr_SetString(PyExc_RuntimeError, "Unable to load Ocean core module (pyOcean_cpu)");
      status = -1; goto final;
   }

   /* Load Numpy */
   import_array();

   /* Create the module */
   #if PY_MAJOR_VERSION >= 3
      module = PyModule_Create(&py_ocean_numpy_module);
      if (module == NULL) { status = -1; goto final; }
   #else
      module = Py_InitModule3(PyMODULE_NAME_STR("pyOceanNumpy"), py_ocean_numpy_methods,
                              "pyOceanNumpy provides conversion routines between Numpy and Ocean");
      if (module == NULL) { status = -1; goto final; }
   #endif

   /* Finalization */
   #if PY_MAJOR_VERSION < 3
   pyOceanNumpy_original_dealloc = Py_TYPE(module) -> tp_dealloc;
   Py_TYPE(module) -> tp_dealloc = pyOceanNumpy_dealloc;
   #endif

   /* Register the import functions */
   status = pyOcean_registerConverter(&py_ocean_numpy_format);

final : ;   
   /* Free the local reference to pyOcean */
   Py_XDECREF(py_oc_module_ocean); py_oc_module_ocean = NULL;

   /* Check for errors */
   if (status != 0)
   {  Py_XDECREF(module);
      PyMODULE_INIT_ERROR;
   }

   /* Return the module */
   #if PY_MAJOR_VERSION >= 3
   return module;
   #endif
}


/* ===================================================================== */
/* Finalization                                                          */
/* ===================================================================== */

#if PY_MAJOR_VERSION < 3
/* -------------------------------------------------------------------- */
static void pyOceanNumpy_dealloc(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (pyOceanNumpy_original_dealloc) pyOceanNumpy_original_dealloc(obj);

   if (obj == module) pyOceanNumpy_free((void *)obj);
}
#endif


/* -------------------------------------------------------------------- */
static void pyOceanNumpy_free(void *data)
/* -------------------------------------------------------------------- */
{
   /* ------------------------------------------------------ */
   /* Note: the original code kept a global variable for the */
   /*       pyOcean module, but this is not needed as Python */
   /*       maintains an internal reference to modules that  */
   /*       have been loaded.                                */
   /* ------------------------------------------------------ */
   /* Decref the Ocean module */
   /*
   if (py_oc_module_ocean)
   {  Py_DECREF(py_oc_module_ocean); py_oc_module_ocean = NULL;
   }
   */
}


/* ===================================================================== */
/* Import and export functions                                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int  pyOceanNumpy_getImportDType(int np_dtype, OcDType *dtype)
/* -------------------------------------------------------------------- */
{   switch (np_dtype)
   {  case NPY_BOOL       : *dtype = OcDTypeBool;    return 0;
      case NPY_INT8       : *dtype = OcDTypeInt8;    return 0;
      case NPY_INT16      : *dtype = OcDTypeInt16;   return 0;
      case NPY_INT32      : *dtype = OcDTypeInt32;   return 0;
      case NPY_INT64      : *dtype = OcDTypeInt64;   return 0;
      case NPY_UINT8      : *dtype = OcDTypeUInt8;   return 0;
      case NPY_UINT16     : *dtype = OcDTypeUInt16;  return 0;
      case NPY_UINT32     : *dtype = OcDTypeUInt32;  return 0;
      case NPY_UINT64     : *dtype = OcDTypeUInt64;  return 0;
      case NPY_FLOAT16    : *dtype = OcDTypeHalf;    return 0;
      case NPY_FLOAT32    : *dtype = OcDTypeFloat;   return 0;
      case NPY_FLOAT64    : *dtype = OcDTypeDouble;  return 0;
      case NPY_COMPLEX64  : *dtype = OcDTypeCFloat;  return 0;
      case NPY_COMPLEX128 : *dtype = OcDTypeCDouble; return 0;
      default : break;
   }

   return -1;
}


/* -------------------------------------------------------------------- */
static int pyOceanNumpy_getExportDType(OcDType dtype, int *np_dtype)
/* -------------------------------------------------------------------- */
{
   switch (dtype)
   {
      case OcDTypeBool    : *np_dtype = NPY_BOOL;       return 0;
      case OcDTypeUInt8   : *np_dtype = NPY_UINT8;      return 0;
      case OcDTypeUInt16  : *np_dtype = NPY_UINT16;     return 0;
      case OcDTypeUInt32  : *np_dtype = NPY_UINT32;     return 0;
      case OcDTypeUInt64  : *np_dtype = NPY_UINT64;     return 0;
      case OcDTypeInt8    : *np_dtype = NPY_INT8;       return 0;
      case OcDTypeInt16   : *np_dtype = NPY_INT16;      return 0;
      case OcDTypeInt32   : *np_dtype = NPY_INT32;      return 0;
      case OcDTypeInt64   : *np_dtype = NPY_INT64;      return 0;
      case OcDTypeHalf    : *np_dtype = NPY_FLOAT16;    return 0;
      case OcDTypeFloat   : *np_dtype = NPY_FLOAT32;    return 0;
      case OcDTypeDouble  : *np_dtype = NPY_FLOAT64;    return 0;
      case OcDTypeCFloat  : *np_dtype = NPY_COMPLEX64;  return 0;
      case OcDTypeCDouble : *np_dtype = NPY_COMPLEX128; return 0;
      default : break;
   }

   return -1;
}


/* -------------------------------------------------------------------- */
static int pyOceanNumpy_getScalarDType(PyObject *obj, OcDType *dtype)
/* -------------------------------------------------------------------- */
{  PyArray_Descr* desc;
   int np_type;

   /* Check the object type and extract the type number */
   if (!PyArray_IsScalar(obj, Generic)) return 0;
   if (!PyArray_DescrConverter(obj, &desc)) return -1;
   np_type = desc -> type_num;
   Py_DECREF(desc);

   /* Set the numpy type */
   if (pyOceanNumpy_getImportDType(np_type, dtype) == 0) return 1;

   return -1;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_isScalar(PyObject *obj)
/* -------------------------------------------------------------------- */
{  OcDType dtype;

   return (pyOceanNumpy_getScalarDType(obj, &dtype) != 1) ? 0 : 1;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_isTensor(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   return (PyArray_Check(obj)) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_isScalarTensor(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (!PyArray_Check(obj)) return 0;
   return (PyArray_Size(obj) == 1) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_getScalar(PyObject *obj, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcDType dtype;
   int     result;

   /* Get the data type */
   result =  pyOceanNumpy_getScalarDType(obj, &dtype);
   if (result != 1) return result;

   /* Create the scalar */
   scalar -> dtype = dtype;

   /* Set the scalar value */
   PyArray_ScalarAsCtype(obj, (void *)&(scalar -> value));

   return 1;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_getScalarTensor(PyObject *obj, OcTensor **tensor)
/* -------------------------------------------------------------------- */
{
   if (!PyArray_Check(obj)) return 0;
   if (PyArray_Size(obj) != 1) return 0;

   return pyOceanNumpy_getTensor(obj, tensor);
}


/* -------------------------------------------------------------------- */
void pyOceanNumpy_freeObject(void *object, void *data)
/* -------------------------------------------------------------------- */
{
   (void)data;
   Py_DECREF((PyObject *)object);
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_getTensor(PyObject *obj, OcTensor **tensor)
/* -------------------------------------------------------------------- */
{  PyArrayObject *np_array;
   PyArray_Descr *np_descr;
   char          *np_data;

   OcStorage     *storage;
   OcSize         size[OC_TENSOR_MAX_DIMS];
   OcIndex        strides[OC_TENSOR_MAX_DIMS];
   OcSize         offset, extent;
   OcDevice      *device;
   OcDType        dtype;
   int            byteswapped;
   int            readonly;
   int            ndims;
   int            result;

   /* Deal with scalars */
   if (!PyArray_Check(obj)) return 0;

   /* Get tensor information */
   result = pyOceanNumpy_getTensorLayout(obj, &ndims, size, strides, &dtype, &device, 1);
   if (result != 1) return result;
   
   /* Get additional fields */
   np_array = (PyArrayObject *)obj;
   np_descr = PyArray_DESCR(np_array);

   /* Determine the byte order */
   switch (np_descr -> byteorder)
   {  case '=' : byteswapped = 0; break;
      case '|' : byteswapped = 0; break;
      case '<' : byteswapped = (OcCPU -> endianness == 0) ? 0 : 1; break;
      case '>' : byteswapped = (OcCPU -> endianness == 1) ? 0 : 1; break;
      default:
         OcError(-1, "Invalid byte-order symbol '%c'", np_descr -> byteorder);
   }

   /* Get the data */
   np_data = PyArray_BYTES(np_array);

   /* Compute the size and offset */
   result = OcShape_extent(ndims, size, strides, OcDType_size(dtype), &offset, &extent);
   if (result != 0) return result;

   /* Create storage from object */
   storage = OcStorage_createFromObject(extent, OcDTypeUInt8, OcCPU,
                                  (void *)np_array, (void *)((char *)np_data - offset),
                                  byteswapped, pyOceanNumpy_freeObject, NULL);
   if (storage == NULL) return -1;

   /* Set the read-only flag */
   readonly = (PyArray_FLAGS(np_array) & NPY_ARRAY_WRITEABLE) ? 0 : 1;
   OcStorage_setReadOnly(storage, readonly);

   /* Increment the array reference counter */
   Py_INCREF((PyObject *)np_array);

   /* Create tensor from storage */
   *tensor = OcTensor_createFromStorage(storage, ndims, size, strides, offset, dtype);
   if (*tensor == NULL) { OcDecrefStorage(storage); return -1; }

   /* Update the storage data type */
   if (OcTensor_isAligned(*tensor))
        OcStorage_setDType(storage, dtype);
   else OcStorage_setDTypeRaw(storage);

   /* Decrement the storage reference count */
   OcDecrefStorage(storage);

   /* Success */
   return 1;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_getScalarType(PyObject *obj, OcDType *dtype, int verbose)
/* -------------------------------------------------------------------- */
{  int result;

   result = pyOceanNumpy_getScalarDType(obj, dtype);
   if (result == 0) return 0;
   if (result != 1)
   {  if (!verbose) return -1;
      OcError(-1, "Numpy data type is not supported in ocean");
   }

   return 1;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_getTensorLayout(PyObject *obj, int *ndims, OcSize *size, OcIndex *strides,
                                 OcDType *dtype, OcDevice **device, int verbose)
/* -------------------------------------------------------------------- */
{  PyArrayObject *np_array;
   npy_intp      *np_dims;
   npy_intp      *np_strides;
   int            np_ndims;

   OcSize         _size[OC_TENSOR_MAX_DIMS];
   OcIndex        _strides[OC_TENSOR_MAX_DIMS];
   OcDType        _dtype;
   OcSize         offset, extent;
   int            i, elemsize;

   /* Check the object type */
   if (!PyArray_Check(obj)) return 0;

   /* Get the array information */
   np_array   = (PyArrayObject *)obj;
   np_ndims   = PyArray_NDIM(np_array);
   np_dims    = PyArray_DIMS(np_array);
   np_strides = PyArray_STRIDES(np_array);

   /* Check the number of dimensions */
   if (np_ndims > OC_TENSOR_MAX_DIMS)
   {  if (!verbose) return -1;
      OcError(-1, "Array dimensions exceed the maximum supported by Ocean");
   }

   /* Set size and stride arrays if needed */
   if (size == NULL) size = _size;
   if (strides == NULL) strides = _strides;

   /* Copy the size and stride information */
   for (i = 0; i < np_ndims; i++)
   {  
      if ((size[i] = (OcSize)np_dims[i]) > 1)
           strides[i] = (OcIndex)np_strides[i];
      else strides[i] = 0;
   }

   /* Convert the data type */
   if (dtype == NULL) dtype = &_dtype;
   if (pyOceanNumpy_getImportDType(PyArray_TYPE(np_array), dtype) != 0)
   {  if (!verbose) return -1;
      OcError(-1, "Numpy array data type is not supported by Ocean");
   }

   /* Compute the size and offset */
   elemsize = OcDType_size(*dtype);
   if (OcShape_extent(np_ndims, size, strides, elemsize, &offset, &extent) != 0) return -1;

   /* Set the device type */
   if (device) *device = OcCPU;

   /* Set the number of dimensions */
   if (ndims) *ndims = np_ndims;

   /* Success */
   return 1;
}


/* -------------------------------------------------------------------- */
int pyOceanNumpy_exportTensor(OcTensor *tensor, PyObject **obj, int deepcopy)
/* -------------------------------------------------------------------- */
{  PyArray_Descr *np_descr;
   PyObject      *np_array;
   npy_intp       np_dims[OC_TENSOR_MAX_DIMS];
   npy_intp       np_strides[OC_TENSOR_MAX_DIMS];
   int            np_dtype, np_flags;
   int            readonly;

   PyObject      *storageObj;
   OcTensor      *t;
   int            i;

   /* Get the numpy type */
   if (pyOceanNumpy_getExportDType(tensor -> dtype, &np_dtype) != 0)
      OcError(-1, "Ocean tensor data type is not supported by Numpy");
      
   /* Copy the read-only flag if needed */
   if (OcTensor_isReadOnly(tensor))
        readonly = ((deepcopy) || (OcTensor_device(tensor) != OcCPU)) ? 0 : 1;
   else readonly = 0;

   /* Make sure that the tensor resides on the CPU */
   if (deepcopy)
        t = OcTensor_cloneTo(tensor, OcCPU);
   else OcTensor_ensureDevice(&tensor, OcCPU, &t);
   if (t == NULL) return -1; else tensor = t;

   /* Synchronize the tensor */
   OcTensor_synchronize(tensor);

   /* Create a Python object for the storage */
   storageObj = PyOceanStorage_New(tensor -> storage);
   if (storageObj == NULL) goto error;

   /* Copy the size and strides */
   for (i = 0; i < tensor -> ndims; i++)
   {  np_dims[i]    = tensor -> size[i];
      np_strides[i] = tensor -> strides[i];
   }

   /* Create a new numpy array */
   np_flags = (readonly) ? 0 : NPY_ARRAY_WRITEABLE;
   np_array = PyArray_New(&PyArray_Type, tensor -> ndims, np_dims, np_dtype, np_strides,
                          (void *)(OcTensor_data(tensor)), tensor -> elemsize, np_flags, NULL);
   if (np_array == NULL)
   {  OcErrorMessage("Unable to create the numpy array");
      goto error;
   }

   /* Set the base object */
   if (PyArray_SetBaseObject((PyArrayObject *)np_array, storageObj) == 0)
   {  storageObj = NULL; /* PyArray_SetBaseObject steals the reference */
   }

   /* Set the byte order */
   if (OcTensor_isByteswapped(tensor))
   {   np_descr = PyArray_DESCR((PyArrayObject *)np_array);
       np_descr -> byteorder = (OcCPU -> endianness == 0) ? '>' : '<';
   }

   /* Successful tensor export */
   *obj = np_array;
   return 0;

error: ; 
   Py_XDECREF(storageObj);
   OcDecrefTensor(tensor);
   return -1;
}

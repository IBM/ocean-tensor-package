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

#include "pyOcean.h"
#include "pyOcean_index.h"
#include "pyOcean_core.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions - pyOcTensorIndex */
static void      pyOcTensorIndex_dealloc      (pyOcTensorIndex *self);
static PyObject *pyOcTensorIndex_new          (PyTypeObject *subtype, PyObject *args, PyObject *kwargs);
static PyObject *pyOcTensorIndex_richcompare  (PyObject *o1, PyObject *o2, int opid);
static PyObject *pyOcTensorIndex_str          (pyOcTensorIndex *self);
static PyObject *pyOcTensorIndex_create       (PyObject *args, PyObject *kwargs);

/* Standard functions - pyOcIndexCreate */
static PyObject *pyOcIndexCreate_call         (PyObject *callable_object, PyObject *args, PyObject *kwargs);
static PyObject *pyOcIndexCreate_str          (pyOcIndexCreate *self);

/* Mapping protocol functions */
static PyObject *pyOcIndexCreate_mp_subscript (pyOcIndexCreate *self, PyObject *args);

/* Get and set functions */
static PyObject *pyOcTensorIndex_getnelem     (pyOcTensorIndex *self, void *closure);
static PyObject *pyOcTensorIndex_getinputsize (pyOcTensorIndex *self, void *closure);
static PyObject *pyOcTensorIndex_getoutputsize(pyOcTensorIndex *self, void *closure);
static PyObject *pyOcTensorIndex_getstrides   (pyOcTensorIndex *self, void *closure);

/* Member functions */
static PyObject *pyOcTensorIndex_clone        (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_clear        (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_append       (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_bind         (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_setDevice    (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_isScalar     (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_isView       (pyOcTensorIndex *self, PyObject *args);
static PyObject *pyOcTensorIndex_isBound      (pyOcTensorIndex *self, PyObject *args);



/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

struct PyGetSetDef py_oc_tensor_index_getseters[] = {
   {"nelem",      (getter)pyOcTensorIndex_getnelem,       NULL, "Number of index elements", NULL},
   {"inputSize",  (getter)pyOcTensorIndex_getinputsize,   NULL, "Input size",               NULL},
   {"outputSize", (getter)pyOcTensorIndex_getoutputsize,  NULL, "Output size",              NULL},
   {"strides",    (getter)pyOcTensorIndex_getstrides,     NULL, "Input strides",            NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef py_oc_tensor_index_methods[] = {
   {"clone",     (PyCFunction)pyOcTensorIndex_clone,     METH_NOARGS,  "Clone tensor index"},
   {"clear",     (PyCFunction)pyOcTensorIndex_clear,     METH_NOARGS,  "Clear tensor index"},
   {"append",    (PyCFunction)pyOcTensorIndex_append,    METH_VARARGS, "Append index elements to the tensor index"},
   {"bind",      (PyCFunction)pyOcTensorIndex_bind,      METH_VARARGS, "Bind the tensor index size (and strides)"},
   {"setDevice", (PyCFunction)pyOcTensorIndex_setDevice, METH_VARARGS, "Set the device for all tensors in the index"},
   {"isScalar",  (PyCFunction)pyOcTensorIndex_isScalar,  METH_NOARGS,  "Check if applying the index gives a scalar"},
   {"isView",    (PyCFunction)pyOcTensorIndex_isView,    METH_NOARGS,  "Check if applying the index gives a view"},
   {"isBound",   (PyCFunction)pyOcTensorIndex_isBound,   METH_NOARGS,  "Check if the index size/strides are set"},
   {NULL}  /* Sentinel */
};

static PyMappingMethods py_oc_index_create_as_mapping = {0};

PyTypeObject py_oc_tensor_index_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.tensorIndex",        /* tp_name      */
   sizeof(pyOcTensorIndex),    /* tp_basicsize */
};

PyTypeObject py_oc_index_create_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.indexCreate",        /* tp_name      */
   sizeof(pyOcIndexCreate),    /* tp_basicsize */
};

PyTypeObject *PyOceanTensorIndex;
PyTypeObject *PyOceanIndexCreate;
static int    py_oc_tensor_index_magic;


/* -------------------------------------------------------------------- */
int pyOcIndex_Initialize(void)
/* -------------------------------------------------------------------- */
{  PyMappingMethods *mp = &py_oc_index_create_as_mapping;

   /* ---------------------------------------- */
   /*  Construct the index create type object  */
   /* ---------------------------------------- */
   PyOceanIndexCreate = &py_oc_index_create_type;
   PyOceanIndexCreate -> tp_flags      = Py_TPFLAGS_DEFAULT;
   PyOceanIndexCreate -> tp_alloc      = PyType_GenericAlloc;
   PyOceanIndexCreate -> tp_call       = (ternaryfunc)pyOcIndexCreate_call;
   PyOceanIndexCreate -> tp_str        = (reprfunc)pyOcIndexCreate_str;
   PyOceanIndexCreate -> tp_repr       = (reprfunc)pyOcIndexCreate_str; /* [sic] */
   PyOceanIndexCreate -> tp_doc        = "Ocean tensor index constructor";
   PyOceanIndexCreate -> tp_as_mapping = mp;

   /* Mapping functions */
   mp -> mp_subscript     = (binaryfunc)pyOcIndexCreate_mp_subscript;
   mp -> mp_ass_subscript = (objobjargproc)0;

   /* ---------------------------------------- */
   /*  Construct the tensor index type object  */
   /* ---------------------------------------- */
   PyOceanTensorIndex = &py_oc_tensor_index_type;
   PyOceanTensorIndex -> tp_flags       = Py_TPFLAGS_DEFAULT;
   PyOceanTensorIndex -> tp_alloc       = PyType_GenericAlloc;
   PyOceanTensorIndex -> tp_dealloc     = (destructor)pyOcTensorIndex_dealloc;
   PyOceanTensorIndex -> tp_new         = (newfunc)pyOcTensorIndex_new;
   PyOceanTensorIndex -> tp_str         = (reprfunc)pyOcTensorIndex_str;
   PyOceanTensorIndex -> tp_repr        = (reprfunc)pyOcTensorIndex_str; /* [sic] */
   PyOceanTensorIndex -> tp_richcompare = pyOcTensorIndex_richcompare;
   PyOceanTensorIndex -> tp_getset      = py_oc_tensor_index_getseters;
   PyOceanTensorIndex -> tp_methods     = py_oc_tensor_index_methods;
   PyOceanTensorIndex -> tp_doc         = "Ocean tensor index";

   /* Finalize the type */
   if (PyType_Ready(PyOceanTensorIndex) < 0) return -1;
   if (PyType_Ready(PyOceanIndexCreate) < 0) return -1;

   /* Get the magic number for tensor index types */
   py_oc_tensor_index_magic = pyOcOpaque_CreateMagic();

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcIndex_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{  PyObject *index;

   Py_INCREF(PyOceanTensorIndex); /* Static object - do not delete */
   Py_INCREF(PyOceanIndexCreate); /* Static object - do not delete */

   /* Create the index constructor */
   index = PyOceanIndexCreate -> tp_alloc(PyOceanIndexCreate,0);
   if (index == NULL) return -1;
   PyModule_AddObject(module, "index", index);

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcIndex_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Empty */
}



/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanTensorIndex_New(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  pyOcTensorIndex *obj;

   if (index == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcTensorIndex *)PyOceanTensorIndex -> tp_alloc(PyOceanTensorIndex,0);
   if (obj == NULL) return NULL;

   /* Set the index */
   obj -> index = OcIncrefTensorIndex(index);

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanTensorIndex_Wrap(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  pyOcTensorIndex *obj;

   if (index == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcTensorIndex *)PyOceanTensorIndex -> tp_alloc(PyOceanTensorIndex,0);
   if (obj == NULL)
   {  /* Decrement the reference count */
      OcDecrefTensorIndex(index);
      return NULL;
    }

   /* Set the index (do not increment the reference count) */
   obj -> index = index;

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanTensorIndex_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanTensorIndex);
}



/* ===================================================================== */
/* Class functions - get and set methods                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_getnelem(pyOcTensorIndex *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> index -> n);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_getinputsize(pyOcTensorIndex *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *result = NULL;
   OcSize   *size = NULL;
   int       i, ndims;

   /* Determine the number of dimensions */
   ndims = OcTensorIndex_getNumInputDims(self -> index);
   if (ndims < 0) Py_RETURN_NONE;

   /* Allocate memory for the size */
   size = (OcSize *)OcMalloc(sizeof(OcSize) * (ndims == 0 ? 1 : ndims));
   if (size == NULL) OcError(NULL, "Error allocating memory for the size array");

   /* Get the size */
   if (OcTensorIndex_getInputDims(self -> index, size, &ndims) < 0) goto final;
   if (ndims < 0) { Py_INCREF(Py_None); result = Py_None; goto final; }

   /* Create a tupe */
   if ((result = PyTuple_New(ndims)) == NULL)
   {  OcErrorMessage("Error creating size tuple");
      goto final;
   }

   /* Add the dimensions */
   for (i = 0; i < ndims; i++)
   {  if (PyTuple_SetItem(result, i, PyInt_FromLong((long)size[i])) != 0)
      {  OcErrorMessage("Error creating size tuple");
         Py_DECREF(result); result = NULL;
         goto final;
      }
   }

final : ;
   if (size) OcFree(size);
   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_getoutputsize(pyOcTensorIndex *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *result = NULL;
   OcSize   *size = NULL;
   int       i, ndims;

   /* Determine the number of dimensions */
   ndims = OcTensorIndex_getNumOutputDims(self -> index);
   if (ndims < 0) Py_RETURN_NONE;

   /* Allocate memory for the size */
   size = (OcSize *)OcMalloc(sizeof(OcSize) * (ndims == 0 ? 1 : ndims));
   if (size == NULL) OcError(NULL, "Error allocating memory for the size array");

   /* Get the size */
   if (OcTensorIndex_getOutputDims(self -> index, size, &ndims) < 0) goto final;
   if (ndims < 0) { Py_INCREF(Py_None); result = Py_None; goto final; }

   /* Create a tupe */
   if ((result = PyTuple_New(ndims)) == NULL)
   {  OcErrorMessage("Error creating size tuple");
      goto final;
   }

   /* Add the dimensions */
   for (i = 0; i < ndims; i++)
   {  if (PyTuple_SetItem(result, i, PyInt_FromLong((long)size[i])) != 0)
      {  OcErrorMessage("Error creating size tuple");
         Py_DECREF(result); result = NULL;
         goto final;
      }
   }

final : ;
   if (size) OcFree(size);
   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_getstrides(pyOcTensorIndex *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *result = NULL;
   OcIndex  *strides = NULL;
   int       i, ndims;

   /* Determine the number of dimensions */
   ndims = OcTensorIndex_getNumInputDims(self -> index);
   if (ndims < 0) Py_RETURN_NONE;

   /* Allocate memory for the strides */
   strides = (OcIndex *)OcMalloc(sizeof(OcIndex) * (ndims == 0 ? 1 : ndims));
   if (strides == NULL) OcError(NULL, "Error allocating memory for the stride array");

   /* Get the strides */
   if (OcTensorIndex_getInputStrides(self -> index, strides, &ndims) < 0) goto final;
   if (ndims < 0) { Py_INCREF(Py_None); result = Py_None; goto final; }

   /* Create a tupe */
   if ((result = PyTuple_New(ndims)) == NULL)
   {  OcErrorMessage("Error creating stride tuple");
      goto final;
   }

   /* Add the dimensions */
   for (i = 0; i < ndims; i++)
   {  if (PyTuple_SetItem(result, i, PyInt_FromLong((long)strides[i])) != 0)
      {  OcErrorMessage("Error creating stride tuple");
         Py_DECREF(result); result = NULL;
         goto final;
      }
   }

final : ;
   if (strides) OcFree(strides);
   return result;
}



/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcTensorIndex_dealloc(pyOcTensorIndex *self)
/* -------------------------------------------------------------------- */
{
   if (self -> index) OcDecrefTensorIndex(self -> index);

   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static OcTensorIndex *pyOcTensorIndex_parse(PyObject *args)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index = NULL;

   /* Parse the indices */
   if (pyOcean_convertIndices(args, &index) != 0) return NULL;

   /* Detach the tensors */
   if (OcTensorIndex_detach(&index, NULL) != 0) return NULL;

   return index;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_create(PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{  OcTensorIndex  *index;

   /* Make sure there are no keyword arguments */
   if (kwargs != NULL) OcError(NULL, "The tensor-index constructor does not take keyword arguments");

   /* Parse the argument */
   index = pyOcTensorIndex_parse(args);

   /* Return the result */
   return PyOceanTensorIndex_Wrap(index);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{
   return pyOcTensorIndex_create(args, kwargs);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_str(pyOcTensorIndex *self)
/* -------------------------------------------------------------------- */
{  PyObject *obj = NULL;
   char     *str = NULL;

   /* Format the content string */
   if (OcTensorIndex_format(self -> index, &str, "<", ">") != 0) goto final;
   
   /* Create the result string */
   obj = PyString_FromString(str);

final : ;
   /* Free the formatted string */
   if (str) free(str);

   return obj;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index1, *index2;

   /* Make sure the object is an Ocean index */
   if (!PyOceanTensorIndex_Check(obj))
   {  if (opid == Py_EQ) Py_RETURN_FALSE;
      if (opid == Py_NE) Py_RETURN_TRUE;
   }
   else
   {  /* Get the indexes */
      index1 = ((pyOcTensorIndex *)self) -> index;
      index2 = ((pyOcTensorIndex *)obj) -> index;

      /* Evaluate supported comparison operations */
      if (opid == Py_EQ) return PyBool_FromLong(index1 == index2);
      if (opid == Py_NE) return PyBool_FromLong(index1 != index2);
   }

   OcError(NULL, "The given comparison operation is not implemented");
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcIndexCreate_str(pyOcIndexCreate *self)
/* -------------------------------------------------------------------- */
{
   return PyString_FromString("<index-constructor>");
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcIndexCreate_call(PyObject *callable_object, PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{
   return pyOcTensorIndex_create(args, kwargs);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcIndexCreate_mp_subscript(pyOcIndexCreate *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return pyOcTensorIndex_create(args, NULL);
}



/* ===================================================================== */
/* Class functions - member functions                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_clone(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index;

   index = OcTensorIndex_shallowCopy(self -> index);
   return PyOceanTensorIndex_Wrap(index);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_clear(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int result;

   result = OcTensorIndex_clear(self -> index);
   if (result != 0) return NULL; else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_append(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index;
   int result;

   /* ============================ */
   /* Syntax: index.append(index)  */
   /* ============================ */

   /* Parse the argument */
   index = pyOcTensorIndex_parse(args);
   if (index == NULL) return NULL;

   /* Append the index */
   result = OcTensorIndex_addIndex(self -> index, index);

   /* Free the index parameter */
   OcDecrefTensorIndex(index);

   /* Return the result */
   if (result != 0) return NULL; else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_bind(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   PyObject      *result = NULL;
   OcSize        *size = NULL;
   OcIndex       *strides = NULL;
   OcTensorIndex *index = NULL;
   int            nSize = -1, nStrides = -1;
   int            flagInplace = 0;
   int            status;

   /* ===================================================== */
   /* Syntax: index.bind(size [,strides] [,inplace=False])  */
   /* ===================================================== */

   PyOceanArgs_Init(&param, args, "index.bind");
   PyOceanArgs_GetTensorSize(&param, &size, &nSize, 1);
   if (PyOceanArgs_GetBool(&param, &flagInplace, 0) != 1)
   {  PyOceanArgs_GetTensorStrides(&param, &strides, &nStrides, 0);
      PyOceanArgs_GetBool(&param, &flagInplace, 0);
   }
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Basic parameter checks */
   if ((strides != NULL) && (nStrides != nSize))
   {  OcErrorMessage("Mismatch in stride and size dimensions");
      goto final;
   }

   /* Bind the index */
   status = OcTensorIndex_bind(&(self -> index), 1, nSize, size, strides, &index);
   if (status != 0) goto final;

   /* Replace existing index or create a new one */
   if (flagInplace)
   {  OcDecrefTensorIndex(PYOC_GET_TENSOR_INDEX(self));
      PYOC_GET_TENSOR_INDEX(self) = index;
      Py_INCREF(Py_None); result = Py_None;
   }
   else
   {  result = PyOceanTensorIndex_Wrap(index);
   }

final : ;
   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_setDevice(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   OcDevice      *device;
   int            flagInplace = 0;

   /* ===================================================== */
   /* Syntax: index.setDevice(device [,inplace=False])  */
   /* ===================================================== */

   PyOceanArgs_Init(&param, args, "index.setDevice");
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Create a new index object */
   return pyOceanCore_intrnl_ensure((PyObject *)self, OcDTypeNone, device, flagInplace);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_isScalar(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int result = OcTensorIndex_isScalar(self -> index);
   if (result < 0)
        return NULL;
   else return PyBool_FromLong((long int)result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_isView(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int result = OcTensorIndex_isView(self -> index);
   if (result < 0)
        return NULL;
   else return PyBool_FromLong((long int)result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensorIndex_isBound(pyOcTensorIndex *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyObject *flag1, *flag2, *tuple;
   int result1, result2;

   /* Get the bound flags */
   result1 = OcTensorIndex_isBound(self -> index, 0);
   result2 = OcTensorIndex_isBound(self -> index, 1);
   if ((result1 < 0) || (result2 < 0)) return NULL;

   /* Create the Python objects */
   flag1 = PyBool_FromLong((long int)result1);
   flag2 = PyBool_FromLong((long int)result2);
   tuple = PyTuple_New(2);
   if ((flag1 == NULL) || (flag2 == NULL) || (tuple == NULL))
   {  Py_XDECREF(flag1);
      Py_XDECREF(flag2);
      Py_XDECREF(tuple);
      return NULL;
   }

   /* Finalize the tuple (transfer ownership of booleans) */
   PyTuple_SET_ITEM(tuple, 0, flag1);
   PyTuple_SET_ITEM(tuple, 1, flag2);
   return tuple;
}

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
#include "pyOcean_scalar.h"
#include "pyOcean_storage.h"
#include "pyOcean_tensor.h"
#include "pyOcean_core.h"
#include "pyOcean_type_macros.h"

#include "ocean.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

static PyObject *pyOcDType_call       (PyObject *callable_object, PyObject *args, PyObject *kw);

static PyObject *pyOcDType_getname    (pyOcDType *self, void *closure);
static PyObject *pyOcDType_getsize    (pyOcDType *self, void *closure);
static PyObject *pyOcDType_getnbits   (pyOcDType *self, void *closure);
static PyObject *pyOcDType_getmin     (pyOcDType *self, void *closure);
static PyObject *pyOcDType_getmax     (pyOcDType *self, void *closure);
static PyObject *pyOcDType_geteps     (pyOcDType *self, void *closure);
static PyObject *pyOcDType_getbasetype(pyOcDType *self, void *closure);
static PyObject *pyOcDType_isnumber   (pyOcDType *self, void *closure);
static PyObject *pyOcDType_issigned   (pyOcDType *self, void *closure);
static PyObject *pyOcDType_isfloat    (pyOcDType *self, void *closure);
static PyObject *pyOcDType_iscomplex  (pyOcDType *self, void *closure);
static PyObject *pyOcDType_str        (pyOcDType *self);

static PyObject *pyOcDType_richcompare(PyObject *o1, PyObject *o2, int opid);

/* Member functions */
static PyObject *pyOcDType_setDefault (pyOcDType *self, PyObject *args);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

/* Define the getset function table */
#define PYOC_DTYPE_GETTERS(field, function, description) {field, (getter)pyOcDType_##function, NULL, description, NULL}
struct PyGetSetDef py_oc_dtype_getseters[] = {
   PYOC_DTYPE_GETTERS("name",      getname,     "name of the data type"                                  ),
   PYOC_DTYPE_GETTERS("size",      getsize,     "size in bytes"                                          ),
   PYOC_DTYPE_GETTERS("nbits",     getnbits,    "size in bits"                                           ),
   PYOC_DTYPE_GETTERS("min",       getmin,      "minimum value"                                          ),
   PYOC_DTYPE_GETTERS("max",       getmax,      "maximum value"                                          ),
   PYOC_DTYPE_GETTERS("eps",       geteps,      "epsilon"                                                ),
   PYOC_DTYPE_GETTERS("basetype",  getbasetype, "base type"                                              ),
   PYOC_DTYPE_GETTERS("isnumber",  isnumber,    "flag indicating whether the data type is a number"      ),
   PYOC_DTYPE_GETTERS("issigned",  issigned,    "flag indicating whether the data type is signed"        ),
   PYOC_DTYPE_GETTERS("isfloat",   isfloat,     "flag indicating whether the data type is floating point"),
   PYOC_DTYPE_GETTERS("iscomplex", iscomplex,   "flag indicating whether the data type is complex"       ),
   {NULL}  /* Sentinel */
};
#undef PYOC_DTYPE_GETTERS

static PyMethodDef py_oc_dtype_methods[] = {
   {"setDefault", (PyCFunction)pyOcDType_setDefault, METH_NOARGS, "Set the default data type"},
   {NULL} /* Sentinel */
};


/* Define the data type object */
PyTypeObject py_oc_dtype_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.dtype",              /* tp_name      */
   sizeof(pyOcDType),          /* tp_basicsize */
};

/* Define a pointer to data-type type */
PyTypeObject *PyOceanDType;


/* -------------------------------------------------------------------- */
int pyOcDType_Initialize(void)
/* -------------------------------------------------------------------- */
{
   /* Set the data-type pointer */
   PyOceanDType = &py_oc_dtype_type;

   /* Initialize all type objects */
   PyOceanDType -> tp_flags       = Py_TPFLAGS_DEFAULT;
   PyOceanDType -> tp_alloc       = PyType_GenericAlloc;
   PyOceanDType -> tp_call        = (ternaryfunc)pyOcDType_call;
   PyOceanDType -> tp_str         = (reprfunc)pyOcDType_str;
   PyOceanDType -> tp_repr        = (reprfunc)pyOcDType_str; /* [sic] */
   PyOceanDType -> tp_richcompare = pyOcDType_richcompare;
   PyOceanDType -> tp_getset      = py_oc_dtype_getseters;
   PyOceanDType -> tp_methods     = py_oc_dtype_methods;
   PyOceanDType -> tp_doc         = "Ocean dtype";

   if (PyType_Ready(PyOceanDType) < 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcDType_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
   Py_INCREF(PyOceanDType);
   PyModule_AddObject(module, "dtype", (PyObject *)PyOceanDType);

   #define PYOC_DECLARE_TYPE(dtype, name, id, ctype) \
      PyModule_AddObject(module, #name, PyOceanDType_New(id));

   PYOC_DECLARE_ALL_TYPES
   #undef PYOC_DECLARE_TYPE

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcDType_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Empty */
}



/* ===================================================================== */
/* Global function definitions                                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanDType_New(OcDType dtype)
/* -------------------------------------------------------------------- */
{  pyOcDType *obj;

   /* Ensure that the None data type is never exposed */
   if (dtype == OcDTypeNone)
      OcError(NULL, "Internal error: attempt to return the None data type");

   /* Create the new object */
   obj = (pyOcDType *)PyOceanDType -> tp_alloc(PyOceanDType,0);
   if (obj == NULL) return NULL;

   /* Set the data type */
   obj -> dtype = dtype;

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanDType_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanDType);
}


/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_call(PyObject *callable_object, PyObject *args, PyObject *kw)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *obj, *result = NULL;
   OcDType      dtype;
   int          flagInplace = 0;

   /* ============================================== */
   /* Syntax: ocean.dtype(storage [,inplace=False])  */
   /* Syntax: ocean.dtype(scalar  [,inplace=False])  */
   /* Syntax: ocean.dtype(tensor  [,inplace=False])  */
   /* ============================================== */

   /* Parameter checks */
   if (kw != NULL) OcError(NULL, "Keyword arguments are not supported");

   /* Extract the device */
   dtype = PYOC_GET_DTYPE(callable_object);

   /* Parse the parameters */
   PyOceanArgs_Init(&param, args, "dtype.call");
   PyOceanArgs_GetPyObject(&param, &obj, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the internal ensure function */
   result = pyOceanCore_intrnl_ensure(obj, dtype, NULL, flagInplace);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   return result;
}



/* ===================================================================== */
/* Internal function definitions                                         */
/* ===================================================================== */


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_getname(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyString_FromString(OcDType_name(self -> dtype));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_getsize(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long)(OcDType_size(self -> dtype)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_getmin(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   OcScalar_setMin(&scalar, self -> dtype);
   return PyOceanScalar_New(&scalar);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_getmax(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   OcScalar_setMax(&scalar, self -> dtype);
   return PyOceanScalar_New(&scalar);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_geteps(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   OcScalar_setEps(&scalar, self -> dtype);
   return PyOceanScalar_New(&scalar);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_getnbits(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long)(OcDType_nbits(self -> dtype)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_getbasetype(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanDType_New(OcDType_getBaseType(self -> dtype));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_isnumber(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(OcDType_isNumber(self -> dtype)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_issigned(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(OcDType_isSigned(self -> dtype)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_isfloat(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(OcDType_isFloat(self -> dtype)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_iscomplex(pyOcDType *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(OcDType_isComplex(self -> dtype)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_str(pyOcDType *self)
/* -------------------------------------------------------------------- */
{  char buffer[128];

   snprintf(buffer, 128, "<dtype '%s'>", OcDType_name(self -> dtype));

   return PyString_FromString(buffer);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  OcDType dtype1, dtype2;

   /* Make sure the object is an Ocean device */
   if (!PyOceanDType_Check(obj))
   {  if (opid == Py_EQ) Py_RETURN_FALSE;
      if (opid == Py_NE) Py_RETURN_TRUE;
   }
   else
   {  /* Get the data types */
      dtype1 = ((pyOcDType *)self) -> dtype;
      dtype2 = ((pyOcDType *)obj ) -> dtype;

      /* Evaluate supported comparison operations */
      if (opid == Py_EQ) return PyBool_FromLong(dtype1 == dtype2);
      if (opid == Py_NE) return PyBool_FromLong(dtype1 != dtype2);
      if (opid == Py_LT) return PyBool_FromLong(dtype1 <  dtype2);
      if (opid == Py_LE) return PyBool_FromLong(dtype1 <= dtype2);
      if (opid == Py_GT) return PyBool_FromLong(dtype1 >  dtype2);
      if (opid == Py_GE) return PyBool_FromLong(dtype1 >= dtype2);
   }

   OcError(NULL,  "The given comparison operation is not implemented");
}


/* ===================================================================== */
/* Member functions                                                      */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcDType_setDefault(pyOcDType *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   OcDType_setDefault(self -> dtype);

   Py_RETURN_NONE;
}

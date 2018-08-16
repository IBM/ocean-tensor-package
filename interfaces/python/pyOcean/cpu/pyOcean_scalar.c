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
#include "pyOcean_convert.h"
#include "pyOcean_scalar.h"
#include "pyOcean_tensor.h"
#include "pyOcean_core.h"

#include <math.h>
#include <limits.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions */
static void      pyOcScalar_dealloc    (pyOcScalar *self);
static PyObject *pyOcScalar_new        (PyTypeObject *subtype, PyObject *args, PyObject *kwargs);
static PyObject *pyOcScalar_richcompare(PyObject *obj1, PyObject *obj2, int opid);
static PyObject *pyOcScalar_str        (pyOcScalar *self);

/* Numeric functions */
static PyObject *pyOcScalar_nb_int         (pyOcScalar *self);
#if PY_MAJOR_VERSION < 3
static PyObject *pyOcScalar_nb_long        (pyOcScalar *self);
#endif
static PyObject *pyOcScalar_nb_float       (pyOcScalar *self);
static PyObject *pyOcScalar_complex        (pyOcScalar *self, PyObject *args);
static int       pyOcScalar_nb_nonzero     (pyOcScalar *self);
static PyObject *pyOcScalar_nb_positive    (pyOcScalar *self);
static PyObject *pyOcScalar_nb_negative    (pyOcScalar *self);
static PyObject *pyOcScalar_nb_invert      (pyOcScalar *self);

static PyObject *pyOcScalar_nb_add                 (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_subtract            (PyObject *self, PyObject *other);
static PyObject *pyOcScalar_nb_multiply            (PyObject *self, PyObject *other);
static PyObject *pyOcScalar_nb_remainder           (PyObject *self, PyObject *other);
static PyObject *pyOcScalar_nb_and                 (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_or                  (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_xor                 (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_lshift              (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_rshift              (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_add         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_subtract    (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_multiply    (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_remainder   (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_and         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_or          (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_xor         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_lshift      (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_rshift      (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_power               (PyObject *obj1, PyObject *args, PyObject *keywords);
static PyObject *pyOcScalar_nb_inplace_power       (PyObject *obj1, PyObject *args, PyObject *keyworks);

#ifdef PY_VERSION_2_2
static PyObject *pyOcScalar_nb_true_divide         (PyObject *self, PyObject *other);
static PyObject *pyOcScalar_nb_floor_divide        (PyObject *self, PyObject *other);
static PyObject *pyOcScalar_nb_inplace_true_divide (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcScalar_nb_inplace_floor_divide(PyObject *obj1, PyObject *obj2);
#endif

/* Get and set functions */
static PyObject *pyOcScalar_getdtype      (pyOcScalar *self, void *closure);
static PyObject *pyOcScalar_getreal       (pyOcScalar *self, void *closure);
static PyObject *pyOcScalar_getimag       (pyOcScalar *self, void *closure);
static int       pyOcScalar_setreal       (pyOcScalar *self, PyObject *value, void *closure);
static int       pyOcScalar_setimag       (pyOcScalar *self, PyObject *value, void *closure);
static PyObject *pyOcScalar_getelemsize   (pyOcScalar *self, void *closure);

/* Additional functions */
static PyObject *pyOcScalar_asPython   (pyOcScalar *self, PyObject *args);
static PyObject *pyOcScalar_asTensor   (pyOcScalar *self, PyObject *args);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

struct PyGetSetDef py_oc_scalar_getseters[] = {
   {"dtype",    (getter)pyOcScalar_getdtype,    NULL,                       "data type",      NULL},
   {"real",     (getter)pyOcScalar_getreal,     (setter)pyOcScalar_setreal, "real part",      NULL},
   {"imag",     (getter)pyOcScalar_getimag,     (setter)pyOcScalar_setimag, "imaginary part", NULL},
   {"elemsize", (getter)pyOcScalar_getelemsize, NULL,                       "element size",   NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef py_oc_scalar_methods[] = {
   {"__complex__", (PyCFunction)pyOcScalar_complex,  METH_NOARGS,  "Convert to a Python complex number"},
   {"asPython",    (PyCFunction)pyOcScalar_asPython, METH_NOARGS,  "Convert to a Python scalar"},
   {"asTensor",    (PyCFunction)pyOcScalar_asTensor, METH_VARARGS, "Convert to a tensor"},
   {NULL}  /* Sentinel */
};

static PyNumberMethods py_oc_scalar_as_number = {0};

PyTypeObject py_oc_scalar_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.scalar",            /* tp_name      */
   sizeof(pyOcScalar),        /* tp_basicsize */
};

PyTypeObject *PyOceanScalar;


/* -------------------------------------------------------------------- */
int pyOcScalar_Initialize(void)
/* -------------------------------------------------------------------- */
{  PyNumberMethods *nb = &py_oc_scalar_as_number;

   /* Construct the scalar type object */
   PyOceanScalar = &py_oc_scalar_type;

   #if PY_MAJOR_VERSION >= 3
   PyOceanScalar -> tp_flags       = Py_TPFLAGS_DEFAULT;
   #else
   PyOceanScalar -> tp_flags       = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES;
   #endif

   PyOceanScalar -> tp_alloc       = PyType_GenericAlloc;
   PyOceanScalar -> tp_dealloc     = (destructor)pyOcScalar_dealloc;
   PyOceanScalar -> tp_new         = (newfunc)pyOcScalar_new;
   PyOceanScalar -> tp_str         = (reprfunc)pyOcScalar_str;
   PyOceanScalar -> tp_repr        = (reprfunc)pyOcScalar_str;
   PyOceanScalar -> tp_richcompare = pyOcScalar_richcompare;
   PyOceanScalar -> tp_getset      = py_oc_scalar_getseters;
   PyOceanScalar -> tp_methods     = py_oc_scalar_methods;
   PyOceanScalar -> tp_as_number   = nb;
   PyOceanScalar -> tp_doc         = "Ocean scalar";

   /* Number functions */
   nb -> nb_int                  = (unaryfunc)pyOcScalar_nb_int;
   nb -> nb_float                = (unaryfunc)pyOcScalar_nb_float;
   nb -> nb_positive             = (unaryfunc)pyOcScalar_nb_positive;
   nb -> nb_negative             = (unaryfunc)pyOcScalar_nb_negative;
   nb -> nb_invert               = (unaryfunc)pyOcScalar_nb_invert;

   nb -> nb_add                  = (binaryfunc)pyOcScalar_nb_add;
   nb -> nb_subtract             = (binaryfunc)pyOcScalar_nb_subtract;
   nb -> nb_multiply             = (binaryfunc)pyOcScalar_nb_multiply;
   nb -> nb_remainder            = (binaryfunc)pyOcScalar_nb_remainder;
   nb -> nb_and                  = (binaryfunc)pyOcScalar_nb_and;
   nb -> nb_or                   = (binaryfunc)pyOcScalar_nb_or;
   nb -> nb_xor                  = (binaryfunc)pyOcScalar_nb_xor;
   nb -> nb_lshift               = (binaryfunc)pyOcScalar_nb_lshift;
   nb -> nb_rshift               = (binaryfunc)pyOcScalar_nb_rshift;

   nb -> nb_inplace_add          = (binaryfunc)pyOcScalar_nb_inplace_add;
   nb -> nb_inplace_subtract     = (binaryfunc)pyOcScalar_nb_inplace_subtract;
   nb -> nb_inplace_multiply     = (binaryfunc)pyOcScalar_nb_inplace_multiply;
   nb -> nb_inplace_remainder    = (binaryfunc)pyOcScalar_nb_inplace_remainder;
   nb -> nb_inplace_and          = (binaryfunc)pyOcScalar_nb_inplace_and;
   nb -> nb_inplace_or           = (binaryfunc)pyOcScalar_nb_inplace_or;
   nb -> nb_inplace_xor          = (binaryfunc)pyOcScalar_nb_inplace_xor;
   nb -> nb_inplace_lshift       = (binaryfunc)pyOcScalar_nb_inplace_lshift;
   nb -> nb_inplace_rshift       = (binaryfunc)pyOcScalar_nb_inplace_rshift;

   nb -> nb_power                = (ternaryfunc)pyOcScalar_nb_power;
   nb -> nb_inplace_power        = (ternaryfunc)pyOcScalar_nb_inplace_power;

   #ifdef PY_VERSION_2_2
   nb -> nb_true_divide          = (binaryfunc)pyOcScalar_nb_true_divide;
   nb -> nb_floor_divide         = (binaryfunc)pyOcScalar_nb_floor_divide;
   nb -> nb_inplace_true_divide  = (binaryfunc)pyOcScalar_nb_inplace_true_divide;
   nb -> nb_inplace_floor_divide = (binaryfunc)pyOcScalar_nb_inplace_floor_divide;
   #endif

   #if PY_MAJOR_VERSION >= 3
   nb -> nb_matrix_multiply         = (binaryfunc)pyOcScalar_nb_multiply;
   nb -> nb_inplace_matrix_multiply = (binaryfunc)pyOcScalar_nb_inplace_multiply;
   #endif

   /* Type conversion */
   #if PY_MAJOR_VERSION >= 3
   nb -> nb_bool                 = (inquiry  )pyOcScalar_nb_nonzero;
   #else
   nb -> nb_long                 = (unaryfunc)pyOcScalar_nb_long;
   nb -> nb_nonzero              = (inquiry  )pyOcScalar_nb_nonzero;
   nb -> nb_divide               = (binaryfunc)pyOcScalar_nb_true_divide;
   nb -> nb_inplace_divide       = (binaryfunc)pyOcScalar_nb_inplace_true_divide;
   #endif

   if (PyType_Ready(PyOceanScalar) < 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcScalar_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   OcScalar scalar;

   Py_INCREF(PyOceanScalar); /* Static object - do not delete */
   PyModule_AddObject(module, "scalar", (PyObject *)PyOceanScalar);

   /* Add scalar constants (Python objects) */
   OcScalar_doubleInf(&scalar);
   if ((obj = PyFloat_FromDouble(scalar.value.sDouble)) == NULL) return -1;
   PyModule_AddObject(module, "inf", obj);

   OcScalar_doubleNaN(&scalar);
   if ((obj = PyFloat_FromDouble(scalar.value.sDouble)) == NULL) return -1;
   PyModule_AddObject(module, "nan", obj);

   if ((obj = PyFloat_FromDouble(M_PI)) == NULL) return -1;
   PyModule_AddObject(module, "pi", obj);

   if ((obj = PyFloat_FromDouble(M_E)) == NULL) return -1;
   PyModule_AddObject(module, "e", obj);

   /* True and false */
   scalar.dtype = OcDTypeBool; OcScalar_fromInt64(&scalar, 1);
   if ((obj = PyOceanScalar_New(&scalar)) == NULL) return -1;
   PyModule_AddObject(module, "true", obj);
   
   scalar.dtype = OcDTypeBool; OcScalar_fromInt64(&scalar, 0);
   if ((obj = PyOceanScalar_New(&scalar)) == NULL) return -1;
   PyModule_AddObject(module, "false", obj);

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcScalar_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Empty */
}


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanScalar_Create(void)
/* -------------------------------------------------------------------- */
{
   /* Construct the object */
   return (PyObject *)PyOceanScalar -> tp_alloc(PyOceanScalar,0);
}



/* -------------------------------------------------------------------- */
PyObject *PyOceanScalar_Wrap(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  PyObject *result;

   result = PyOceanScalar_New(scalar);
   if (scalar) OcScalar_free(scalar);
   
   return result;
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanScalar_New(OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  pyOcScalar  *obj;

   if (scalar == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcScalar *)PyOceanScalar -> tp_alloc(PyOceanScalar,0);
   if (obj == NULL) return NULL;

   /* Copy the scalar */
   obj -> scalar.dtype = scalar -> dtype;
   OcScalar_copy(scalar, &(obj -> scalar));

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanScalar_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanScalar);
}


/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcScalar_dealloc(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{
   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   PyObject   *result = NULL;
   OcScalar   *value;
   OcDType     dtype;

   /* ================================== */
   /* Syntax: Scalar([value] [, dtype])  */
   /* ================================== */

   /* Make sure there are no key-word arguments */
   if (kwargs != NULL) OcError(NULL, "The scalar constructor does not take keyword arguments");

   /* Parse the parameters */
   PyOceanArgs_Init(&param, args, "ocean.scalar");
   PyOceanArgs_GetScalarLike(&param, &value, 0);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;
   
   /* Determine the data type */
   if (dtype == OcDTypeNone)
   {  if (value)
      {  dtype = value -> dtype;
      }
      else
      {  if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) goto final;
      }
   }

   /* Create the new scalar */
   if (value)
   {  if (value -> dtype == dtype)
      {  result = PyOceanScalar_New(value);
      }
      else
      {  result =PyOceanScalar_Wrap(OcScalar_castDType(value, dtype));
      }
   }
   else
   {  result = PyOceanScalar_Wrap(OcScalar_create(dtype));;
   }

final : ;
   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_str(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{  PyObject *obj = NULL;
   char     *str = NULL;

   /* Format the content string */
   if (OcScalar_format(&(self -> scalar), &str, NULL, NULL) != 0) return NULL;

   /* Create the result string */
   obj = PyString_FromString(str);

   /* Free the formatted string */
   if (str) free(str);

   return obj;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_richcompare(PyObject *obj1, PyObject *obj2, int opid)
/* -------------------------------------------------------------------- */
{  OcScalar scalar1;
   OcScalar scalar2;
   int result = -1;

   /* Make sure the object is an Ocean scalar */
   if ((!pyOcean_isScalar(obj1)) || (!pyOcean_isScalar(obj2)))
   {  PyTypeObject *type = NULL;
      if (pyOcean_isTensor(obj1)) type = Py_TYPE(obj1);
      if (pyOcean_isTensor(obj2)) type = Py_TYPE(obj2);
      if (type && type -> tp_richcompare)
         return type -> tp_richcompare(obj1, obj2, opid);
      else
         OcErrorMessage("Unsupported type in comparison operation");
   }
   else
   {  /* Get the scalars */
      if ((pyOcean_getScalar(obj1, &scalar1) != 1) ||
          (pyOcean_getScalar(obj2, &scalar2) != 1))
         OcError(NULL, "Error converting the right-hand side to a scalar");

      /* Evaluate supported comparison operations */
      switch (opid)
      {  case Py_LT : result = OcScalar_isLT(&scalar1, &scalar2); break;
         case Py_LE : result = OcScalar_isLE(&scalar1, &scalar2); break;
         case Py_EQ : result = OcScalar_isEQ(&scalar1, &scalar2); break;
         case Py_NE : result = OcScalar_isNE(&scalar1, &scalar2); break;
         case Py_GE : result = OcScalar_isGE(&scalar1, &scalar2); break;
         case Py_GT : result = OcScalar_isGT(&scalar1, &scalar2); break;
         default :
            result = -1;
            OcErrorMessage("The given comparison operation is not implemented");
      }
   }

   /* Return the result */
   if (result == 0)
      Py_RETURN_FALSE;
   else if (result == 1)
      Py_RETURN_TRUE;
   else return NULL;
}


/* ===================================================================== */
/* Class functions - numeric functions                                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_int(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{  OcInt64 value;

   value = OcScalar_asInt64(&(self -> scalar));
   return PyInt_FromLong(value);
}


#if PY_MAJOR_VERSION < 3
/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_long(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{  OcInt64 value;

   value = OcScalar_asInt64(&(self -> scalar));
   return PyLong_FromLong(value);
}
#endif


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_float(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{  double value;

   value = OcScalar_asDouble(&(self -> scalar));
   return PyFloat_FromDouble(value);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_complex(pyOcScalar *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcCDouble value;

   value = OcScalar_asCDouble(&(self -> scalar));
   return PyComplex_FromDoubles(value.real, value.imag);
}


/* -------------------------------------------------------------------- */
static int pyOcScalar_nb_nonzero(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{
   return OcScalar_asBool(&(self -> scalar));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_positive(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{
   return PyOceanScalar_New(&(self -> scalar));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_negative(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   if (OcScalar_negative(&(self -> scalar), &scalar) != 0) return NULL;

   return PyOceanScalar_New(&scalar);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_invert(pyOcScalar *self)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   if (OcScalar_bitwiseNot(&(self -> scalar), &scalar) != 0) return NULL;

   return PyOceanScalar_New(&scalar);
}



/* --------------------------------------------------------------------------- */
/* static PyObject *pyOcScalar_nb_add         (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_subtract    (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_multiply    (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_remainder   (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_and         (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_or          (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_xor         (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_lshift      (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_rshift      (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_true_divide (PyObject *obj1, PyObject *obj2) */
/* static PyObject *pyOcScalar_nb_floor_divide(PyObject *obj1, PyObject *obj2) */
/* --------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, INTRNL_OP) \
static PyObject *pyOcScalar_nb_##OP(PyObject *obj1, PyObject *obj2) \
{  \
   return pyOceanCore_intrnl_##INTRNL_OP(obj1, obj2, NULL); \
}

OC_TEMPLATE(add,          add          )
OC_TEMPLATE(subtract,     subtract     )
OC_TEMPLATE(multiply,     scale        )
OC_TEMPLATE(remainder,    mod          )
OC_TEMPLATE(and,          bitwiseAnd   )
OC_TEMPLATE(or,           bitwiseOr    )
OC_TEMPLATE(xor,          bitwiseXor   )
OC_TEMPLATE(lshift,       bitshiftLeft )
OC_TEMPLATE(rshift,       bitshiftRight)

#ifdef PY_VERSION_2_2
OC_TEMPLATE(true_divide,  divide       )
OC_TEMPLATE(floor_divide, floorDivide  )
#endif

#undef OC_TEMPLATE



/* ---------------------------------------------------------------------------- */
/* PyObject *pyOcScalar_nb_inplace_add         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_subtract    (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_multiply    (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_remainder   (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_and         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_or          (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_xor         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_lshift      (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_rshift      (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_true_divide (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcScalar_nb_inplace_floor_divide(PyObject *obj1, PyObject *obj2) */
/* ---------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP_NAME, OP, OCEAN_OP) \
static PyObject *pyOcScalar_nb_inplace_##OP(PyObject *obj1, PyObject *obj2) \
{  OcScalar *scalar1, scalar2; \
   \
   /* The input object must be a scalar */ \
   scalar1 = PYOC_GET_SCALAR(obj1); \
  if (pyOcean_getScalar(obj2, &scalar2) != 1) \
  {  OcError(NULL, "Unsupported operand types for "#OP_NAME); \
  } \
  \
  /* Apply the scalar operation */ \
  if (OcScalar_##OCEAN_OP(scalar1, &scalar2, scalar1) != 0) return NULL; \
  \
  /* Return a new reference */ \
  Py_INCREF(obj1); \
  return obj1; \
}

OC_TEMPLATE("+=",  add,          add          )
OC_TEMPLATE("-=",  subtract,     subtract     )
OC_TEMPLATE("*=",  multiply,     multiply     )
OC_TEMPLATE("%%=", remainder,    mod          )
OC_TEMPLATE("&=",  and,          bitwiseAnd   )
OC_TEMPLATE("|=",  or,           bitwiseOr    )
OC_TEMPLATE("^=",  xor,          bitwiseXor   )
OC_TEMPLATE("<<=", lshift,       bitshiftLeft )
OC_TEMPLATE(">>=", rshift,       bitshiftRight)

#ifdef PY_VERSION_2_2
OC_TEMPLATE("/=",  true_divide,  divide     )
OC_TEMPLATE("//=", floor_divide, floorDivide)
#endif

#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_power(PyObject *obj1, PyObject *obj2, PyObject *obj3)
/* -------------------------------------------------------------------- */
{  char mode = OcTensor_getDefaultMathMode();

   /* Check the modulo parameter */
   if (obj3 != Py_None)
      OcError(NULL,"The modulo parameter in power is not supported");

   /* Call the function */
   return pyOceanCore_intrnl_power(obj1, obj2, NULL, mode);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_nb_inplace_power(PyObject *obj1, PyObject *obj2, PyObject *obj3)
/* -------------------------------------------------------------------- */
{  OcScalar *scalar1, scalar2;

   /* Check the modulo parameter */
   if (obj3 != Py_None)
      OcError(NULL,"The modulo parameter in power is not supported");

   /* The input object must be a scalar */
   scalar1 = PYOC_GET_SCALAR(obj1);
  if (pyOcean_getScalar(obj2, &scalar2) != 1)
  {  OcError(NULL, "Unsupported operand types for power");
  }

  /* Apply the scalar operation */
  if (OcScalar_power(scalar1, &scalar2, scalar1) != 0) return NULL;

  /* Return a new reference */
  Py_INCREF(obj1);
  return obj1;
}



/* ===================================================================== */
/* Class functions - get and set functions                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_getdtype(pyOcScalar *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanDType_New(self -> scalar.dtype);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_getreal(pyOcScalar *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *result = PyOceanScalar_Create();

   if (result)
      OcScalar_getReal(&(self -> scalar), PYOC_GET_SCALAR(result));

   return result;
}

/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_getimag(pyOcScalar *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *result = PyOceanScalar_Create();

   if (result)
      OcScalar_getImag(&(self -> scalar), PYOC_GET_SCALAR(result));

   return result;
}


/* -------------------------------------------------------------------- */
static int pyOcScalar_setreal(pyOcScalar *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;
   int result;

  if (pyOcean_getScalar(value, &scalar) != 1)
     OcError(1, "Unsupported type in assignment");

  result = OcScalar_setReal(PYOC_GET_SCALAR(self), &scalar);
  return (result == 0) ? 0 : 1;
}


/* -------------------------------------------------------------------- */
static int pyOcScalar_setimag(pyOcScalar *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;
   int result;

  if (pyOcean_getScalar(value, &scalar) != 1)
     OcError(1, "Unsupported type in assignment");

  result = OcScalar_setImag(PYOC_GET_SCALAR(self), &scalar);
  return (result == 0) ? 0 : 1;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_getelemsize(pyOcScalar *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcScalar *scalar = PYOC_GET_SCALAR(self);

   return PyInt_FromLong(OcDType_size(scalar -> dtype));
}


/* ===================================================================== */
/* Class functions - additional functions                                */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_asPython(pyOcScalar *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcScalar *scalar = &(self -> scalar);

   /* Boolean scalars */
   if (scalar -> dtype == OcDTypeBool)
   {  if (scalar -> value.sBool == 0)
           Py_RETURN_FALSE;
      else Py_RETURN_TRUE;
   }

   /* Integer scalars */
   if (!OcDType_isFloat(scalar -> dtype))
   {  return pyOcScalar_nb_int(self);
   }

   /* Float scalars */
   if (!OcDType_isComplex(scalar -> dtype))
   {  return pyOcScalar_nb_float(self);
   }

   /* Complex scalars */
   return pyOcScalar_complex(self, NULL);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcScalar_asTensor(pyOcScalar *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcDevice   *device;
   OcTensor   *tensor;

   /* ======================================== */
   /* Syntax: scalar.asTensor([device=OcCPU])  */
   /* ======================================== */

   /* Parse the parameters */
   PyOceanArgs_Init(&param, args, "scalar.asTensor");
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;
   
   /* Apply the default device */
   if (device == NULL) device = OcCPU;

   /* Create the tensor */
   tensor = OcTensor_createFromScalar(&(self -> scalar), self -> scalar.dtype, device, 0);
   return PyOceanTensor_Wrap(tensor);
}

#undef OC_TEMPLATE

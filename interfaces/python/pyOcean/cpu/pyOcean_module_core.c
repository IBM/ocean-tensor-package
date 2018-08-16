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

#include "pyOcean_module_core.h"
#include "pyOcean_core.h"
#include "pyOcean_args.h"
#include "pyOcean_dtype.h"
#include "pyOcean_device.h"
#include "pyOcean_stream.h"
#include "pyOcean_storage.h"
#include "pyOcean_scalar.h"
#include "pyOcean_tensor.h"
#include "pyOcean_convert.h"

#include <stdint.h>


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Format settings */
PyObject *pyOceanCore_setDisplayWidth      (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setDebugMode         (PyObject *self, PyObject *args);

/* Data type functions */
PyObject *pyOceanCore_getDefaultDType      (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setDefaultDType      (PyObject *self, PyObject *args);

/* Device functions */
PyObject *pyOceanCore_getDeviceList        (PyObject *self, PyObject *args);
PyObject *pyOceanCore_getDefaultDevice     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setDefaultDevice     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_getMaxBufferSize     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setMaxBufferSize     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_getBufferCount       (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setBufferCount       (PyObject *self, PyObject *args);

/* Warning configuration */
PyObject *pyOceanCore_getWarningMode       (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setWarningMode       (PyObject *self, PyObject *args);

/* Scalar configuration */
PyObject *pyOceanCore_getScalarCastMode    (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setScalarCastMode    (PyObject *self, PyObject *args);

/* Tensor configuration */
PyObject *pyOceanCore_getAutoBroadcast     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setAutoBroadcast     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_getAutoTypecast      (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setAutoTypecast      (PyObject *self, PyObject *args);
PyObject *pyOceanCore_getDefaultMathMode   (PyObject *self, PyObject *args);
PyObject *pyOceanCore_setDefaultMathMode   (PyObject *self, PyObject *args);

/* Import and export data types */
PyObject *pyOceanCore_getImportTypes       (PyObject *self, PyObject *args);
PyObject *pyOceanCore_getExportTypes       (PyObject *self, PyObject *args);

/* Device and type casting */
PyObject *pyOceanCore_ensure               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_cast                 (PyObject *self, PyObject *args);

/* Tensor construction */
PyObject *pyOceanCore_asTensor             (PyObject *self, PyObject *args);
PyObject *pyOceanCore_zeros                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_ones                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_full                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_tensorLike           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_zerosLike            (PyObject *self, PyObject *args);
PyObject *pyOceanCore_onesLike             (PyObject *self, PyObject *args);
PyObject *pyOceanCore_fullLike             (PyObject *self, PyObject *args);
PyObject *pyOceanCore_arange               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_linspace             (PyObject *self, PyObject *args);
PyObject *pyOceanCore_eye                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_diag                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_merge                (PyObject *self, PyObject *args);

/* Unary tensor operations */
#define OC_TEMPLATE(NAME,X,Y,Z) \
PyObject *pyOceanCore_##NAME               (PyObject *self, PyObject *args);
#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE

/* Binary tensor operations */
PyObject *pyOceanCore_add                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_subtract             (PyObject *self, PyObject *args);
PyObject *pyOceanCore_scale                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_divide               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_trueDivide           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_floorDivide          (PyObject *self, PyObject *args);
PyObject *pyOceanCore_power                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_mod                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_fmod                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_min                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_max                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_fmin                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_fmax                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_bitwiseAnd           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_bitwiseOr            (PyObject *self, PyObject *args);
PyObject *pyOceanCore_bitwiseXor           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_logicalAnd           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_logicalOr            (PyObject *self, PyObject *args);
PyObject *pyOceanCore_logicalXor           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_bitshiftLeft         (PyObject *self, PyObject *args);
PyObject *pyOceanCore_bitshiftRight        (PyObject *self, PyObject *args);

/* Tensor domain checks */
PyObject *pyOceanCore_allLT                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_allLE                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_allGE                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_allGT                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_allInRange           (PyObject *self, PyObject *args);

/* Tensor global reduction */
PyObject *pyOceanCore_any                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_all                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_allFinite            (PyObject *self, PyObject *args);
PyObject *pyOceanCore_anyInf               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_anyNaN               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_nnz                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_nnzNaN               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_sum                  (PyObject *self, PyObject *args);
PyObject *pyOceanCore_prod                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_sumNaN               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_prodNaN              (PyObject *self, PyObject *args);
PyObject *pyOceanCore_sumAbs               (PyObject *self, PyObject *args);
PyObject *pyOceanCore_sumAbsNaN            (PyObject *self, PyObject *args);
PyObject *pyOceanCore_maximum              (PyObject *self, PyObject *args);
PyObject *pyOceanCore_minimum              (PyObject *self, PyObject *args);
PyObject *pyOceanCore_maximumAbs           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_minimumAbs           (PyObject *self, PyObject *args);
PyObject *pyOceanCore_norm                 (PyObject *self, PyObject *args);
PyObject *pyOceanCore_norm1                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_norm2                (PyObject *self, PyObject *args);
PyObject *pyOceanCore_normInf              (PyObject *self, PyObject *args);
PyObject *pyOceanCore_normNaN              (PyObject *self, PyObject *args);

/* Tensor find */
PyObject *pyOceanCore_find                 (PyObject *self, PyObject *args);

/* Tensor multiplication */
PyObject *pyOceanCore_multiply             (PyObject *self, PyObject *args);
PyObject *pyOceanCore_gemm                 (PyObject *self, PyObject *args);

/* Internal Ocean functions for unit testing */
PyObject *pyOceanCore_checkSelfOverlap     (PyObject *self, PyObject *args);
PyObject *pyOceanCore_checkOverlap         (PyObject *self, PyObject *args);


/* ===================================================================== */
/* Module definition                                                     */
/* ===================================================================== */

PyMethodDef py_ocean_core_methods[] = {
   {"devices",               pyOceanCore_getDeviceList,         METH_NOARGS,  "List of available devices"},

   /* General configuration */
   {"setDisplayWidth",       pyOceanCore_setDisplayWidth,       METH_VARARGS, "Set the width of the display"},
   {"setDebugMode",          pyOceanCore_setDebugMode,          METH_VARARGS, "Set the debug mode"},

   /* Default device and data type */
   {"setDefaultDevice",      pyOceanCore_setDefaultDevice,      METH_VARARGS, "Set the default device"},
   {"getDefaultDevice",      pyOceanCore_getDefaultDevice,      METH_NOARGS,  "Returns the default device"},
   {"setDefaultDType",       pyOceanCore_setDefaultDType,       METH_VARARGS, "Set the default data type"},
   {"getDefaultDType",       pyOceanCore_getDefaultDType,       METH_NOARGS,  "Returns the default data type"},
   {"getMaxBufferSize",      pyOceanCore_getMaxBufferSize,      METH_VARARGS, "Get the maximum temporary buffer size on a device"},
   {"setMaxBufferSize",      pyOceanCore_setMaxBufferSize,      METH_VARARGS, "Set the maximum temporary buffer size for a device or device type"},
   {"getBufferCount",        pyOceanCore_getBufferCount,        METH_VARARGS, "Get the maximum number of temporary buffers on a device"},
   {"setBufferCount",        pyOceanCore_setBufferCount,        METH_VARARGS, "Set the maximum number of temporary buffers on device type"},

   /* Warning configuration */
   {"getWarningMode",        pyOceanCore_getWarningMode,        METH_NOARGS,  "Get the warning mode (0 = off, 1 = once, 2 = always)"},
   {"setWarningMode",        pyOceanCore_setWarningMode,        METH_VARARGS, "Set the warning mode (0 = off, 1 = once, 2 = always)"},

   /* Scalar configuration */
   {"getScalarCastMode",     pyOceanCore_getScalarCastMode,     METH_NOARGS,  "Get the scalar cast mode"},
   {"setScalarCastMode",     pyOceanCore_setScalarCastMode,     METH_VARARGS, "Set the scalar cast mode"},

   /* Tensor configuration */
   {"getAutoBroadcast",      pyOceanCore_getAutoBroadcast,      METH_NOARGS,  "Current status of automatic broadcasting of tensor dimensions"},
   {"setAutoBroadcast",      pyOceanCore_setAutoBroadcast,      METH_VARARGS, "Enable or disable automatic broadcasting of tensor dimensions"},
   {"getAutoTypecast",       pyOceanCore_getAutoTypecast,       METH_NOARGS,  "Get the tensor cast mode"},
   {"setAutoTypecast",       pyOceanCore_setAutoTypecast,       METH_VARARGS, "Set the tensor cast mode"},
   {"getDefaultMathMode",    pyOceanCore_getDefaultMathMode,    METH_NOARGS,  "Get the default math mode"},
   {"setDefaultMathMode",    pyOceanCore_setDefaultMathMode,    METH_VARARGS, "Set the default math mode"},

   /* Import and export data types */
   {"getImportTypes",        pyOceanCore_getImportTypes,        METH_NOARGS,   "List of all supported import types"},
   {"getExportTypes",        pyOceanCore_getExportTypes,        METH_NOARGS,   "List of all supported export types"},

   /* Device and type casting */
   {"ensure",                pyOceanCore_ensure,                METH_VARARGS, "Ensure that the data type and/or device matches"},
   {"cast",                  pyOceanCore_cast,                  METH_VARARGS, "Cast to the given data type and/or device"},

   /* Creation of tensor from data */
   {"asTensor",              pyOceanCore_asTensor,              METH_VARARGS,  "Create a tensor from data"},
   {"zeros",                 pyOceanCore_zeros,                 METH_VARARGS,  "Create a tensor of zeros"},
   {"ones",                  pyOceanCore_ones,                  METH_VARARGS,  "Create a tensor of ones"},
   {"full",                  pyOceanCore_full,                  METH_VARARGS,  "Create a tensor filled with a scalar"},
   {"tensorLike",            pyOceanCore_tensorLike,            METH_VARARGS,  "Create a tensor with matching size, data type, and device"},
   {"zerosLike",             pyOceanCore_zerosLike,             METH_VARARGS,  "Create a tensor of zeros with matching size, data type, and device"},
   {"onesLike",              pyOceanCore_onesLike,              METH_VARARGS,  "Create a tensor of ones with matching size, data type, and device"},
   {"fullLike",              pyOceanCore_fullLike,              METH_VARARGS,  "Create a tensor filled with a scalar, with matching size, data type, and device"},
   {"arange",                pyOceanCore_arange,                METH_VARARGS,  "Create a number range"},
   {"linspace",              pyOceanCore_linspace,              METH_VARARGS,  "Linear interpolation"},
   {"eye",                   pyOceanCore_eye,                   METH_VARARGS,  "Create an identity matrix"},
   {"diag",                  pyOceanCore_diag,                  METH_VARARGS,  "Create a diagonal matrix"},
   {"merge",                 pyOceanCore_merge,                 METH_VARARGS,  "Merge tensors along an axis"},

   /* Unary tensor operations */
   #define OC_TEMPLATE(NAME,X,Y,DESC) \
   {#NAME, pyOceanCore_##NAME, METH_VARARGS, "Compute the "#DESC},
   #include "ocean/core/generic/generate_tensor_unary.h"
   #undef OC_TEMPLATE

   /* Binary tensor operations */
   {"add",                  pyOceanCore_add,                   METH_VARARGS,  "Addition of tensors or scalars"},
   {"subtract",             pyOceanCore_subtract,              METH_VARARGS,  "Subtraction of tensors or scalars"},
   {"scale",                pyOceanCore_scale,                 METH_VARARGS,  "Elementwise multiplication of tensors or scalars"},
   {"divide",               pyOceanCore_divide,                METH_VARARGS,  "Elementwise division of tensors or scalars"},
   {"trueDivide",           pyOceanCore_trueDivide,            METH_VARARGS,  "Elementwise true division"},
   {"floorDivide",          pyOceanCore_floorDivide,           METH_VARARGS,  "Elementwise floor division"},
   {"power",                pyOceanCore_power,                 METH_VARARGS,  "Elementwise power"},
   {"mod",                  pyOceanCore_mod,                   METH_VARARGS,  "Elementwise modulo"},
   {"fmod",                 pyOceanCore_fmod,                  METH_VARARGS,  "Elementwise modulo (C-style)"},
   {"min",                  pyOceanCore_min,                   METH_VARARGS,  "Elementwise minimum"},
   {"max",                  pyOceanCore_max,                   METH_VARARGS,  "Elementwise maximum"},
   {"fmin",                 pyOceanCore_fmin,                  METH_VARARGS,  "Elementwise minimum (checks for NaN)"},
   {"fmax",                 pyOceanCore_fmax,                  METH_VARARGS,  "Elementwise maximum (checks for NaN)"},
   {"bitwiseAnd",           pyOceanCore_bitwiseAnd,            METH_VARARGS,  "Elementwise bitwise-AND"},
   {"bitwiseOr",            pyOceanCore_bitwiseOr,             METH_VARARGS,  "Elementwise bitwise-OR"},
   {"bitwiseXor",           pyOceanCore_bitwiseXor,            METH_VARARGS,  "Elementwise bitwise-XOR"},
   {"logicalAnd",           pyOceanCore_logicalAnd,            METH_VARARGS,  "Elementwise logical-AND"},
   {"logicalOr",            pyOceanCore_logicalOr,             METH_VARARGS,  "Elementwise logical-OR"},
   {"logicalXor",           pyOceanCore_logicalXor,            METH_VARARGS,  "Elementwise logical-XOR"},
   {"bitshiftLeft",         pyOceanCore_bitshiftLeft,          METH_VARARGS,  "Elementwise bitshift left"},
   {"bitshiftRight",        pyOceanCore_bitshiftRight,         METH_VARARGS,  "Elementwise bitshift right"},

   /* Domain checks */
   {"allLT",                pyOceanCore_allLT,                 METH_VARARGS,  "Check if all elements are strictly less than the given value"},
   {"allLE",                pyOceanCore_allLE,                 METH_VARARGS,  "Check if all elements are less than or equal to given value"},
   {"allGT",                pyOceanCore_allGT,                 METH_VARARGS,  "Check if all elements are strictly greater than the given value"},
   {"allGE",                pyOceanCore_allGE,                 METH_VARARGS,  "Check if all elements are greater than or equal to given value"},
   {"allInRange",           pyOceanCore_allInRange,            METH_VARARGS,  "Check if all elements are in the given range"},

   /* Reduction operators */
   {"any",                  pyOceanCore_any,                   METH_VARARGS,  "Check if any element in the tensor or along the given axis is non-zero"},
   {"all",                  pyOceanCore_all,                   METH_VARARGS,  "Check if all elements in the tensor or along the given axis are non-zero"},
   {"allFinite",            pyOceanCore_allFinite,             METH_VARARGS,  "Check if all elements in the tensor or along the given axis are finite"},
   {"anyInf",               pyOceanCore_anyInf,                METH_VARARGS,  "Check if any element in the tensor or along the given axis is infinite"},
   {"anyNaN",               pyOceanCore_anyNaN,                METH_VARARGS,  "Check if any element in the tensor or along the given axis is NaN"},
   {"nnz",                  pyOceanCore_nnz,                   METH_VARARGS,  "Number of non-zero entries in the tensor"},
   {"nnzNaN",               pyOceanCore_nnzNaN,                METH_VARARGS,  "Number of non-zero entries in the tensor, excluding NaN entries"},
   {"sum",                  pyOceanCore_sum,                   METH_VARARGS,  "Sum of all tensor elements"},
   {"prod",                 pyOceanCore_prod,                  METH_VARARGS,  "Product of all tensor elements"},
   {"sumNaN",               pyOceanCore_sumNaN,                METH_VARARGS,  "Sum of all tensor elements, excluding NaN entries"},
   {"prodNaN",              pyOceanCore_prodNaN,               METH_VARARGS,  "Product of all tensor elements, excluding NaN entries"},
   {"sumAbs",               pyOceanCore_sumAbs,                METH_VARARGS,  "Sum of the magnitudes of all tensor elements"},
   {"sumAbsNaN",            pyOceanCore_sumAbsNaN,             METH_VARARGS,  "Sum of the magnitudes of all tensor elements, excluding NaN entries"},
   {"maximum",              pyOceanCore_maximum,               METH_VARARGS,  "Maximum tensor element"},
   {"minimum",              pyOceanCore_minimum,               METH_VARARGS,  "Minimum tensor element"},
   {"maximumAbs",           pyOceanCore_maximumAbs,            METH_VARARGS,  "Maximum tensor element magnitude"},
   {"minimumAbs",           pyOceanCore_minimumAbs,            METH_VARARGS,  "Minimum tensor element magnitude"},
   {"norm",                 pyOceanCore_norm,                  METH_VARARGS,  "Norm of vectorized slices along given axes"},
   {"norm1",                pyOceanCore_norm1,                 METH_VARARGS,  "1-Norm of vectorized slices along given axes"},
   {"norm2",                pyOceanCore_norm2,                 METH_VARARGS,  "2-Norm of vectorized slices along given axes"},
   {"normInf",              pyOceanCore_normInf,               METH_VARARGS,  "Inf-norm of vectorized slices along given axes"},
   {"normNaN",              pyOceanCore_normNaN,               METH_VARARGS,  "Norm of vectorized slices along given axes, excluding NaN entries"},

   /* Tensor find */
   {"find",                 pyOceanCore_find,                  METH_VARARGS,  "Find the indices of the non-zero indices"},

   /* Tensor multiplication */
   {"multiply",             pyOceanCore_multiply,              METH_VARARGS,  "Tensor-tensor multiplication"},
   {"gemm",                 pyOceanCore_gemm,                  METH_VARARGS,  "Tensor-tensor multiplication"},

   /* Internal functions */
   {"checkSelfOverlap",     pyOceanCore_checkSelfOverlap,      METH_VARARGS,  "Check for self overlap"},
   {"checkOverlap",         pyOceanCore_checkOverlap,          METH_VARARGS,  "Check for tensor overlap"},

   {NULL}  /* Sentinel */
};



/* ===================================================================== */
/* Function definitions - Format settings                                */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setDisplayWidth(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int width;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"i", &width)) return NULL;
   if (width <= 0) OcError(NULL, "Display width must be positive");

   /* Set the display width */
   oc_format_linewidth = width;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setDebugMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  unsigned char mode;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"B", &mode)) return NULL;

   /* Set the debug mode */
   oc_debug_mode = (mode == 0) ? 0 : 1;

   Py_RETURN_NONE;
}



/* ===================================================================== */
/* Function definitions - Data types                                     */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getDefaultDType(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcDType dtype;

   if ((dtype = OcDType_getDefault()) == OcDTypeNone) Py_RETURN_NONE;
   
   return PyOceanDType_New(dtype);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setDefaultDType(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyObject *dtype;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"O", &dtype)) return NULL;

   /* Check the parameter type */
   if (dtype == Py_None)
   {  OcDType_setDefault(OcDTypeNone);
   }
   else if (PyOceanDType_Check(dtype))
   {  OcDType_setDefault(PYOC_GET_DTYPE(dtype));
   }
   else
   {  OcError(NULL, "Invalid input argument to function setDefaultDType.");
   }

   Py_RETURN_NONE;
}



/* ===================================================================== */
/* Function definitions - Devices                                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getDeviceList(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyObject       *list;
   Py_int_ssize_t  i, n;

   /* Create the tuple */
   n = OcDeviceCount();
   list = PyTuple_New(n);
   if (list == NULL) return NULL;

   /* Add the devices */
   for (i = 0; i < n; i++)
   {  if (PyTuple_SetItem(list, i, PyOceanDevice_New(OcDeviceByIndex(i))) != 0)
      {  Py_DECREF(list); return NULL;
      }
   }

   /* Return the list */
   return list;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getDefaultDevice(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcDevice *device;

   if ((device = OcDevice_getDefault()) == NULL) Py_RETURN_NONE;
   
   return PyOceanDevice_New(device);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setDefaultDevice(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyObject *device;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"O", &device)) return NULL;

   /* Check the parameter type */
   if (device == Py_None)
   {  OcDevice_setDefault(NULL);
   }
   else if (PyOceanDevice_Check(device))
   {  OcDevice_setDefault(PYOC_GET_DEVICE(device));
   }
   else
   {  OcError(NULL, "Invalid input argument to function setDefaultDevice.");
   }

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getMaxBufferSize(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcDevice    *device;
   OcSize       size;

   /* ======================================= */
   /* Syntax: ocean.getMaxBufferSize(device)  */
   /* ======================================= */
   PyOceanArgs_Init(&param, args, "ocean.getMaxBufferSize");
   PyOceanArgs_GetOcDevice(&param, &device, 1);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Return the maximum buffer size */
   if (OcDevice_getMaxBufferSize(device, &size) != 0) return NULL;
   return PyLong_FromLong((long)size);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setMaxBufferSize(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs   param;
   OcDeviceType *deviceType;
   OcDevice     *device = NULL;
   char         *type = NULL;
   long int      size;
   int           i, n, result = -1;

   /* ================================================= */
   /* Syntax: ocean.setMaxBufferSize(size)              */
   /* Syntax: ocean.setMaxBufferSize(device, size)      */
   /* Syntax: ocean.setMaxBufferSize(deviceType, size)  */
   /* ================================================= */
   PyOceanArgs_Init(&param, args, "ocean.setMaxBufferSize");
   if (PyOceanArgs_GetOcDevice(&param, &device, 0) == 0)
   {  PyOceanArgs_GetAsciiString(&param, &type, 0);
   }
   PyOceanArgs_GetScalarInt(&param, &size, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Set the maximum buffer size */
   if (device != NULL)
   {  result = OcDevice_setMaxBufferSize(device, size);
   }
   else if (type != NULL)
   {  /* Loop over all devices with the given device type */
      if ((deviceType = OcDeviceTypeByName(type)) == NULL)
      {  OcErrorMessage("Device type '%s' does not exist", type);
         goto final;
      }

      /* Get the number of device instances */
      n = OcDeviceType_deviceCount(deviceType);
      for (i = 0; i < n; i++)
      {  device = OcDeviceType_getDevice(deviceType, i);
         if (device == NULL) goto final;
         if (OcDevice_setMaxBufferSize(device, size) != 0) goto final;
      }
   }
   else
   {  /* Loop over all devices */
      n = OcDeviceCount();
      for (i = 0; i < n; i++)
      {  device = OcDeviceByIndex(i);
         if (device == NULL) goto final;
         if (OcDevice_setMaxBufferSize(device, size) != 0) goto final;
      }
   }

   /* Success */
   result = 0;

final : ;
   PyOceanArgs_Finalize(&param);
   if (result == 0) Py_RETURN_NONE; else return NULL;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getBufferCount(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcDevice    *device;
   int          count;

   /* ===================================== */
   /* Syntax: ocean.getBufferCount(device)  */
   /* ===================================== */
   PyOceanArgs_Init(&param, args, "ocean.getBufferCount");
   PyOceanArgs_GetOcDevice(&param, &device, 1);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Return the buffer count */
   if (OcDevice_getBufferCount(device, &count) != 0) return NULL;
   return PyInt_FromLong((long)count);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setBufferCount(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs   param;
   OcDeviceType *deviceType;
   OcDevice     *device = NULL;
   char         *type = NULL;
   long int      count;
   int           i, n, result = -1;

   /* ================================================ */
   /* Syntax: ocean.setBufferCount(count)              */
   /* Syntax: ocean.setBufferCount(device, count)      */
   /* Syntax: ocean.setBufferCount(deviceType, count)  */
   /* ================================================ */
   PyOceanArgs_Init(&param, args, "ocean.setBufferCount");
   if (PyOceanArgs_GetOcDevice(&param, &device, 0) == 0)
   {  PyOceanArgs_GetAsciiString(&param, &type, 0);
   }
   PyOceanArgs_GetScalarInt(&param, &count, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Set the maximum number of buffers */
   if (device != NULL)
   {  result = OcDevice_setBufferCount(device, count);
   }
   else if (type != NULL)
   {  /* Loop over all devices with the given device type */
      if ((deviceType = OcDeviceTypeByName(type)) == NULL)
      {  OcErrorMessage("Device type '%s' does not exist", type);
         goto final;
      }

      /* Get the number of device instances */
      n = OcDeviceType_deviceCount(deviceType);
      for (i = 0; i < n; i++)
      {  device = OcDeviceType_getDevice(deviceType, i);
         if (device == NULL) goto final;
         if (OcDevice_setBufferCount(device, count) != 0) goto final;
      }
   }
   else
   {  /* Loop over all devices */
      n = OcDeviceCount();
      for (i = 0; i < n; i++)
      {  device = OcDeviceByIndex(i);
         if (device == NULL) goto final;
         if (OcDevice_setBufferCount(device, count) != 0) goto final;
      }
   }

   /* Success */
   result = 0;

final : ;
   PyOceanArgs_Finalize(&param);
   if (result == 0) Py_RETURN_NONE; else return NULL;
}



/* ===================================================================== */
/* Function definitions - Warning configuration                          */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getWarningMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long)(OcWarning_getMode()));
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setWarningMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int mode;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"i", &mode)) return NULL;

   /* Set the warning mode */
   if ((mode < 0) || (mode > 2)) mode = 2;
   OcWarning_setMode((OcWarningMode)mode);

   Py_RETURN_NONE;
}



/* ===================================================================== */
/* Function definitions - Scalar configuration                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getScalarCastMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long)(OcScalar_getCastMode()));
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setScalarCastMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int mode;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"i", &mode)) return NULL;

   /* Set the scalar cast mode */
   OcScalar_setCastMode(mode);

   Py_RETURN_NONE;
}



/* ===================================================================== */
/* Function definitions - Tensor configuration                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getAutoBroadcast(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long)(OcTensor_getAutoBroadcastMode()));
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setAutoBroadcast(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int mode;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"i", &mode)) return NULL;

   /* Set the automatic tensor extension mode */
   OcTensor_setAutoBroadcastMode(mode);

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getAutoTypecast(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long)(OcTensor_getAutoTypecastMode()));
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setAutoTypecast(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  int mode;

   /* Get the input argument */
   if (!PyArg_ParseTuple(args,"i", &mode)) return NULL;

   /* Set the auto typecast mode */
   OcTensor_setAutoTypecastMode(mode);

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getDefaultMathMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  char buffer[2] = {0x00, 0x00};
   buffer[0] = OcTensor_getDefaultMathMode();
   return PyString_FromString(buffer);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_setDefaultMathMode(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   char mode;

   /* ======================================= */
   /* Syntax: ocean.setDefaultMathMode(mode)   */
   /* ======================================= */
   PyOceanArgs_Init(&param, args, "ocean.setDefaultMathMode");
   PyOceanArgs_GetChar(&param, &mode, 1);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Set the auto typecast mode */
   if (OcTensor_setDefaultMathMode(mode) != 0) return NULL;

   Py_RETURN_NONE;
}



/* ===================================================================== */
/* Function definitions - Import and export data types                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getImportTypes(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return pyOcean_getImportTypes();
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_getExportTypes(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return pyOcean_getExportTypes();
}



/* ===================================================================== */
/* Function definitions - Device and type casting                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_ensure(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *obj, *result = NULL;
   OcDevice    *device;
   OcDType      dtype;
   int          flagInplace = 0;

   /* ================================================================== */
   /* Syntax: ocean.ensure(storage [,device] [,dtype] [,inplace=False])  */
   /* Syntax: ocean.ensure(tensor  [,device] [,dtype] [,inplace=False])  */
   /* Syntax: ocean.ensure(scalar  [,device] [,dtype] [,inplace=False])  */
   /* ================================================================== */
   PyOceanArgs_Init(&param, args, "ocean.ensure");
   PyOceanArgs_GetPyObject(&param, &obj, 1);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the internal ensure function */
   result = pyOceanCore_intrnl_ensure(obj, dtype, device, flagInplace);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   return result;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_cast(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *obj, *result = NULL;
   OcDevice    *device;
   OcDType      dtype;

   /* =============================================== */
   /* Syntax: ocean.cast(storage [,device] [,dtype])  */
   /* Syntax: ocean.cast(tensor  [,device] [,dtype])  */
   /* Syntax: ocean.cast(scalar  [,device] [,dtype])  */
   /* =============================================== */
   PyOceanArgs_Init(&param, args, "ocean.cast");
   PyOceanArgs_GetPyObject(&param, &obj, 1);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the internal cast function */
   result = pyOceanCore_intrnl_cast(obj, dtype, device);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   return result;
}


/* ===================================================================== */
/* Function definitions - Creation of tensor from data                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_asTensor(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyObject    *obj;
   OcDType      dtype;
   OcDevice    *device = NULL;
   OcScalar    *scalar = NULL;
   char         order = 'F';
   PyOceanArgs  param;
   OcTensor    *tensor = NULL;
   int          deepcopy = 0;
   int          result;

   /* ============================================================================== */
   /* Syntax: ocean.asTensor(tensor, [, dtype] [, device] [, deepcopy] )             */
   /* Syntax: ocean.asTensor(tensor-like [, scalar] [, order] [, dtype] [, device] ) */
   /* ============================================================================== */
   PyOceanArgs_Init(&param, args, "ocean.asTensor");
   if (PyOceanArgs_GetTensor(&param, &tensor, 0) == 0)
   {  PyOceanArgs_GetPyObject(&param, &obj, 1);
      PyOceanArgs_GetScalar(&param, &scalar, 0);
      PyOceanArgs_GetTensorStrideType(&param, &order, 0);
   }
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (tensor != NULL) PyOceanArgs_GetBool(&param, &deepcopy, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   if (tensor != NULL)
   {  result = OcTensor_ensure(&tensor, dtype, device, &tensor);
      if ((result != 0) || ((deepcopy) && (OcTensor_detachStorage(tensor) != 0)))
      {  OcXDecrefTensor(tensor); tensor = NULL;
         OcErrorMessage("Unable to create a deep copy of the input tensor");
      }
   }
   else
   {  /* Convert the data to a tensor */
      result = pyOcean_intrnlGetTensorLike(obj, &tensor, scalar, dtype, device, order);
      if (result == 0) OcErrorMessage("Invalid data type for data parameter");
   }

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_zeros(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcSize      *size;
   OcDevice    *device;
   OcDType      dtype;
   int          ndims;

   /* =============================================================== */
   /* Syntax: ocean.zeros(size [, dtype] [, device])                  */
   /* =============================================================== */
   PyOceanArgs_Init(&param, args, "ocean.zeros");
   PyOceanArgs_GetTensorSize(&param, &size, &ndims, 1);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Create the tensor */
   tensor = OcTensor_zeros(ndims, size, dtype, device);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_ones(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcSize      *size;
   OcDevice    *device;
   OcDType      dtype;
   int          ndims;

   /* =============================================================== */
   /* Syntax: ocean.ones(size [, dtype] [, device])                   */
   /* =============================================================== */
   PyOceanArgs_Init(&param, args, "ocean.ones");
   PyOceanArgs_GetTensorSize(&param, &size, &ndims, 1);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Create the tensor */
   tensor = OcTensor_ones(ndims, size, dtype, device);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_full(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcSize      *size;
   OcScalar    *value;
   OcDevice    *device;
   OcDType      dtype;
   int          ndims;

   /* =============================================================== */
   /* Syntax: ocean.full(size, value, [, dtype] [, device])           */
   /* =============================================================== */
   PyOceanArgs_Init(&param, args, "ocean.full");
   PyOceanArgs_GetTensorSize(&param, &size, &ndims, 1);
   PyOceanArgs_GetScalarLike(&param, &value, 1);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Determine the data type */
   if (dtype == OcDTypeNone) dtype = value -> dtype;

   /* Create the tensor and fill it with ones */
   tensor = OcTensor_full(ndims, size, value, dtype, device);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_tensorLike(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcTensor    *result;

   /* ================================= */
   /* Syntax: ocean.tensorLike(tensor)  */
   /* ================================= */

   PyOceanArgs_Init(&param, args, "ocean.tensorLike");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Copy the data */
   result = OcTensor_emptyLike(tensor);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_zerosLike(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcTensor    *result;

   /* ================================ */
   /* Syntax: ocean.zerosLike(tensor)  */
   /* ================================ */

   PyOceanArgs_Init(&param, args, "ocean.zerosLike");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Copy the data */
   result = OcTensor_zerosLike(tensor);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_onesLike(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcTensor    *result;

   /* =============================== */
   /* Syntax: ocean.onesLike(tensor)  */
   /* =============================== */

   PyOceanArgs_Init(&param, args, "ocean.onesLike");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Copy the data */
   result = OcTensor_onesLike(tensor);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_fullLike(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcScalar    *value;
   OcTensor    *result;

   /* ====================================== */
   /* Syntax: ocean.fullLike(tensor, value)  */
   /* ====================================== */

   PyOceanArgs_Init(&param, args, "ocean.fullLike");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   PyOceanArgs_GetScalarLike(&param, &value, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Copy the data */
   result = OcTensor_fullLike(tensor, value);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_arange(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   OcScalar    *value1, *value2, *value3;
   OcScalar    *start, *stop, *step;
   OcDevice    *device;
   OcDType      dtype;

   /* =============================================================== */
   /* Syntax: ocean.arange([start,] stop [,step] [,dtype] [,device])  */
   /* =============================================================== */
   PyOceanArgs_Init(&param, args, "ocean.arange");
   PyOceanArgs_GetScalarLike(&param, &value1, 1);
   PyOceanArgs_GetScalarLike(&param, &value2, 0);
   PyOceanArgs_GetScalarLike(&param, &value3, 0);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Determine the data type */
   if (dtype == OcDTypeNone)
   {  dtype = value1 -> dtype;
      if (value2) dtype = OcDType_getCommonType(dtype, value2 -> dtype);
      if (value3) dtype = OcDType_getCommonType(dtype, value3 -> dtype);
   }

   /* Determine the start, stop, and step parameters */
   start = (value2) ? value1 : NULL;
   stop  = (value2) ? value2 : value1;
   step  = (value3) ? value3 : NULL;

   /* Create the tensor */
   tensor = OcTensor_range(start, stop, step, dtype, device);
   
   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_linspace(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   PyObject      *tensorObj = NULL, *spacingObj = NULL, *tupleObj = NULL;
   OcTensor      *tensor = NULL;
   OcScalar      *value1, *value2, spacing;
   int            flagEndpoint = 1;
   int            flagSpacing = 0;
   long unsigned  num = 50, intervals;
   OcDevice      *device;
   OcDType        dtype;

   /* ===================================================================================== */
   /* Syntax: ocean.linspace(start, stop [,num] [,endpoint [,spacing]] [,dtype] [,device])  */
   /* ===================================================================================== */
   PyOceanArgs_Init(&param, args, "ocean.linear");
   PyOceanArgs_GetScalarLike(&param, &value1, 1);
   PyOceanArgs_GetScalarLike(&param, &value2, 1);
   PyOceanArgs_GetScalarUInt(&param, &num, 0);
   if (PyOceanArgs_GetBool(&param, &flagEndpoint, 0) == 1)
   {  PyOceanArgs_GetBool(&param, &flagSpacing, 0);
   }   
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Determine the data type */
   if (dtype == OcDTypeNone)
   {  dtype = OcDType_getCommonType(value1 -> dtype, value2 -> dtype);
      if (OcDType_isInteger(dtype)) dtype = OcDTypeDouble;
   }

   /* Determine the number of intervals */
   if ((!flagEndpoint) && (num == 0))
   {  OcErrorMessage("Cannot create linspace of zero points without end point");
      goto final;
   }
   else
   {  intervals = (flagEndpoint) ? num-1 : num;
   }

   /* Create the tensor */
   tensor = OcTensor_linspace(value1, value2, num, intervals,
                              (flagSpacing ? &spacing : NULL),
                              dtype, device);

final : ;
   /* Finalize */
   PyOceanArgs_Finalize(&param);

   /* Prepare the tensor object */
   if (tensor == NULL) return NULL;
   tensorObj = PyOceanTensor_Wrap(tensor);
   
   /* Return only the tensor */
   if (!flagSpacing) return tensorObj;

   /* Create the spacing object */
   spacingObj = PyOceanScalar_New(&spacing);

   /* Create the tuple */
   tupleObj = PyTuple_New(2);

   /* Prepare the tuple */
   if ((tensorObj == NULL) || (spacingObj == NULL) || (tupleObj == NULL))
   {  Py_XDECREF(tensorObj);
      Py_XDECREF(spacingObj);
      Py_XDECREF(tupleObj);
      return NULL;
   }
   else
   {  PyTuple_SET_ITEM(tupleObj, 0, tensorObj);
      PyTuple_SET_ITEM(tupleObj, 1, spacingObj);
   }

   return tupleObj;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_eye(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   unsigned long  rows, columns;
   long int       index = 0;
   OcDevice      *device;
   OcDType        dtype;

   /* ================================================================= */
   /* Syntax: ocean.eye(rows [, columns [, index]] [,dtype] [,device])  */
   /* ================================================================= */
   PyOceanArgs_Init(&param, args, "ocean.eye");
   PyOceanArgs_GetScalarUInt(&param, &rows, 1);
   if (PyOceanArgs_GetScalarUInt(&param, &columns, 0) == 1)
   {  PyOceanArgs_GetScalarInt(&param, &index, 0);
   }
   else
   {  columns = rows;
   }
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Create and return the tensor */
   return PyOceanTensor_Wrap(OcTensor_eye(rows, columns, index, dtype, device));
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_diag(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   OcTensor      *tensor, *values;
   long int       index = 0;
   OcDevice      *device;
   OcDType        dtype;

   /* ================================================================= */
   /* Syntax: ocean.diag(values [, index] [,dtype] [,device])           */
   /* ================================================================= */
   PyOceanArgs_Init(&param, args, "ocean.diag");
   PyOceanArgs_GetTensorLike(&param, &values, 1);
   PyOceanArgs_GetScalarInt(&param, &index, 0);
   PyOceanArgs_GetOcDType(&param, &dtype, 0);
   PyOceanArgs_GetOcDevice(&param, &device, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Create the tensor */
   tensor = OcTensor_diagonal(values, index, dtype, device);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_merge(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs     param;
   PyObject       *tensorList;
   Py_int_ssize_t  i, n;
   OcDevice       *device = NULL, *tensorDevice;
   OcDType         dtype = OcDTypeNone, tensorDType;
   OcTensor       *result, *tensor1, *tensor2;
   OcTensor      **buffer = NULL;
   OcSize          tensorSize, tensorSizePrev;
   OcIndex         offset;
   long int        axis;
   int             flagResult;
   int             success = 0;
   int             j;
   
   /* =============================================================== */
   /* Syntax: ocean.merge(tensor-list, axis, result)                  */
   /* Syntax: ocean.merge(tensor-list, axis, [, dtype] [, device])    */
   /* =============================================================== */

   PyOceanArgs_Init(&param, args, "ocean.merge");
   PyOceanArgs_GetTensorList(&param, &tensorList, 1);
   PyOceanArgs_GetScalarInt(&param, &axis, 1);
   if (PyOceanArgs_GetTensorNone(&param, &result, 0) == 0)
   {  PyOceanArgs_GetOcDType(&param, &dtype, 0);
      PyOceanArgs_GetOcDevice(&param, &device, 0);
   }
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Get the first tensor for reference */
   flagResult   = (result != NULL);
   n            = PySequence_Size(tensorList);
   tensor1      = PYOC_GET_TENSOR(PySequence_GetItem(tensorList, 0));
   tensorDType  = tensor1 -> dtype;
   tensorDevice = OcTensor_device(tensor1);
   tensorSize   = tensor1 -> size[axis];

   /* Check consistency */
   for (i = 1; i < n; i++)
   {
      /* Compare the dimensions */
      tensor2 = PYOC_GET_TENSOR(PySequence_GetItem(tensorList, i));
      if (tensor2 -> ndims != tensor1 -> ndims)
      {  OcErrorMessage("Inconsistent tensor dimensions"); goto final;
      }
      for (j = 0; j < tensor1 -> ndims; j++)
      {  if ((j != axis) && (tensor2 -> size[j] != tensor1 -> size[j]))
         {  OcErrorMessage("Inconsistent tensor dimensions"); goto final;
         }
      }

      /* Update the data type and device */
      if (tensor2 -> dtype  != tensorDType) tensorDType = OcDTypeNone;
      if (OcTensor_device(tensor2) != tensorDevice) tensorDevice = NULL;
      
      /* Update the size */
      tensorSizePrev = tensorSize;
      tensorSize += tensor2 -> size[axis];
      if (tensorSize < tensorSizePrev)
      {  OcErrorMessage("Overflow detected while adding dimensions"); goto final;
      }
   }

   /* Create a new result tensor if needed */
   if (result == NULL)
   {  OcSize size[OC_TENSOR_MAX_DIMS];

      /* Determine data type and device */
      if ((tensorDType == OcDTypeNone) || (dtype != OcDTypeNone)) tensorDType = dtype;
      if ((tensorDevice == NULL) || (device != NULL)) tensorDevice = device;

      /* Set the final size */
      for (j = 0; j < tensor1 -> ndims; j++) size[j] = tensor1 -> size[j];
      size[axis] = tensorSize;

      /* Create the new tensor */
      result = OcTensor_create(tensor1 -> ndims, size, NULL, tensorDType, tensorDevice);
      if (result == NULL) goto final;
   }
   else
   {  /* Check the size of the result tensor */
      if (result -> ndims != tensor1 -> ndims)
      {  OcErrorMessage("Invalid number of dimensions in result tensor (%d instead of %d)",
                        result -> ndims, tensor1 -> ndims);
         goto final;
      }
      for (j = 0; j < tensor1 -> ndims; j++)
      {  if (((j == axis) && (result -> size[j] != tensorSize)) ||
             ((j != axis) && (result -> size[j] != tensor1 -> size[j])))
         {  OcErrorMessage("Mismatch in dimension %d (expected %"OC_FORMAT_LU" instead of %"OC_FORMAT_LU")",
                           j, (long int)(j == axis ? tensorSize : tensor1 -> size[j]), (long int)(result -> size[j]));
            goto final;
         }
      }
   }

   /* Immidiately return if all slice sizes add up to zero. */
   if (tensorSize == 0) { success = 1; goto final; }

   /* When the destination axis has zero stride the input slices must  */
   /* be guaranteed to be identical. We do not check the contents of   */
   /* the slice data itself but make sure that all input tensors have  */
   /* stride zero or dimension zero or one for the given axis and all  */
   /* point to the same data (with same device, data type, and byte    */
   /* order). When this condition is satisfied we copy the first slice */
   /* that has a non-zero size.                                        */
   if ((result -> strides[axis] == 0) && (result -> size[axis] > 1))
   {
      /* Tensor slices are already guaranteed to be consistent */
      for (i = 0; i < n; i++)
      {  tensor2 = PYOC_GET_TENSOR(PySequence_GetItem(tensorList,i));
         if (tensor2 -> size[axis] == 0) continue;
         if ((tensor2 -> size[axis] <= 1) || (tensor2 -> strides[axis] == 0))
         {  if (tensor1 == NULL)
            { tensor1 = tensor2; continue; }

            /* Compare the data types */
            if ((tensor2 -> dtype  == tensor1 -> dtype) &&
                (OcTensor_device(tensor2) == OcTensor_device(tensor1)) &&
                (OcTensors_haveSameByteOrder(tensor1, tensor2)))
            {  if (tensor2 -> size[axis] == 0) continue;
               if (tensor1 -> size[axis] == 0) { tensor1 = tensor2; continue; }
               if (OcTensor_data(tensor1) == OcTensor_data(tensor2)) continue;
            }
         }

         /* Potentially inconsistent data */
         OcErrorMessage("Input slices must share identical data when the result tensor has zero stride along the merge axis");
         goto final;
      }

      /* Create the slice (offset is always zero) */
      tensor2 = OcTensor_slice(result, axis, 0, tensor1 -> size[axis]);
      if (tensor2 == NULL) goto final;
      
      /* Copy the data */
      success = (OcTensor_copy(tensor1, tensor2) == 0) ? 1 : 0;
      OcDecrefTensor(tensor2);
      goto final;
   }

   /* When slices of an existing tensor are copied back in a different */
   /* order there is a possible problem that writing one slice of the  */
   /* output overwrites one of the input tensors before it was copied, */
   /* thereby obtaining an undesired result. One way to avoid this is  */
   /* to compare the input tensors pairwise, which would results in    */
   /* an O(n^2) overlap determination complexity in number of tensors. */
   /* To avoid this we instead make copies of all tensors that overlap */
   /* with the output data but are not identical to the corresponding  */
   /* slice.                                                           */

   /* Reserve a buffer for tensors */
   buffer = (OcTensor **)malloc(sizeof(OcTensor *) * n);
   if (buffer == NULL) goto final;
   for (i = 0; i < n; i++) buffer[i] = NULL;

   /* Make all necessary copies */
   offset = 0;
   for (i = 0; i < n; i++)
   {
      /* Get the input tensor */
      tensor1 = PYOC_GET_TENSOR(PySequence_GetItem(tensorList, i));

      /* Check for overlap with the result tensor */
      if (!OcTensors_overlap(tensor1, result))
      {  buffer[i] = OcIncrefTensor(tensor1);
         continue;
      }
      
      /* Create the corresponding slice */
      tensor2 = OcTensor_slice(result, axis, offset, tensor1 -> size[axis]);
      if (tensor2 == NULL) goto final;

      /* Avoid copy if the slices are identical */
      if (OcTensors_match(tensor1, tensor2))
      {  buffer[i] = NULL;
      }
      else
      {  if ((buffer[i] = OcTensor_clone(tensor1)) == NULL)
         {  OcDecrefTensor(tensor2);
            goto final;
         }
      }

      /* Free the intermediate tensor */
      OcDecrefTensor(tensor2);

      /* Update the offset */
      offset += tensor1 -> size[axis];
   }

   /* Copy the slices */
   offset = 0;
   for (i = 0; i < n; i++)
   {
      /* Get the input tensor */
      tensor1 = buffer[i];

      /* Avoid copy when the slices are identical slices */
      if (tensor1 == NULL)
      {  tensor1 = PYOC_GET_TENSOR(PySequence_GetItem(tensorList, i));
         offset += tensor1 -> size[axis];
         continue;
      }

      /* Copy into the corresponding slice */
      tensor2 = OcTensor_slice(result, axis, offset, tensor1 -> size[axis]);
      if ((tensor2 == NULL) || (OcTensor_copy(tensor1, tensor2) != 0))
      {  if (tensor2 != NULL) OcDecrefTensor(tensor2);
         goto final;
      }

      /* Free the intermediate tensor */
      OcDecrefTensor(tensor2);

      /* Update the offset */
      offset += tensor1 -> size[axis];
   }

   /* Success */
   success = 1;

final : ;
   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Clean up the tensor buffer */
   if (buffer)
   {  for (i = 0; i < n; i++)
      {  if (buffer[i]) OcDecrefTensor(buffer[i]);
      }
      free(buffer);
   }

   /* Clean up in case of error */
   if (!success)
   {  if ((!flagResult) && (result)) OcDecrefTensor(result);
      return NULL;
   }

   /* Return result or None */
   if (!flagResult) return PyOceanTensor_Wrap(result); else Py_RETURN_NONE;
}



/* ===================================================================== */
/* Function definitions - Unary tensor operations - unconstrained        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_unary(PyObject *self, PyObject *args)          */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME,X,FLAG,DESC) OC_TEMPLATE_B##FLAG(OPNAME,DESC)
#define OC_TEMPLATE_B1(OPNAME,DESC) /* Empty */
#define OC_TEMPLATE_B0(OPNAME,DESC) \
PyObject *pyOceanCore_##OPNAME(PyObject *self, PyObject *args)  \
{  PyOceanArgs    param; \
   PyObject      *obj; \
   OcScalar       scalar, output; \
   OcTensor      *tensor = NULL, *result = NULL; \
   OcDType        dtype = OcDTypeNone; \
   int            flagResult = 0; \
   int            flagScalar = 0; \
   int            status = -1; \
   \
   /* ================================================================= */ \
   /* Syntax: ocean.opname(scalar)                                      */ \
   /* Syntax: ocean.opname(tensor [,result])                            */ \
   /* Syntax: ocean.opname(tensor [,dtype])                             */ \
   /* ================================================================= */ \
   PyOceanArgs_Init(&param, args, "ocean." # OPNAME); \
   PyOceanArgs_GetPyObject(&param, &obj, 1); \
   if (PyOceanArgs_GetTensorNone(&param, &result, 0) == 1) \
   {  flagResult = 1; \
   } \
   else \
   {  PyOceanArgs_GetOcDType(&param, &dtype, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Check operation type */ \
   if ((PyOceanArgs_Length(&param) == 1) && (pyOcean_isScalar(obj))) \
   {  /* Parse the scalar */ \
      if (pyOcean_getScalar(obj, &scalar) != 1) goto final; \
      \
      /* Apply the scalar operation */ \
      flagScalar = 1; \
      status = OcScalar_##OPNAME(&scalar, &output); \
   } \
   else \
   {  /* Parse the tensor object */ \
      if (pyOcean_getTensorLike(obj, &tensor, OcDTypeNone, NULL) != 1) \
      {  OcErrorMessage("First input argument to ocean." # OPNAME " must be a tensor"); \
         goto final; \
      } \
      \
      /* Create the result if needed */ \
      if (dtype != OcDTypeNone) \
      {  result = OcTensor_create(tensor -> ndims, tensor -> size, NULL, dtype, OcTensor_device(tensor)); \
         if (result == NULL) goto final; \
      } \
      \
      /* Apply the tensor operation */ \
      status = OcTensor_##OPNAME(tensor, &result); \
   } \
   \
final: ; \
   /* Finalize */ \
   OcXDecrefTensor(tensor); \
   PyOceanArgs_Finalize(&param); \
   \
   /* Return result or None */ \
   if (status != 0) \
   {  if (flagResult == 0) OcXDecrefTensor(result); \
      return NULL; \
   } \
   \
   if (flagScalar) return PyOceanScalar_New(&output); \
   if (!flagResult) return PyOceanTensor_Wrap(result); else Py_RETURN_NONE; \
}

#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE_B0
#undef OC_TEMPLATE_B1
#undef OC_TEMPLATE



/* ===================================================================== */
/* Function definitions - Unary tensor operations - domain checks        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_unary(PyObject *self, PyObject *args)          */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME,X,FLAG,DESC) OC_TEMPLATE_B##FLAG(OPNAME,DESC)
#define OC_TEMPLATE_B0(OPNAME,DESC) /* Empty */
#define OC_TEMPLATE_B1(OPNAME,DESC) \
PyObject *pyOceanCore_##OPNAME(PyObject *self, PyObject *args)  \
{  PyOceanArgs    param; \
   PyObject      *obj; \
   OcScalar       scalar, output; \
   OcTensor      *tensor = NULL, *result = NULL; \
   OcDType        dtype = OcDTypeNone; \
   char           mode = OcTensor_getDefaultMathMode(); \
   int            flagResult = 0; \
   int            flagScalar = 0; \
   int            status = -1; \
   \
   /* ================================================================= */ \
   /* Syntax: ocean.opname(scalar)                                      */ \
   /* Syntax: ocean.opname(tensor [,mode] [,result])                    */ \
   /* Syntax: ocean.opname(tensor [,mode] [,dtype])                     */ \
   /* ================================================================= */ \
   PyOceanArgs_Init(&param, args, "ocean." # OPNAME); \
   PyOceanArgs_GetPyObject(&param, &obj, 1); \
   PyOceanArgs_GetChar(&param, &mode, 0); \
   if (PyOceanArgs_GetTensorNone(&param, &result, 0) == 1) \
   {  flagResult = 1; \
   } \
   else \
   {  PyOceanArgs_GetOcDType(&param, &dtype, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Check operation type */ \
   if ((PyOceanArgs_Length(&param) == 1) && (pyOcean_isScalar(obj))) \
   {  /* Parse the scalar */ \
      if (pyOcean_getScalar(obj, &scalar) != 1) goto final; \
      \
      /* Apply the scalar operation */ \
      flagScalar = 1; \
      status = OcScalar_##OPNAME(&scalar, &output); \
   } \
   else \
   {  /* Parse the tensor object */ \
      if (pyOcean_getTensorLike(obj, &tensor, OcDTypeNone, NULL) != 1) \
      {  OcErrorMessage("First input argument to ocean." # OPNAME " must be a tensor"); \
         goto final; \
      } \
      \
      /* Create the result if needed */ \
      if (dtype != OcDTypeNone) \
      {  result = OcTensor_create(tensor -> ndims, tensor -> size, NULL, dtype, OcTensor_device(tensor)); \
         if (result == NULL) goto final; \
      } \
      \
      /* Apply the tensor operation */ \
      status = OcTensor_##OPNAME(tensor, &result, mode); \
   } \
   \
final: ; \
   /* Finalize */ \
   OcXDecrefTensor(tensor); \
   PyOceanArgs_Finalize(&param); \
   \
   /* Return result or None */ \
   if (status != 0) \
   {  if (flagResult == 0) OcXDecrefTensor(result); \
      return NULL; \
   } \
   \
   if (flagScalar) return PyOceanScalar_New(&output); \
   if (!flagResult) return PyOceanTensor_Wrap(result); else Py_RETURN_NONE; \
}

#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE_B0
#undef OC_TEMPLATE_B1
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_add          (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_subtract     (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_scale        (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_divide       (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_trueDivide   (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_floorDivide  (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_mod          (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_fmod         (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_min          (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_max          (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_fmin         (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_fmax         (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_bitwiseAnd   (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_bitwiseOr    (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_bitwiseXor   (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_logicalAnd   (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_logicalOr    (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_logicalXor   (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_bitshiftLeft (PyObject *self, PyObject *args)  */
/* PyObject *pyOceanCore_bitshiftRight(PyObject *self, PyObject *args)  */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs  param; \
   PyObject    *obj1, *obj2; \
   PyObject    *result; \
   OcTensor    *dst; \
   \
   /* ================================================================= */ \
   /* Syntax: ocean.OP(src1, src2 [, dst])                              */ \
   /* ================================================================= */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetPyObject(&param, &obj1, 1); \
   PyOceanArgs_GetPyObject(&param, &obj2, 1); \
   PyOceanArgs_GetTensor(&param, &dst, 0); \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   result = pyOceanCore_intrnl_##OP(obj1, obj2, dst); \
   \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   \
   return result; \
}

OC_TEMPLATE(add)
OC_TEMPLATE(subtract)
OC_TEMPLATE(scale)
OC_TEMPLATE(divide)
OC_TEMPLATE(trueDivide)
OC_TEMPLATE(floorDivide)
OC_TEMPLATE(mod)
OC_TEMPLATE(fmod)
OC_TEMPLATE(min)
OC_TEMPLATE(max)
OC_TEMPLATE(fmin)
OC_TEMPLATE(fmax)
OC_TEMPLATE(bitwiseAnd)
OC_TEMPLATE(bitwiseOr)
OC_TEMPLATE(bitwiseXor)
OC_TEMPLATE(logicalAnd)
OC_TEMPLATE(logicalOr)
OC_TEMPLATE(logicalXor)
OC_TEMPLATE(bitshiftLeft)
OC_TEMPLATE(bitshiftRight)
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_power(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *obj1, *obj2;
   PyObject    *result;
   OcTensor    *dst;
   char         mode = OcTensor_getDefaultMathMode();

   /* ================================================================= */
   /* Syntax: ocean.OP(src1, src2 [, mode] [, dst])                     */
   /* ================================================================= */
   PyOceanArgs_Init(&param, args, "ocean.power");
   PyOceanArgs_GetPyObject(&param, &obj1, 1);
   PyOceanArgs_GetPyObject(&param, &obj2, 1);
   PyOceanArgs_GetChar(&param, &mode, 0);
   PyOceanArgs_GetTensor(&param, &dst, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = pyOceanCore_intrnl_power(obj1, obj2, dst, mode);

   /* Finalize */
   PyOceanArgs_Finalize(&param);

   return result;
}



/* ===================================================================== */
/* Function definitions - Range checks                                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_allLT(PyObject *self, PyObject *args)          */
/* PyObject *pyOceanCore_allLE(PyObject *self, PyObject *args)          */
/* PyObject *pyOceanCore_allGT(PyObject *self, PyObject *args)          */
/* PyObject *pyOceanCore_allGE(PyObject *self, PyObject *args)          */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, OCEAN_OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs    param; \
   OcTensor      *tensor; \
   OcScalar      *value; \
   int            result; \
   \
   /* ====================================================== */ \
   /* Syntax: ocean.OP(tensor, value)                        */ \
   /* ====================================================== */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetTensorLike(&param, &tensor, 1); \
   PyOceanArgs_GetScalar(&param, &value, 1); \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   result = OcTensor_##OCEAN_OP(tensor, value); \
   \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   \
   if (result < 0) return NULL; else return PyBool_FromLong(result); \
}

OC_TEMPLATE(allLT, allLessThan)
OC_TEMPLATE(allLE, allLessEqual)
OC_TEMPLATE(allGT, allGreaterThan)
OC_TEMPLATE(allGE, allGreaterEqual)
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_allInRange(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   OcTensor      *tensor;
   OcScalar      *lower, *upper;
   int            lowerInclusive = 1, upperInclusive = 1;
   int            result;

   /* =============================================================== */
   /* Syntax: ocean.allInRange(tensor, lower [, lowerInclusive=True], */
   /*                                  upper [, upperInclusive=True]) */
   /* =============================================================== */
   PyOceanArgs_Init(&param, args, "ocean.allInRange");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   PyOceanArgs_GetScalarOrNone(&param, &lower, 1);
   PyOceanArgs_GetBool(&param, &lowerInclusive, 0);
   PyOceanArgs_GetScalarOrNone(&param, &upper, 1);
   PyOceanArgs_GetBool(&param, &upperInclusive, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = OcTensor_allInRange(tensor, lower, lowerInclusive, upper, upperInclusive);

   /* Finalize */
   PyOceanArgs_Finalize(&param);

   if (result < 0) return NULL; else return PyBool_FromLong(result);
}



/* ===================================================================== */
/* Function definitions - Reduction operators                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_any      (PyObject *self, PyObject *args)      */
/* PyObject *pyOceanCore_all      (PyObject *self, PyObject *args)      */
/* PyObject *pyOceanCore_allFinite(PyObject *self, PyObject *args)      */
/* PyObject *pyOceanCore_anyInf   (PyObject *self, PyObject *args)      */
/* PyObject *pyOceanCore_anyNaN   (PyObject *self, PyObject *args)      */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, AXIS_OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs    param; \
   PyObject      *result = NULL; \
   OcTensor      *tensor, *dest = NULL; \
   int            axes[OC_TENSOR_MAX_DIMS], n; \
   int            flagAxes, flagDest = 0, v, keepdims = 0; \
   \
   /* ====================================================== */ \
   /* Syntax: ocean.OP(tensor [, axes [,keepdims] [, dest]]) */ \
   /* ====================================================== */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetTensorLike(&param, &tensor, 1); \
   if ((flagAxes = PyOceanArgs_GetAxes(&param, axes, &n, 1, 1, 0)) == 1) \
   {  PyOceanArgs_GetBool(&param, &keepdims, 0); \
      flagDest = PyOceanArgs_GetTensor(&param, &dest, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   if (flagAxes) \
   {  if (OcTensor_##AXIS_OP(tensor, n, axes, keepdims, &dest) != 0) goto final; \
      \
      /* Prepare the result */ \
      if (flagDest) \
      {  Py_INCREF(Py_None); \
         result = Py_None; \
      } \
      else \
      {  result = PyOceanTensor_Wrap(dest); \
      } \
   } \
   else \
   {  if (OcTensor_##OP(tensor, &v) != 0) goto final; \
      result = PyBool_FromLong(v); \
   } \
   \
final : ; \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   return result; \
}

OC_TEMPLATE(any,       axisAny      )
OC_TEMPLATE(all,       axisAll      )
OC_TEMPLATE(allFinite, axisAllFinite)
OC_TEMPLATE(anyInf,    axisAnyInf   )
OC_TEMPLATE(anyNaN,    axisAnyNaN   )
#undef OC_TEMPLATE



/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_nnz   (PyObject *self, PyObject *args)         */
/* PyObject *pyOceanCore_nnzNaN(PyObject *self, PyObject *args)         */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, AXIS_OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs    param; \
   PyObject      *result = NULL; \
   OcTensor      *tensor, *dest = NULL; \
   OcScalar       value; \
   OcUInt64       v; \
   int            axes[OC_TENSOR_MAX_DIMS], n; \
   int            flagAxes, flagDest = 0, keepdims = 0; \
   \
   /* ====================================================== */ \
   /* Syntax: ocean.OP(tensor [, axis [,keepdims] [, dest]]) */ \
   /* ====================================================== */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetTensorLike(&param, &tensor, 1); \
   if ((flagAxes = PyOceanArgs_GetAxes(&param, axes, &n, 1, 1, 0)) == 1) \
   {  PyOceanArgs_GetBool(&param, &keepdims, 0); \
      flagDest = PyOceanArgs_GetTensor(&param, &dest, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   if (flagAxes) \
   {  if (OcTensor_##AXIS_OP(tensor, n, axes, keepdims, &dest) != 0) goto final; \
      \
      /* Prepare the result */ \
      if (flagDest) \
      {  Py_INCREF(Py_None); \
         result = Py_None; \
      } \
      else \
      {  result = PyOceanTensor_Wrap(dest); \
      } \
   } \
   else \
   {  if (OcTensor_##OP(tensor, &v) != 0) goto final; \
      value.dtype = OcDTypeUInt64; \
      OcScalar_fromUInt64(&value, v); \
      result = PyOceanScalar_New(&value); \
   } \
   \
final : ; \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   return result; \
}

OC_TEMPLATE(nnz,    axisNnz   )
OC_TEMPLATE(nnzNaN, axisNnzNaN)
#undef OC_TEMPLATE



/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_sum       (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_prod      (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_sumNaN    (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_prodNaN   (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_sumAbs    (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_sumAbsNaN (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_maximum   (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_minimum   (PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_maximumAbs(PyObject *self, PyObject *args)     */
/* PyObject *pyOceanCore_minimumAbs(PyObject *self, PyObject *args)     */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, AXIS_OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs    param; \
   PyObject      *result = NULL; \
   OcTensor      *tensor, *dest = NULL; \
   OcScalar       value; \
   int            axes[OC_TENSOR_MAX_DIMS], n; \
   int            flagAxes, flagDest = 0, keepdims = 0; \
   \
   /* ====================================================== */ \
   /* Syntax: ocean.OP(tensor [, axes [,keepdims] [, dest]]) */ \
   /* ====================================================== */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetTensorLike(&param, &tensor, 1); \
   if ((flagAxes = PyOceanArgs_GetAxes(&param, axes, &n, 1, 1, 0)) == 1) \
   {  PyOceanArgs_GetBool(&param, &keepdims, 0); \
      flagDest = PyOceanArgs_GetTensor(&param, &dest, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   if (flagAxes) \
   {  if (OcTensor_##AXIS_OP(tensor, n, axes, keepdims, &dest) != 0) goto final; \
      \
      /* Prepare the result */ \
      if (flagDest) \
      {  Py_INCREF(Py_None); \
         result = Py_None; \
      } \
      else \
      {  result = PyOceanTensor_Wrap(dest); \
      } \
   } \
   else \
   {  if (OcTensor_##OP(tensor, &value) != 0) goto final; \
      result = PyOceanScalar_New(&value); \
   } \
   \
final : ; \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   return result; \
}

OC_TEMPLATE(sum,        axisSum       )
OC_TEMPLATE(prod,       axisProd      )
OC_TEMPLATE(sumNaN,     axisSumNaN    )
OC_TEMPLATE(prodNaN,    axisProdNaN   )
OC_TEMPLATE(sumAbs,     axisSumAbs    )
OC_TEMPLATE(sumAbsNaN,  axisSumAbsNaN )
OC_TEMPLATE(maximum,    axisMaximum   )
OC_TEMPLATE(minimum,    axisMinimum   )
OC_TEMPLATE(maximumAbs, axisMaximumAbs)
OC_TEMPLATE(minimumAbs, axisMinimumAbs)
#undef OC_TEMPLATE



/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_norm   (PyObject *self, PyObject *args)        */
/* PyObject *pyOceanCore_normNaN(PyObject *self, PyObject *args)        */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, AXIS_OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs    param; \
   PyObject      *result = NULL; \
   OcTensor      *tensor, *dest = NULL;\
   OcScalar       value; \
   double         p = 2; \
   int            axes[OC_TENSOR_MAX_DIMS], n;\
   int            flagAxes, flagDest = 0, keepdims = 0;\
   \
   /* ================================================================ */ \
   /* Syntax: ocean.norm(tensor [,p=2] [, axis [,keepdims] [, dest]])  */ \
   /* ================================================================ */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetTensorLike(&param, &tensor, 1); \
   PyOceanArgs_GetScalarDouble(&param, &p, 0); \
   if ((flagAxes = PyOceanArgs_GetAxes(&param, axes, &n, 1, 1, 0)) == 1) \
   {  PyOceanArgs_GetBool(&param, &keepdims, 0); \
      flagDest = PyOceanArgs_GetTensor(&param, &dest, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   if (flagAxes) \
   {  if (OcTensor_##AXIS_OP(tensor, p, n, axes, keepdims, &dest) != 0) goto final; \
      \
      /* Prepare the result */ \
      if (flagDest) \
      {  Py_INCREF(Py_None);\
         result = Py_None; \
      } \
      else \
      {  result = PyOceanTensor_Wrap(dest); \
      } \
   } \
   else \
   {  if (OcTensor_##OP(tensor, p, &value) != 0) goto final; \
      result = PyOceanScalar_New(&value); \
   } \
   \
final : ; \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   return result; \
}

OC_TEMPLATE(norm,    axisNorm   )
OC_TEMPLATE(normNaN, axisNormNaN)
#undef OC_TEMPLATE



/* -------------------------------------------------------------------- */
/* PyObject *pyOceanCore_norm1  (PyObject *self, PyObject *args)        */
/* PyObject *pyOceanCore_norm2  (PyObject *self, PyObject *args)        */
/* PyObject *pyOceanCore_normInf(PyObject *self, PyObject *args)        */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, AXIS_OP) \
PyObject *pyOceanCore_##OP(PyObject *self, PyObject *args) \
{  PyOceanArgs    param; \
   PyObject      *result = NULL; \
   OcTensor      *tensor, *dest = NULL;\
   OcScalar       value; \
   int            axes[OC_TENSOR_MAX_DIMS], n;\
   int            flagAxes, flagDest = 0, keepdims = 0;\
   \
   /* ================================================================ */ \
   /* Syntax: ocean.norm(tensor [, axis [,keepdims] [, dest]])         */ \
   /* ================================================================ */ \
   PyOceanArgs_Init(&param, args, "ocean."#OP); \
   PyOceanArgs_GetTensorLike(&param, &tensor, 1); \
   if ((flagAxes = PyOceanArgs_GetAxes(&param, axes, &n, 1, 1, 0)) == 1) \
   {  PyOceanArgs_GetBool(&param, &keepdims, 0); \
      flagDest = PyOceanArgs_GetTensor(&param, &dest, 0); \
   } \
   if (!PyOceanArgs_Success(&param)) return NULL; \
   \
   /* Call the function */ \
   if (flagAxes) \
   {  if (OcTensor_##AXIS_OP(tensor, n, axes, keepdims, &dest) != 0) goto final; \
      \
      /* Prepare the result */ \
      if (flagDest) \
      {  Py_INCREF(Py_None);\
         result = Py_None; \
      } \
      else \
      {  result = PyOceanTensor_Wrap(dest); \
      } \
   } \
   else \
   {  if (OcTensor_##OP(tensor, &value) != 0) goto final; \
      result = PyOceanScalar_New(&value); \
   } \
   \
final : ; \
   /* Finalize */ \
   PyOceanArgs_Finalize(&param); \
   return result; \
}

OC_TEMPLATE(norm1,   axisNorm1 )
OC_TEMPLATE(norm2,   axisNorm2  )
OC_TEMPLATE(normInf, axisNormInf)
#undef OC_TEMPLATE



/* ===================================================================== */
/* Function definitions - Tensor find                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_find(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   OcTensor      *tensor, *result;

   /* =========================== */
   /* Syntax: ocean.find(tensor)  */
   /* =========================== */
   PyOceanArgs_Init(&param, args, "ocean.find");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = OcTensor_find(tensor);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return PyOceanTensor_Wrap(result);
}



/* ===================================================================== */
/* Function definitions - Tensor multiplication                          */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_multiply(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   PyObject      *result;
   OcTensor      *A, *B, *C = NULL;
   char           transA = 'N', transB = 'N';

   /* ========================================================= */
   /* Syntax: ocean.multiply(A [, transA], B [, transB] [, C])  */
   /* ========================================================= */
   PyOceanArgs_Init(&param, args, "ocean.multiply");
   PyOceanArgs_GetTensorLike(&param, &A, 1);
   PyOceanArgs_GetChar(&param, &transA, 0);
   PyOceanArgs_GetTensorLike(&param, &B, 1);
   PyOceanArgs_GetChar(&param, &transB, 0);
   PyOceanArgs_GetTensorLike(&param, &C, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = pyOceanCore_intrnl_gemm(NULL, A, transA, B, transB, NULL, C);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return result;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_gemm(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   PyObject      *alpha, *beta = NULL;
   PyObject      *result;
   OcTensor      *A, *B, *C = NULL;
   char           transA = 'N', transB = 'N';

   /* ================================================================= */
   /* Syntax: ocean.gemm(alpha, A [, transA], B [, transB] [, beta, C])   */
   /* ================================================================= */
   PyOceanArgs_Init(&param, args, "ocean.gemm");
   PyOceanArgs_GetPyObject(&param, &alpha, 1);
   PyOceanArgs_GetTensorLike(&param, &A, 1);
   PyOceanArgs_GetChar(&param, &transA, 0);
   PyOceanArgs_GetTensorLike(&param, &B, 1);
   PyOceanArgs_GetChar(&param, &transB, 0);
   if (PyOceanArgs_GetPyObject(&param, &beta, 0) == 1)
   {  PyOceanArgs_GetTensorLike(&param, &C, 1);
   }
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = pyOceanCore_intrnl_gemm(alpha, A, transA, B, transB, beta, C);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   return result;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_checkSelfOverlap(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcSize      *size;
   OcIndex     *strides;
   OcDType      dtype;
   long int     value;
   int          nSize, nStrides, elemsize = 1;
   int          result;

   /* ================================================================= */
   /* Syntax: ocean.checkSelfOverlap(size, strides, [elemsize=1])       */
   /* Syntax: ocean.checkSelfOverlap(size, strides, dtype)              */
   /* ================================================================= */
   PyOceanArgs_Init(&param, args, "ocean.checkSelfOverlap");
   PyOceanArgs_GetTensorSize(&param, &size, &nSize, 1);
   PyOceanArgs_GetTensorStrides(&param, &strides, &nStrides, 1);
   if (PyOceanArgs_GetOcDType(&param, &dtype, 0) == 1)
   {  elemsize = OcDType_size(dtype);
   }
   else if (PyOceanArgs_GetScalarInt(&param, &value, 0) == 1)
   {  elemsize = (int)value;
   }
   PyOceanArgs_ErrorIf(&param, (nSize != nStrides), "Mismatch in size and stride lengths");
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = OcShape_isSelfOverlapping(nSize, size, strides, elemsize);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   if (result == -1) return NULL;
   if (result == 1) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_checkOverlap(PyObject *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcSize      *size1, *size2;
   OcIndex     *strides1, *strides2;
   OcDType      dtype;
   long int     offset1, offset2, value;
   int          nSize1, nStrides1, elemsize1 = 0;
   int          nSize2, nStrides2, elemsize2 = 0;
   int          result;

   /* ========================================================================== */
   /* Syntax: ocean.checkOverlap(size1, strides1, offset1, {dtype1 | elemsize1}, */
   /*                            size2, strides2, offset2, {dtype1 | elemsize2}) */
   /* ========================================================================== */
   PyOceanArgs_Init(&param, args, "ocean.checkOverlap");

   /* Dimensions #1 */
   PyOceanArgs_GetTensorSize(&param, &size1, &nSize1, 1);
   PyOceanArgs_GetTensorStrides(&param, &strides1, &nStrides1, 1);
   PyOceanArgs_GetScalarInt(&param, &offset1, 1);
   if (PyOceanArgs_GetOcDType(&param, &dtype, 0) == 1)
   {  elemsize1 = OcDType_size(dtype);
   }
   else if (PyOceanArgs_GetScalarInt(&param, &value, 1) == 1)
   {  elemsize1 = (int)value;
   }

   /* Dimensions #2 */
   PyOceanArgs_GetTensorSize(&param, &size2, &nSize2, 1);
   PyOceanArgs_GetTensorStrides(&param, &strides2, &nStrides2, 1);
   PyOceanArgs_GetScalarInt(&param, &offset2, 1);
   if (PyOceanArgs_GetOcDType(&param, &dtype, 0) == 1)
   {  elemsize2 = OcDType_size(dtype);
   }
   else if (PyOceanArgs_GetScalarInt(&param, &value, 1) == 1)
   {  elemsize2 = (int)value;
   }

   /* Parameter checks */
   PyOceanArgs_ErrorIf(&param, (nSize1 != nStrides1), "Mismatch in size and stride lengths");
   PyOceanArgs_ErrorIf(&param, (nSize2 != nStrides2), "Mismatch in size and stride lengths");
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Call the function */
   result = OcShapes_overlap(nSize1, size1, strides1, offset1, elemsize1,
                             nSize2, size2, strides2, offset2, elemsize2);

   /* Finalize */
   PyOceanArgs_Finalize(&param);
   if (result == -1) return NULL;
   if (result == 1) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

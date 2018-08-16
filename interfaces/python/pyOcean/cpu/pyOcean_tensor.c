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
#include "pyOcean_convert.h"
#include "pyOcean_index.h"
#include "pyOcean_compatibility.h"
#include "pyOcean_core.h"
#include "pyOcean_module_core.h"

#include <string.h>


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Standard functions */
static void      pyOcTensor_dealloc    (pyOcTensor *self);
static PyObject *pyOcTensor_new        (PyTypeObject *subtype, PyObject *args, PyObject *kwargs);
static PyObject *pyOcTensor_richcompare(PyObject *o1, PyObject *o2, int opid);
static PyObject *pyOcTensor_str        (pyOcTensor *self);

/* Numeric protocol functions */
static PyObject *pyOcTensor_nb_int                 (pyOcTensor *self);
#if PY_MAJOR_VERSION < 3
static PyObject *pyOcTensor_nb_long                (pyOcTensor *self);
#endif
static PyObject *pyOcTensor_nb_float               (pyOcTensor *self);
static int       pyOcTensor_nb_nonzero             (pyOcTensor *self);
static PyObject *pyOcTensor_nb_negative            (pyOcTensor *self);
static PyObject *pyOcTensor_nb_positive            (pyOcTensor *self);
static PyObject *pyOcTensor_nb_invert              (pyOcTensor *self);

/* Internal math operations */
static PyObject *pyOcTensor_nb_add                 (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_subtract            (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_multiply            (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_remainder           (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_and                 (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_or                  (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_xor                 (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_lshift              (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_rshift              (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_add         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_subtract    (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_multiply    (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_remainder   (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_and         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_or          (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_xor         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_lshift      (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_rshift      (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_power               (PyObject *obj1, PyObject *obj2, PyObject *obj3);
static PyObject *pyOcTensor_nb_inplace_power       (PyObject *obj1, PyObject *obj2, PyObject *obj3);

#ifdef PY_VERSION_2_2
static PyObject *pyOcTensor_nb_true_divide         (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_floor_divide        (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_true_divide (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_floor_divide(PyObject *obj1, PyObject *obj2);
#endif

#if PY_MAJOR_VERSION >= 3
static PyObject *pyOcTensor_nb_scale               (PyObject *obj1, PyObject *obj2);
static PyObject *pyOcTensor_nb_inplace_scale       (PyObject *obj1, PyObject *obj2);
#endif

/* Mapping protocol functions */
static PyObject *pyOcTensor_mp_subscript    (pyOcTensor *self, PyObject *args);
static int       pyOcTensor_mp_ass_subscript(pyOcTensor *self, PyObject *index, PyObject *value);

/* Get and set functions */
static PyObject *pyOcTensor_getdevice        (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getdtype         (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getstorage       (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getobject        (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getptr           (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getndims         (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getsize          (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getstrides       (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getoffset        (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getelemsize      (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getnelem         (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getreal          (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getimag          (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getbyteswapped   (pyOcTensor *self, void *closure);
static int       pyOcTensor_setbyteswapped   (pyOcTensor *self, PyObject *value, void *closure);
static PyObject *pyOcTensor_getreadonly      (pyOcTensor *self, void *closure);
static int       pyOcTensor_setreadonly      (pyOcTensor *self, PyObject *value, void *closure);
static PyObject *pyOcTensor_getrefcount      (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getfooter        (pyOcTensor *self, void *closure);

static PyObject *pyOcTensor_gettranspose     (pyOcTensor *self, void *closure);
static PyObject *pyOcTensor_getctranspose    (pyOcTensor *self, void *closure);


/* Member functions */
static PyObject *pyOcTensor_copy             (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_clone            (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_replicate        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_convertTo        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_sync             (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_detach           (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_deallocate       (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_shallowCopy      (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_asContiguous     (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_asScalar         (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_asPython         (pyOcTensor *self, PyObject *args);

static PyObject *pyOcTensor_complex          (pyOcTensor *self, PyObject *args);

/* Member functions -- Shape and layout */
static PyObject *pyOcTensor_reshape          (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_reshapeLike      (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_broadcastTo      (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_broadcastLike    (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_flipAxis         (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_fliplr           (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_flipud           (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_transpose        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_ctranspose       (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_swapAxes         (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_reverseAxes      (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_reverseAxes2     (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_permuteAxes      (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_squeeze          (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_unsqueeze        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_flatten          (pyOcTensor *self, PyObject *args);

/* Member functions -- Subtensors */
static PyObject *pyOcTensor_diag             (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_slice            (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_split            (pyOcTensor *self, PyObject *args);

/* Tensor operations */
static PyObject *pyOcTensor_byteswap         (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_zero             (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_fill             (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_fillNaN          (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_conj             (pyOcTensor *self, PyObject *args);

/* Tensor properties */
static PyObject *pyOcTensor_isEmpty          (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isScalar         (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isContiguous     (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isLinear         (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isAligned        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isFortran        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isReal           (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isComplex        (pyOcTensor *self, PyObject *args);
static PyObject *pyOcTensor_isSelfOverlapping(pyOcTensor *self, PyObject *args);


/* ===================================================================== */
/* Module setup                                                          */
/* ===================================================================== */

struct PyGetSetDef py_oc_tensor_getseters[] = {
   {"device",      (getter)pyOcTensor_getdevice,       NULL, "device type",                NULL},
   {"dtype",       (getter)pyOcTensor_getdtype,        NULL, "data type",                  NULL},
   {"storage",     (getter)pyOcTensor_getstorage,      NULL, "storage",                    NULL},
   {"obj",         (getter)pyOcTensor_getobject,       NULL, "pointer to OcTensor object", NULL},
   {"ptr",         (getter)pyOcTensor_getptr,          NULL, "pointer to the data",        NULL},
   {"ndims",       (getter)pyOcTensor_getndims,        NULL, "number of dimensions",       NULL},
   {"size",        (getter)pyOcTensor_getsize,         NULL, "tensor size",                NULL},
   {"strides",     (getter)pyOcTensor_getstrides,      NULL, "strides",                    NULL},
   {"offset",      (getter)pyOcTensor_getoffset,       NULL, "offset",                     NULL},
   {"elemsize",    (getter)pyOcTensor_getelemsize,     NULL, "element size",               NULL},
   {"nelem",       (getter)pyOcTensor_getnelem,        NULL, "number of elements",         NULL},
   {"real",        (getter)pyOcTensor_getreal,         NULL, "real part of the data",      NULL},
   {"imag",        (getter)pyOcTensor_getimag,         NULL, "imaginary part of the data", NULL},
   {"T",           (getter)pyOcTensor_gettranspose,    NULL, "transpose",                  NULL},
   {"H",           (getter)pyOcTensor_getctranspose,   NULL, "conjugate transpose",        NULL},
   {"byteswapped", (getter)pyOcTensor_getbyteswapped,
                   (setter)pyOcTensor_setbyteswapped,        "byteswapped",                NULL},
   {"readonly",    (getter)pyOcTensor_getreadonly,
                   (setter)pyOcTensor_setreadonly,           "read-only",                  NULL},
   {"refcount",    (getter)pyOcTensor_getrefcount,     NULL, "reference count",            NULL},
   {"footer",      (getter)pyOcTensor_getfooter,       NULL, "tensor footer string",       NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef py_oc_tensor_methods[] = {
   {"clone",             (PyCFunction)pyOcTensor_clone,             METH_VARARGS, "Clone tensor"},
   {"replicate",         (PyCFunction)pyOcTensor_replicate,         METH_VARARGS, "Replicate tensor"},
   {"copy",              (PyCFunction)pyOcTensor_copy,              METH_VARARGS, "Copy tensor"},
   {"convertTo",         (PyCFunction)pyOcTensor_convertTo,         METH_VARARGS, "Convert tensor to another format"},
   {"sync",              (PyCFunction)pyOcTensor_sync,              METH_NOARGS,  "Synchronizes the tensor"},
   {"detach",            (PyCFunction)pyOcTensor_detach,            METH_NOARGS,  "Ensure that the tensor storage is not shared"},
   {"dealloc",           (PyCFunction)pyOcTensor_deallocate,        METH_NOARGS,  "Deallocates the tensor data"},
   {"shallowCopy",       (PyCFunction)pyOcTensor_shallowCopy,       METH_NOARGS,  "Create a shallow copy of the tensor"},
   {"asContiguous",      (PyCFunction)pyOcTensor_asContiguous,      METH_VARARGS, "Convert tensor to C or Fortran contiguous"},
   {"asScalar",          (PyCFunction)pyOcTensor_asScalar,          METH_NOARGS,  "Convert single-element tensor to scalar"},
   {"asPython",          (PyCFunction)pyOcTensor_asPython,          METH_VARARGS, "Convert tensor to a Python object"},
   {"__complex__",       (PyCFunction)pyOcTensor_complex,           METH_NOARGS,  "Cast single-element tensor to Python complex"},
   {"reshape",           (PyCFunction)pyOcTensor_reshape,           METH_VARARGS, "Reshape the tensor"},
   {"reshapeLike",       (PyCFunction)pyOcTensor_reshapeLike,       METH_VARARGS, "Reshape the tensor to the reference tensor size"},
   {"broadcastTo",       (PyCFunction)pyOcTensor_broadcastTo,       METH_VARARGS, "Broadcast the tensor dimensions to match the input size"},
   {"broadcastLike",     (PyCFunction)pyOcTensor_broadcastLike,     METH_VARARGS, "Broadcast the tensor dimensions to match the input tensor"},
   {"flipAxis",          (PyCFunction)pyOcTensor_flipAxis,          METH_VARARGS, "Flip the direction of the given axis"},
   {"fliplr",            (PyCFunction)pyOcTensor_fliplr,            METH_VARARGS, "Flip the direction of the last axis"},
   {"flipud",            (PyCFunction)pyOcTensor_flipud,            METH_VARARGS, "Flip the direction of the first axis"},
   {"transpose",         (PyCFunction)pyOcTensor_transpose,         METH_VARARGS, "Transpose the tensor"},
   {"ctranspose",        (PyCFunction)pyOcTensor_ctranspose,        METH_VARARGS, "Conjugate transpose the tensor"},
   {"swapAxes",          (PyCFunction)pyOcTensor_swapAxes,          METH_VARARGS, "Swap the given axes"},
   {"reverseAxes",       (PyCFunction)pyOcTensor_reverseAxes,       METH_VARARGS, "Reverse the order of the tensor axes"},
   {"reverseAxes2",      (PyCFunction)pyOcTensor_reverseAxes2,      METH_VARARGS, "Reverse the order of the tensor axes and transpose"},
   {"permuteAxes",       (PyCFunction)pyOcTensor_permuteAxes,       METH_VARARGS, "Permute the order of the tensor axes"},
   {"squeeze",           (PyCFunction)pyOcTensor_squeeze,           METH_VARARGS, "Squeeze out one or more unit dimensions"},
   {"unsqueeze",         (PyCFunction)pyOcTensor_unsqueeze,         METH_VARARGS, "Insert a unit dimension"},
   {"flatten",           (PyCFunction)pyOcTensor_flatten,           METH_VARARGS, "Flatten the tensor to a vector"},

   {"diag",              (PyCFunction)pyOcTensor_diag,              METH_VARARGS, "Extract a diagonal of the tensor"},
   {"slice",             (PyCFunction)pyOcTensor_slice,             METH_VARARGS, "Extract a slice of the tensor"},
   {"split",             (PyCFunction)pyOcTensor_split,             METH_VARARGS, "Partition the tensor along an axis"},

   {"byteswap",          (PyCFunction)pyOcTensor_byteswap,          METH_NOARGS,  "Byteswap the tensor data"},
   {"zero",              (PyCFunction)pyOcTensor_zero,              METH_VARARGS, "Zeros out the elements of the tensor"},
   {"fill",              (PyCFunction)pyOcTensor_fill,              METH_VARARGS, "Fill the tensor with a scalar value"},
   {"fillNaN",           (PyCFunction)pyOcTensor_fillNaN,           METH_VARARGS, "Fill all NaN tensor elements with a scalar value"},
   {"conj",              (PyCFunction)pyOcTensor_conj,              METH_NOARGS,  "Elementwise complex conjugate"},

   {"isEmpty",           (PyCFunction)pyOcTensor_isEmpty,           METH_NOARGS,  "Checks if the tensor is empty"},
   {"isScalar",          (PyCFunction)pyOcTensor_isScalar,          METH_NOARGS,  "Checks if the tensor is a scalar"},
   {"isContiguous",      (PyCFunction)pyOcTensor_isContiguous,      METH_NOARGS,  "Checks if the tensor is contiguous"},
   {"isLinear",          (PyCFunction)pyOcTensor_isLinear,          METH_NOARGS,  "Checks if the tensor is linear in memory"},
   {"isAligned",         (PyCFunction)pyOcTensor_isAligned,         METH_NOARGS,  "Checks if the tensor is memory aligned"},
   {"isFortran",         (PyCFunction)pyOcTensor_isFortran,         METH_NOARGS,  "Checks if the tensor is in Fortran order"},
   {"isReal",            (PyCFunction)pyOcTensor_isReal,            METH_NOARGS,  "Checks if the tensor has a non-complex data type"},
   {"isComplex",         (PyCFunction)pyOcTensor_isComplex,         METH_NOARGS,  "Checks if the tensor has a complex data type"},
   {"isSelfOverlapping", (PyCFunction)pyOcTensor_isSelfOverlapping, METH_NOARGS,  "Checks if the tensor is self overlapping in memory"},
   {NULL}  /* Sentinel */
};

static PyNumberMethods  py_oc_tensor_as_number = {0};
static PyMappingMethods py_oc_tensor_as_mapping = {0};

PyTypeObject py_oc_tensor_type = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "ocean.tensor",             /* tp_name      */
   sizeof(pyOcTensor),         /* tp_basicsize */
};

PyTypeObject    *PyOceanTensor;
static OcTensor *py_oc_void_tensor = NULL;
static int       py_oc_tensor_magic;


/* -------------------------------------------------------------------- */
int pyOcTensor_Initialize(void)
/* -------------------------------------------------------------------- */
{  PyNumberMethods  *nb = &py_oc_tensor_as_number;
   PyMappingMethods *mp = &py_oc_tensor_as_mapping;

   /* Construct the tensor type object */
   PyOceanTensor = &py_oc_tensor_type;

   #if PY_MAJOR_VERSION >= 3
   PyOceanTensor -> tp_flags       = Py_TPFLAGS_DEFAULT;
   #else
   PyOceanTensor -> tp_flags       = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES;
   #endif
   PyOceanTensor -> tp_alloc       = PyType_GenericAlloc;
   PyOceanTensor -> tp_dealloc     = (destructor)pyOcTensor_dealloc;
   PyOceanTensor -> tp_new         = (newfunc)pyOcTensor_new;
   PyOceanTensor -> tp_str         = (reprfunc)pyOcTensor_str;
   PyOceanTensor -> tp_repr        = (reprfunc)pyOcTensor_str;
   PyOceanTensor -> tp_richcompare = pyOcTensor_richcompare;
   PyOceanTensor -> tp_getset      = py_oc_tensor_getseters;
   PyOceanTensor -> tp_methods     = py_oc_tensor_methods;
   PyOceanTensor -> tp_as_number   = nb;
   PyOceanTensor -> tp_as_mapping  = mp;
   PyOceanTensor -> tp_doc         = "Ocean tensor";

   /* Number functions */
   nb -> nb_int                  = (unaryfunc )pyOcTensor_nb_int;
   nb -> nb_float                = (unaryfunc )pyOcTensor_nb_float;
   nb -> nb_negative             = (unaryfunc )pyOcTensor_nb_negative;
   nb -> nb_positive             = (unaryfunc )pyOcTensor_nb_positive;
   nb -> nb_invert               = (unaryfunc )pyOcTensor_nb_invert;

   nb -> nb_add                  = (binaryfunc)pyOcTensor_nb_add;
   nb -> nb_subtract             = (binaryfunc)pyOcTensor_nb_subtract;
   nb -> nb_multiply             = (binaryfunc)pyOcTensor_nb_multiply;
   nb -> nb_remainder            = (binaryfunc)pyOcTensor_nb_remainder;
   nb -> nb_and                  = (binaryfunc)pyOcTensor_nb_and;
   nb -> nb_or                   = (binaryfunc)pyOcTensor_nb_or;
   nb -> nb_xor                  = (binaryfunc)pyOcTensor_nb_xor;
   nb -> nb_lshift               = (binaryfunc)pyOcTensor_nb_lshift;
   nb -> nb_rshift               = (binaryfunc)pyOcTensor_nb_rshift;

   nb -> nb_inplace_add          = (binaryfunc)pyOcTensor_nb_inplace_add;
   nb -> nb_inplace_subtract     = (binaryfunc)pyOcTensor_nb_inplace_subtract;
   nb -> nb_inplace_multiply     = (binaryfunc)pyOcTensor_nb_inplace_multiply;
   nb -> nb_inplace_remainder    = (binaryfunc)pyOcTensor_nb_inplace_remainder;
   nb -> nb_inplace_and          = (binaryfunc)pyOcTensor_nb_inplace_and;
   nb -> nb_inplace_or           = (binaryfunc)pyOcTensor_nb_inplace_or;
   nb -> nb_inplace_xor          = (binaryfunc)pyOcTensor_nb_inplace_xor;
   nb -> nb_inplace_lshift       = (binaryfunc)pyOcTensor_nb_inplace_lshift;
   nb -> nb_inplace_rshift       = (binaryfunc)pyOcTensor_nb_inplace_rshift;

   nb -> nb_power                = (ternaryfunc)pyOcTensor_nb_power;
   nb -> nb_inplace_power        = (ternaryfunc)pyOcTensor_nb_inplace_power;

   #ifdef PY_VERSION_2_2
   nb -> nb_true_divide          = (binaryfunc)pyOcTensor_nb_true_divide;
   nb -> nb_floor_divide         = (binaryfunc)pyOcTensor_nb_floor_divide;
   nb -> nb_inplace_true_divide  = (binaryfunc)pyOcTensor_nb_inplace_true_divide;
   nb -> nb_inplace_floor_divide = (binaryfunc)pyOcTensor_nb_inplace_floor_divide;
   #endif

   #if PY_MAJOR_VERSION >= 3
   nb -> nb_matrix_multiply         = (binaryfunc)pyOcTensor_nb_scale;
   nb -> nb_inplace_matrix_multiply = (binaryfunc)pyOcTensor_nb_inplace_scale;
   #endif

   /* Type conversion */
   #if PY_MAJOR_VERSION >= 3
   nb -> nb_bool                = (inquiry   )pyOcTensor_nb_nonzero;
   #else
   nb -> nb_long                = (unaryfunc )pyOcTensor_nb_long;
   nb -> nb_nonzero             = (inquiry   )pyOcTensor_nb_nonzero;
   nb -> nb_divide              = (binaryfunc)pyOcTensor_nb_true_divide;
   nb -> nb_inplace_divide      = (binaryfunc)pyOcTensor_nb_inplace_true_divide;
   #endif

   /* Mapping functions */
   mp -> mp_subscript           = (binaryfunc)pyOcTensor_mp_subscript;
   mp -> mp_ass_subscript       = (objobjargproc)pyOcTensor_mp_ass_subscript;

   /* Finalize the type */
   if (PyType_Ready(PyOceanTensor) < 0) return -1;

   /* Get the magic number for tensor types */
   py_oc_tensor_magic = pyOcOpaque_CreateMagic();

   /* Create a read-only placeholder tensor to deallocate tensors */
   py_oc_void_tensor = OcTensor_createEmpty(OcDTypeInt8, OcCPU);
   if (py_oc_void_tensor == NULL) return -1;
   OcTensor_setReadOnly(py_oc_void_tensor, 1);
   OcStorage_setReadOnly(py_oc_void_tensor -> storage, 1);

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcTensor_InitializeModule(PyObject *module)
/* -------------------------------------------------------------------- */
{
   Py_INCREF(PyOceanTensor); /* Static object - do not delete */
   PyModule_AddObject(module, "tensor", (PyObject *)PyOceanTensor);

   return 0;
}


/* -------------------------------------------------------------------- */
void pyOcTensor_Finalize(void)
/* -------------------------------------------------------------------- */
{
   /* Decrement the void tensor */
   OcXDecrefTensor(py_oc_void_tensor);
}


/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
PyObject *PyOceanTensor_New(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  pyOcTensor   *obj;

   if (tensor == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcTensor *)PyOceanTensor -> tp_alloc(PyOceanTensor,0);
   if (obj == NULL) return NULL;

   /* Set the tensor */
   obj -> tensor = OcIncrefTensor(tensor);

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
PyObject *PyOceanTensor_Wrap(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  pyOcTensor   *obj;

   if (tensor == NULL) return NULL;

   /* Construct the object */
   obj = (pyOcTensor *)PyOceanTensor -> tp_alloc(PyOceanTensor,0);
   if (obj == NULL)
   {  /* Decrement the reference count */
      OcDecrefTensor(tensor);
      return NULL;
    }

   /* Set the tensor (do not increment the reference count) */
   obj -> tensor = tensor;

   return (PyObject *)obj;
}


/* -------------------------------------------------------------------- */
int PyOceanTensor_Check(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (obj == NULL) return 0;

   return PyObject_IsInstance(obj, (PyObject *)PyOceanTensor);
}



/* ===================================================================== */
/* Class functions - standard methods                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void pyOcTensor_dealloc(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{
   if (self -> tensor) OcDecrefTensor(self -> tensor);

   Py_TYPE(self)->tp_free((PyObject *)self);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
/* -------------------------------------------------------------------- */
{  OcDevice    *device = NULL;
   OcStream    *stream = NULL;
   OcStorage   *storage = NULL;
   OcDType      dtype = OcDTypeNone;
   PyOceanArgs  param;
   int          nSize = -1, nStrides = -1;
   char         strideType = 0x00;
   OcSize      *size = NULL;
   OcIndex     *strides = NULL;
   OcIndex      strideBuffer[OC_TENSOR_MAX_DIMS];
   long int     offset = 0;
   long int     unitsize = 0;
   OcTensor    *tensor = NULL;
   int          i;

   /* ========================================================================== */
   /* Syntax: Tensor(size [,strides [, unitsize]] [,dtype] [,device] [, stream]) */
   /* Syntax: Tensor(storage, [,offset, [size [,strides [,unitsize]]]] [,dtype])  */
   /* ========================================================================== */

   /* Make sure there are no keyword arguments */
   if (kwargs != NULL) OcError(NULL, "The tensor constructor does not take keyword arguments");

   PyOceanArgs_Init(&param, args, "ocean.tensor");
   if (PyOceanArgs_GetOcStorage(&param, &storage, 0) == 1)
   {  if (PyOceanArgs_GetScalarInt(&param, &offset, 0) == 1)
      {  if (PyOceanArgs_GetTensorSize(&param, &size, &nSize, 0) == 1)
         {  if ((PyOceanArgs_GetTensorStrides(&param, &strides, &nStrides, 0) == 1) ||
                (PyOceanArgs_GetTensorStrideType(&param, &strideType, 0) == 1))
            {  PyOceanArgs_GetScalarInt(&param, &unitsize, 0);
            }
         }
      }
      PyOceanArgs_GetOcDType(&param, &dtype, 0);
   }
   else if (PyOceanArgs_GetTensorSize(&param, &size, &nSize, 0) == 1)
   {  if (PyOceanArgs_GetTensorStrides(&param, &strides, &nStrides, 0) == 1)
      {  PyOceanArgs_GetScalarInt(&param, &unitsize, 0);
      }
      else PyOceanArgs_GetTensorStrideType(&param, &strideType, 0);
      PyOceanArgs_GetOcDType(&param, &dtype, 0);
      PyOceanArgs_GetOcDevice(&param, &device, 0);
      PyOceanArgs_GetOcStream(&param, &stream, 0);
   }
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Basic parameter checks */
   if ((strides != NULL) && (nStrides != nSize))
   {  OcErrorMessage("Mismatch in stride and size dimensions");
      goto final;
   }
      
   /* Processing */
   if (storage)
   {  /* Determine the data type */
      if (dtype == OcDTypeNone) 
      {  if (!OcStorage_isRaw(storage))
         {  dtype = storage -> dtype;
         }
         else
         {  if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) goto final;
         }
      }

      /* Scale offset and strides by the unit size */
      if (unitsize == 0) unitsize = OcDType_size(dtype);
      for (i = 0; i < nStrides; i++) strides[i] *= unitsize;
      offset *= unitsize;

      /* Convert stride character to strides */
      if ((size != NULL) && (strideType != 0x00))
      {  strides = strideBuffer;
         if (OcShape_getStrides(nSize, size, OcDType_size(dtype), strides, strideType) != 0) goto final;
         nStrides = nSize;
      }
      
      /* Create the tensor from storage */
      tensor = OcTensor_createFromStorage(storage,
                                          (nSize >= 0) ? nSize : -1,
                                          (nSize >  0) ? size : NULL,
                                          (nStrides > 0) ? strides : NULL,
                                          (OcIndex)offset, dtype);
   }
   else
   {
      /* Apply the default data type if needed */
      if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) goto final;

      /* Scale strides by the unit size */
      if (unitsize == 0) unitsize = OcDType_size(dtype);
      for (i = 0; i < nStrides; i++) strides[i] *= unitsize;

      /* Convert stride character to strides */
      if ((size) && (strideType != 0x00))
      {  strides = strideBuffer;
         if (OcShape_getStrides(nSize, size, OcDType_size(dtype), strides, strideType) != 0) goto final;
         nStrides = nSize;
      }

      /* Create the tensor */
      if (stream == NULL)
      {
         /* Create the tensor */
         tensor = OcTensor_create(nSize,
                                  (nSize > 0) ? size : NULL,
                                  (nStrides > 0) ? strides : NULL,
                                  dtype, device);
      }
      else
      {  /* Make sure the device is consistent */
         if ((device != NULL) && (stream -> device != device))
         {  OcErrorMessage("Device does not match the stream device");
            goto final;
         }

         /* Create the tensor */
         tensor = OcTensor_createWithStream(nSize, (nSize > 0) ? size : NULL,
                                            (nStrides > 0) ? strides : NULL,
                                            dtype, stream);
      }
   }


final : ;
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   return PyOceanTensor_Wrap(tensor);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_str(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  PyObject *obj = NULL;
   char     *str = NULL, *footer = NULL;

   /* Format the footer */
   if ((OcTensor_formatFooter(self -> tensor, &footer, "<",">")) != 0) return NULL;

   /* Format the content string */
   if (OcTensor_format(self -> tensor, &str, NULL, footer) != 0) goto final;
   
   /* Create the result string */
   obj = PyString_FromString(str);

final : ;
   /* Free the formatted string */
   if (footer) free(footer);
   if (str) free(str);

   return obj;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_richcompare(PyObject *self, PyObject *obj, int opid)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor1 = NULL, *tensor2 = NULL, *output = NULL;
   OcUInt64  value;
   int       result;

   /* Make sure the object is an Ocean tensor */
   pyOcean_getTensorLike(self, &tensor1, OcDTypeNone, NULL);
   pyOcean_getTensorLike(obj, &tensor2, OcDTypeNone, NULL);

   /* Deal with the case of unrecognized objects */
   if ((tensor1 == NULL) || (tensor2 == NULL))
   {  
      /* Make sure tensor1 is a tensor */
      if (tensor1 == NULL) { tensor1 = tensor2; tensor2 = NULL; }
      if (tensor1 == NULL) OcError(NULL, "Invalid parameters to rich compare");

      /* Compare tensor1 with unrecognized object */
      result = 0;
      switch(opid)
      {  case Py_LT : value = ((void *)self) <  ((void *)obj); break;
         case Py_LE : value = ((void *)self) <= ((void *)obj); break;
         case Py_EQ : value = 0; break;
         case Py_NE : value = 1; break;
         case Py_GE : value = ((void *)self) >= ((void *)obj); break;
         case Py_GT : value = ((void *)self) >  ((void *)obj); break;
         default    : OcErrorMessage("The given comparison operation is not implemented");
                      goto final;
      }

      /* Create the result tensor */
      output = OcTensor_create(tensor1 -> ndims, tensor1 -> size, NULL,
                               OcDTypeBool, tensor1 -> device);
      if (output == NULL) goto final;

      /* Fill the output tensor */
      result = OcTensor_fillUInt64(output, value);
      if (result != 0) { OcDecrefTensor(output); output = NULL; }
   }
   else
   {  /* Evaluate supported comparison operations */
      switch(opid)
      {  case Py_LT : result = OcTensor_elemwiseLT(tensor1, tensor2, &output); break;
         case Py_LE : result = OcTensor_elemwiseLE(tensor1, tensor2, &output); break;
         case Py_EQ : result = OcTensor_elemwiseEQ(tensor1, tensor2, &output); break;
         case Py_NE : result = OcTensor_elemwiseNE(tensor1, tensor2, &output); break;
         case Py_GE : result = OcTensor_elemwiseGE(tensor1, tensor2, &output); break;
         case Py_GT : result = OcTensor_elemwiseGT(tensor1, tensor2, &output); break;
         default    : OcErrorMessage("The given comparison operation is not implemented");
      }
   }

final :
   OcXDecrefTensor(tensor1);
   OcXDecrefTensor(tensor2);

   return PyOceanTensor_Wrap(output);
}



/* ===================================================================== */
/* Class functions - numeric functions                                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
/* PyObject *pyOcTensor_nb_add         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_subtract    (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_multiply    (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_remainder   (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_and         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_or          (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_xor         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_lshift      (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_rshift      (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_true_divide (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_floor_divide(PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_scale       (PyObject *obj1, PyObject *obj2) */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP_NAME, OP, INTRNL_OP) \
PyObject *pyOcTensor_nb_##OP(PyObject *obj1, PyObject *obj2) \
{  \
   return pyOceanCore_intrnl_##INTRNL_OP(obj1, obj2, NULL); \
}

OC_TEMPLATE("+",  add,          add          )
OC_TEMPLATE("-",  subtract,     subtract     )
OC_TEMPLATE("*",  multiply,     mtimes       )
OC_TEMPLATE("%",  remainder,    mod          )
OC_TEMPLATE("&",  and,          bitwiseAnd   )
OC_TEMPLATE("|",  or,           bitwiseOr    )
OC_TEMPLATE("^",  xor,          bitwiseXor   )
OC_TEMPLATE("<<", lshift,       bitshiftLeft )
OC_TEMPLATE(">>", rshift,       bitshiftRight)


#ifdef PY_VERSION_2_2
OC_TEMPLATE("/",  true_divide,  divide       )
OC_TEMPLATE("//", floor_divide, floorDivide  )
#endif

#if PY_MAJOR_VERSION >= 3
OC_TEMPLATE("@",  scale,        scale        )
#endif
#undef OC_TEMPLATE


/* ---------------------------------------------------------------------------- */
/* PyObject *pyOcTensor_nb_inplace_add         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_subtract    (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_multiply    (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_remainder   (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_and         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_or          (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_xor         (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_lshift      (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_rshift      (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_true_divide (PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_floor_divide(PyObject *obj1, PyObject *obj2) */
/* PyObject *pyOcTensor_nb_inplace_scale       (PyObject *obj1, PyObject *obj2) */
/* ---------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP_NAME, OP, INTRNL_OP) \
PyObject *pyOcTensor_nb_##OP(PyObject *obj1, PyObject *obj2) \
{ \
   if (pyOceanCore_intrnl_##INTRNL_OP(PYOC_GET_TENSOR(obj1), obj2) == 0) \
   {  Py_INCREF(obj1); \
      return obj1; \
   } \
   else \
   {  return NULL; \
   } \
}

OC_TEMPLATE("+=",  inplace_add,         iadd          )
OC_TEMPLATE("-=",  inplace_subtract,    isubtract     )
OC_TEMPLATE("*=",  inplace_multiply,    iscale        )
OC_TEMPLATE("%=",  inplace_remainder,   imod          )
OC_TEMPLATE("&=",  inplace_and,         ibitwiseAnd   )
OC_TEMPLATE("|=",  inplace_or,          ibitwiseOr    )
OC_TEMPLATE("^=",  inplace_xor,         ibitwiseXor   )
OC_TEMPLATE("<<=", inplace_lshift,      ibitshiftLeft )
OC_TEMPLATE(">>=", inplace_rshift,      ibitshiftRight)

#ifdef PY_VERSION_2_2
OC_TEMPLATE("/=", inplace_true_divide,  idivide      )
OC_TEMPLATE("//=",inplace_floor_divide, ifloorDivide )
#endif

#if PY_MAJOR_VERSION >= 3
OC_TEMPLATE("@=", inplace_scale,        iscale       )
#endif
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_power(PyObject *obj1, PyObject *obj2, PyObject *obj3)
/* -------------------------------------------------------------------- */
{
   /* Check the modulo parameter */
   if (obj3 != Py_None)
      OcError(NULL,"The modulo parameter in power is not supported");

   /* Call the function */
   return pyOceanCore_intrnl_power(obj1, obj2, NULL, OcTensor_getDefaultMathMode());
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_inplace_power(PyObject *obj1, PyObject *obj2, PyObject *obj3)
/* -------------------------------------------------------------------- */
{  char mode = OcTensor_getDefaultMathMode();

   /* Check the modulo parameter */
   if (obj3 != Py_None)
      OcError(NULL,"The modulo parameter in power is not supported");

   /* Call the function */
   if (pyOceanCore_intrnl_ipower(PYOC_GET_TENSOR(obj1), obj2, mode) == 0)
   {  Py_INCREF(obj1);
      return obj1;
   }
   else
   {  return NULL;
   }
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_int(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   /* Get the scalar value */
   if (OcTensor_toScalar(self -> tensor, &scalar) != 0) return NULL;
   
   /* Convert the data */
   return PyInt_FromLong(OcScalar_asInt64(&scalar));
}


#if PY_MAJOR_VERSION < 3
/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_long(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;
   
   /* Get the scalar value */
   if (OcTensor_toScalar(self -> tensor, &scalar) != 0) return NULL;
   
   /* Convert the data */
   return PyInt_FromLong(OcScalar_asInt64(&scalar));
}
#endif


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_float(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   /* Get the scalar value */
   if (OcTensor_toScalar(self -> tensor, &scalar) != 0) return NULL;
   
   /* Convert the data */
   return PyFloat_FromDouble(OcScalar_asDouble(&scalar));
}


/* -------------------------------------------------------------------- */
static int pyOcTensor_nb_nonzero(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   /* Make sure the tensor has exactly one element */
   if (self -> tensor -> nelem != 1)
   {  OcError(-1, "Cannot convert non-scalar tensors to Boolean (use any or all instead)");
   }

   /* Get the scalar value */
   OcTensor_toScalar(self -> tensor, &scalar);

   /* Convert the data */
   return OcScalar_asBool(&scalar);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_negative(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  OcTensor *result = NULL;

   if (OcTensor_negative(self -> tensor, &result) != 0) return NULL;

   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_positive(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{
   return PyOceanTensor_Wrap(OcTensor_shallowCopy(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_nb_invert(pyOcTensor *self)
/* -------------------------------------------------------------------- */
{  OcTensor *result = NULL;

   if (OcTensor_bitwiseNot(self -> tensor, &result) != 0) return NULL;

   return PyOceanTensor_Wrap(result);
}


/* ===================================================================== */
/* Class functions - get and set functions                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getdevice(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
  return PyOceanDevice_New(OcTensor_device(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getdtype(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanDType_New(OcTensor_dtype(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getstorage(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanStorage_New(OcTensor_storage(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getobject(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanOpaque_New((void *)(self -> tensor), py_oc_tensor_magic);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getptr(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromVoidPtr((void *)(OcTensor_data(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getndims(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> tensor -> ndims);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getsize(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *tuple;
   int       i, n;
   
   /* Create a tupe */
   n = OcTensor_ndims(self -> tensor);
   if ((tuple = PyTuple_New(n)) == NULL) OcError(NULL, "Error allocating size tuple");

   /* Add the dimensions */
   for (i = 0; i < n; i++)
   {  if (PyTuple_SetItem(tuple, i, PyInt_FromLong((long)(self -> tensor -> size[i]))) != 0)
      {  Py_DECREF(tuple);
         OcError(NULL, "Error filling size tuple");
      }
   }

   /* Return the tuple */
   return tuple;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getstrides(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *tuple;
   int       i, n;
   
   /* Create a tuple */
   n = OcTensor_ndims(self -> tensor);
   if ((tuple = PyTuple_New(n)) == NULL) OcError(NULL, "Error allocating strides tuple");

   /* Add the dimensions */
   for (i = 0; i < n; i++)
   {  if (PyTuple_SetItem(tuple, i, PyInt_FromLong((long)(self -> tensor -> strides[i]))) != 0)
      {  Py_DECREF(tuple);
         OcError(NULL, "Error filling strides tuple");
      }
   }

   /* Return the tuple */
   return tuple;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getoffset(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyLong_FromLong((long int)(self -> tensor -> offset));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getelemsize(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> tensor -> elemsize);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getnelem(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong(self -> tensor -> nelem);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getreal(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanTensor_Wrap(OcTensor_real(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getimag(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyOceanTensor_Wrap(OcTensor_imag(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_gettranspose(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcTensor *result = NULL;
   OcTensor_transpose(&(self -> tensor), &result);
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getctranspose(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{  OcTensor *result = NULL;
   OcTensor_ctranspose(&(self -> tensor), &result);
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getbyteswapped(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isByteswapped(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static int pyOcTensor_setbyteswapped(pyOcTensor *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{  OcDevice *device = self -> tensor -> device;
   int flag;

   if (PyBool_Check(value) == 0)
      OcError(1, "Byte-swap value must be Boolean");

   /* Determine the byteswap flag */
   flag = (value == Py_True) ? 1 : 0;
   if (flag && !OcDevice_supportsTensorByteswap(device))
      OcError(1, "Byteswapped tensor data is not supported on device %s", device -> name);
   OcTensor_setByteswapped(self -> tensor, flag);

   return 0;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getreadonly(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isReadOnly(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static int pyOcTensor_setreadonly(pyOcTensor *self, PyObject *value, void *closure)
/* -------------------------------------------------------------------- */
{
   if (PyBool_Check(value) == 0)
      OcError(1, "Read-only value must be Boolean");

   /* Determine the tensor read-only flag - note that the tensor */
   /* read-only flag has no effect if the storage is read-only.  */
   OcTensor_setReadOnly(self -> tensor, (value == Py_True) ? 1 : 0);

   return 0;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getrefcount(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{
   return PyInt_FromLong((long int)(self -> tensor -> refcount));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_getfooter(pyOcTensor *self, void *closure)
/* -------------------------------------------------------------------- */
{  PyObject *obj = NULL;
   char     *footer = NULL;

   /* Format the footer */
   if ((OcTensor_formatFooter(self -> tensor, &footer, "<",">")) != 0) return NULL;

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
static PyObject *pyOcTensor_copy(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   OcTensor    *tensor;
   int          result;

   /* ============================ */
   /* Syntax: tensor.copy(tensor)  */
   /* ============================ */

   PyOceanArgs_Init(&param, args, "Tensor.copy");
   PyOceanArgs_GetTensorLike(&param, &tensor, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Copy the data */
   result = OcTensor_copy(tensor, self -> tensor);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Success */
   if (result != 0) return NULL; else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_clone(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   PyObject      *deviceList = NULL;
   OcDevice      *device = NULL;
   PyObject      *tensor, *obj;
   PyObject      *result;
   Py_int_ssize_t  n, i;
   int             flagList;
   
   /* ========================================= */
   /* Syntax: tensor.clone([device | devices])  */
   /* ========================================= */

   PyOceanArgs_Init(&param, args, "Tensor.clone");
   if ((PyOceanArgs_GetOcDevice(&param, &device, 0) == 0) &&
       (PyOceanArgs_GetOcDeviceList(&param, &deviceList, 0) == 0))
   {  device = self -> tensor -> device;
   }
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Clone the tensor */
   if (deviceList == NULL)
   {  /* Single device */
      result = PyOceanTensor_Wrap(OcTensor_cloneTo(self -> tensor, device));
   }
   else
   {  
      /* Create a new list */
      flagList = PyList_Check(deviceList);
      n = (flagList) ? PyList_Size(deviceList) : PyTuple_Size(deviceList);
      if ((result = PyList_New(n)) == NULL) return NULL;

      /* Fill the list */
      for (i = 0; i < n; i++)
      {  /* Clone the tensor and add it to the result list */
         obj = (flagList) ? PyList_GetItem(deviceList,i) : PyTuple_GetItem(deviceList,i);
         tensor = PyOceanTensor_Wrap(OcTensor_cloneTo(self -> tensor, PYOC_GET_DEVICE(obj)));
         if (tensor == NULL) { Py_DECREF(result); return NULL; }
         PyList_SetItem(result, i, tensor);
      }
   }
   
   /* Return the result */
   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_replicate(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *deviceList = NULL;
   OcDevice    *device = NULL;
   PyObject    *tensor, *obj;
   PyObject    *result;

   /* ======================================== */
   /* Syntax: tensor.replicate([device-list])  */
   /* ======================================== */

   PyOceanArgs_Init(&param, args, "Tensor.replicate");
   if ((PyOceanArgs_GetOcDevice(&param, &device, 0) == 0) &&
       (PyOceanArgs_GetOcDeviceList(&param, &deviceList, 0) == 0))
   {  device = OcTensor_device(self -> tensor);
   }
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Replicate the tensor */
   if (deviceList == NULL)
   {  /* Single device */
      result = PyOceanTensor_Wrap(OcTensor_replicateTo(self -> tensor, device));
   }
   else
   {  Py_int_ssize_t n, i;
      int flagList;

      /* Create a new list */
      flagList = PyList_Check(deviceList);
      n = (flagList) ? PyList_Size(deviceList) : PyTuple_Size(deviceList);
      if ((result = PyList_New(n)) == NULL) return NULL;

      /* Fill the list */
      for (i = 0; i < n; i++)
      {  /* Replicate the tensor and add it to the result list */
         obj = (flagList) ? PyList_GetItem(deviceList,i) : PyTuple_GetItem(deviceList,i);
         tensor = PyOceanTensor_Wrap(OcTensor_replicateTo(self -> tensor, PYOC_GET_DEVICE(obj)));
         if (tensor == NULL) { Py_DECREF(result); return NULL; }
         PyList_SetItem(result, i, tensor);
      }
   }
   
   /* Return the result */
   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_convertTo(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   PyObject    *obj = NULL;
   char        *typename;
   int          deepcopy = 0;
   int          result;

   /* ===================================================== */
   /* Syntax: tensor.convertTo(typename [,deepcopy=False])  */
   /* ===================================================== */

   PyOceanArgs_Init(&param, args, "Tensor.convertTo");
   PyOceanArgs_GetAsciiString(&param, &typename, 1);
   PyOceanArgs_GetBool(&param, &deepcopy, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Export the tensor */
   result = pyOcean_exportTensor(self -> tensor, typename, &obj, deepcopy);
   if (result != 0) obj = NULL;

   /* Finalize the parameters and return the result */
   PyOceanArgs_Finalize(&param);
   return obj;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_sync(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   if (OcTensor_synchronize(self -> tensor) != 0) return NULL;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_detach(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   /* ======================================= */
   /* Syntax: tensor.detach()                 */
   /* ======================================= */

   if (OcTensor_detachStorage(self -> tensor) != 0) return NULL;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_deallocate(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   /* ======================================= */
   /* Syntax: tensor.dealloc()                */
   /* ======================================= */
   OcIncrefTensor(py_oc_void_tensor);
   OcDecrefTensor(self -> tensor);
   self -> tensor = py_oc_void_tensor;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_shallowCopy(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyOceanTensor_Wrap(OcTensor_shallowCopy(self -> tensor));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_asContiguous(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   char        strideType = 'F';

   /* ======================================== */
   /* Syntax: tensor.asContiguous([mode='F'])  */
   /* ======================================== */
   PyOceanArgs_Init(&param, args, "tensor.asContiguous");
   PyOceanArgs_GetTensorStrideType(&param, &strideType, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   return PyOceanTensor_Wrap(OcTensor_contiguousType(self -> tensor, strideType));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_asScalar(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcScalar scalar;

   if (OcTensor_toScalar(self -> tensor, &scalar) != 0) return NULL;

   return PyOceanScalar_New(&scalar);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_intrnlAsPython(OcTensor *tensor, char *ptr, int mode, int index)
/* -------------------------------------------------------------------- */
{  PyObject       *list, *obj;
   Py_int_ssize_t  nelem, i;

   OcScalar        scalar;
   OcIndex         stride;
   char           *data;
   int             idx;
   int             byteswapped;

   /* Supported modes: 0 = Fortran; 1 = C; 2 = row-based Fortran */

   /* Create the list */
   if (index >= 0)
   {  /* Convert the logical index to a real one */
      if (mode == 0)
      {  idx = index;
      }
      else if (mode == 1)
      {  idx = ((tensor -> ndims) - (index + 1));
      }
      else /* if (mode == 2) */
      {  if ((index < 2) && (tensor -> ndims >= 2))
              idx = 1 - index;
         else idx = index;
      }
      
      /* Get the size and strides */
      nelem  = tensor -> size[idx];
      stride = tensor -> strides[idx];
      if ((list = PyList_New(nelem)) == NULL) return NULL;
   }
   else
   {  /* Scalar case */
      nelem  = 1;
      stride = 0;
      list   = NULL;
   }

   /* --- General level --- */
   if (index > 0)
   {  for (i = 0; i < nelem; i++)
      {  obj  = pyOcTensor_intrnlAsPython(tensor, ptr, mode, index-1);
         if (obj == NULL) { Py_DECREF(list); return NULL; }
         PyList_SetItem(list, i, obj);
         ptr += stride;         
      }
      return list;
   }

   /* --- Lowest level --- */
   
   /* Initialize the scalar */
   scalar.dtype = tensor -> dtype;
   byteswapped = OcTensor_hasHostByteOrder(tensor) ? 0 : 1;
   data = ptr;

   /* Boolean scalars */
   if (tensor -> dtype == OcDTypeBool)
   {  for (i = 0; i < nelem; i++)
      {  obj = ((*((OcBool *)data)) == 0) ? Py_False : Py_True;
         Py_INCREF(obj);
         if (list)
         {  PyList_SetItem(list, i, obj);
            data += stride;
         }
         else
         {  list = obj;
         }
      }
      return list;
   }

   /* Integer scalars */
   if (!OcDType_isFloat(tensor -> dtype))
   {  for (i = 0; i < nelem; i++)
      {  OcScalar_importData(&scalar, data, byteswapped);
         obj = PyInt_FromLong(OcScalar_asInt64(&scalar));
         if (list)
         {  if (obj == NULL) { Py_DECREF(list); return NULL; }
            PyList_SetItem(list, i, obj);
            data += stride;
         }
         else
         {  list = obj;
         }
      }
      return list;
   }

   /* Float scalars */
   if (!OcDType_isComplex(tensor -> dtype))
   {  for (i = 0; i < nelem; i++)
      {  OcScalar_importData(&scalar, data, byteswapped);
         obj = PyFloat_FromDouble(OcScalar_asDouble(&scalar));
         if (list)
         {  if (obj == NULL) { Py_DECREF(list); return NULL; }
            PyList_SetItem(list, i, obj);
            data += stride;
         }
         else
         {  list = obj;
         }
      }
      return list;
   }

   /* Complex scalars */
   {  OcCDouble value;

      for (i = 0; i < nelem; i++)
      {  OcScalar_importData(&scalar, data, byteswapped);
         value = OcScalar_asCDouble(&scalar);
         obj = PyComplex_FromDoubles(value.real, value.imag);
         if (list)
         {  if (obj == NULL) { Py_DECREF(list); return NULL; }
            PyList_SetItem(list, i, obj);
            data += stride;
         }
         else
         {  list = obj;
         }
      }
      return list;
   }
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_asPython(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcTensor   *tensor;
   PyObject   *result;
   char        strideType = 'F';
   int         mode;

   /* ======================================== */
   /* Syntax: tensor.asPython([mode='F'])  */
   /* ======================================== */
   PyOceanArgs_Init(&param, args, "tensor.asPython");
   PyOceanArgs_GetTensorStrideType(&param, &strideType, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Convert stride type to mode */
   if (strideType == 'F')
      mode = 0;
   else if (strideType == 'C')
      mode = 1;
   else if (strideType == 'R')
      mode = 2;
   else 
      OcError(NULL, "Unrecognized stride type in tensor.asPython");

   /* Cast tensor to CPU */
   tensor = OcTensor_castDevice(self -> tensor, OcCPU);
   if (tensor == NULL) return NULL;

   /* Synchronize the tensor */
   OcTensor_synchronize(tensor);

   /* Create the Python object */
   result = pyOcTensor_intrnlAsPython(tensor, OcTensor_data(tensor), mode, tensor -> ndims - 1);

   /* Decref the tensor */
   OcDecrefTensor(tensor);

   return result;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_complex(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcScalar  scalar;
   OcCDouble value;

   /* Create a scalar object */
   if (OcTensor_toScalar(self -> tensor, &scalar) != 0) return NULL;

   /* Convert the data */
   value = OcScalar_asCDouble(&scalar);
   return PyComplex_FromDoubles(value.real, value.imag);
}



/* ===================================================================== */
/* Member functions -- Shape and layout                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_reshape(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcSize      buffer[OC_TENSOR_MAX_DIMS];
   OcSize     *size;
   long int    value;
   int         ndims;
   OcTensor   *result = NULL;
   int         flagInplace = 0;
   int         status;

   /* ===================================================================== */
   /* Syntax: tensor.reshape(size [, inplace = False])                      */
   /* Syntax: tensor.reshape(dim1 [,dim2 [, dim3 ...] [, inplace = False])  */
   /* ===================================================================== */
   PyOceanArgs_Init(&param, args, "tensor.reshape");

   ndims = 0; size = buffer;
   while ((status = PyOceanArgs_GetScalarInt(&param, &value, 0)) == 1)
   {  if (ndims == OC_TENSOR_MAX_DIMS)
      {  PyOceanArgs_ErrorIf(&param, 1, "Size exceeds the maximum number of tensor dimensions");
      }
      else
      {  buffer[ndims] = value;
         ndims++;
      }
   }
   if (ndims == 0) PyOceanArgs_GetTensorSize(&param, &size, &ndims, 1);

   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Reshape the tensor */
   status = OcTensor_reshape(&(self -> tensor), ndims, (ndims > 0) ? size : NULL, &result);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   if (status != 0) return NULL;
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_reshapeLike(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcTensor   *reference;
   OcTensor   *result;
   int         flagInplace = 0;
   int         status;

   /* ========================================================== */
   /* Syntax: tensor.reshapeLike(reference [, inplace = False])  */
   /* ========================================================== */

   PyOceanArgs_Init(&param, args, "tensor.broadcastLike");
   PyOceanArgs_GetTensorLike(&param, &reference, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Reshape the tensor */
   status = OcTensor_reshapeLike(&(self -> tensor), reference, &result);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   if (status != 0) return NULL;
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_broadcastTo(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcSize     *size;
   int         ndims;
   OcTensor   *result;
   long int    mode = 0;
   int         flagInplace = 0;
   int         status;

   /* ================================================================== */
   /* Syntax: tensor.broadcastTo(size [, mode = 0] [, inplace = False])  */
   /* ================================================================== */

   PyOceanArgs_Init(&param, args, "tensor.broadcastTo");
   PyOceanArgs_GetTensorSize(&param, &size, &ndims, 1);
   PyOceanArgs_GetScalarInt(&param, &mode, 0);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Broadcast the tensor dimensions */
   status = OcTensor_broadcastTo(&(self -> tensor), ndims, size, mode, &result);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   if (status != 0) return NULL;
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_broadcastLike(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcTensor   *reference;
   OcTensor   *result;
   long int    mode = 0;
   int         flagInplace = 0;
   int         status;

   /* ========================================================================= */
   /* Syntax: tensor.broadcastLike(reference [, mode = 0] [, inplace = False])  */
   /* ========================================================================= */

   PyOceanArgs_Init(&param, args, "tensor.broadcastLike");
   PyOceanArgs_GetTensorLike(&param, &reference, 1);
   PyOceanArgs_GetScalarInt(&param, &mode, 0);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Broadcast the tensor dimensions */
   status = OcTensor_broadcastLike(&(self -> tensor), reference, mode, &result);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   if (status != 0) return NULL;
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_flipAxis(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   long int    axis;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* ================================================== */
   /* Syntax: tensor.flipAxis(axis [, inplace = False])  */
   /* ================================================== */

   PyOceanArgs_Init(&param, args, "tensor.flipAxis");
   PyOceanArgs_GetScalarInt(&param, &axis, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Flip the tensor axis */
   status = OcTensor_flipAxis(&(self -> tensor), axis, &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_fliplr(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* =========================================== */
   /* Syntax: tensor.flipAxis([inplace = False])  */
   /* =========================================== */

   PyOceanArgs_Init(&param, args, "tensor.fliplr");
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Flip the tensor left-right */
   status = OcTensor_fliplr(&(self -> tensor), &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_flipud(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* =========================================== */
   /* Syntax: tensor.flipAxis([inplace = False])  */
   /* =========================================== */

   PyOceanArgs_Init(&param, args, "tensor.flipud");
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Flip the tensor up-down */
   status = OcTensor_flipud(&(self -> tensor), &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_transpose(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* ============================================ */
   /* Syntax: tensor.transpose([inplace = False])  */
   /* ============================================ */

   PyOceanArgs_Init(&param, args, "tensor.transpose");
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Transpose the tensor */
   status = OcTensor_transpose(&(self -> tensor), &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_ctranspose(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* ============================================= */
   /* Syntax: tensor.ctranspose([inplace = False])  */
   /* ============================================= */

   PyOceanArgs_Init(&param, args, "tensor.ctranspose");
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Transpose the tensor */
   status = OcTensor_ctranspose(&(self -> tensor), &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_swapAxes(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   long int    axis1, axis2;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* =========================================================== */
   /* Syntax: tensor.swapAxes(axis1, axis2, [, inplace = False])  */
   /* =========================================================== */

   PyOceanArgs_Init(&param, args, "tensor.swapAxes");
   PyOceanArgs_GetScalarInt(&param, &axis1, 1);
   PyOceanArgs_GetScalarInt(&param, &axis2, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Flip the tensor axis */
   status = OcTensor_swapAxes(&(self -> tensor), axis1, axis2, &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_reverseAxes(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* ============================================== */
   /* Syntax: tensor.reverseAxes([inplace = False])  */
   /* ============================================== */

   PyOceanArgs_Init(&param, args, "tensor.reverseAxes");
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Reverse the tensor dimensions */
   status = OcTensor_reverseAxes(&(self -> tensor), &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_reverseAxes2(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   int         flagInplace = 0;
   OcTensor   *result;
   int         status;

   /* =============================================== */
   /* Syntax: tensor.reverseAxes2([inplace = False])  */
   /* =============================================== */

   PyOceanArgs_Init(&param, args, "tensor.reverseAxes2");
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Reverse the tensor dimensions */
   status = OcTensor_reverseAxes2(&(self -> tensor), &result);
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_permuteAxes(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs     param;
   Py_int_ssize_t  n;
   OcIndex        *dims = NULL;
   int             flagInplace = 0;
   OcTensor       *result;
   int             status;

   /* ===================================================== */
   /* Syntax: tensor.permuteAxes(order [,inplace = False])  */
   /* ===================================================== */

   PyOceanArgs_Init(&param, args, "tensor.reverseAxes");
   PyOceanArgs_GetIndexList(&param, &dims, &n, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   PyOceanArgs_ErrorIf(&param, (n > OC_TENSOR_MAX_DIMS), "Invalid number of dimensions");
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Transpose the tensor */
   status = OcTensor_permuteAxes(&(self -> tensor), (int)n, dims, &result);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Check for success */
   if (status != 0) return NULL;

   /* Return the result */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_squeeze(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   long int     axis = 0;
   int          flagAxis, flagInplace = 0;
   OcTensor    *result;
   int          status;

   /* =========================================================== */
   /* Syntax: tensor.squeeze([axis] [, inplace])                  */
   /* =========================================================== */
   PyOceanArgs_Init(&param, args, "tensor.squeeze");
   flagAxis = PyOceanArgs_GetScalarInt(&param, &axis, 0);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Squeeze the tensor */
   if (flagAxis)
   {  status = OcTensor_squeezeDim(&(self -> tensor), axis, &result);
   }
   else
   {  status = OcTensor_squeeze(&(self -> tensor), &result);
   }

   /* Return the tensor or None */
   if (status != 0) return NULL;
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_unsqueeze(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   long int     axis = 0;
   int          flagInplace = 0;
   OcTensor    *result;
   int          status;

   /* =========================================================== */
   /* Syntax: tensor.unsqueeze(axis [, inplace])                  */
   /* =========================================================== */
   PyOceanArgs_Init(&param, args, "tensor.unsqueeze");
   PyOceanArgs_GetScalarInt(&param, &axis, 1);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Unsqueeze the tensor */
   status = OcTensor_unsqueezeDim(&(self -> tensor), axis, &result);
   if (status != 0) return NULL;

   /* Return the tensor or None */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_flatten(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   char         type = 'F';
   int          flagInplace = 0;
   OcTensor    *result;
   int          status;

   /* =========================================================== */
   /* Syntax: tensor.flatten([type] [, inplace])                  */
   /* =========================================================== */
   PyOceanArgs_Init(&param, args, "tensor.flatten");
   PyOceanArgs_GetChar(&param, &type, 0);
   PyOceanArgs_GetBool(&param, &flagInplace, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Flatten the tensor */
   status = OcTensor_flatten(&(self -> tensor), type, &result);
   if (status != 0) return NULL;

   /* Return the tensor or None */
   if (!flagInplace) return PyOceanTensor_Wrap(result);

   /* Replace the existing tensor */
   OcDecrefTensor(self -> tensor);
   self -> tensor = result;
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_mp_subscript(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index;
   OcTensor      *tensor;
   OcScalar       scalar;
   OcIndex        indices[OC_TENSOR_MAX_DIMS];
   int            flagScalar, status;

   /* Check for direct extraction of a single element */
   if (pyOcean_convertScalarIndices(args, self -> tensor -> ndims, indices) == 1)
   {  status = OcTensor_getIndexValue(self -> tensor, indices, &scalar);
      return (status == 0) ? PyOceanScalar_New(&scalar) : NULL;
   }

   /* Convert the indices */
   if (pyOcean_convertIndices(args, &index) != 0) return NULL;

   /* Get the tensor index */
   tensor = OcTensor_getIndexFlags(self -> tensor, index, &flagScalar, NULL);

   /* Free the indices */
   OcDecrefTensorIndex(index);

   if ((flagScalar) && (tensor != NULL))
   {  status = OcTensor_toScalar(tensor, &scalar);
      OcDecrefTensor(tensor);
      return (status == 0) ? PyOceanScalar_New(&scalar) : NULL;
   }
   else
   {  return PyOceanTensor_Wrap(tensor);
   }
}


/* -------------------------------------------------------------------- */
static int pyOcTensor_mp_ass_subscript(pyOcTensor *self, PyObject *args, PyObject *value)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index;
   OcTensor      *tensor = NULL;
   OcScalar       scalar;
   OcIndex        indices[OC_TENSOR_MAX_DIMS];
   int            resultScalar;
   int            result = 0;

   /* Check if value is a scalar */
   resultScalar = pyOcean_getScalar(value, &scalar);

   /* Check for direct assignment of a single element */
   if ((resultScalar == 1) &&
       (pyOcean_convertScalarIndices(args, self -> tensor -> ndims, indices) == 1))
   {  result = OcTensor_setIndexValue(self -> tensor, indices, &scalar);
      return (result == 0) ? 0 : -1;
   }

   /* Convert the indices */
   if (pyOcean_convertIndices(args, &index) != 0) return -1;

   /* Parse the value */
   if (resultScalar != 0)
   {  if (resultScalar == 1)
           result = OcTensor_fillIndex(self -> tensor, index, &scalar);
      else result = -1;
   }
   else if ((result = pyOcean_getTensorLike(value, &tensor, OcDTypeNone, NULL)) != 0)
   {  if (result == 1)
         result = OcTensor_setIndex(self -> tensor, index, tensor);
   }
   else
   {  OcErrorMessage("Invalid type for tensor assignment");
      result = -1;
   }

   /* Free intermediate data */
   if (tensor) OcDecrefTensor(tensor);
   OcDecrefTensorIndex(index);

   return result;
}




/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_diag(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;
   long int     axis1 = 0;
   long int     axis2 = 1;
   long int     offset = 0;
   OcTensor    *result;

   /* =========================================================== */
   /* Syntax: tensor.diag([offset = 0 [, axis1 = 0, axis2 = 1]])  */
   /* =========================================================== */
   PyOceanArgs_Init(&param, args, "tensor.diag");
   if (PyOceanArgs_GetScalarInt(&param, &offset, 0) == 1)
   {  PyOceanArgs_GetScalarInt(&param, &axis1, 0);
      PyOceanArgs_GetScalarInt(&param, &axis2, 0);
   }
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Extract and return the diagonal */
   result = OcTensor_diag(PYOC_GET_TENSOR(self), (OcIndex)offset, (int)axis1, (int)axis2);

   /* Return the result */
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_slice(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs    param;
   long int       axis;
   unsigned long  offset;
   unsigned long  size = 1;
   OcTensor       *result;

   /* =========================================================== */
   /* Syntax: tensor.slice(axis, offset [, size = 1])             */
   /* =========================================================== */

   PyOceanArgs_Init(&param, args, "tensor.slice");
   PyOceanArgs_GetScalarInt(&param, &axis, 1);
   PyOceanArgs_GetScalarUInt(&param, &offset, 1);
   PyOceanArgs_GetScalarUInt(&param, &size, 0);
   if (!PyOceanArgs_Finalize(&param)) return NULL;

   /* Extract and return the diagonal */
   result = OcTensor_slice(PYOC_GET_TENSOR(self), axis, offset, size);

   /* Return the result */
   return PyOceanTensor_Wrap(result);
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_split(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs     param;
   PyObject       *result = NULL, *obj;
   PyObject       *deviceList = NULL;
   Py_int_ssize_t  n, i;
   OcSize         *sizes = NULL, s, r, k;
   OcIndex         offset;
   OcDevice       *device;
   OcTensor       *tensor, *t1;
   long int        axis = 0;
   unsigned long   parts = 0;
   int             detach = 0;
   int             success = 1;


   /* =============================================================== */
   /* Syntax: tensor.split(axis, parts [, detach=False])              */
   /* Syntax: tensor.split(axis, sizes [, detach=False])              */
   /* Syntax: tensor.split(axis, devices [sizes, ] [, detach=False])  */
   /* =============================================================== */

   PyOceanArgs_Init(&param, args, "tensor.split");
   PyOceanArgs_GetScalarInt(&param, &axis, 1);
   if (PyOceanArgs_GetScalarUInt(&param, &parts, 0) == 1)
   {  n = parts;
   }
   else if (PyOceanArgs_GetOcDeviceList(&param, &deviceList, 0) == 1)
   {  if (PyOceanArgs_GetSizeList(&param, &sizes, &n, 0) == 1)       
      {  PyOceanArgs_ErrorIf(&param, (n != PySequence_Size(deviceList)), "Mismatch in number of devices and sizes");
      }
      else
      {  n = PySequence_Size(deviceList);
      }
   }
   else if (PyOceanArgs_GetSizeList(&param, &sizes, &n, 1) == 1)
   {  /* Done */
   }
   PyOceanArgs_GetBool(&param, &detach, 0);
   PyOceanArgs_ErrorIf(&param, (n <= 0), "Number of parts must be positive");
   if (!PyOceanArgs_Success(&param)) return NULL;

   /* Make sure that the sizes add up to the tensor dimension */
   tensor = PYOC_GET_TENSOR(self);
   s = tensor -> size[axis];
   if (sizes)
   {  offset = 0;
      for (i = 0; i < n; i++)
      {  if (s - offset < sizes[i]) /* Avoid overflows */
         {  OcErrorMessage("Sum of sizes exceeds the axis dimension (%"OC_FORMAT_LU")", (unsigned long)s);
            goto final;
         }
         offset += sizes[i];
      }
   }

   /* Create the result list */
   if ((result = PyList_New(n)) == NULL) goto final;

   /* Create the slices */
   device = OcTensor_device(tensor);
   offset = 0;
   r = s % n;
   s /= n;
   for (i = 0; i < n; i++)
   {
      /* Determine the device */
      if (deviceList) device = PYOC_GET_DEVICE(PySequence_GetItem(deviceList,i));

      /* Determine the slice size */
      if (sizes)
           k = sizes[i];
      else k = s + ((i < r) ? 1 : 0);

      /* Create the tensor */
      t1 = OcTensor_slice(tensor, axis, offset, k);
      if (t1 == NULL) { success = 0; break; }

      if (device != OcTensor_device(tensor))
      {  if (OcTensor_ensureDevice(&t1, device, NULL) != 0)
         {  success = 0; break;
         }
      }
      else if (detach)
      {  if (OcTensor_detachStorage(t1) != 0)
         {  OcDecrefTensor(t1);
            success = 0; break;
         }
      }

      /* Wrap the tensor */
      obj = PyOceanTensor_Wrap(t1);
      if (obj == NULL) { success = 0; break; }

      /* Add the tensor to the tuple */
      PyList_SetItem(result, i, obj);

      /* Update the offset */
      offset += k;
   }

   if (success == 0)
   {  Py_DECREF(result); result = NULL;
      goto final;
   }

final : ;
   PyOceanArgs_Finalize(&param);

   /* Return the result */
   return result;
}


/* ===================================================================== */
/* Member functions -- Tensor operations                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_byteswap(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   if (OcTensor_byteswap(self -> tensor) != 0) return NULL;

   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_zero(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs  param;

   /* ====================== */
   /* Syntax: tensor.zero()  */
   /* ====================== */

   PyOceanArgs_Init(&param, args, "tensor.zero");
   if (!PyOceanArgs_Finalize(&param)) return NULL;
 
   /* Fill the storage with the given value */
   if (OcTensor_zero(PYOC_GET_TENSOR(self)) != 0) return NULL;

   /* Return */
   Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_fill(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcScalar   *scalar;
   OcTensor   *mask = NULL;
   int         result = -1;

   /* ==================================== */
   /* Syntax: tensor.fill(scalar [,mask])  */
   /* ==================================== */

   PyOceanArgs_Init(&param, args, "tensor.fill");
   PyOceanArgs_GetScalarLike(&param, &scalar, 1);
   PyOceanArgs_GetTensorLike(&param, &mask, 0);
   if (!PyOceanArgs_Success(&param)) return NULL;
 
   /* Fill the storage with the given value */
   if (mask == NULL)
        result = OcTensor_fill(PYOC_GET_TENSOR(self), scalar);
   else result = OcTensor_maskedFill(PYOC_GET_TENSOR(self), mask, scalar);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return */
   if (result != 0) return NULL; else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_fillNaN(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  PyOceanArgs param;
   OcScalar   *scalar;
   int         result = -1;

   /* =============================== */
   /* Syntax: tensor.fillNaN(scalar)  */
   /* =============================== */

   PyOceanArgs_Init(&param, args, "tensor.fillNaN");
   PyOceanArgs_GetScalarLike(&param, &scalar, 1);
   if (!PyOceanArgs_Success(&param)) return NULL;
 
   /* Fill the storage with the given value */
   result = OcTensor_fillNaN(PYOC_GET_TENSOR(self), scalar);

   /* Finalize the parameters */
   PyOceanArgs_Finalize(&param);

   /* Return */
   if (result != 0) return NULL; else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_conj(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{  OcTensor *dst = NULL;
   int result;

   /* ====================== */
   /* Syntax: tensor.conj()  */
   /* ====================== */

   /* Fill the storage with the given value */
   result = OcTensor_conj(PYOC_GET_TENSOR(self), &dst);

   /* Return */
   return (result != 0) ? NULL : PyOceanTensor_Wrap(dst);
}



/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isEmpty(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isEmpty(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isScalar(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isScalar(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isContiguous(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isContiguous(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isLinear(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isLinear(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isAligned(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcTensor_isAligned(self -> tensor)));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isFortran(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)OcTensor_hasOrder(self -> tensor, 'F'));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isReal(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)(OcDType_isComplex(self -> tensor -> dtype) ? 0 : 1));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isComplex(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)OcDType_isComplex(self -> tensor -> dtype));
}


/* -------------------------------------------------------------------- */
static PyObject *pyOcTensor_isSelfOverlapping(pyOcTensor *self, PyObject *args)
/* -------------------------------------------------------------------- */
{
   return PyBool_FromLong((long int)OcTensor_isSelfOverlapping(self -> tensor));
}

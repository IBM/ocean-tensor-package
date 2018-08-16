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


/* ===================================================================== */
/* Local defines                                                         */
/* ===================================================================== */

#define PYOC_FLAGS_SIZE_FIXED        0x0001
#define PYOC_FLAGS_CONTAINS_SCALAR   0x0002


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

static pyOceanConvert *py_ocean_convert_types = NULL;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

int pyOcean_getTensorLikeLayoutFlags(PyObject *obj, int *ndims, OcSize *size,
                                     OcDType *dtype, OcDevice **device,
                                     int *flags, int padded, int verbose);
int pyOcean_intrnlGetTensorLayout (PyObject *obj, int *ndims, OcSize *size,
                                   int offset, OcDType *dtype, OcDevice **device,
                                   int *flags, int padded, int verbose);
int pyOcean_intrnlImportTensorData(PyObject *obj, OcTensor *tensor, int padded,
                                   OcDeviceType *deviceType, int flagExclude);



/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int pyOcean_isScalar(PyObject *obj)   
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;

   /* Python types */
   if (PyBool_Check(obj)) return 1;
   if (PyInt_Check(obj)) return 1;
   if (PyLong_Check(obj)) return 1;
   if (PyFloat_Check(obj)) return 1;
   if (PyComplex_Check(obj)) return 1;

   /* Ocean scalar */
   if (PyOceanScalar_Check(obj)) return 1;

   /* Registered types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if ((type -> isScalar) && (type -> isScalar(obj))) return 1;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_isWeakScalar(PyObject *obj)   
/* -------------------------------------------------------------------- */
{
   /* Python types */
   if (PyBool_Check(obj)) return 0;
   if (PyInt_Check(obj)) return 1;
   if (PyLong_Check(obj)) return 1;
   if (PyFloat_Check(obj)) return 1;
   if (PyComplex_Check(obj)) return 1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_isScalarLike(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (pyOcean_isScalar(obj)) return 1;
   if (pyOcean_isScalarTensorLike(obj)) return 1;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_isScalarTensor(PyObject *obj)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;

   /* Ocean tensor */
   if (PyOceanTensor_Check(obj))
   {  return (PYOC_GET_TENSOR(obj) -> nelem == 1) ? 1 : 0;
   }

   /* Registered types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if ((type -> isScalarTensor) && (type -> isScalarTensor(obj))) return 1;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_isScalarTensorLike(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   /* Python list */
   if (PyList_Check(obj))
   {  if (PyList_GET_SIZE(obj) != 1)
           return 0;
      else return pyOcean_isScalarLike(PyList_GET_ITEM(obj, 0));
   }

   /* Python tuple */
   if (PyTuple_Check(obj))
   {  if (PyTuple_GET_SIZE(obj) != 1)
           return 0;
      else return pyOcean_isScalarLike(PyTuple_GET_ITEM(obj, 0));
   }

   /* Scalar */
   if (pyOcean_isScalar(obj)) return 1;

   /* All other types */
   return pyOcean_isScalarTensor(obj);
}


/* -------------------------------------------------------------------- */
int pyOcean_isTensor(PyObject *obj)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;

   /* Ocean tensor */
   if (PyOceanTensor_Check(obj)) return 1;

   /* Registered types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if ((type -> isTensor) && (type -> isTensor(obj))) return 1;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_isTensorLike(PyObject *obj)
/* -------------------------------------------------------------------- */
{
   if (pyOcean_isScalar(obj)) return 1;

   return pyOcean_isTensorLikeOnly(obj);
}


/* -------------------------------------------------------------------- */
int pyOcean_isTensorLikeOnly(PyObject *obj)
/* -------------------------------------------------------------------- */
{  int       ndims;
   OcSize    size[OC_TENSOR_MAX_DIMS];
   OcDType   dtype;
   OcDevice *device;
   int       result;

   /* Tensor */
   if (pyOcean_isTensor(obj)) return 1;

   /* Python sequences and iterators */
   if (PySequence_Check(obj) || PyIter_Check(obj))
   {
      /* Exclude the Unicode type */
      if (PyUnicode_Check(obj)) return 0;

      result = pyOcean_getTensorLikeLayout(obj, &ndims, size, &dtype, &device, 0, 0);
      return (result == 1) ? 1 : 0;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_getScalar(PyObject *obj, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   int result = 1;

   /* Ocean scalar */
   if (PyOceanScalar_Check(obj))
   {  *scalar = *(PYOC_GET_SCALAR(obj));
      return 1;
   }

   /* Python types */
   PyErr_Clear();
   if (PyFloat_Check(obj))
   {  scalar -> dtype = OcDTypeDouble;
      OcScalar_fromDouble(scalar, PyFloat_AsDouble(obj));
   }
   else if (PyBool_Check(obj))
   {  scalar -> dtype = OcDTypeBool;
      OcScalar_fromInt64(scalar, (obj == Py_True) ? 1 : 0);
   }
   else if (PyInt_Check(obj))
   {  scalar -> dtype = OcDTypeInt64;
      OcScalar_fromInt64(scalar, PyInt_AsLong(obj));
   }
   else if (PyLong_Check(obj))
   {  scalar -> dtype = OcDTypeInt64;
      OcScalar_fromInt64(scalar, PyLong_AsLong(obj));
   }
   else if (PyComplex_Check(obj))
   {  scalar -> dtype = OcDTypeCDouble;
      OcScalar_fromComplex(scalar, PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
   }
   else
   {  result = 0;
   }
   if (result == 1)
   {  if (PyErr_Occurred())
      {  scalar -> dtype = OcDTypeNone;
         OcError(-1, "Error converting Python scalar");
      }

      return 1;
   }
   
   /* Registered type */
   scalar -> dtype = OcDTypeNone;
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if (type -> getScalar)
      {  result = type -> getScalar(obj, scalar);
         if (result != 0) break;
      }
   }

   return result;
}


/* -------------------------------------------------------------------- */
int pyOcean_getScalarLike(PyObject *obj, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  int result;

   if ((result = pyOcean_getScalar(obj, scalar)) != 0) return result;
   if ((result = pyOcean_getScalarTensorLike(obj, scalar)) != 0) return result;

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_getScalarTensor(PyObject *obj, OcTensor **tensor)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   OcTensor *t;
   int result;

   /* Ocean tensor */
   if (PyOceanTensor_Check(obj))
   {  t = PYOC_GET_TENSOR(obj);
      if (t -> nelem != 1) return 0;
      *tensor = OcIncrefTensor(t);
      return 1;
   }

   /* Registered types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if (type -> getScalarTensor)
      {  result = type -> getScalarTensor(obj, tensor);
         if (result != 0) return result;
      }
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_getScalarTensorLike(PyObject *obj, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor;
   int result;

   /* Python list */
   if (PyList_Check(obj))
   {  if (PyList_GET_SIZE(obj) != 1)
           return 0;
      else return pyOcean_getScalarLike(PyList_GET_ITEM(obj, 0), scalar);
   }

   /* Python tuple */
   if (PyTuple_Check(obj))
   {  if (PyTuple_GET_SIZE(obj) != 1)
           return 0;
      else return pyOcean_getScalarLike(PyTuple_GET_ITEM(obj, 0), scalar);
   }

   /* Get a scalar tensor */
   result = pyOcean_getScalarTensor(obj, &tensor);
   if (result != 1) return result;
   if (OcTensor_toScalar(tensor, scalar) != 0) result = -1;
   OcDecrefTensor(tensor);
   return result;
}


/* -------------------------------------------------------------------- */
int pyOcean_getTensor(PyObject *obj, OcTensor **tensor)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   int result;

   /* Ocean tensor */
   if (PyOceanTensor_Check(obj))
   {  *tensor = OcIncrefTensor(PYOC_GET_TENSOR(obj));
      return 1;
   }

   /* Registered types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if (type -> getTensor)
      {  result = type -> getTensor(obj, tensor);
         if (result != 0) return result;
      }
   }
   
   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_getTensorLike(PyObject *obj, OcTensor **tensor, OcDType dtype, OcDevice *device)
/* -------------------------------------------------------------------- */
{  return pyOcean_intrnlGetTensorLike(obj, tensor, NULL, dtype, device, 'F');
}


/* -------------------------------------------------------------------- */
int pyOcean_intrnlGetTensorLike(PyObject *obj, OcTensor **tensor, OcScalar *padding,
                                OcDType dtype, OcDevice *device, char order)
/* -------------------------------------------------------------------- */
{  OcScalar s;
   int      result;

   /* When padding is not NULL tensors at the lowest level can  */
   /* differ in size. The given order applies only to newly     */
   /* constructed tensors if the object is already a tensor we  */
   /* just ensure that the data type and device match.          */

   /* Scalar */
   if ((result = pyOcean_getScalar(obj, &s)) != 0)
   {  if (result != 1) return result;
      
      /* Set data type and device */
      if (device == NULL) device = OcCPU;
      if (dtype == OcDTypeNone) dtype = s.dtype;

      /* Create the tensor */
      *tensor = OcTensor_createFromScalar(&s, dtype, device, 0);
      if ((*tensor) == NULL) return -1;

      return 1;
   }
   
   /* Tensor types */
   result = pyOcean_getTensor(obj, tensor);
   if (result != 0)
   {  if (result != 1) return result;

      /* Set data type and device */
      if (dtype == OcDTypeNone) dtype = (*tensor) -> dtype;
      if (device == NULL) device = OcTensor_device(*tensor);

      /* Ignore the requested data order */
      return (OcTensor_ensure(tensor, dtype, device, NULL) == 0) ? 1 : -1;
   }

   /* Tensor like */
   return pyOcean_getTensorLikeOnly(obj, tensor, padding, dtype, device, order);
}


/* -------------------------------------------------------------------- */
int pyOcean_getTensorLikeOnly(PyObject *obj, OcTensor **tensor, OcScalar *padding,
                              OcDType dtype, OcDevice *device, char order)
/* -------------------------------------------------------------------- */
{  OcTensor *t;
   OcSize    size[OC_TENSOR_MAX_DIMS];
   OcIndex   strides[OC_TENSOR_MAX_DIMS];
   OcDType   tensorDType;
   OcDevice *tensorDevice = NULL;
   int       tensorFlags;
   int       ndims;
   int       padded = (padding == NULL) ? 0 : 1;
   int       result;

   /* =============================== */
   /* Python sequences and iterators  */
   /* =============================== */
   if (PySequence_Check(obj) || PyIter_Check(obj))
   {
      /* Exclude the Unicode type */
      if (PyUnicode_Check(obj)) return 0;

      /* Analyze the object */
      result = pyOcean_getTensorLikeLayoutFlags(obj, &ndims, size,
                                                &tensorDType, &tensorDevice,
                                                &tensorFlags, padded, 1);
      if (result != 1) return -1;

      /* Determine the final device type */
      if (device == NULL)
      {   device = tensorDevice ? tensorDevice : OcCPU;
      }

      /* Determine the intermediate device type */
      if (tensorFlags & PYOC_FLAGS_CONTAINS_SCALAR)
           tensorDevice = OcCPU;
      else tensorDevice = device;
 
      /* Determine the data type */
      if (dtype != OcDTypeNone)
      {  tensorDType = dtype;
      }
      else if (tensorDType == OcDTypeNone)
      {  /* Empty tensor - use smallest possible data type */
         tensorDType = OcDTypeBool;
      }

      /* Allocate the tensor on the intermediate device type */
      if ((OcShape_getStrides(ndims, size, OcDType_size(tensorDType), strides, order) != 0) ||
          ((t = OcTensor_create(ndims, size, strides, tensorDType, tensorDevice)) == NULL))
      {  return -1;
      }

      /* Set the default value */
      if (padded)
      {  result = OcTensor_fill(t, padding);
         if (result != 0) { OcDecrefTensor(t); return -1; }
         OcTensor_synchronize(t);
      }

      /* Set the tensor data */
      if (tensorDevice -> type != device -> type)
      {  /* Exclude tensors that already reside on the destination device */
         result = pyOcean_intrnlImportTensorData(obj, t, padded, device -> type, 1);
      }
      else
      {  /* Import all data */
         result = pyOcean_intrnlImportTensorData(obj, t, padded, NULL, 0);
      }
      if (result != 0) { OcDecrefTensor(t); return -1; }
      
      /* Ensure that the device is correct */
      if ((device != NULL) && (OcTensor_device(t) != device))
      {  OcTensor_ensureDevice(&t, device, tensor);
         OcDecrefTensor(t);
      }
      else
      {  *tensor = t;
      }

      /* Set additional tensor data */
      if ((*tensor) && (tensorDevice -> type != device -> type))
      {  /* Include only tensors that already reside on the destination device */
         result = pyOcean_intrnlImportTensorData(obj, *tensor, padded, device -> type, 0);
         if (result != 0) { OcDecrefTensor(*tensor); *tensor = NULL; }
      }

      /* Transpose the tensor if needed */
      if ((*tensor) && ((order == 'R') || (order == 'r')))
      {  OcTensor_transpose(tensor, NULL);
      }
      if ((*tensor) && ((order == 'C') || (order == 'c')))
      {  OcTensor_reverseAxes(tensor, NULL);
      }

      /* Success */
      return (*tensor == NULL) ? -1 : 1;
   }
   
   return 0;
}


/* ===================================================================== */
/* Functions for tensor indexing                                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int pyOcean_convertIndex(PyObject *obj, OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcScalar           scalar1, *ptrScalar1 = &scalar1;
   OcScalar           scalar2, *ptrScalar2 = &scalar2;
   OcScalar           scalar3, *ptrScalar3 = &scalar3;
   OcTensor          *tensor;
   int                result;

   if (PyOceanTensorIndex_Check(obj))
   {  /* Merge the tensor index elements */
      result = OcTensorIndex_addIndex(index, PYOC_GET_TENSOR_INDEX(obj));
      return result;
   }

   if (PySlice_Check(obj))
   {  /* Check for all */
      if ((((PySliceObject *)obj) -> start == Py_None) &&
          (((PySliceObject *)obj) -> stop  == Py_None) &&
          (((PySliceObject *)obj) -> step  == Py_None))
      {  result = OcTensorIndex_addAll(index,1);
         return (result >= 0) ? 0 : -1;
      }

      /* Parse the start object */
      if (((PySliceObject *)obj) -> start == Py_None)
      {  ptrScalar1 = NULL;
      }
      else if (pyOcean_getScalar(((PySliceObject *)obj) -> start, &scalar1) != 1)
      {  OcError(-1, "Invalid type for slice start");
      }

      /* Parse the stop object */
      if (((PySliceObject *)obj) -> stop == Py_None)
      {  ptrScalar2 = NULL;
      }
      else if (pyOcean_getScalar(((PySliceObject *)obj) -> stop, &scalar2) != 1)
      {  OcError(-1, "Invalid type for slice stop");
      }

      /* Parse the step object */
      if (((PySliceObject *)obj) -> step == Py_None)
      {  ptrScalar3 = NULL;
      }
      else if (pyOcean_getScalar(((PySliceObject *)obj) -> step, &scalar3) != 1)
      {  OcError(-1, "Invalid type for slice step");
      }

      /* Add a new range index */
      result = OcTensorIndex_addRange(index, ptrScalar1, ptrScalar2, ptrScalar3);
   }
   else if (obj == Py_Ellipsis)
   {
      result = OcTensorIndex_addEllipsis(index);
   }
   else if (obj == Py_None)
   {
      result = OcTensorIndex_addInsert(index, 1);
   }
   else if ((result = pyOcean_getScalar(obj, &scalar1)) != 0)
   {  if (result == -1) return result;
      result = OcTensorIndex_addScalar(index, &scalar1);
   }
   else if ((result = pyOcean_getTensorLike(obj, &tensor, OcDTypeNone, NULL)) != 0)
   {  if (result == -1) return result;
      result = OcTensorIndex_addTensor(index, tensor);
      OcDecrefTensor(tensor);
   }
   else
   {  OcError(-1, "Unrecognized index type");
   }

   return (result >= 0) ? 0 : -1;
}


/* -------------------------------------------------------------------- */
int pyOcean_convertIndices(PyObject *args, OcTensorIndex **index)
/* -------------------------------------------------------------------- */
{  Py_int_ssize_t i,n;
   OcTensorIndex *idx;
   int result = 0;

   /* Determine the number of indices */
   if (PyTuple_Check(args))
   {  n = PyTuple_Size(args);
   }
   else
   {  n = 1;
   }

   /* Sanity check in the number of indices */
   if (n > OC_TENSOR_MAX_DIMS * 2)
   {  OcError(-1, "Too many indices");
   }

   /* Create the index */
   idx = OcTensorIndex_createWithCapacity((int)n);
   if (idx == NULL) return -1;

   /* Convert the indices */
   if (PyTuple_Check(args))
   {  
      for (i = 0; i < n; i++)
      {  if (pyOcean_convertIndex(PyTuple_GetItem(args, i), idx) != 0)
         {  result = -1; break;  }
      }
   }
   else
   {  result = pyOcean_convertIndex(args, idx);
   }

   /* Check for errors */
   if (result == 0)
        *index = idx;
   else OcDecrefTensorIndex(idx);

   return result;
}


/* -------------------------------------------------------------------- */
int pyOcean_convertScalarIndices(PyObject *args, int ndims, OcIndex *indices)
/* -------------------------------------------------------------------- */
{  PyObject *obj;
   int i;

   /* -------------------------------------- */
   /* Fast mode for indexing single elements */
   /* -------------------------------------- */

   /* Make sure the input argument is a tuple with ndims elements */
   if ((!PyTuple_Check(args)) || (PyTuple_Size(args) != ndims)) return 0;

   /* Check all indices */
   for (i = 0; i < ndims; i++)
   {  obj = PyTuple_GetItem(args, i);

      if (PyInt_Check(obj))
      {  indices[i] = PyInt_AsLong(obj);
      }
      else if (PyLong_Check(obj))
      {  indices[i] = PyLong_AsLong(obj);
      }
      else return 0;
   }

   return 1;
}


/* ===================================================================== */
/* Data type and layout functions                                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int pyOcean_getScalarType(PyObject *obj, OcDType *dtype, int verbose)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   OcDType _dtype;
   int result = 1;
   
   /* Initialize */
   if (dtype == NULL) dtype = &_dtype;

   /* Ocean scalar */
   if (PyOceanScalar_Check(obj))
   {  *dtype = PYOC_GET_SCALAR(obj) -> dtype;
   }

   /* Python types */
   else if (PyFloat_Check(obj))
   {  *dtype = OcDTypeDouble;  /* Double */
   }
   else if (PyBool_Check(obj))
   {  *dtype = OcDTypeBool;    /* Boolean */
   }
   else if ((PyInt_Check(obj)) || (PyLong_Check(obj)))
   {  *dtype = OcDTypeInt64;   /* Long integer */
   }
   else if (PyComplex_Check(obj))
   {  *dtype = OcDTypeCDouble; /* Complex */
   }

   /* Registered type */
   else
   {  result = 0;
      for (type = py_ocean_convert_types; type != NULL; type = type -> next)
      {  if (type -> getScalarType)
         {  result = type -> getScalarType(obj, dtype, verbose);
            if (result != 0) return result;
         }
      }
   }

   return result;
}


/* -------------------------------------------------------------------- */
int pyOcean_getTensorLayout(PyObject *obj, int *ndims, OcSize *size,
                            OcIndex *strides, OcDType *dtype,
                            OcDevice **device, int verbose)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   OcTensor       *tensor;
   int             i, result;

   /* Ocean tensor */
   if (PyOceanTensor_Check(obj))
   {  
      /* Extract the tensor */
      tensor = PYOC_GET_TENSOR(obj);

      /* Set the size and strides */
      for (i = 0; i < tensor -> ndims; i++)
      {  if (size) size[i] = tensor -> size[i];
         if (strides) strides[i] = tensor -> strides[i];
      }

      /* Copy the remaining fields */
      if (ndims)  *ndims  = tensor -> ndims;
      if (dtype)  *dtype  = OcTensor_dtype(tensor);
      if (device) *device = OcTensor_device(tensor);

      return 1;
   }

   /* Registered tensor types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  if (type -> getTensorLayout)
      {  result = type -> getTensorLayout(obj, ndims, size, strides, dtype, device, verbose);
         if (result != 0) return result;
      }
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_getTensorLikeLayout(PyObject *obj, int *ndims, OcSize *size,
                                OcDType *dtype, OcDevice **device,
                                int padded, int verbose)
/* -------------------------------------------------------------------- */
{  int flags = 0;

   return pyOcean_getTensorLikeLayoutFlags(obj, ndims, size, dtype, device,
                                           &flags, padded, verbose);
}


/* -------------------------------------------------------------------- */
int pyOcean_getTensorLikeLayoutFlags(PyObject *obj, int *ndims, OcSize *size,
                                     OcDType *dtype, OcDevice **device,
                                     int *flags, int padded, int verbose)
/* -------------------------------------------------------------------- */
{  OcSize s;
   int    result;
   int    i,n;

   /* Initialize */
   if (ndims)  *ndims  = 0;
   if (dtype)  *dtype  = OcDTypeNone;
   if (device) *device = NULL;
   if (flags)  *flags  = 0;

   /* Determine the tensor size in reverse order */
   result = pyOcean_intrnlGetTensorLayout(obj, ndims, size, 0, dtype, device,
                                          flags, padded, verbose);
   if (result != 1) return result;

   /* Reverse the size */
   for (n = *ndims, i = 0; i < n / 2; i++)
   {  s = size[i]; size[i] = size[n - (i+1)]; size[n - (i+1)] = s;
   }

   return 1;
}


/* -------------------------------------------------------------------- */
int pyOcean_intrnlGetTensorLayout(PyObject *obj, int *ndims, OcSize *size,
                                  int offset, OcDType *dtype, OcDevice **device,
                                  int *flags, int padded, int verbose)
/* -------------------------------------------------------------------- */
{  Py_int_ssize_t  idx, n;
   PyObject       *elem;
   int             tensorNDims;
   OcDType         tensorDType, scalarDType;
   OcSize          tensorSize[OC_TENSOR_MAX_DIMS];
   OcDevice       *tensorDevice;
   int             i, j;
   int             result;


   /* ============================= */
   /* Scalar types                  */
   /* ============================= */
   result = pyOcean_getScalarType(obj, &scalarDType, verbose);
   if (result != 0)
   {  if (result != 1) return result;

      /* Verify the size */
      if ((*flags) & PYOC_FLAGS_SIZE_FIXED)
      {  if (offset != *ndims)
         {  OcError(-1, "Scalar values are allowed only at the final tensor dimension");
         }
      }
      else
      {  /* Fix the dimensions */
         (*flags) |= PYOC_FLAGS_SIZE_FIXED;
      }

      /* Update the data type */
      *dtype = OcDType_getCommonType(*dtype, scalarDType);

      /* Update flags */
      (*flags) |= PYOC_FLAGS_CONTAINS_SCALAR;

      return 1;
   }

   /* ============================= */
   /* Tensor types                  */
   /* ============================= */

   /* Note: we must deal with tensor types before we deal with */
   /* sequence types to avoid iterating over tensor types that */
   /* have implemented the sequence protocol (such as Numpy).  */
   result = pyOcean_getTensorLayout(obj, &tensorNDims, tensorSize, NULL,
                                    &tensorDType, &tensorDevice, verbose);
   if (result != 0)
   {  if (result != 1) return result;

      /* Verify or update the size */
      if ((*flags) & PYOC_FLAGS_SIZE_FIXED)
      {  
         if (offset + tensorNDims != *ndims)
         {  if (!verbose) return -1;
            OcError(-1, "Mismatch in tensors dimensions: expected %d got %d", (int)(*ndims - offset), tensorNDims);
         }

         /* Verify or update dimensions */
         if (!padded)
         {  for (i = offset, j = 0; i < *ndims; i++, j++)
            {  if (tensorSize[tensorNDims - (j+1)] != size[i])
               {  if (!verbose) return -1;
                  OcError(-1, "Incompatible tensor size at dimension %d expected %"OC_FORMAT_LU" got %"OC_FORMAT_LU,
                              *ndims - (i+1), (long unsigned)(size[i]),
                                              (long unsigned)(tensorSize[tensorNDims-(j+1)]));
               }
            }
         }
         else
         {  for (i = offset, j = 0; i < *ndims; i++, j++)
            {  if (tensorSize[tensorNDims - (j+1)] > size[i]) size[i] = tensorSize[tensorNDims - (j+1)];
            }
         }
      }
      else
      {  if (*ndims + tensorNDims >= OC_TENSOR_MAX_DIMS)
         {  if (!verbose) return -1;
            OcError(-1, "Data exceeds the maximum tensor dimension (%d)", OC_TENSOR_MAX_DIMS);
         }

         /* Set the dimensions */
         for (j = 0; j < tensorNDims; j++)
         {  size[offset++] = tensorSize[tensorNDims - (j+1)];
         }
         *ndims += tensorNDims;

         /* Fix the number of dimensions */
         *flags |= PYOC_FLAGS_SIZE_FIXED;
      }

      /* Set the device, in case of a mismatch we select the CPU. */
      if (*device != tensorDevice)
      {  if (*device == NULL)
              *device = tensorDevice;
         else *device = OcCPU;
      }

      /* Set the data type */
      *dtype = OcDType_getCommonType(*dtype, tensorDType);

      return 1;
   }


   /* ====================== */
   /* Python sequence types  */
   /* ====================== */
   if (PySequence_Check(obj) && ((n = PySequence_Size(obj)) != -1))
   {
      if (PyUnicode_Check(obj))
      {  if (!verbose) return -1;
         OcError(-1, "Unicode strings are not supported");
      }

#if PY_MAJOR_VERSION < 3
      if (PyString_Check(obj))
      {  if (!verbose) return -1;
         OcError(-1, "Strings are not supported");
      }
#endif

      /* Verify or update the tensor size */
      if ((*flags) & PYOC_FLAGS_SIZE_FIXED)
      {  if (*ndims == offset)
         {  OcError(-1, "Sequences are not allowed at the final dimension");
         }
         
         if (!padded)
         {  if (size[offset] != n)
            {  if (!verbose) return -1;
               OcError(-1, "Incompatible size at dimension %d expected %"OC_FORMAT_LU" got %"OC_FORMAT_LU"",
                          offset+1, (long unsigned)(size[offset]), (long unsigned)(n));
            }
         }
         else
         {  if (size[offset] < n) size[offset] = n;
         }
      }
      else
      {  if (*ndims + 1 >= OC_TENSOR_MAX_DIMS)
         {  if (!verbose) return -1;
            OcError(-1, "Data exceeds the maximum tensor dimension (%d)", OC_TENSOR_MAX_DIMS);
         }

         /* Set the size and increment the number of dimensions */
         size[offset] = (OcSize)n;
         *ndims += 1;

         /* Fix the dimensions if the sequence is empty */
         if (n == 0) *flags |= PYOC_FLAGS_SIZE_FIXED;
      }

      /* Iterate over the elements */
      offset ++;
      for (idx = 0; idx < n; idx++)
      {  /* Process the element */
         elem   = PySequence_ITEM(obj, idx); 
         result = pyOcean_intrnlGetTensorLayout(elem, ndims, size, offset, dtype, device,
                                                flags, padded, verbose);
         if (result != 1) return result;
      }

      return 1;
   }


   /* ============================= */
   /* Python iterable types         */
   /* ============================= */
   if (PyIter_Check(obj))
   {  PyObject *iterator;
      PyObject *item;
      int       flagFixed = (*flags) & PYOC_FLAGS_SIZE_FIXED;

      /* Check validity of the iterable type */
      if (flagFixed)
      {  if (*ndims == offset)
            OcError(-1, "Iterable types are not allowed at the final dimension");
      }
      else
      {  /* Make sure at least one dimension can be added */
         if (offset + 1 >= OC_TENSOR_MAX_DIMS)
         {  if (!verbose) return -1;
            OcError(-1, "Data exceeds the maximum number of tensor dimensions (%d)", OC_TENSOR_MAX_DIMS);
         }

         /* Increase the number of dimensions - size is not yet known */
         *ndims += 1;
      }

      /* Create the iterator */
      if ((iterator = PyObject_GetIter(obj)) == NULL)
      {  if (!verbose) return -1;
         OcError(-1, "Error initializing iterator object");
      }

      /* Iterate over the elements */
      n = 0; result = 1;
      while ((item = PyIter_Next(iterator)) != NULL)
      {  /* Process the element */
         n ++;
         result = pyOcean_intrnlGetTensorLayout(item, ndims, size, offset+1, dtype, device,
                                                flags, padded, verbose);
         if (result != 1) break;
      }
      Py_DECREF(iterator);

      /* Check the result */
      if (result != 1) return result;

      /* Verify or update the tensor size */
      if (flagFixed)
      {  if (!padded)
         {  if (size[offset] != n)
            {  if (!verbose) return -1;
               OcError(-1, "Incompatible size at dimension %d expected %"OC_FORMAT_LU" got %"OC_FORMAT_LU"",
                           offset+1, (long unsigned)(size[offset]), (long unsigned)(n));
            }
         }
         else
         {  if (n > size[offset]) size[offset] = n;
         }
      }
      else
      {  size[offset] = n;
         *flags |= PYOC_FLAGS_SIZE_FIXED;
      }

      return result;
   }


   /* Unrecognized type */
   if (offset != 0)
   {  if (!verbose) return -1;
      OcError(-1, "Unsupported data type encountered while parsing data");
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int pyOcean_exportTensor(OcTensor *tensor, const char *name, PyObject **obj, int deepcopy)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
 
   /* Parameter checks */ 
   if ((name == NULL) || (obj == NULL))
      OcError(-1, "NULL pointer in pyOcean_exportTensor");

   /* Registered tensor types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {  
      /* Export the tensor when a match is found */
      if ((type -> name != NULL) && (strcmp(name, type -> name) == 0))
         return (type -> exportTensor == 0) ? 0 : type -> exportTensor(tensor, obj, deepcopy);
   }

   OcError(-1, "Unrecognize tensor type ('%s')", name);
}


/* -------------------------------------------------------------------- */
int pyOcean_registerConverter(pyOceanConvert *type)
/* -------------------------------------------------------------------- */
{
   /* Add the import structure */
   type -> next = py_ocean_convert_types;
   py_ocean_convert_types = type;

   return 0;
}


/* -------------------------------------------------------------------- */
PyObject *pyOcean_getImportTypes(void)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   PyObject *list = NULL, *name = NULL;

   /* Create the list */
   list = PyList_New(0);
   if (list == NULL) OcError(NULL, "Error creating list object");

   /* Add the supported import types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {
      if ((type -> getTensor) || (type -> getScalar))
      {  /* Create and add a new string object */
         name = PyString_FromString(type -> name);
         if ((name == NULL) || (PyList_Append(list, name) != 0))
         {  Py_XDECREF(name);
            Py_XDECREF(list);
            OcError(NULL, "Error creating list of import types");
         }
      }
   }

   /* Return the list */
   return list;
}


/* -------------------------------------------------------------------- */
PyObject *pyOcean_getExportTypes(void)
/* -------------------------------------------------------------------- */
{  pyOceanConvert *type;
   PyObject *list = NULL, *name = NULL;

   /* Create the list */
   list = PyList_New(0);
   if (list == NULL) OcError(NULL, "Error creating list object");

   /* Add the supported export types */
   for (type = py_ocean_convert_types; type != NULL; type = type -> next)
   {
      if (type -> exportTensor)
      {  /* Create and add a new string object */
         name = PyString_FromString(type -> name);
         if ((name == NULL) || (PyList_Append(list, name) != 0))
         {  Py_XDECREF(name);
            Py_XDECREF(list);
            OcError(NULL, "Error creating list of export types");
         }
      }
   }

   /* Return the list */
   return list;
}


/* -------------------------------------------------------------------- */
int pyOcean_intrnlImportTensorData(PyObject *obj, OcTensor *tensor, int padded,
                                   OcDeviceType *deviceType, int flagExclude)
/* -------------------------------------------------------------------- */
{  Py_int_ssize_t  idx, n;
   PyObject       *elem;
   OcTensor       *elemTensor;
   OcScalar        elemScalar, s;
   OcTensorFlags   flags;
   OcSize          size[OC_TENSOR_MAX_DIMS];
   OcSize          nelem = 0;
   int             result, changed, i;

   /* ------------------------------------------------------------- */
   /* The device type parameter can be set to any device type other */
   /* than that of OcCPU to indicate that the data of this device   */
   /* type should be included or excluded. When the included, only  */
   /* data from tensors that have the given device type is copied.  */
   /* ------------------------------------------------------------- */

   /* ------------------------------------------------------------- */
   /* Assumption: The given tensor is assumed to be new and without */
   /* any self overlap. Being a completely new tensor also means    */
   /* that there can be no overlap with existing tensors, and that  */
   /* the extent does not need to be updated when temporarily       */
   /* resizing the tensor.                                          */
   /* ------------------------------------------------------------- */

   /* ============================= */
   /* Supported scalar types        */
   /* ============================= */
   result = pyOcean_getScalar(obj, &elemScalar);
   if (result < 0) return result;
   if (result == 1)
   {  if ((deviceType != NULL) && (flagExclude == 0)) return 0;
      OcScalar_castTo(&elemScalar, tensor -> dtype, &s);
      OcScalar_exportData(&s, OcTensor_data(tensor), 0);
      return 0;
   }


   /* ============================= */
   /* Supported tensor types        */
   /* ============================= */
   result = pyOcean_getTensor(obj, &elemTensor);
   if (result < 0) return result;
   if (result == 1)
   {
      /* Make sure the number of dimensions match */
      if (tensor -> ndims != elemTensor -> ndims)
         OcError(-1, "Mismatch in tensor dimensions");

      /* Check whether to copy the tensor */
      if (deviceType)
      {  if ((( flagExclude) && (elemTensor -> device -> type == deviceType)) ||
             ((!flagExclude) && (elemTensor -> device -> type != deviceType)))
         {  OcDecrefTensor(elemTensor);
            return 0;
         }
      }
   
      /* Update the input tensor size */
      if (padded)
      {  changed = 0; flags = tensor -> flags;
         for (i = 0; i < tensor -> ndims; i++)
         {  if (tensor -> size[i] != elemTensor -> size[i]) changed = 1;
            size[i] = tensor -> size[i];
            tensor -> size[i] = elemTensor -> size[i];
         }
         nelem = tensor -> nelem;
         tensor -> nelem = elemTensor -> nelem;
         if (changed)
         {  /* Reset tensor shape flags - destination tensor   */
            /* is assumed to be new and have no self-overlaps. */
            OcTensor_updateShapeFlags(tensor, 0);
         }
      }

      /* Copy the data */
      result = OcTensor_copy(elemTensor, tensor);

      /* Restore the tensor size */
      if (padded)
      {  if (changed)
         {  for (i = 0; i < tensor -> ndims; i++)
            {  tensor -> size[i] = size[i];
            }
            tensor -> flags = flags; /* Restore flags */
         }
         tensor -> nelem = nelem;
      }
      
      OcDecrefTensor(elemTensor);
      return result;
   }


   /* ====================== */
   /* Python sequence types  */
   /* ====================== */
   if (PySequence_Check(obj) && ((n = PySequence_Length(obj)) != -1))
   {
      /* Check the number of items */
      if (n == 0) return 0;

      /* Exclude unicode */
      if (PyUnicode_Check(obj))
         OcError(-1, "Unicode strings are not supported");
#if PY_MAJOR_VERSION < 3
      if (PyString_Check(obj))
         OcError(-1, "Strings are not supported");
#endif

      /* Remove final dimension */
      tensor -> ndims -= 1;
      tensor -> nelem /= tensor -> size[tensor -> ndims];
      
      /* Iterate over the elements */
      result = 1;
      for (idx = 0; idx < n; idx++)
      {
         /* Get the element */
         elem = PySequence_ITEM(obj, idx);
   
         /* Initialize the tensor */
         result = pyOcean_intrnlImportTensorData(elem, tensor, padded, deviceType, flagExclude);
         if (result != 0) break;

         /* Update the data pointers */
         tensor -> offset += tensor -> strides[tensor -> ndims];
      }
      
      /* Restore the original data pointers */
      tensor -> offset -= idx * tensor -> strides[tensor -> ndims];

      /* Restore final dimension */
      tensor -> nelem *= tensor -> size[tensor -> ndims];
      tensor -> ndims += 1;

      return result;
   }


   /* ============================= */
   /* Python iterator types         */
   /* ============================= */
   if (PyIter_Check(obj))
   {  PyObject *iterator;

      /* Get the number of items */
      n = tensor -> size[0];
      if (n == 0) return 0;

      /* Remove final dimension */
      tensor -> ndims -= 1;
      tensor -> nelem /= tensor -> size[tensor -> ndims];

      /* Create the iterator */
      if ((iterator = PyObject_GetIter(obj)) == NULL)
      {  OcError(-1, "Error initializing the iterator object");
      }

      /* Count the number of elements */
      idx = 0; result = 1;
      while ((idx < n) && ((elem = PyIter_Next(iterator)) != NULL))
      {  idx ++;
      
         /* Initialize the tensor */
         result = pyOcean_intrnlImportTensorData(elem, tensor, padded, deviceType, flagExclude);
         if (result != 0) break;

         /* Update the data pointers */
         tensor -> offset += tensor -> strides[tensor -> ndims];
      }

      /* Finalize the iterator */
      Py_DECREF(iterator);

      /* Check the results */
      if ((!padded) && (result == 0) && (idx != n))
      {  OcError(-1, "Inconsistent number of elements in iterator object");
      }
      
      /* Restore the original data pointers */
      tensor -> offset -= idx * tensor -> strides[tensor -> ndims];
 
       /* Restore final dimension */
      tensor -> nelem *= tensor -> size[tensor -> ndims];
      tensor -> ndims += 1;

      return result;
   }

   /* Unrecognized types */
   OcError(-1, "Unsupported data type encountered while parsing data");
}

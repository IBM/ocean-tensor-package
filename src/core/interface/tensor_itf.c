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

#include "ocean/core/cpu/device_cpu.h"

#include "ocean/core/interface/module_core.h"
#include "ocean/core/interface/device_itf.h"
#include "ocean/core/interface/scalar_itf.h"
#include "ocean/core/interface/storage_itf.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/interface/index_itf.h"

#include "ocean/core/generic/tensor_op.h"

#include "ocean/base/format.h"
#include "ocean/base/malloc.h"
#include "ocean/base/scalar.h"
#include "ocean/base/shape.h"
#include "ocean/base/platform.h"
#include "ocean/base/warning.h"
#include "ocean/base/error.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <ctype.h>


/* ===================================================================== */
/* Global variables                                                      */
/* ===================================================================== */

/* Modes */
int  oc_tensor_broadcast_mode = 1;    /* 0 = no broadcasting; 1 = auto broadcast */
int  oc_tensor_typecast_mode  = 1;    /* 0 = no casting; 1 = auto typecast */
char oc_tensor_math_mode      = '-';  /* '-' = quiet, 'w' = warning, 'e' = error, 'c' = cast */

/* Warnings */
int oc_warning_tensor_sqrt;
int oc_warning_tensor_reciprocal;
int oc_warning_tensor_arcsin;
int oc_warning_tensor_arccos;
int oc_warning_tensor_arccosh;
int oc_warning_tensor_arctanh;
int oc_warning_tensor_log;
int oc_warning_tensor_log2;
int oc_warning_tensor_log10;
int oc_warning_tensor_log1p;
int oc_warning_tensor_division_zero;
int oc_warning_tensor_modulo_zero;
int oc_warning_tensor_discard_imag;
int oc_warning_tensor_discard_imag2;


/* ===================================================================== */
/* Module initialization                                                 */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcModuleCore_initializeTensorItf(void)
/* --------------------------------------------------------------------- */
{
   /* Register warning types:                                  */
   /* Warning message --------------------------------------+  */
   /* Raise only once ----------------------------------+   |  */
   /* Variable for unique identifier ---+               |   |  */
   /*                                   |               |   |  */
   OcWarning_register(&oc_warning_tensor_sqrt,          1, "Tensor elements must be nonnegative for square-root");
   OcWarning_register(&oc_warning_tensor_reciprocal,    1, "Tensor elements must be nonzero for reciprocal");
   OcWarning_register(&oc_warning_tensor_arcsin,        1, "Tensor elements must be in [-1,1] for arcsin");
   OcWarning_register(&oc_warning_tensor_arccos,        1, "Tensor elements must be in [-1,1] for arccos");
   OcWarning_register(&oc_warning_tensor_arccosh,       1, "Tensor elements must be >= 1 for arccosh");
   OcWarning_register(&oc_warning_tensor_arctanh,       1, "Tensor elements must be in [-1,1] for arctanh");
   OcWarning_register(&oc_warning_tensor_log,           1, "Tensor elements must be >= 0 for log");
   OcWarning_register(&oc_warning_tensor_log2,          1, "Tensor elements must be >= 0 for log");
   OcWarning_register(&oc_warning_tensor_log10,         1, "Tensor elements must be >= 0 for log");
   OcWarning_register(&oc_warning_tensor_log1p,         1, "Tensor elements must be >= -1 for log1p");
   OcWarning_register(&oc_warning_tensor_modulo_zero,   1, "Tensor elements must be nonzero for modulo");
   OcWarning_register(&oc_warning_tensor_division_zero, 1, "Division by zero encountered");

   OcWarning_register(&oc_warning_tensor_discard_imag,  1, "Casting complex to real discards the imaginary part");
   OcWarning_register(&oc_warning_tensor_discard_imag2, 1, "Casting complex intermediate value to real output discards the imaginary part");
   return 0;
}


/* ===================================================================== */
/* Function implementation - Tensor creation                             */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
OcTensor *OcTensor_intrnlCreate(int ndims, OcSize *size, OcIndex *strides,
                                 OcDType dtype, OcDevice *device,
                                 OcStream *stream, int flagTemporary)
/* --------------------------------------------------------------------- */
{  OcSize      offset, extent;
   OcTensor   *tensor;
   OcStorage  *storage;
   int         flagAligned;

   /* Parameter checks */
   if (ndims == 0) { size = NULL; strides = NULL; }
   else if (size == NULL) OcError(NULL, "Size parameter cannot be NULL");

   /* Make sure all dimensions are nonnegative */
   if (!OcShape_isValidSize(ndims, size))
      OcError(NULL, "Negative tensor sizes are not allowed");

   /* Compute the offset and extent, and make sure that all index */
   /* and size computations can be done without integer overflow. */
   if (OcShape_nelem(ndims, size, NULL) != 0) return NULL;
   if (OcShape_extent(ndims, size, strides, OcDType_size(dtype), &offset, &extent) != 0) return NULL;

   /* Create the tensor object */
   if ((tensor = OcAllocateTensor(ndims, dtype)) == NULL) return NULL;

   /* Create the storage object */
   if (stream == NULL)
   {  if (!flagTemporary)
           storage = OcStorage_create(extent, OcDTypeUInt8, device);
      else storage = OcStorage_createTemporary(extent, OcDTypeUInt8, device);
   }
   else
   {  storage = OcStorage_createWithStream(extent, OcDTypeUInt8, stream);
   }
   if (storage == NULL) goto error;

   /* Set the tensor shape - update the shape flags, update the self-overlap */
   /* information, do not update the extent (will set manually below)        */
   if (OcTensor_updateShape(tensor, ndims, size, strides, 1, 1, 0) != 0) goto error;

   /* Initialize the tensor fields */
   tensor -> offset  = offset;
   tensor -> storage = storage;
   tensor -> device  = device;

   /* Set the extent information */
   tensor -> blockOffset = offset;
   tensor -> blockExtent = extent;
   tensor -> flags      |= OC_TENSOR_EXTENT;

   /* Check data alignment */
   flagAligned = OcTensor_isAligned(tensor);
   if ((device -> requiresAlignedData) && (!flagAligned))
   {  OcErrorMessage("Device %s requires data to be aligned", device -> name);
      goto error;
   }

   /* Set the storage data type */
   if (flagAligned)
        OcStorage_setDType(storage, dtype);
   else OcStorage_setDTypeRaw(storage);

   /* Success */
   return tensor;

error : ;
   OcDecrefTensor(tensor);
   return NULL;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_create(int ndims, OcSize *size, OcIndex *strides,
                          OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{
   /* Apply default data type and device */
   if ((dtype == OcDTypeNone) && ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone)) return NULL;
   if ((device == NULL) && ((device = OcDevice_applyDefault(device)) == NULL)) return NULL;

   /* Create the tensor */
   return OcTensor_intrnlCreate(ndims, size, strides, dtype, device, NULL, 0);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_createTemporary(int ndims, OcSize *size, OcIndex *strides,
                                   OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{
   /* Apply default data type and device */
   if ((dtype == OcDTypeNone) && ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone)) return NULL;
   if ((device == NULL) && ((device = OcDevice_applyDefault(device)) == NULL)) return NULL;

   /* Create the tensor */
   return OcTensor_intrnlCreate(ndims, size, strides, dtype, device, NULL, 1);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_createWithStream(int ndims, OcSize *size, OcIndex *strides,
                                    OcDType dtype, OcStream *stream)
/* --------------------------------------------------------------------- */
{
   /* Apply default data type */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;

   /* Create the tensor */
   return OcTensor_intrnlCreate(ndims, size, strides, dtype, stream -> device, stream, 0);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_createFromStorage(OcStorage *storage, int ndims, OcSize *size,
                                     OcIndex *strides, OcIndex offset, OcDType dtype)
/* --------------------------------------------------------------------- */
{  OcSize    tensorOffset, tensorExtent, s;
   OcTensor *tensor;
   int       flagAligned;

   /* Parameter checks */
   if (storage == NULL)
       OcError(NULL, "Storage parameter cannot be NULL");
   if (offset < 0)
       OcError(NULL, "The storage offset cannot be negative");
   if (offset > storage -> size)
       OcError(NULL, "The storage offset exceeds the storage size");
   if (!OcShape_isValidSize(ndims, size))
      OcError(NULL, "Negative tensor sizes are not allowed");

   /* Determine the data type */
   if (dtype == OcDTypeNone)
   {  if (!OcStorage_isRaw(storage))
      {  dtype = storage -> dtype;
      }
      else
      {  dtype = OcDType_applyDefault(dtype);
         if (dtype == OcDTypeNone) return NULL;
      }
   }

   /* Determine the size and strides. When empty dimensions are given */
   /* we create a one-dimensional tensor with size matching that of   */
   /* the storage. To reduce the chance of inadvertedly using this    */
   /* mode we require that the number of dimensions is set to -1 and  */
   /* the strides are set to NULL.                                    */
   if (ndims == 0)
   {  /* Make sure size and strides are empty */
      size    = NULL;
      strides = NULL;
   }
   else if (size == NULL)
   {  /* Check requirements */
      if (ndims != -1)
         OcError(NULL, "Number of dimensions must be -1 in canonical tensor creation from storage");
      if (strides != NULL)
         OcError(NULL, "The strides must be NULL in canonical tensor creation from storage");

      /* Set the effective dimensions */
      s = (storage -> size - offset) / OcDType_size(dtype);
      size  = &s;
      ndims = 1;
   }

   /* Compute the offset and extent, and make sure that all index */
   /* and size computations can be done without integer overflow. */
   if (OcShape_nelem(ndims, size, NULL) != 0) return NULL;
   if (OcShape_extent(ndims, size, strides, OcDType_size(dtype),
                      &tensorOffset, &tensorExtent) != 0) return NULL;

   /* Create the tensor object */
   if ((tensor = OcAllocateTensor(ndims, dtype)) == NULL) return NULL;

   /* Set the tensor shape - update the shape flags, update the self-overlap */
   /* information, do not update the extent (will set manually below)        */
   if (OcTensor_updateShape(tensor, ndims, size, strides, 1, 1, 0) != 0) goto error;

   /* Make sure the tensor fits the storage */
   if ((offset < (tensorOffset)) ||
       (offset - (tensorOffset) + (tensorExtent) > (storage -> size)))
   {  OcErrorMessage("The tensor extent exceeds the storage bounds");
      goto error;
   }

   /* Initialize the tensor fields */
   tensor -> offset  = offset;
   tensor -> storage = OcIncrefStorage(storage);
   tensor -> device  = storage -> stream -> device;

   /* Set the extent information */
   tensor -> blockOffset = tensorOffset;
   tensor -> blockExtent = tensorExtent;
   tensor -> flags      |= OC_TENSOR_EXTENT;

   /* Check data alignment */
   flagAligned = OcTensor_isAligned(tensor);
   if ((tensor -> device -> requiresAlignedData) && (!flagAligned))
   {  OcErrorMessage("Device %s requires data to be aligned", tensor -> device -> name);
      goto error;
   }

   /* Success */
   return tensor;

error : ;
   OcDecrefTensor(tensor);
   return NULL;
}


/* --------------------------------------------------------------------- */
OC_API OcTensor *OcTensor_createFromScalar(OcScalar *scalar, OcDType dtype,
                                           OcDevice *device, int flagTemporary)
/* --------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;
   OcTensor *tensor = NULL;

   /* Apply default device and data type */
   if (device == NULL) device = OcCPU;
   if (dtype == OcDTypeNone) dtype = scalar -> dtype;

   /* Create a new tensor */
   if (flagTemporary)
   {  /* Get the core module device context */
      ctx = (OcModuleCore_Context *)(OC_GET_DEVICE_CONTEXT(device, oc_module_core));
      if (ctx != NULL)
      {  tensor = OcXIncrefTensor(ctx -> scalarList[dtype][ctx -> scalarIndex[dtype]]);
         if (tensor == NULL)
         {  tensor = OcTensor_create(0, NULL, NULL, dtype, device);
            ctx -> scalarList[dtype][ctx -> scalarIndex[dtype]] = OcXIncrefTensor(tensor);
         }
         ctx -> scalarIndex[dtype] = (ctx -> scalarIndex[dtype] + 1) % (ctx -> scalarCount);
      }
   }
   if (tensor == NULL) tensor = OcTensor_create(0, NULL, NULL, dtype, device);
   if (tensor == NULL) return NULL;

   /* Fill the tensor */
   if (OcTensor_fill(tensor, scalar) != 0)
   {  OcDecrefTensor(tensor);
      return NULL;
   }

   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_createFromData(void *data, OcSize size, OcDType dtype,
                                  OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;

   /* Make sure all dimensions are nonnegative */
   if (!(size >= 0))
      OcError(NULL, "Negative tensor sizes are not allowed");

   /* Create a tensor on the CPU */
   tensor = OcTensor_create(1, &size, NULL, dtype, OcCPU);
   if (tensor == NULL) return NULL;

   /* Set the entries */
   memcpy((void *)OcTensor_data(tensor), data, size * OcDType_size(dtype));

   /* Copy the tensor to the desired device */
   if (device != NULL) OcTensor_ensureDevice(&tensor, device, NULL);

   return tensor;
}



/* --------------------------------------------------------------------- */
OcTensor *OcTensor_intrnlContiguousLike(OcTensor *tensor, OcDType dtype,
                                        OcDevice *device, int flagTemporary)
/* --------------------------------------------------------------------- */
{  OcIndex   strides[OC_TENSOR_MAX_DIMS], s;
   OcIndex  *ptrStrides;
   int       elemsize, i;
   OcTensor *result;

   /* Create a new tensor with the same strides as the reference tensor */
   /* if it is contiguous and has the same element size. Otherwise,     */
   /* create a new tensor that is in column-major format except for all */
   /* zero-stride dimensions, which are preserved in the new tensor.    */

   /* Apply default data type*/
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;

   /* Get the element size */
   elemsize = OcDType_size(dtype);

   /* Determine the strides */
   if ((tensor -> elemsize == elemsize) && (OcTensor_isContiguous(tensor)))
   {
      /* Use the original strides */
      ptrStrides = tensor -> strides;
   }
   else
   {  /* Default strides with zero strides maintained */
      s = elemsize;
      for (i = 0; i < tensor -> ndims; i++)
      {  if ((tensor -> strides[i] == 0) || (tensor -> size[i] == 1))
         {  strides[i] = 0;
         }
         else
         {  strides[i] = s;
            s *= tensor -> size[i];
         }
      }

      ptrStrides = strides;
   }

   /* Create a new tensor */
   if (!flagTemporary)
        result = OcTensor_create(tensor -> ndims, tensor -> size, ptrStrides, dtype, device);
   else result = OcTensor_createTemporary(tensor -> ndims, tensor -> size, ptrStrides, dtype, device);
   if (result == NULL) return NULL;

   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_createContiguousLike(OcTensor *tensor, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  return OcTensor_intrnlContiguousLike(tensor, dtype, device, 0);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_createEmpty(OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcSize s = 0;

   return OcTensor_intrnlCreate(1, &s, NULL, dtype, device, NULL, 0);
}


/* --------------------------------------------------------------------- */
int OcTensor_setResult(OcTensor **tensorPtr, OcTensor **result, OcTensor *tensor, int status)
/* --------------------------------------------------------------------- */
{
   /* Delete intermediate results */
   if ((status != 0) && (tensor != NULL))
   {  OcDecrefTensor(tensor);
      tensor = NULL;
   }

   /* Always assign the tensor */
   if (result == NULL)
   {  OcDecrefTensor(*tensorPtr);
      *tensorPtr = tensor;
   }
   else
   {  *result = tensor;
   }

   return status;
}


/* ===================================================================== */
/* Configuration                                                         */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_getAutoTypecastMode(void)
/* --------------------------------------------------------------------- */
{  return oc_tensor_typecast_mode;
}


/* --------------------------------------------------------------------- */
void OcTensor_setAutoTypecastMode(int flag)
/* --------------------------------------------------------------------- */
{  oc_tensor_typecast_mode = flag ? 1 : 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_checkAutoTypecast(OcTensor *tensor, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{
   if (oc_tensor_typecast_mode == 0)
   {  if ((dtype != OcDTypeNone) && (tensor -> dtype != dtype))
         OcError(-1, "Tensor data type mismatch (automatic type casting is disabled)");
      if ((device != NULL) && (tensor -> device != device))
         OcError(-1, "Tensor device mismatch (automatic type casting is disabled)");
   }

   return 0;
}


/* --------------------------------------------------------------------- */
char OcTensor_getDefaultMathMode(void)
/* --------------------------------------------------------------------- */
{  return oc_tensor_math_mode;
}


/* --------------------------------------------------------------------- */
int OcTensor_setDefaultMathMode(char mode)
/* --------------------------------------------------------------------- */
{  int result;
   result = OcTensor_validateMathMode(&mode);
   if (result != 0) return -1;

   oc_tensor_math_mode = mode;

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_validateMathMode(char *mode)
/* --------------------------------------------------------------------- */
{  char m = *mode;

   if (m == 0x0) m = '-';
   if (m == 'W') m = 'w';
   if (m == 'E') m = 'e';
   if (m == 'C') m = 'c';

   /* Check validity of the mode */   
   if ((m != '-') && (m != 'w') && (m != 'e') && (m != 'c'))
      OcError(-1, "Invalid math mode ('-' = quiet, 'w' = warning, 'e' = error, 'c' = cast)");

   /* Update the normalized mode */
   *mode = m;
   return 0;
}



/* ===================================================================== */
/* Function Implementations - Additional tensor creation                 */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
OcTensor *OcTensor_zeros(int ndims, OcSize *size, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;

   /* Create a new tensor and zero out the entries */
   if ((tensor = OcTensor_create(ndims, size, NULL, dtype, device)) != NULL)
   {  if (OcTensor_zero(tensor) != 0)
      { OcDecrefTensor(tensor); tensor = NULL; }
   }

   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_ones(int ndims, OcSize *size, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;

   /* Create the tensor and fill it with ones */
   if ((tensor = OcTensor_create(ndims, size, NULL, dtype, device)) != NULL)
   {  if (OcTensor_fillOnes(tensor) != 0)
      {  OcDecrefTensor(tensor); tensor = NULL;
      }
   }

   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_full(int ndims, OcSize *size, OcScalar *value,
                        OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;

   /* Create the tensor and fill it with the given value */
   if ((tensor = OcTensor_create(ndims, size, NULL, dtype, device)) != NULL)
   {  if (OcTensor_fill(tensor, value) != 0)
      {  OcDecrefTensor(tensor); tensor = NULL;
      }
   }

   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_emptyLike(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  return OcTensor_create(tensor -> ndims, tensor -> size, NULL,
                         tensor -> dtype, tensor -> device);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_zerosLike(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  return OcTensor_zeros(tensor -> ndims, tensor -> size,
                         tensor -> dtype, tensor -> device);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_onesLike(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  return OcTensor_ones(tensor -> ndims, tensor -> size,
                        tensor -> dtype, tensor -> device);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_fullLike(OcTensor *tensor, OcScalar *value)
/* --------------------------------------------------------------------- */
{
   return OcTensor_full(tensor -> ndims, tensor -> size, value,
                        tensor -> dtype, tensor -> device);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_eye(OcSize rows, OcSize columns, OcIndex index,
                       OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor, *diag;
   OcSize    size[2];
   int       result = -1;
   
   /* Create the tensor and fill it with ones */
   size[0] = rows;
   size[1] = columns;
   if ((tensor = OcTensor_create(2, size, NULL, dtype, device)) != NULL)
   {  if ((OcTensor_zero(tensor) == 0) &&
          ((diag = OcTensor_diag(tensor, index, 0, 1)) != NULL))
      {  result = OcTensor_fillOnes(diag);
         OcDecrefTensor(diag);
         if (result == 0) return tensor;
      }
      OcDecrefTensor(tensor);
   }

   return NULL;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_diagonal(OcTensor *tensor, OcIndex index,
                            OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcTensor *result, *diag = NULL;
   OcSize    size[2], s;
   int       success = 1;

   /* Check the input tensor */
   if (tensor -> ndims > 1)
      OcError(NULL, "The diagonal values must be a one-dimensional tensor");

   /* Update dtype and device */
   if (dtype == OcDTypeNone) dtype = tensor -> dtype;
   if (device == NULL) device = tensor -> device;

   /* Create the result tensor */
   s = (tensor -> ndims == 1) ? tensor -> size[0] : 1;
   size[0] = s + ((index < 0) ? (-1 * index) : index);
   size[1] = size[0];
   result = OcTensor_create(2, size, NULL, dtype, device);
   if (result == NULL) return NULL;

   /* Create the diagonal */
   if ((OcTensor_zero(result) != 0) ||
       ((diag = OcTensor_diag(result, index, 0, 1)) == NULL) ||
       (OcTensor_copy(tensor, diag) != 0))
   {  success = 0;
   }

   /* Finalize */
   if (diag) OcDecrefTensor(diag);
   if ((!success) && (result)) { OcDecrefTensor(result); result = NULL; }
   return result;
}



/* ===================================================================== */
/* Function implementations - Detach operations                          */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_detach(OcTensor **tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *result;

   /* Single tensor reference count: detach the storage only */
   if ((*tensor) -> refcount == 1)
      return OcTensor_detachStorage(*tensor);

   /* Multiple tensor references - replicate the tensor */
   result = OcTensor_clone(*tensor);
   if (result == NULL) return -1;

   /* Replace the original */
   OcDecrefTensor(*tensor);
   *tensor = result;

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_detachTensor(OcTensor **tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *result;

   /* Return the the tensor is already detached */
   if ((*tensor) -> refcount == 1) return 0;

   /* Create a shallow copy */
   result = OcTensor_shallowCopy(*tensor);
   if (result == NULL) return -1;

   /* Update the tensor */
   OcDecrefTensor(*tensor);
   *tensor = result;

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_detachStorage(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *t;

   /* Replace the tensor with a new one if the storage is shared */
   if ((tensor -> storage != NULL) &&
       ((tensor -> storage -> refcount > 1) || (!OcStorage_isOwner(tensor -> storage))))
   {
      /* Create a new tensor and copy the contents */
      if (((t = OcTensor_createContiguousLike(tensor, tensor -> dtype, tensor -> device)) == NULL) ||
           (OcTensor_copy(tensor, t) != 0))
      {  OcXDecrefTensor(t); return -1;  }

      /* Replace the storage */
      OcDecrefStorage(tensor -> storage);
      tensor -> storage = OcIncrefStorage(t -> storage);

      /* Update the offset */
      tensor -> offset = t -> offset;

      /* Update the shape -- set the flags, update self-overlap, update extent */
      OcTensor_updateShape(tensor, t -> ndims, t -> size, t -> strides, 1, 1, 1);
      
      /* Free the intermediate tensor */
      OcDecrefTensor(t);
   }

   return 0;
}



/* ===================================================================== */
/* Function implementations - Copy operations                            */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_copy(OcTensor *src, OcTensor *dst)
/* --------------------------------------------------------------------- */
{  OcTensor *temporary0 = NULL;
   OcTensor *temporary1 = NULL;
   OcTensor *temporary2 = NULL;
   OcTensor *temporary3 = NULL;
   int       result = 0;

   /* Check validity of destination */
   if (OcTensor_isReadOnly(dst))
      OcError(-1, "Cannot copy to a read-only tensor");
   if (OcTensor_isSelfOverlapping(dst))
      OcError(-1, "The destination tensor cannot contain self overlaps");

   /* Extend the source tensor, if needed */
   if (src -> nelem != dst -> nelem)
   {  src = temporary0 = OcTensor_autoBroadcastLike(src, dst);
      if (src == NULL) { result = -1; goto final; }
   }

   /* Check if the tensors are empty */
   if (src -> nelem == 0) { result = 0; goto final; }

   /* Check if the tensors match */
   if (OcTensors_match(src, dst))
   {  /* No need to copy the tensor - only add synchronization */
      OcTensor_startRead(dst,src);
      OcTensor_update(dst);
      OcTensor_finishRead(dst,src);
      result = 0; goto final;
   }

   /* Avoid copying repeated data */
   if (OcTensor_hasZeroStrides(dst))
   {  /* Remove matching zero strides */
      temporary1 = OcTensor_removeMatchingRepeats(src, dst);
      temporary2 = OcTensor_removeMatchingRepeats(dst, src);
      src = temporary1; dst = temporary2;
      if ((src == NULL) || (dst == NULL)) { result = -1; goto final; }

      /* The destination tensor should have no zero strides */
      if (OcTensor_hasZeroStrides(dst))
      {  OcErrorMessage("Zero-stride dimensions in destination tensor must be matched by the source tensor");
         result = -1; goto final;
      }
   }

   /* Make sure the tensors have no overlap */
   if (OcTensors_overlap(src, dst))
   {  src = temporary3 = OcTensor_cloneFlags(src, OcDTypeNone, NULL, 1, 1);
      if (src == NULL) { result = -1; goto final; }
   }

   /* Call the internal copy function */
   result = OcTensor_intrnlCopy(src, dst);

final: ;
   if (temporary0) OcDecrefTensor(temporary0);
   if (temporary1) OcDecrefTensor(temporary1);
   if (temporary2) OcDecrefTensor(temporary2);
   if (temporary3) OcDecrefTensor(temporary3);

   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_intrnlCopy(OcTensor *src, OcTensor *dst)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *src, OcTensor *dst);

   /* -------------------------------------------------------------- */
   /* The intrnlCopy function can assume that the destination tensor */
   /* is writable, has no self overlap, and has no zero strides. In  */
   /* addition, the source and destination can be assumed to either  */
   /* have the same shape or otherwise the same number of elements,  */
   /* and do not have any overlap.                                   */
   /* -------------------------------------------------------------- */

   /* Copy between tensors on the same device type */
   if (src -> device -> type == dst -> device -> type)
   {  
      /* Look up the TensorCopy function */
      if ((funptr = OC_GET_CORE_FUNCTION(dst -> device, Tensor_copy)) == 0)
      {  OcError(-1, "Tensor copy is not supported on device %s", dst -> device -> type -> name);
      }

      return funptr(src, dst);
   }

   /* Copy from CPU to device */
   if (src -> device == OcCPU)
   {  
      /* Look up the TensorCopyFromCPU function */
      if ((funptr = OC_GET_CORE_FUNCTION(dst -> device, Tensor_copyFromCPU)) == 0)
      {  OcError(-1, "Tensor copy from CPU is not supported on device %s", dst -> device -> type -> name);
      }

      return funptr(src, dst);
   }

   /* Copy from device to CPU */
   if (dst -> device == OcCPU)
   {
      /* Look up the TensorCopyToCPU function */
      if ((funptr = OC_GET_CORE_FUNCTION(src -> device, Tensor_copyToCPU)) == 0)
      {  OcError(-1, "Tensor copy to CPU is not supported on device %s", src -> device -> type -> name);
      }

      return funptr(src, dst);
   }

   /* Copy between two device types other than CPU */
   OcError(-1, "Copy between different device types is not yet supported");
}


/* --------------------------------------------------------------------- */
OC_API int OcTensor_intrnlCopyDevices(OcTensor *src, OcTensor *dst)
/* --------------------------------------------------------------------- */
{  OcTensor     *srcTensor = NULL, *dstTensor = NULL;
   OcIndex       strides[OC_TENSOR_MAX_DIMS];
   int           flagByteswapped, flagCompressed;
   int           i, j, result = 0;

   /* ---------------------------------------------------------------- */
   /* The intrnlCopyDevices function can assume that the destination   */
   /* tensor is writable, has no self overlap, and has no zero strides.*/
   /* In addition, the source and destination can be assumed to either */
   /* have the same shape or otherwise the same number of elements.    */
   /* ---------------------------------------------------------------- */

   /* Check for empty tensors */
   if (src -> nelem == 0) return 0;

   /* Check if a byte swap is needed */
   flagByteswapped = !OcTensors_haveSameByteOrder(src, dst);

   /* Try to transfer the smallest number of bytes. In case of a tie */
   /* we prefer to transfer from or to a contiguous tensor. In order */
   /* to determine the number of elements to transfer we need to     */
   /* exclude dimensions with zero stride.                           */
   srcTensor = OcTensor_removeRepeats(src);      
   if (srcTensor == NULL) { result = -1; goto final; }
   flagCompressed = (srcTensor == src) ? 0 : 1;

   /* Create an intermediate tensor on the source device if needed */
   if (dst -> elemsize < src -> elemsize)
   {  /* Create a new contiguous tensor with destination data type */
      dstTensor = OcTensor_createTemporary(srcTensor -> ndims, srcTensor -> size, NULL,
                                           dst -> dtype, src -> device);
      if (dstTensor == NULL) { result = -1; goto final; }
   }
   else if (!OcTensor_isContiguous(srcTensor) ||
            (flagByteswapped && !OcDevice_supportsTensorByteswap(dst -> device)))
   {  /* Create a new contiguous tensor with source data type */
      if (flagCompressed)
           dstTensor = OcTensor_intrnlContiguousLike(srcTensor, src -> dtype, src -> device, 1);
      else dstTensor = OcTensor_intrnlContiguousLike(dst, src -> dtype, src -> device, 1);
      if (dstTensor == NULL) { result = -1; goto final; }
   }
   else
   {  /* The source tensor is contiguous and has a data type    */
      /* that is no larger than that of the destination tensor. */
      dstTensor = NULL;
   }

   /* Copy to the intermediate tensor on the source device if needed */
   if (dstTensor)
   {
      if (flagByteswapped && !OcDevice_supportsTensorByteswap(dst -> device))
      {  /* Reverse the original byte order to match that of the destination tensor */
         OcTensor_setByteswapped(dstTensor, OcTensor_isByteswapped(src) ? 0 : 1);
         flagByteswapped = 0;
      }
      else
      {  /* Maintain the original byte order */
         OcTensor_setByteswapped(dstTensor, OcTensor_isByteswapped(src));
      }

      /* Copy the data to the intermediate tensor */
      result = OcTensor_intrnlCopy(srcTensor, dstTensor);
      if (result != 0) goto final;

      /* Replace srcTensor by dstTensor */
      OcDecrefTensor(srcTensor);
      srcTensor = dstTensor;
      dstTensor = NULL;
   }

   /* =============================================================== */
   /* The srcTensor object is contigous and ready for transfer to the */
   /* destination device                                              */
   /* =============================================================== */

   /* Create an intermediate tensor on the destination device if needed */
   if ((flagCompressed) ||
       (srcTensor -> dtype != dst -> dtype) ||
       (!OcTensors_haveCompatibleLayout(srcTensor, dst)))
   {  /* Create the tensor */
      dstTensor = OcTensor_createTemporary(srcTensor -> ndims,
                                           srcTensor -> size, srcTensor -> strides,
                                           srcTensor -> dtype, dst -> device);
      if (dstTensor == NULL) { result = -1; goto final; }
   }
   else
   {  dstTensor = NULL;
   }

   /* Copy to the intermediate tensor if needed */
   if (dstTensor)
   {
      result = OcTensor_intrnlCopy(srcTensor, dstTensor);
      if (result != 0) goto final;

      /* Decompress the intermediate tensor if needed */
      if (flagCompressed)
      {  /* Insert all zero strides in the stride array */
         for (i = 0, j = 0; i < src -> ndims; i++)
         {  if (src -> strides[i] == 0)
                 strides[i] = 0;
            else strides[i] = dstTensor -> strides[j++];
         }
      
         /* Update the shape -- update the flags, do not update the self */
         /* overlap information, do not update the extent                */
         result = OcTensor_updateShape(dstTensor, src -> ndims, src -> size, strides, 1, 0, 0);
         if (result != 0) goto final;
      }

      /* Replace srcTensor by dstTensor */
      OcDecrefTensor(srcTensor);
      srcTensor = dstTensor;
      dstTensor = NULL;
   }

   /* =============================================================== */
   /* Final copy from srcTensor to the destination tensor             */
   /* =============================================================== */

   /* Flip the byte swap flag if needed */
   if (flagByteswapped)
   {  OcTensor_setByteswapped(dst, OcTensor_isByteswapped(dst) ? 0 : 1);
   }

   /* Copy from the intermediate destination tensor to the final one */
   result = OcTensor_intrnlCopy(srcTensor, dst);

   /* Byte swap the data */
   if (flagByteswapped)
   {  OcTensor_setByteswapped(dst, OcTensor_isByteswapped(dst) ? 0 : 1);

      /* Byte swap the data */
      if (result == 0) result = OcTensor_byteswapNoFlag(dst);
   }


final: ;

   /* Delete intermediate tensors */
   if (srcTensor != NULL) OcDecrefTensor(srcTensor);
   if (dstTensor != NULL) OcDecrefTensor(dstTensor);

   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_clone(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  return OcTensor_cloneFlags(tensor, OcDTypeNone, NULL, 1, 0);
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_cloneTo(OcTensor *tensor, OcDevice *device)
/* -------------------------------------------------------------------- */
{  return OcTensor_cloneFlags(tensor, OcDTypeNone, device, 1, 0);
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_cloneFlags(OcTensor *tensor, OcDType dtype, OcDevice *device,
                              int flagByteswapped, int flagTemporary)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   /* Determine the data type and device */
   if (dtype == OcDTypeNone) dtype = tensor -> dtype;
   if (device == NULL) device = tensor -> device;

   /* Create a new contiguous tensor */
   result = OcTensor_intrnlContiguousLike(tensor, dtype, device, flagTemporary);
   if (result == NULL) return NULL;

   /* Copy byteswap information */
   if ((flagByteswapped) && (tensor -> device -> type == device -> type))
   {  OcStorage_setByteswapped(result -> storage, OcStorage_isByteswapped(tensor -> storage));
      OcTensor_setByteswapped(result, OcTensor_isByteswapped(tensor));
   }

   /* Copy the data */
   if (OcTensor_copy(tensor, result) != 0)
   {  OcDecrefTensor(result);
      result = NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_replicate(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  return OcTensor_replicateTo(tensor, tensor -> device);
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_replicateTo(OcTensor *tensor, OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   /* Create a new tensor */
   result = OcTensor_create(tensor -> ndims, tensor -> size, NULL,
                            tensor -> dtype, device);
   if (result == NULL) return NULL;

   /* Copy the data */
   if (OcTensor_copy(tensor, result) != 0)
   {  OcDecrefTensor(result);
      result = NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_contiguous(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   /* Increment the reference count and return if already contiguous */
   if (OcTensor_isContiguous(tensor))
      return OcIncrefTensor(tensor);

   /* Create the new tensor */
   result = OcTensor_createContiguousLike(tensor, tensor -> dtype, tensor -> device);
   if (result == NULL) return NULL;

   /* Copy the tensor */
   if (OcTensor_copy(tensor, result) != 0)
   {  OcDecrefTensor(result);
      result = NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_contiguousType(OcTensor *tensor, char type)
/* -------------------------------------------------------------------- */
{  OcIndex   strides[OC_TENSOR_MAX_DIMS];
   OcTensor *result;
   int       status;

   /* Increment the reference count and return if the tensor */
   /* already has the given type of contiguousness.          */
   status = OcTensor_hasOrder(tensor, type);
   if (status == -1) return NULL;
   if (status ==  1) return OcIncrefTensor(tensor);

   /* Determine the strides */
   status = OcShape_getStrides(tensor -> ndims, tensor -> size,
                               OcDType_size(tensor -> dtype),
                               strides, type);
   if (status != 0) return NULL;

   /* Create the new tensor */
   result = OcTensor_create(tensor -> ndims, tensor -> size, strides,
                            tensor -> dtype, tensor -> device);
   if (result == NULL) return NULL;


   /* Update the flags */
   result -> flags |= (OC_TENSOR_CONTIGUOUS | OC_TENSOR_CONTIGUOUS_SET);
   
   /* Copy the tensor */
   if (OcTensor_copy(tensor, result) != 0)
   {  OcDecrefTensor(result);
      result = NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensor_toScalar(OcTensor *tensor, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is a scalar */
   if (tensor -> nelem != 1)
      OcError(-1, "Cannot convert non-scalar tensor to scalar");

   /* Make sure the tensor is on the CPU (new or incref) */
   if (OcTensor_ensureDevice(&tensor, OcCPU, &tensor) != 0) return -1;

   /* Synchronize the tensor */
   OcTensor_synchronize(tensor);

   /* Copy the data */
   scalar -> dtype = tensor -> dtype;
   OcScalar_importData(scalar, OcTensor_data(tensor), OcTensor_isByteswapped(tensor));

   /* Decref the tensor */
   OcDecrefTensor(tensor);

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensor_ensureDType(OcTensor **tensorPtr, OcDType dtype, OcTensor **result)
/* -------------------------------------------------------------------- */
{
   return OcTensor_ensureFlags(tensorPtr, dtype, NULL, result, 0);
}


/* -------------------------------------------------------------------- */
int OcTensor_ensureDevice(OcTensor **tensorPtr, OcDevice *device, OcTensor **result)
/* -------------------------------------------------------------------- */
{
   return OcTensor_ensureFlags(tensorPtr, OcDTypeNone, device, result, 0);
}


/* -------------------------------------------------------------------- */
int OcTensor_ensure(OcTensor **tensorPtr, OcDType dtype, OcDevice *device,
                    OcTensor **result)
/* -------------------------------------------------------------------- */
{  return OcTensor_ensureFlags(tensorPtr, dtype, device, result, 0);
}


/* -------------------------------------------------------------------- */
int OcTensor_ensureFlags(OcTensor **tensorPtr, OcDType dtype, OcDevice *device,
                         OcTensor **result, int flagTemporary)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = *tensorPtr;
   int       status = -1;

   /* Note that tensorPtr and result can be equal. */

   /* Set the device and data type */
   if (device == NULL) device = tensor -> device;
   if (dtype == OcDTypeNone) dtype = tensor -> dtype;

   /* Return when the data type and device matches */
   if ((tensor -> dtype == dtype) && (tensor -> device == device))
   {  /* Set result if needed */
      if (result != NULL) *result = OcIncrefTensor(tensor);
      return 0;
   }

   /* Create a new tensor */
   tensor = OcTensor_intrnlContiguousLike(tensor, dtype, device, flagTemporary);
   if (tensor == NULL) goto final;

   /* Copy the tensor */
   if (OcTensor_copy(*tensorPtr, tensor) != 0)
   {  OcDecrefTensor(tensor); tensor = NULL;
      goto final;
   }

   /* Success */
   status = 0;
   
final :
   if (result != NULL)
   {  *result = tensor;
   }
   else
   {  OcDecrefTensor(*tensorPtr);
      *tensorPtr = tensor;
   }
   return status;
}


/* -------------------------------------------------------------------- */
int OcTensor_ensureByteOrder(OcTensor **tensorPtr, OcTensor **result)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = NULL;
   int status = -1;

   /* Check whether the tensor already has the native byte order */
   if (OcTensor_hasHostByteOrder(*tensorPtr))
   {  if (result) *result = OcIncrefTensor(*tensorPtr);
      return 0;
   }

   /* Create a new tensor */
   tensor = OcTensor_createContiguousLike(*tensorPtr, (*tensorPtr)->dtype, (*tensorPtr)->device);
   if (tensor == NULL) goto final;

   /* Copy the tensor */
   if (OcTensor_copy(*tensorPtr, tensor) != 0)
   {  OcDecrefTensor(tensor); tensor = NULL;
      goto final;
   }

   /* Success */
   status = 0;
   
final :
   if (result != NULL)
   {  *result = tensor;
   }
   else
   {  OcDecrefTensor(*tensorPtr);
      *tensorPtr = tensor;
   }
   return status;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_castDType(OcTensor *tensor, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   OcTensor_ensureDType(&tensor, dtype, &result);
   if (result != NULL)
   {  if (OcTensor_detach(&result) != 0)
      {  OcDecrefTensor(result);
         result = NULL;
      }
   }
   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_castDevice(OcTensor *tensor, OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   OcTensor_ensureDevice(&tensor, device, &result);
   if (result != NULL)
   {  if (OcTensor_detach(&result) != 0)
      {  OcDecrefTensor(result);
         result = NULL;
      }
   }
   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_cast(OcTensor *tensor, OcDType dtype, OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   OcTensor_ensure(&tensor, dtype, device, &result);
   if (result != NULL)
   {  if (OcTensor_detach(&result) != 0)
      {  OcDecrefTensor(result);
         result = NULL;
      }
   }
   return result;
}



/* ===================================================================== */
/* Function implementations - Byte order                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensor_byteswap(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  int result;

   if (OcTensor_isReadOnly(tensor))
      OcError(-1, "Cannot byteswap read-only tensor");
   if (OcDType_baseSize(tensor -> dtype) == 1) return 0;

   /* Byte swap the data */
   result = OcTensor_byteswapNoFlag(tensor);

   /* Update the byte-swapped flag */
   if (result == 0)
   {  if (tensor -> flags & OC_TENSOR_BYTESWAPPED)
           tensor -> flags &= ~OC_TENSOR_BYTESWAPPED;
      else tensor -> flags |=  OC_TENSOR_BYTESWAPPED;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensor_byteswapNoFlag(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *tensor);

   if (OcDType_baseSize(tensor -> dtype) == 1) return 0;

   /* Look up the TensorByteswap function */
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_byteswapNoFlag)) == 0)
   {  OcError(-1, "Tensor byteswap is not supported on device %s", tensor -> device -> type -> name);
   }

   /* Call the function */
   return funptr(tensor);
}


/* -------------------------------------------------------------------- */
int OcTensor_hasHostByteOrder(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  int byteswapped = OcTensor_isByteswapped(tensor);

   if (tensor -> elemsize == 1) return 1;

   if (tensor -> device -> endianness == OcCPU -> endianness)
        return byteswapped ? 0 : 1;
   else return byteswapped;
}


/* -------------------------------------------------------------------- */
void OcTensor_setByteswapped(OcTensor *tensor, int flag)
/* -------------------------------------------------------------------- */
{
   /* Make sure the final byteswapped status is as indicated   */
   /* when the underlying storage is byteswapped this means we */
   /* have to negate the given flag.                           */
   if (OcStorage_isByteswapped(tensor -> storage))
        flag = (flag == 0) ? 1 : 0;

   tensor -> flags &= ~OC_TENSOR_BYTESWAPPED;
   if (flag) tensor -> flags |= OC_TENSOR_BYTESWAPPED;
}


/* ===================================================================== */
/* Function implementations - Broadcasting                               */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_broadcastTo(OcTensor **tensorPtr, int ndims, OcSize *size, int mode, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   OcIndex   strides[OC_TENSOR_MAX_DIMS];
   int       identical = 1;
   int       status = -1;
   int       i, j;

   /* Make sure that the mode and dimensions are valid */
   if ((mode < 0) || (mode > 1))
   {  OcErrorMessage("Invalid broadcast mode (%d)", mode); goto final; }
   if (ndims > OC_TENSOR_MAX_DIMS)
   {  OcErrorMessage("Number of broadcast dimensions exceeds the maximum tensor dimensions"); goto final; }
   if (ndims < input -> ndims)
   {  OcErrorMessage("Broadcast cannot lower the number of tensor dimensions"); goto final; }
   if (!OcShape_isValidSize(ndims, size))
   {  OcErrorMessage("Negative tensor sizes are not allowed"); goto final; }


   /* Ensure compatibility and find the broadcast shape */
   if (mode == 0)
   {  /* Dimension broadcast on the right */
      for (i = 0; i < input -> ndims; i++)
      {  if (input -> size[i] != size[i])
         {  if (input -> size[i] != 1)
            {  OcErrorMessage("Tensor dimensions are incompatible with broadcast dimensions");
               goto final;
            }
            else
            {  strides[i] = 0;
               identical = 0;
            }
         }
         else
         {  strides[i] = input -> strides[i];
         }
      }
      /* Remaining dimensions */
      if (ndims > input -> ndims)
      {  identical = 0;
         for ( ; i < ndims; i++) strides[i] = 0;
      }
   }
   else
   {  /* Dimension broadcast on the left */
      j = ndims - 1;
      for (i = input -> ndims - 1; i >= 0; i--, j--)
      {  if (input -> size[i] != size[j])
         {  if (input -> size[i] != 1)
            {  OcErrorMessage("Tensor dimensions are incompatible with broadcast dimensions");
               goto final;
            }
            else
            {  strides[j] = 0;
               identical = 0;
            }
         }
         else
         {  strides[j] = input -> strides[i];
         }
      }
      /* Remaining dimensions */
      if (ndims > input -> ndims)
      {  identical = 0;
         for ( ; j >= 0; j--) strides[j] = 0;
      }
   }

   /* Check for identical tensors */
   if (identical)
   {  if (result) *result = OcIncrefTensor(input);
      return 0;
   }

   /* Make sure that the number of elements can be */
   /* computed without integer overflow.           */
   if (OcShape_nelem(ndims, size, NULL) != 0) goto final;

   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;

   /* Set the shape -- update shape flags, do not update self */
   /* overlap information, do not update extent               */
   if (OcTensor_updateShape(tensor, ndims, size, strides, 1, 0, 0) != 0) goto final;

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* --------------------------------------------------------------------- */
int OcTensor_broadcastLike(OcTensor **tensorPtr, OcTensor *reference, int mode, OcTensor **result)
/* --------------------------------------------------------------------- */
{
   /* Check if the tensor needs to be extended */
   if (((*tensorPtr) -> nelem == reference -> nelem) &&
       (OcTensors_haveSameSize(*tensorPtr, reference)))
   {  if (result != NULL) *result = OcIncrefTensor(*tensorPtr);
      return 0;
   }

   /* Call broadcast with size */
   return OcTensor_broadcastTo(tensorPtr, reference -> ndims, reference -> size, mode, result);
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_autoBroadcastLike(OcTensor *tensor, OcTensor *reference)
/* --------------------------------------------------------------------- */
{  OcTensor *result;

   /* Check if the tensor needs to be extended */
   if ((tensor -> nelem == reference -> nelem) &&
       (OcTensors_haveSameSize(tensor, reference)))
      return OcIncrefTensor(tensor);

   if ((!oc_tensor_broadcast_mode) && (!OcTensor_isScalar(tensor)))
      OcError(NULL, "Tensor size mismatch (automatic broadcasting of tensor dimensions is disabled)");

   /* Broadcast the tensor dimensions */
   if (OcTensor_broadcastTo(&tensor, reference -> ndims, reference -> size, 0, &result) != 0) return NULL;

   /* Return the result */
   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_canAutoBroadcast(OcTensor *tensor, OcTensor *reference)
/* --------------------------------------------------------------------- */
{
   if (oc_tensor_broadcast_mode)
        return OcTensor_canBroadcast(tensor, reference);
   else return OcTensors_haveSameSize(tensor, reference);
}


/* --------------------------------------------------------------------- */
int OcTensor_canBroadcast(OcTensor *tensor, OcTensor *reference)
/* --------------------------------------------------------------------- */
{  OcSize s;
   int    i, n;

   if ((n = tensor -> ndims) > reference -> ndims) return 0;

   for (i = 0; i < n; i++)
   {  s = tensor -> size[i];
      if ((reference -> size[i] != s) && (s != 1)) return 0;
   }

   return 1;
}


/* --------------------------------------------------------------------- */
void OcTensor_setAutoBroadcastMode(int flag)
/* --------------------------------------------------------------------- */
{  oc_tensor_broadcast_mode = (flag != 0);
}


/* --------------------------------------------------------------------- */
int OcTensor_getAutoBroadcastMode(void)
/* --------------------------------------------------------------------- */
{  return oc_tensor_broadcast_mode;
}



/* ===================================================================== */
/* Function implementations - Shape and layout                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensor_reshape(OcTensor **tensorPtr, int ndims, OcSize *size, OcTensor **result)
/* -------------------------------------------------------------------- */
{  OcIndex   strides[OC_TENSOR_MAX_DIMS];
   OcIndex   stride;
   OcSize    s, s1, s2, nelem;
   OcTensor *tensor = NULL, *input = *tensorPtr;
   int       compatible, identical;
   int       status = -1;
   int       i,j;

   /* Parameter checks */
   if (ndims > OC_TENSOR_MAX_DIMS)
   {  OcErrorMessage("Number of dimensions (%d) exceeds the maximum of %d", ndims, OC_TENSOR_MAX_DIMS);
      goto final;
   }
   if (!OcShape_isValidSize(ndims, size))
   {  OcErrorMessage("Negative tensor sizes are not allowed");
      goto final;
   }

   /* Check if resize is needed */
   if (ndims == input -> ndims)
   {  identical = 1;
      for (i = 0; i < ndims; i++)
      {  if (size[i] != input -> size[i])
         {  identical = 0; break;  }
      }
      if (identical)
      {  if (result) *result = OcIncrefTensor(input);
         return 0;
      }
   }
   
   /* Make sure the number of elements match */
   for (nelem = 1, i = 0; i < ndims; i++) nelem *= size[i];
   if (input -> nelem != nelem)
   {  OcErrorMessage("Mismatch in number of elements in tensor resize");
      goto final;
   }

   /* Check compatibility */
   if (input -> nelem > 0)
   {  i = 0; j = 0; compatible = 1;
      while (i < ndims)
      {
         /* Deal with singleton timensions -- we do not need to check   */
         /* whether the j indices reach the number of dimensions as the */
         /* number of elements match, and i would reach ndims first.    */
         if ((s1 = size[i]) == 1) { strides[i] = 0; i++; continue; }
         while ((s2 = input -> size[j]) == 1) { j ++; }

         /* Set the base stride size */
         stride = input -> strides[j];
         strides[i] = stride;

         /* Find a block of matching size -- provided the strides match */
         /* this is guaranteed to exist as the total number of elements */
         /* matches and we therefore do not need to check validity of   */
         /* the i and j indices.                                        */
         while (s1 != s2)
         {
            if (s1 < s2)
            {  i++; s = size[i];
               strides[i] = s1 * stride; s1 *= s;
            }
            else
            {  j++; s = input -> size[j];
               if ((s > 1) && (input -> strides[j] != stride * s2))
               {  compatible = 0; break;  }
               s2 *= s;
            }
         }
         if (!compatible) break;

         /* Move to the next block */
         i++; j++;
      }
   }
   else
   {  compatible = 1;
      for (i = 0; i < ndims; i++) strides[i] = 0;
   }

   /* Create a new tensor or a new view */
   if (compatible)
        tensor = OcTensor_createFromStorage(input -> storage, ndims, size, strides, input -> offset, input -> dtype);
   else tensor = OcTensor_create(ndims, size, NULL, input -> dtype, input -> device);
   if (tensor == NULL) goto final;

   /* Preserve the byteswap flag */
   OcTensor_setByteswapped(tensor, OcTensor_isByteswapped(input));

   /* Copy the data if needed */
   if (!compatible)
   {  status = OcTensor_copy(input, tensor);
      if (status != 0) goto final;
   }

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* -------------------------------------------------------------------- */
int OcTensor_reshapeLike(OcTensor **tensorPtr, OcTensor *reference, OcTensor **result)
/* -------------------------------------------------------------------- */
{
   return OcTensor_reshape(tensorPtr, reference -> ndims, reference -> size, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_flipAxis(OcTensor **tensorPtr, int axis, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   OcIndex   offset;
   int       status = -1;

   if (input -> ndims == 0)
   {  OcErrorMessage("Flip axis does not apply to scalar tensors");
      goto final;
    }
   if ((axis < 0) || (axis >= input -> ndims))
   {  OcErrorMessage("Invalid axis (%d) to flip axis: valid range is 0-%d", axis, input -> ndims-1);
      goto final;
   }

   /* Check if update is needed */
   if (input -> size[axis] <= 1)
   {  if (result) *result = OcIncrefTensor(input);
      return 0;
   }
      
   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;

   /* Update the tensor */
   offset = tensor -> strides[axis] * (tensor -> size[axis] - 1);
   tensor -> offset += offset;
   tensor -> strides[axis] *= -1;

   /* Update flags, do not update self-overlap information */
   OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* --------------------------------------------------------------------- */
int OcTensor_fliplr(OcTensor **tensorPtr, OcTensor **result)
/* --------------------------------------------------------------------- */
{  int ndims;

   ndims = (*tensorPtr) -> ndims;
   if (ndims == 0)
      OcError(-1, "Fliplr does not apply to scalar tensors");

   return OcTensor_flipAxis(tensorPtr, (ndims == 1) ? 0 : 1, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_flipud(OcTensor **tensorPtr, OcTensor **result)
/* --------------------------------------------------------------------- */
{
   if ((*tensorPtr) -> ndims == 0)
      OcError(-1, "Flipud does not apply to scalar tensors");

   return OcTensor_flipAxis(tensorPtr, 0, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_transpose(OcTensor **tensorPtr, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   int ndims, keepdims = 0;
   int status = -1, flags;

   /* Get the number of dimensions */
   ndims = input -> ndims;

   if (ndims == 0)
   {  if (result) *result = OcIncrefTensor(input);
      return 0;
   }
   else if (ndims == 1)
   {  /* Create a shallow copy */
      tensor = OcTensor_shallowCopy(input);
      if (tensor == NULL) goto final;

      /* Extend the number of dimensions */
      ndims = 2;
      if (OcTensor_allocDims(tensor, ndims) != 0) goto final;

      /* Transpose the tensor */
      tensor -> ndims = ndims;
      tensor -> size[1] = tensor -> size[0];
      tensor -> size[0] = 1;
      tensor -> strides[1] = tensor -> strides[0];
      tensor -> strides[0] = 0;

      /* Update flags */
      flags = 1;
   }
   else
   {  /* Transpose the tensor */
      status = OcTensor_swapAxes(tensorPtr, 0, 1, &tensor);
      if (status != 0) goto final;
   }

   /* Simplify the dimensions */
   if ((!keepdims) && (ndims <= 2))
   {  while ((ndims > 0) && (tensor -> size[ndims-1] == 1)) { flags = 1; ndims --; }
      tensor -> ndims = ndims;
   }

   /* Update the shape flags, do not update self-overlap information */
   if (flags) OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* --------------------------------------------------------------------- */
int OcTensor_ctranspose(OcTensor **tensorPtr, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL;
   int       status = -1;

   /* Non-complex tensors */
   if (OcTensor_isReal(*tensorPtr))
      return OcTensor_transpose(tensorPtr, result);

   /* Complex tensors */
   tensor = OcTensor_clone(*tensorPtr);
   if (OcTensor_transpose(&tensor, NULL) != 0) goto final;
   if (OcTensor_conj(tensor, NULL) != 0) goto final;

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* --------------------------------------------------------------------- */
int OcTensor_swapAxes(OcTensor **tensorPtr, int axis1, int axis2, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   OcIndex   stride;
   OcSize    size;
   int       status = -1;

   if (input -> ndims == 0)
   {  OcErrorMessage("Swap axes does not apply to scalar tensors"); goto final; }
   if ((axis1 < 0) || (axis1 >= input -> ndims))
   {  OcErrorMessage("Invalid axis1 (%d) to swap axes: valid range is 0-%d", axis1, input -> ndims-1); goto final; }
   if ((axis2 < 0) || (axis2 >= input -> ndims))
   {  OcErrorMessage("Invalid axis2 (%d) to swap axes: valid range is 0-%d", axis2, input -> ndims-1); goto final; }

   /* Check if operation is needed */
   if (axis1 == axis2)
   {  if (result) *result = OcIncrefTensor(input);
      return 0;
   }
   
   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) return -1;

   /* Swap the size and strides */
   size = tensor -> size[axis1];
   tensor -> size[axis1] = tensor -> size[axis2];
   tensor -> size[axis2] = size;
   stride = tensor -> strides[axis1];
   tensor -> strides[axis1] = tensor -> strides[axis2];
   tensor -> strides[axis2] = stride;

   /* Update flags, do not update self-overlap information */
   OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* --------------------------------------------------------------------- */
int OcTensor_reverseAxes(OcTensor **tensorPtr, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   OcSize    d;
   OcIndex   s;
   int       status = 0;
   int       i, j, n;

   n = OcTensor_ndims(input);
   if (n <= 1)
   {  tensor = OcIncrefTensor(input);
   }
   else
   {  tensor = OcTensor_shallowCopy(input);
      if (tensor == NULL) { status = -1; goto final; }

      for (i = 0, j = n-1; i < j; i++, j--)
      {  d = tensor -> size[i];
         tensor -> size[i] = tensor -> size[j];
         tensor -> size[j] = d;

         s = tensor -> strides[i];
         tensor -> strides[i] = tensor -> strides[j];
         tensor -> strides[j] = s;
      }

      /* Update flags, do not update self-overlap information */
      OcTensor_updateShapeFlags(tensor, 0);
   }

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* --------------------------------------------------------------------- */
int OcTensor_reverseAxes2(OcTensor **tensorPtr, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor;
   OcSize    d;
   OcIndex   s;
   int       n;

   if (OcTensor_reverseAxes(tensorPtr, result) != 0) return -1;
   tensor = (result) ? *result : *tensorPtr;
   n = OcTensor_ndims(tensor);

   if (n >= 2)
   {  /* Swap axes 0 and 1 */
      d = tensor -> size[0];
      tensor -> size[0] = tensor -> size[1];
      tensor -> size[1] = d;

      s = tensor -> strides[0];
      tensor -> strides[0] = tensor -> strides[1];
      tensor -> strides[1] = s;

      /* Update flags, do not update self-overlap information */
      OcTensor_updateShapeFlags(tensor, 0);
   }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_permuteAxes(OcTensor **tensorPtr, int ndims, OcIndex *dims, OcTensor **result)
/* --------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   int covered[OC_TENSOR_MAX_DIMS];
   int status = -1;
   int i, k;

   /* Check the number of dimensions */
   if (ndims != input -> ndims)
   {  OcErrorMessage("Mismatch in number of dimensions"); goto final; }

   /* Make sure the dimensions are a permutation */
   for (i = 0; i < ndims; i++) covered[i] = 0;
   for (i = 0; i < ndims; i++)
   {  if ((dims[i] < 0) || (dims[i] >= ndims))
      {  OcError(-1, "Invalid dimension value at index %d", i); goto final; }
      if (covered[dims[i]])
      {  OcError(-1, "Replicate dimension at index %d", i); goto final; }
      covered[dims[i]] = 1;
   }

   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;

   /* Permute the dimension */
   for (i = 0; i < ndims; i++)
   {  k = dims[i];
      tensor -> size[i]    = input -> size[k];
      tensor -> strides[i] = input -> strides[k];
   }

   /* Update flags, do not update self-overlap information */
   OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* -------------------------------------------------------------------- */
int OcTensor_squeeze(OcTensor **tensorPtr, OcTensor **result)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   int status = -1;
   int i, j;

   /* Check if the operation is needed */
   for (i = 0; i < input -> ndims; i++)
   {  if (input -> size[i] == 1) break;
   }
   if (i == input -> ndims)
   {  if (result) *result = OcIncrefTensor(input);
      return 0;
   }
   
   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;

   /* Squeeze all singleton dimensions */
   for (i = 0, j = 0; i < tensor -> ndims; i++)
   {  if (tensor -> size[i] == 1) continue;
      if (i != j)
      {  tensor -> size[j]    = tensor -> size[i];
         tensor -> strides[j] = tensor -> strides[i];
      }
      j++;
   }

   /* Update the number of dimensions and shape flags, */
   /* do not update self-overlap information           */
   tensor -> ndims = j;
   OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* -------------------------------------------------------------------- */
int OcTensor_squeezeDim(OcTensor **tensorPtr, int dim, OcTensor **result)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   int status = -1;
   int i;

   /* Basic checks */
   if ((dim < 0) || (dim >= input -> ndims))
   {  OcErrorMessage("Invalid dimension index for squeeze"); goto final; }
   if (input -> size[dim] != 1)
   {  OcErrorMessage("Squeeze dimension is not a singleton (%"OC_FORMAT_LU")",
                     (long unsigned)(input -> size[dim]));
      goto final;
   }

   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;

   /* Extract the dimension */
   tensor -> ndims -= 1;
   for (i = dim; i < tensor -> ndims; i++)
   {  tensor -> size[i]    = tensor -> size[i+1];
      tensor -> strides[i] = tensor -> strides[i+1];
   }

   /* Update flags, do not update self-overlap information */
   OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* -------------------------------------------------------------------- */
int OcTensor_unsqueezeDim(OcTensor **tensorPtr, int dim, OcTensor **result)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   int       status = -1;
   int       i;

   /* Basic checks */
   if ((dim < 0) || (dim > input -> ndims))
   {  OcErrorMessage("Invalid dimension index for unsqueeze (%d)", dim); goto final; }
   if (input -> ndims == OC_TENSOR_MAX_DIMS)
   {  OcError(-1, "Adding a dimension would exceed the maximum number of tensor dimension (%d)", OC_TENSOR_MAX_DIMS); goto final; }

   /* Create a new tensor */
   if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;

   /* Allocate the dimensions */
   if (OcTensor_allocDims(tensor, tensor -> ndims + 1) != 0) goto final;

   /* Insert a singular dimension */
   for (i = tensor -> ndims; i > dim; i--)
   {  tensor -> size[i]    = tensor -> size[i-1];
      tensor -> strides[i] = tensor -> strides[i-1];
   }
   tensor -> size[dim]    = 1;
   tensor -> strides[dim] = 0;
   tensor -> ndims ++;

   /* Update flags, do not update self-overlap information */
   OcTensor_updateShapeFlags(tensor, 0);

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}


/* -------------------------------------------------------------------- */
int OcTensor_flatten(OcTensor **tensorPtr, char type, OcTensor **result)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = NULL, *input = *tensorPtr;
   OcIndex   strides[OC_TENSOR_MAX_DIMS];
   OcSize    size;
   int       flagOrder = 0;
   int       status = -1;

   /* Supported types are: 'cC' for row-major (C-style) order, 'fF'    */
   /* for column-major (Fortan-style) order, 'aA' for row-major order, */
   /* unless the tensor already has a column-major order, 'kK' for a   */
   /* column-major order, unless the tensor already has a row-major    */
   /* order.                                                           */

   /* Deal with vectors and scalars */
   if (input -> ndims <= 1)
   {  if (result) *result = OcIncrefTensor(input);
      return 0;
   }

   /* Check different orders */
   if ((type == 'a') || (type == 'A'))
   {  type = OcTensor_hasOrder(input, 'F') ? 'F' : 'C';
      flagOrder = (type == 'F');
   }
   else if ((type == 'k') || (type == 'K'))
   {  type = OcTensor_hasOrder(input, 'C') ? 'C' : 'F';
      flagOrder = (type == 'C');
   }
   if ((type != 'c') && (type != 'C') && (type != 'f') && (type != 'F'))
   {  OcErrorMessage("Unsupported order ('%c') in tensor flatten function", type);
      goto final;
   }

   /* Create a view or a new tensor */
   if (flagOrder || OcTensor_hasOrder(input, type))
   {   /* Create a view */
      if ((tensor = OcTensor_shallowCopy(input)) == NULL) goto final;
   }
   else
   {  /* Create a new tensor */
      OcShape_getStrides(input -> ndims, input -> size, input -> elemsize, strides, type);
      tensor = OcTensor_create(input -> ndims, input -> size, strides, input -> dtype, input -> device);
      if (tensor == NULL) goto final;

      /* Copy the data */
      if (OcTensor_copy(input, tensor) != 0) goto final;
   }

   /* Update the tensor shape; update the flags, update */
   /* the self-overlap information, update the extent   */
   size = tensor -> nelem;
   strides[0] = tensor -> elemsize;
   if (OcTensor_updateShape(tensor, 1, &size, strides, 1, 1, 1) != 0) goto final;

   /* Success */
   status = 0;

final : ;
   return OcTensor_setResult(tensorPtr, result, tensor, status);
}



/* ===================================================================== */
/* Function implementations - Tensor indexing                            */
/* ===================================================================== */


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_getIndex(OcTensor *tensor, OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{
   return OcTensor_getIndexFlags(tensor, index, NULL, NULL);
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_getIndexFlags(OcTensor *tensor, OcTensorIndex *index,
                                 int *flagScalarPtr, int *flagViewPtr)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcTensorIndexView *view, OcTensor *dst);
   OcTensorIndex     *boundIndex;
   OcTensorIndexView *view = NULL;
   OcTensor          *result = NULL, *viewTensor;
   int                flagScalar, flagView;
   int                status;

   /* Bind the tensor index to the tensor size and strides */
   status = OcTensorIndex_bind(&index, 1, tensor -> ndims, tensor -> size, tensor -> strides, &boundIndex);
   if (status != 0) return NULL;

   /* Create a tensor view */
   status = OcTensorIndex_createView(boundIndex, tensor, &view);
   if (status != 0) goto final;

   /* Check type index type */
   flagScalar = OcTensorIndex_isScalar(boundIndex);
   flagView   = OcTensorIndex_isView(boundIndex);

   if (flagView)
   {  /* Return the view tensor */
      result = OcIncrefTensor(view -> view);
   }
   else
   {  /* Create the result tensor */
      viewTensor = view -> view;
      result = OcTensor_create(viewTensor -> ndims,
                               viewTensor -> size,
                               NULL,
                               viewTensor -> dtype,
                               viewTensor -> device);
      if (result == NULL) goto final;

      /* Set the byteswap flag if needed */
      if (OcTensor_isByteswapped(tensor))
         OcTensor_setByteswapped(result, 1);

      /* Look up the Tensor_getIndex function */
      if ((funptr = OC_GET_CORE_FUNCTION(viewTensor -> device, Tensor_getIndex)) == 0)
      {  OcErrorMessage("Tensor get index is not supported on device %s", viewTensor -> device -> type -> name);
      }

      /* Call the get index function */
      if ((funptr == 0) || (funptr(view, result) != 0))
      {  OcDecrefTensor(result); result = NULL;
         goto final;
      }
   }

final : ;
   if (result != NULL)
   {  if (flagScalarPtr != NULL) *flagScalarPtr = flagScalar;
      if (flagViewPtr   != NULL) *flagViewPtr   = flagView;
   }

   /* Delete intermediate data structures */
   if (view) OcTensorIndex_deleteView(view);
   OcDecrefTensorIndex(boundIndex);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensor_setIndex(OcTensor *tensor, OcTensorIndex *index, OcTensor *value)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcTensorIndexView *view, OcTensor *src);
   OcTensorIndex     *boundIndex;
   OcTensorIndexView *view = NULL;
   OcTensor          *viewTensor;
   OcTensor          *sourceTensor = NULL;
   OcScalar           scalar;
   int                overlap;
   int                byteswapped1, byteswapped2;
   int                status, success = 0;

   /* Check whether the tensor is read-only */
   if (OcTensor_isReadOnly(tensor))
      OcError(-1, "Cannot write to a read-only tensor");

   /* Check assignment of complex to real */
   if (OcTensor_isComplex(value) && OcTensor_isReal(tensor))
   {  if (OcWarning_raise(oc_warning_tensor_discard_imag) != 0) return -1;
   }

   /* Special case of scalar values */
   if (value -> nelem == 1)
   {  if (OcTensor_toScalar(value, &scalar) == 0)
         return OcTensor_fillIndex(tensor, index, &scalar);
   }

   /* Bind the tensor index to the tensor size and strides */
   status = OcTensorIndex_bind(&index, 1, tensor -> ndims, tensor -> size, tensor -> strides, &boundIndex);
   if (status != 0) return -1;

   /* Create a tensor view */
   status = OcTensorIndex_createView(boundIndex, tensor, &view);
   if (status != 0) goto final;

   /* Use copy if the bound index is a view */
   if (OcTensorIndex_isView(boundIndex))
   {  /* Copy the data */
      status = OcTensor_copy(value, view -> view);
      if (status == 0) success = 1;
      goto final;
   }

   /* Make sure that the devices and data types match */
   viewTensor   = view -> view;
   sourceTensor = OcIncrefTensor(value);
   if (OcTensor_ensure(&sourceTensor, viewTensor -> dtype, viewTensor -> device, NULL) != 0) goto final;

   /* Check byte order */
   byteswapped1 = OcTensor_isByteswapped(viewTensor);
   byteswapped2 = OcTensor_isByteswapped(sourceTensor);
   if ((byteswapped1 < 0) || (byteswapped2 < 0)) goto final;
   if (byteswapped1 != byteswapped2)
   {  /* Detach and byteswap the source tensor */
      if (OcTensor_detach(&sourceTensor) != 0) goto final;
      if (OcTensor_byteswap(sourceTensor) != 0) goto final;
   }

   /* Check for overlap */
   if ((overlap = OcTensors_overlap(sourceTensor, viewTensor)) < 0) goto final;
   if (overlap)
   {  OcTensor *clone;
      if ((clone = OcTensor_clone(sourceTensor)) == NULL) goto final;
      OcDecrefTensor(sourceTensor);
      sourceTensor = clone;
   }

   /* Broadcast the source tensor */
   if (OcTensor_broadcastTo(&sourceTensor, viewTensor -> ndims, viewTensor -> size, 0, NULL) != 0) goto final;

   /* Look up the Tensor_setIndex function */
   if ((funptr = OC_GET_CORE_FUNCTION(viewTensor -> device, Tensor_setIndex)) == 0)
   {  OcErrorMessage("Tensor set index is not supported on device %s", viewTensor -> device -> type -> name);
      goto final;
   }

   /* Call the get index function */
   if (funptr(view, sourceTensor) != 0) goto final;

   /* Success */
   success = 1;

final : ;
   /* Delete intermediate data structures */
   if (view) OcTensorIndex_deleteView(view);
   OcDecrefTensorIndex(boundIndex);
   OcXDecrefTensor(sourceTensor);

   return (success) ? 0 : -1;
}


/* -------------------------------------------------------------------- */
int OcTensor_fillIndex(OcTensor *tensor, OcTensorIndex *index, OcScalar *value)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcTensorIndexView *, OcScalar *);
   OcTensorIndex     *boundIndex;
   OcTensorIndexView *view = NULL;
   OcScalar           scalar;
   int                status, success = 0;

   /* Check if the index is boolean and covers all dimensions */
   if ((index -> n == 1) && (index -> elem[0] -> type == OC_IDX_MASK))
   {  OcTensor *mask = ((OcTensorIndexElem_Mask *)(index -> elem[0])) -> tensor;
      return OcTensor_maskedFill(tensor, mask, value);
   }

   /* Check whether the tensor is read-only */
   if (OcTensor_isReadOnly(tensor))
      OcError(-1, "Cannot write to a read-only tensor");

   /* Check assignment of complex to real */
   if (OcScalar_isComplex(value) && OcTensor_isReal(tensor))
   {  if (OcWarning_raise(oc_warning_tensor_discard_imag) != 0) return -1;
   }

   /* Convert the scalar */
   OcScalar_castTo(value, tensor -> dtype, &scalar);

   /* Look up the Tensor_setIndex function */
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_fillIndex)) == 0)
   {  OcError(-1, "Tensor index fill is not supported on device %s", tensor -> device -> type -> name);
   }

   /* Bind the tensor index to the tensor size and strides */
   status = OcTensorIndex_bind(&index, 1, tensor -> ndims, tensor -> size, tensor -> strides, &boundIndex);
   if (status != 0) return -1;

   /* Create a tensor view */
   status = OcTensorIndex_createView(boundIndex, tensor, &view);
   if (status != 0) goto final;

   /* Byte-swap the scalar if needed */
   if (!OcTensor_hasHostByteOrder(tensor)) OcScalar_byteswap(&scalar);

   /* Call the tensor index fill function */
   status = funptr(view, &scalar);
   if (status != 0) goto final;

   /* Success */
   success = 1;

final : ;
   /* Delete intermediate data structures */
   if (view) OcTensorIndex_deleteView(view);
   OcDecrefTensorIndex(boundIndex);

   return (success) ? 0 : -1;
}


/* -------------------------------------------------------------------- */
int OcTensor_getIndexValue(OcTensor *tensor, OcIndex *indices, OcScalar *value)
/* -------------------------------------------------------------------- */
{  OcSize   s, offset;
   OcIndex  index;
   int      flagIndexError;
   int      i, result;

   /* Determine the offset */
   flagIndexError = 0;
   offset = tensor -> offset;
   for (i = 0; i < tensor -> ndims; i++)
   {  index = indices[i];
      s = tensor -> size[i];

      /* Check the index */
      if (index < 0)
      {  index += s;
         if (index <  0) { flagIndexError = 1; break; }
      }
      else
      {  if (index >= s) { flagIndexError = 1; break; }
      }

      /* Update the offset */
      offset += index * (tensor -> strides[i]);
   }
   if (flagIndexError)
      OcError(-1, "Index out of range at dimension %d", i);

   /* Copy the data */
   result = OcStorage_copyToHost(tensor -> storage, offset,
                                 OcDType_size(tensor -> dtype), 
                                 OcScalar_data(value));
   if (result != 0) return result;

   /* Set the result data type */
   value -> dtype = tensor -> dtype;

   /* Byte-swap the result if needed */   
   if (!OcTensor_hasHostByteOrder(tensor)) OcScalar_byteswap(value);

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensor_setIndexValue(OcTensor *tensor, OcIndex *indices, OcScalar *value)
/* -------------------------------------------------------------------- */
{  OcScalar scalar, *ptrScalar;
   OcSize   s, offset;
   OcIndex  index;
   int      flagIndexError, flagHostOrder;
   int      i;

   /* Check whether the tensor is read-only */
   if (OcTensor_isReadOnly(tensor))
      OcError(-1, "Cannot write to a read-only tensor");

   /* Check assignment of complex to real */
   if (OcScalar_isComplex(value) && OcTensor_isReal(tensor))
   {  if (OcWarning_raise(oc_warning_tensor_discard_imag) != 0) return -1;
   }

   /* Determine the offset */
   flagIndexError = 0;
   offset = tensor -> offset;
   for (i = 0; i < tensor -> ndims; i++)
   {  index = indices[i];
      s = tensor -> size[i];

      /* Check the index */
      if (index < 0)
      {  index += s;
         if (index <  0) { flagIndexError = 1; break; }
      }
      else
      {  if (index >= s) { flagIndexError = 1; break; }
      }

      /* Update the offset */
      offset += index * (tensor -> strides[i]);
   }
   if (flagIndexError)
      OcError(-1, "Index out of range at dimension %d", i);

   /* Make sure the scalar has the correct data type */
   flagHostOrder = OcTensor_hasHostByteOrder(tensor);
   if ((flagHostOrder) && (value -> dtype == tensor -> dtype))
   {  ptrScalar = value;
   }
   else
   {  OcScalar_castTo(value, tensor -> dtype, &scalar);
      if (!flagHostOrder) OcScalar_byteswap(&scalar);
      ptrScalar = &scalar;
   }

   return OcStorage_copyFromHost(tensor -> storage, offset,
                                 OcDType_size(tensor -> dtype),
                                 OcScalar_data(ptrScalar));
}


/* ===================================================================== */
/* Function implementations - Tensor extraction                          */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
OcTensor *OcTensor_diag(OcTensor *tensor, OcIndex offset, int axis1, int axis2)
/* --------------------------------------------------------------------- */
{  OcSize    size[OC_TENSOR_MAX_DIMS];
   OcIndex   strides[OC_TENSOR_MAX_DIMS];
   OcSize    n;
   int       i, j;
   OcTensor *result;

   /* Parameter checks */
   if (tensor -> ndims < 2)
      OcError(NULL, "Tensor.diag requires the tensor to have at least two dimensions");
   if ((axis1 < 0) || (axis1 > tensor -> ndims))
      OcError(NULL, "Axis1 (%d) out of bounds, valid values are 0-%d", axis1, tensor->ndims-1);
   if ((axis2 < 0) || (axis2 > tensor -> ndims))
      OcError(NULL, "Axis1 (%d) out of bounds, valid values are 0-%d", axis1, tensor->ndims-1);
   if (axis1 == axis2)
      OcError(NULL, "Parameters axis1 and axis2 must differ");

   /* Exchange the axes if needed */
   if (offset < 0)
   {  offset *= -1;
      i = axis1; axis1 = axis2; axis2 = i;
   }

   /* Determine the number of elements in the diagonal */
   n = (offset <= tensor -> size[axis2]) ? tensor -> size[axis2] - offset : 0;
   if (n > tensor -> size[axis1]) n = tensor -> size[axis1];

   /* Determine the size and strides */
   for (i = 0, j = 0; i < tensor -> ndims; i++)
   {  if ((i == axis1) || (i == axis2)) continue;

      size[j] = tensor -> size[i];
      strides[j] = tensor -> strides[i];
      j ++;
   }
   size[j] = n;
   strides[j] = tensor -> strides[axis1] + tensor -> strides[axis2];

   /* Create a new tensor from storage */
   offset = (offset * tensor -> strides[axis2]) + (tensor -> offset);
   result = OcTensor_createFromStorage(tensor -> storage, tensor -> ndims - 1,
                                       size, strides, offset, tensor -> dtype);

   /* Copy byte-swap and read-only properties */
   if (result)
   {  result -> flags |= (tensor -> flags) & (OC_TENSOR_BYTESWAPPED | OC_TENSOR_READONLY);
   }

   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_real(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *result;

   /* Direct return for real tensors */
   if (!OcDType_isComplex(tensor -> dtype)) return OcIncrefTensor(tensor);

   /* Create a shallow copy */
   result = OcTensor_shallowCopy(tensor);
   if (result == NULL) return NULL;

   /* Update the data type */
   result -> dtype    = OcDType_getBaseType(tensor -> dtype);
   result -> elemsize = OcDType_size(result -> dtype);

   /* Update the flags and self-overlap information */
   OcTensor_updateShapeFlags(result, 1);

   /* Update the extent */
   OcTensor_updateExtent(result);

   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_imag(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *result;
   OcIndex   strides[OC_TENSOR_MAX_DIMS];
   int       i;

   if (!OcDType_isComplex(tensor -> dtype))
   {
      /* Create a tensor with zero strides */
      for (i = 0; i < tensor -> ndims; i++) strides[i] = 0;
      result = OcTensor_create(tensor -> ndims, tensor -> size, strides, tensor -> dtype, tensor -> device);
      if (result == NULL) return NULL;

      /* Zero out the tensor */
      if (OcTensor_zero(result) != 0)
      {  OcDecrefTensor(result);
         return NULL;
      }

      /* Make the tensor storage read-only */
      OcStorage_setReadOnly(result -> storage, 1);
   }
   else
   {
      /* Create a shallow copy */
      result = OcTensor_shallowCopy(tensor);
      if (result == NULL) return NULL;

      /* Update the data type */
      result -> dtype    = OcDType_getBaseType(tensor -> dtype);
      result -> elemsize = OcDType_size(result -> dtype);
      result -> offset  += result -> elemsize;

      /* Update the flags and self-overlap information */
      OcTensor_updateShapeFlags(result, 1);

      /* Update the extent */
      OcTensor_updateExtent(result);
   }

   return result;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_slice(OcTensor *tensor, int axis, OcSize offset, OcSize size)
/* -------------------------------------------------------------------- */
{  OcTensor *result;

   /* Check parameters */
   if ((axis < 0) || (axis >= tensor -> ndims))
      OcError(NULL, "Invalid axis for slice (%d)", axis);
   if ((offset > tensor -> size[axis]) || ((offset == tensor -> size[axis]) && (size > 0)))
   {  OcError(NULL, "Offset (%"OC_FORMAT_LU") must be smaller than the axis size (%"OC_FORMAT_LU")",
                    (long unsigned)(offset), (long unsigned)(tensor -> size[axis]));
   }
   if (tensor -> size[axis] - offset < size)
   {  OcError(NULL, "Offset plus size (%"OC_FORMAT_LU") exceeds the axis size (%"OC_FORMAT_LU")",
                    (long unsigned)(offset + size), (long unsigned)(tensor -> size[axis]));
   }
   if (!(size >= 0))
   {  OcError(NULL, "The slice size cannot be negative");
   }

   /* Create the new tensor */
   result = OcTensor_shallowCopy(tensor);
   if (result == NULL) return NULL;

   /* Update the size, offset, and number of elements */
   result -> nelem     /= result -> size[axis];
   result -> nelem     *= size;
   result -> size[axis] = size;
   result -> offset    += offset * result -> strides[axis];

   /* Update the flags and self-overlap information */
   OcTensor_updateShapeFlags(result, 1);

   /* Update the extent */
   OcTensor_updateExtent(result);
   
   return result;
}



/* ===================================================================== */
/* Function implementations - Basic tensor operations                    */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_intrnlFill(OcTensor *tensor, OcScalar *value)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *tensor, OcScalar *value);

   /* Look up the TensorFill function */
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_fill)) == 0)
      OcError(-1, "Tensor fill is not supported on device %s", tensor -> device -> type -> name);

   /* Call the function */
   return funptr(tensor, value);
}


/* --------------------------------------------------------------------- */
int OcTensor_zero(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcStorage *, void *, OcDType, OcSize);
   OcScalar value;
   OcSize   offset, extent;

   /* We only make sure that the tensor is not read-only;  */
   /* self-overlapping and byteswapped tensors are allowed */
   if (OcTensor_isReadOnly(tensor))
      OcError(-1, "Destination tensor cannot be read-only");

   /* Fast zero for contiguous tensors */
   if (OcTensor_isContiguous(tensor))
   {  if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Buffer_zero)) != 0)
      {  /* Get the tensor extent */
         if (OcTensor_extent(tensor, &offset, &extent) != 0) return -1;

         /* Fill the buffer with zeros */
         return funptr(tensor -> storage, OcTensor_data(tensor) - offset, OcDTypeInt8, extent);
      }
   }

   /* Apply fill with zero scalar */
   value.dtype = tensor -> dtype;
   OcScalar_fromInt64(&value, 0);
   if (!OcTensor_hasHostByteOrder(tensor)) OcScalar_byteswap(&value);
   return OcTensor_intrnlFill(tensor, &value);
}


/* --------------------------------------------------------------------- */
int OcTensor_fillOnes(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcScalar value;

   /* Fill the tensor with ones */
   value.dtype = tensor -> dtype;
   OcScalar_fromInt64(&value, 1);
   return OcTensor_fill(tensor, &value);
}


/* --------------------------------------------------------------------- */
int OcTensor_fill(OcTensor *tensor, OcScalar *scalar)
/* --------------------------------------------------------------------- */
{  OcScalar value;

   /* Make sure the tensor is a valid destination - allow zero strides */
   if (!OcTensor_isValidDest(tensor, 1)) return -1;

   /* Prepare the scalar */
   value.dtype = tensor -> dtype;
   OcScalar_copy(scalar, &value);
   if (!OcTensor_hasHostByteOrder(tensor)) OcScalar_byteswap(&value);

   /* Fill the tensor */
   return OcTensor_intrnlFill(tensor, &value);
}


/* --------------------------------------------------------------------- */
int OcTensor_fillNaN(OcTensor *tensor, OcScalar *scalar)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *tensor, OcScalar *value);
   OcScalar  value;
   OcTensor *t = tensor;
   int result = -1;

   /* This operation has no effect on integer tensors */
   if (!OcDType_isFloat(tensor -> dtype)) return 0;

   /* Make sure the tensor is a valid destination - allow zero strides */
   if (!OcTensor_isValidDest(tensor, 1)) return -1;

   /* Prepare the scalar */
   value.dtype = tensor -> dtype;
   OcScalar_copy(scalar, &value);

   /* Deal with byte-swapped tensors */
   if (!OcTensor_hasHostByteOrder(tensor))
   {  t = OcTensor_cloneFlags(tensor, tensor -> dtype, tensor -> device, 0, 1);
      if (t == NULL) return -1;
   }

   /* Look up the TensorFill function */
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_fillNaN)) == 0)
   {  OcErrorMessage("Tensor fillNaN is not supported on device %s", tensor -> device -> type -> name);
      goto final;
   }

   /* Call the function */
   if ((result = funptr(t, &value)) != 0) goto final;
   
   /* Copy the tensor */
   if (t != tensor) result = OcTensor_copy(t, tensor);

final :
   if (t != tensor) OcDecrefTensor(t);

   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_fillDouble(OcTensor *tensor, OcDouble value)
/* --------------------------------------------------------------------- */
{  OcScalar s;
   s.dtype = OcDTypeDouble;
   s.value.sDouble = value;

   return OcTensor_fill(tensor, &s);
}


/* --------------------------------------------------------------------- */
int OcTensor_fillInt64(OcTensor *tensor, OcInt64 value)
/* --------------------------------------------------------------------- */
{  OcScalar s;
   s.dtype = OcDTypeInt64;
   s.value.sInt64 = value;

   return OcTensor_fill(tensor, &s);
}


/* --------------------------------------------------------------------- */
int OcTensor_fillUInt64(OcTensor *tensor, OcUInt64 value)
/* --------------------------------------------------------------------- */
{  OcScalar s;
   s.dtype = OcDTypeUInt64;
   s.value.sUInt64 = value;
   
   return OcTensor_fill(tensor, &s);
}


/* --------------------------------------------------------------------- */
int OcTensor_maskedFill(OcTensor *tensor, OcTensor *mask, OcScalar *value)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *, OcTensor *, OcScalar *);
   OcScalar scalar;
   int      status, result = -1;

   /* Check whether the tensor is read-only */
   if (OcTensor_isReadOnly(tensor))
      OcError(-1, "Cannot write to a read-only tensor");

   /* Check assignment of complex to real */
   if (OcScalar_isComplex(value) && OcTensor_isReal(tensor))
   {  if (OcWarning_raise(oc_warning_tensor_discard_imag) != 0) return -1;
   }

   /* Convert the scalar */
   OcScalar_castTo(value, tensor -> dtype, &scalar);

   /* Look up the Tensor_maskedFill function */
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_maskedFill)) == 0)
   {  OcError(-1, "Tensor masked fill is not supported on device %s", tensor -> device -> type -> name);
   }

   /* Broadcast the mask */
   if ((mask = OcTensor_autoBroadcastLike(mask, tensor)) == NULL) return -1;

   /* Ensure the mask data type and device - it is assumed that */
   /* the bool data type is a single byte so no checks on the   */
   /* byte order are neeeded.                                   */
   if (OcTensor_ensure(&mask, OcDTypeBool, tensor -> device, NULL) != 0) goto final;

   /* Byte-swap the scalar if needed */
   if (!OcTensor_hasHostByteOrder(tensor)) OcScalar_byteswap(&scalar);

   /* Call the tensor index fill function */
   status = funptr(tensor, mask, &scalar);
   if (status != 0) goto final;

   /* Success */
   result = 0;

final : ;
   OcDecrefTensor(mask);
   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_range(OcScalar *start, OcScalar *stop, OcScalar *step,
                         OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcDType scalarDType;

   /* Check parameters */
   if (stop == NULL)
      OcError(NULL, "The stop parameter is required");

   /* Apply default data type */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;

   /* Get the data type of the scalars */
   scalarDType = stop -> dtype;
   if (start) scalarDType = OcDType_getCommonType(scalarDType, start -> dtype);
   if (step)  scalarDType = OcDType_getCommonType(scalarDType, step  -> dtype);
   if (scalarDType == OcDTypeBool) scalarDType = OcDTypeInt8;

   /* Convert complex to real */
   if (OcDType_isComplex(scalarDType))
      scalarDType = OcDType_getBaseType(scalarDType);

   /* Choose one of two data types */
   if (OcDType_isFloat(scalarDType))
      scalarDType = OcDTypeDouble;
   else
      scalarDType = OcDTypeInt64;

   /* Make sure we can convert the input parameters */
   if ((start != NULL) && (!OcScalar_inRange(start, scalarDType)))
      OcError(NULL, "The start value is out of range for data type %s", OcDType_name(scalarDType));
   if ((!OcScalar_inRange(stop, scalarDType)))
      OcError(NULL, "The stop value is out of range for data type %s", OcDType_name(scalarDType));
   if ((step != NULL) && (!OcScalar_inRange(step, scalarDType)))
      OcError(NULL, "The step value is out of range for data type %s", OcDType_name(scalarDType));

   /* Generate the range */
   if (scalarDType == OcDTypeDouble)
   {  return OcTensor_rangeDouble((start ? OcScalar_asDouble(start) : 0),
                                  OcScalar_asDouble(stop),
                                  (step ? OcScalar_asDouble(step) : 1),
                                  dtype, device);
   }
   else
   {  return OcTensor_rangeInt64((start ? OcScalar_asInt64(start) : 0),
                                 OcScalar_asInt64(stop),
                                 (step ? OcScalar_asInt64(step) : 1),
                                 dtype, device);
   }
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_rangeDouble(OcDouble start, OcDouble stop, OcDouble step,
                               OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *, double, double);
   OcTensor *tensor;
   OcSize    n;
   double    k;

   /* Make sure the parameters are valid */
   if (step == 0)
      OcError(NULL, "The step parameter cannot be zero");
   if (isnan(step))
      OcError(NULL, "The step parameter cannot be NaN");
   if (!isfinite(start))
      OcError(NULL, "The start parameter must be finite");
   if (!isfinite(stop))
      OcError(NULL, "The stop parameter must be finite");

   /* Apply default data type and device */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;
   if ((device = OcDevice_applyDefault(device)) == NULL) return NULL;

   /* Determine the number of elements */
   if (!isfinite(step))
   {  if (((step > 0) && (stop > start)) ||
          ((step < 0) && (stop < start)))
      {  n = 1; step = 0.0;
      }
      else
      {  n = 0; step = 0.0;
      }
   }
   else
   {  k = ((stop - start) / step);
      if (k < 0)
      {  n = 0;
      }
      else
      {  /* Check for exact division */
         n = (OcSize)k;
         if ((double)n != k) n ++;
      }
   }

   /* Deal with empty tensors */
   if (n == 0) return OcTensor_create(1, &n, NULL, dtype, device);

   /* Look up the TensorStepsDouble function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Tensor_stepsDouble)) == 0)
      OcError(NULL, "Range function is not supported on device %s", device -> type -> name);

   /* Create a new tensor */
   tensor = OcTensor_create(1, &n, NULL, dtype, device);
   if (tensor == NULL) return NULL;

   /* Call the function */
   if (funptr(tensor, start, step) != 0)
   {  OcDecrefTensor(tensor);
      return NULL;
   }

   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_rangeInt64(OcInt64 start, OcInt64 stop, OcInt64 step,
                              OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *, OcInt64, OcInt64);
   OcTensor *tensor;
   OcSize    n;

   /* Make sure the parameters are valid */
   if (step == 0)
      OcError(NULL, "The step parameter cannot be zero");

   /* Apply default data type and device */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;
   if ((device = OcDevice_applyDefault(device)) == NULL) return NULL;

   /* Determine the number of elements */
   n = ((stop - start) / step);
   if ((stop == start) ||
       ((stop < start) && (step > 0)) ||
       ((stop > start) && (step < 0)))
   {  n = 0;
   }
   else
   {  /* Check for exact division */
      if ((start - stop) % step != 0) n++;
   }

   /* Deal with empty tensors */
   if (n == 0) return OcTensor_create(1, &n, NULL, dtype, device);

   /* Look up the TensorStepsInt64 function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Tensor_stepsInt64)) == 0)
      OcError(NULL, "Range function is not supported on device %s", device -> type -> name);

   /* Create a new tensor */
   tensor = OcTensor_create(1, &n, NULL, dtype, device);
   if (tensor == NULL) return NULL;

   /* Call the function */
   if (funptr(tensor, start, step) != 0)
   {  OcDecrefTensor(tensor);
      return NULL;
   }

   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_linspace(OcScalar *start, OcScalar *stop,
                            OcSize nSamples, OcSize nIntervals,
                            OcScalar *spacing, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcDType scalarDType;

   /* Apply default data type and device */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;

   /* Convert start and stop to double or complex double */
   if (OcDType_isComplex(dtype))
        scalarDType = OcDTypeCDouble;
   else scalarDType = OcDTypeDouble;

   /* Make sure we can convert the input parameters */
   if (!OcScalar_inRange(start, scalarDType))
      OcError(NULL, "The start value is out of range for data type %s", OcDType_name(scalarDType));
   if (!OcScalar_inRange(stop, scalarDType))
      OcError(NULL, "The stop value is out of range for data type %s", OcDType_name(scalarDType));

   /* Generate the range */   
   if (scalarDType == OcDTypeDouble)
   {  return OcTensor_linspaceDouble(OcScalar_asDouble(start),
                                     OcScalar_asDouble(stop),
                                     nSamples, nIntervals,
                                     spacing, dtype, device);
   }
   else
   {  return OcTensor_linspaceCDouble(OcScalar_asCDouble(start),
                                      OcScalar_asCDouble(stop),
                                      nSamples, nIntervals,
                                      spacing, dtype, device);
   }
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_linspaceDouble(OcDouble start, OcDouble stop,
                                  OcSize nSamples, OcSize nIntervals,
                                  OcScalar *spacing, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *, double, double);
   OcTensor *tensor = NULL;
   OcSize    size = 0;
   OcDouble  step;

   /* Apply default data type and device */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;
   if ((device = OcDevice_applyDefault(device)) == NULL) return NULL;

   /* Switch to double mode if needed possible */
   if (OcDType_isComplex(dtype))
   {  OcCDouble value1, value2;
      value1.real = start; value1.imag = 0;
      value2.real = stop;  value2.imag = 0;
      return OcTensor_linspaceCDouble(value1, value2,
                                      nSamples, nIntervals,
                                      spacing, dtype, device);
   }
   
   /* Make sure the parameters are valid */
   if (!isfinite(start))
      OcError(NULL, "The start parameter must be finite");
   if (!isfinite(stop))
      OcError(NULL, "The stop parameter must be finite");

   /* Deal with the case of zero intervals */
   if (nIntervals == 0)
   {  if ((nSamples > 0) || (spacing != NULL))
         OcError(NULL, "Cannot determine spacing with zero intervals");
      return OcTensor_create(1, &size, NULL, dtype, device);
   }


   /* Look up the TensorStepsDouble function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Tensor_stepsDouble)) == 0)
      OcError(NULL, "Linspace function (double) is not supported on device %s", device -> type -> name);
      
   /* Compute the spacing */
   step = (stop - start) / nIntervals;

   /* Make sure all values are in-range */
   if (!OcDType_inRangeDouble(start, dtype))
      OcError(NULL, "The start value (%e) is out of range for data type %s", start, OcDType_name(dtype));
   if (!OcDType_inRangeDouble(start + (nSamples-1) * step, dtype))
      OcError(NULL, "The stop value (%e) is out of range for data type %s", (start + (nSamples-1) * step), OcDType_name(dtype));

   /* Create the tensor */
   size = nSamples;
   tensor = OcTensor_create(1, &size, NULL, dtype, device);
   if (tensor == NULL) return NULL;

   /* Call the function */
   if (funptr(tensor, start, step) != 0)
   { OcDecrefTensor(tensor); return NULL; }

   /* Output the spacing */
   if (spacing)
   {  spacing -> dtype = OcDTypeDouble;
      OcScalar_fromDouble(spacing, step);
   }

   /* Return the tensor */
   return tensor;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_linspaceCDouble(OcCDouble start, OcCDouble stop,
                                   OcSize nSamples, OcSize nIntervals,
                                   OcScalar *spacing, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *, OcCDouble, OcCDouble);
   OcTensor *tensor = NULL;
   OcSize    size = 0;
   OcCDouble step;
   OcDType   basetype;
   
   /* Apply default data type and device */
   if ((dtype = OcDType_applyDefault(dtype)) == OcDTypeNone) return NULL;
   if ((device = OcDevice_applyDefault(device)) == NULL) return NULL;

   /* Switch to double mode when possible */
   if (!OcDType_isComplex(dtype))
   {  return OcTensor_linspaceDouble(start.real, stop.real,
                                     nSamples, nIntervals,
                                     spacing, dtype, device);
   }

   /* Make sure the parameters are valid */
   if (!isfinite(start.real) || !isfinite(start.imag))
      OcError(NULL, "The start parameter must be finite");
   if (!isfinite(stop.real) || !isfinite(stop.real))
      OcError(NULL, "The stop parameter must be finite");

   /* Deal with the case of zero intervals */
   if (nIntervals == 0)
   {  if ((nSamples > 0) || (spacing != NULL))
         OcError(NULL, "Cannot determine spacing with zero intervals");
      return OcTensor_create(1, &size, NULL, dtype, device);
   }

   /* Look up the TensorStepsCDouble function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Tensor_stepsCDouble)) == 0)
      OcError(NULL, "Linspace function (complex double) is not supported on device %s", device -> type -> name);
      
   /* Compute the spacing */
   step.real = (stop.real - start.real) / nIntervals;
   step.imag = (stop.imag - start.imag) / nIntervals;

   /* Make sure all values are in-range */
   basetype = OcDType_getBaseType(dtype);
   if (!OcDType_inRangeDouble(start.real, basetype))
      OcError(NULL, "The start.real value (%e) is out of range for data type %s", start.real, OcDType_name(dtype));
   if (!OcDType_inRangeDouble(start.imag, basetype))
      OcError(NULL, "The start.imag value (%e) is out of range for data type %s", start.imag, OcDType_name(dtype));
   if (!OcDType_inRangeDouble(start.real + (nSamples-1) * step.real, basetype))
      OcError(NULL, "The stop.real value (%e) is out of range for data type %s", (start.real + (nSamples-1) * step.real), OcDType_name(dtype));
   if (!OcDType_inRangeDouble(start.imag + (nSamples-1) * step.imag, basetype))
      OcError(NULL, "The stop.imag value (%e) is out of range for data type %s", (start.imag + (nSamples-1) * step.imag), OcDType_name(dtype));

   /* Create the tensor */
   size = nSamples;
   tensor = OcTensor_create(1, &size, NULL, dtype, device);
   if (tensor == NULL) return NULL;

   /* Call the function */
   if (funptr(tensor, start, step) != 0)
   { OcDecrefTensor(tensor); return NULL; }

   /* Output the spacing */
   if (spacing)
   {  spacing -> dtype = OcDTypeCDouble;
      OcScalar_fromCDouble(spacing, step);
      { OcDecrefTensor(tensor); return NULL; }
   }

   /* Return the tensor */
   return tensor;
}



/* ===================================================================== */
/* Function implementations - Index manipulation                         */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
OcTensor *OcTensor_find(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *(*funptr)(OcTensor *tensor);
   OcTensor *local = NULL;
   OcTensor *result = NULL;

   /* The find operation is not defined on scalar tensors */
   if (tensor -> ndims == 0)
      OcError(NULL, "The find operation does not apply to zero-dimensional tensors");

   /* Make sure that the tensor is boolean - note that this  */
   /* creates a new tensor and increases the reference count */
   /* of the existing tensor, and we therefore own the tensor*/
   /* reference from here.                                   */
   if (OcTensor_ensureDType(&tensor, OcDTypeBool, &tensor) != 0)
      OcError(NULL, "Error converting input tensor to Boolean");

   /* Make sure that the tensor has native byte order */
   if (OcTensor_ensureByteOrder(&tensor, NULL) != 0)
   {  OcErrorMessage("Error converting tensor byte-order");
      goto final;
   }

   /* Look up the function on the tensor device */
   funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_find);
   if (funptr) { result = funptr(tensor); goto final; }

   /* -------------------------------------------------- */
   /* Could not find the desired function on the tensor  */
   /* device; fallback to the CPU                        */
   /* -------------------------------------------------- */
   
   /* Create a copy of the tensor on the CPU */
   if (OcTensor_ensureDevice(&tensor, OcCPU, &local) != 0) goto final;

   /* Get the function pointer on the CPU */
   funptr = OC_GET_CORE_FUNCTION(OcCPU, Tensor_find);
   if (funptr == 0)
   {  OcErrorMessage("Tensor find is not supported on device %s", tensor -> device -> name);
      goto final;
   }

   /* Apply the function */
   if ((result = funptr(local)) == NULL) goto final;

   /* Move result to the original device */
   if (OcTensor_ensureDevice(&result, tensor -> device, NULL) != 0) goto final;

final : ;
   /* Clean up intermediate tensors */
   OcDecrefTensor(tensor); /* Updated by ensure dtype */
   OcXDecrefTensor(local);
   return result;
}


/* --------------------------------------------------------------------- */
 OcTensor *OcTensor_maskToOffset(OcTensor *tensor, OcIndex *strides)
/* --------------------------------------------------------------------- */
{  OcTensor *(*funptr)(OcTensor *tensor, OcIndex *strides);
   OcTensor *local = NULL;
   OcTensor *result = NULL;

   /* The find operation is not defined on scalar tensors */
   if (tensor -> ndims == 0)
      OcError(NULL, "The mask to offset operation does not apply to zero-dimensional tensors");

   /* Make sure that the tensor is boolean */
   if (OcTensor_ensureDType(&tensor, OcDTypeBool, &tensor) != 0)
      OcError(NULL, "Error converting input tensor to Boolean");

   /* Make sure that the tensor has native byte order */
   if (OcTensor_ensureByteOrder(&tensor, NULL) != 0)
   {  OcErrorMessage("Error converting tensor byte-order");
      goto final;
   }

   /* Look up the function on the tensor device */
   funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_maskToOffset);
   if (funptr) { result = funptr(tensor, strides); goto final; }

   /* -------------------------------------------------- */
   /* Could not find the desired function on the tensor  */
   /* device; fallback to the CPU                        */
   /* -------------------------------------------------- */

   /* Create a copy of the tensor on the CPU */
   if (OcTensor_ensureDevice(&tensor, OcCPU, &local) != 0) goto final;

   /* Look for the function on the CPU */
   funptr = OC_GET_CORE_FUNCTION(OcCPU, Tensor_maskToOffset);
   if (funptr == 0)
   {  OcErrorMessage("Tensor mask to offset is not supported on device %s", tensor -> device -> name);
      goto final;
   }

   /* Apply the function */
   if ((result = funptr(local, strides)) == NULL) goto final;

   /* Move result to the original device */
   if (OcTensor_ensureDevice(&result, tensor -> device, NULL) != 0) goto final;

final : ;
   /* Clean up intermediate tensors */
   OcDecrefTensor(tensor); /* Updated by ensure dtype */
   OcXDecrefTensor(local);
   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_indexToOffset(OcTensor *tensor, OcIndex *strides, OcTensor **result)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *tensor, OcInt64 *strides, OcTensor *offset);
   OcInt64   data[OC_TENSOR_MAX_DIMS];
   OcTensor *offset = NULL;
   OcSize    size;
   OcSize    ndims;
   int       i, success = 0;

   /* Make sure the tensor is two dimensional */
   if ((tensor -> ndims == 0) || (tensor -> ndims > 2))
      OcError(-1, "The index to offset operation applies only to one- or two-dimensional tensors");

   /* Make sure that the tensor has native byte order */
   if (OcTensor_ensureByteOrder(&tensor, &tensor) != 0)
      OcError(-1, "Error converting tensor byte-order");

   /* Look up the function  */
   funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_indexToOffset);
   if (funptr == 0) goto final;

   /* Determine the size and number of dimensions */
   if (tensor -> ndims == 1)
   {  size  = tensor -> size[0];  ndims = 1; }
   else
   {  size  = tensor -> size[1];  ndims = tensor -> size[0]; }

   /* Make sure that the number of dimensions is valid; */
   /* this should never happen but check, just in case. */
   if (ndims > OC_TENSOR_MAX_DIMS)
   {  OcErrorMessage("Tensor index contains too many dimensions");
      goto final;
   }

   /* Copy the stride information to ensure the correct data type */
   for (i = 0; i < ndims; i++)
   {  if (strides[i] > OC_INT64_MAX)
      {  OcErrorMessage("Strides out of range in index-to-offset conversion"); goto final; }
      data[i] = (OcInt64)(strides[i]);
   }

   /* Check validity of the result tensor */
   if (*result)
   {  /* Check number of elements */
      if ((*result) -> nelem != size)
      {  OcErrorMessage("Incorrect size of result tensor in call to index to offset");
         goto final;
      }

      /* Check dimensions, data type, device, and byte order */
      offset = *result;
      if ((offset -> ndims != 1) ||
          (offset -> dtype != OcDTypeInt64) ||
          (offset -> device != tensor -> device) ||
          (OcTensor_isByteswapped(offset)))
      {  offset = NULL;
      }
   }

   /* Allocate the offset tensor */
   if (offset == NULL)
   {  if ((offset = OcTensor_create(1, &size, NULL, OcDTypeInt64, tensor -> device)) == NULL)
      {  OcErrorMessage("Error allocating offset tensor"); goto final; }
   }

   /* Call the function */
   if (funptr(tensor, data, offset) != 0) goto final;

   /* Assign the result */
   if (*result)
   {  if (offset != *result)
      {  if (OcTensor_copy(offset, *result) != 0) goto final;
      }
   }
   else
   {  *result = offset;
   }

   /* Successful */
   success = 1;
      
final : ;
   /* Clean up intermediate tensors */
   OcDecrefTensor(tensor); /* Updated by ensure byte order */

   /* Clean up the offset tensor */
   if ((success == 0) && (offset))
   {  if ((*result == NULL) || (offset != *result))
         OcDecrefTensor(offset);
   }

   return success ? 0 : -1;
}


/* --------------------------------------------------------------------- */
int OcTensor_addIfNegative(OcTensor *tensor, OcScalar *scalar)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *tensor, OcScalar *value);
   OcScalar value;

   /* Make sure the tensor is a valid destination - do not allow zero strides */
   if (!OcTensor_isValidDest(tensor, 0)) return -1;
   if (!OcTensor_hasHostByteOrder(tensor))
      OcError(-1, "Tensor add-if-negative is not supported on byte-swapped tensors");

   /* Look up the TensorAddIfNegative function */
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_addIfNegative)) == 0)
      OcError(-1, "Tensor add-if-negative is not supported on device %s", tensor -> device -> type -> name);

   /* Prepare the scalar */
   value.dtype = tensor -> dtype;
   OcScalar_copy(scalar, &value);

   /* Call the function */
   return funptr(tensor, &value);
}



/* ===================================================================== */
/* Function implementations - Unary tensor operations                    */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_unary(OcTensor *src, OcTensor **dstptr,
                   OcDType inputType, OcDType outputType,
                   const char *name, OcModule *module, size_t offset)
/* --------------------------------------------------------------------- */
{  OcTensorOp    info;
   OcBinaryFunc  funptr = 0;
   OcDevice     *device = src -> device;
   OcTensor     *dst = NULL;
   char         *moduleData;
   int           idxSrc, idxDst;
   int           result = -1;

   /* Get the function pointer */
   moduleData = OC_GET_MODULE(*module, device);
   if (moduleData) funptr = *((OcBinaryFunc *)(moduleData + offset));
   if (funptr == 0)
   {  OcError(-1, "Function '%s' is not supported on device %s", name, device -> name);
   }

   /* Determine the destination pointer */
   dst = (dstptr) ? *dstptr : src;

   /* Initialize the operation structure */
   OcTensorOp_initialize(&info);
   if (((idxSrc = OcTensorOp_addTensor(&info, &src, OC_OP_READ )) < 0) ||
       ((idxDst = OcTensorOp_addTensor(&info, &dst, OC_OP_WRITE)) < 0)) goto final;

   /* Apply checks and transformations */
   if ((OcTensorOp_broadcastElemwise(&info) != 0) ||
       (OcTensorOp_allocElemwiseIdx(&info, idxDst, outputType) != 0) ||
       (OcTensorOp_ensureDevice(&info, idxDst, device) != 0) ||
       (OcTensorOp_ensureDType(&info, idxSrc, inputType) != 0) ||
       (OcTensorOp_alignTensors(&info) != 0) ||
       (OcTensorOp_overlapElemwise(&info) != 0)) goto final;

   /* Apply the operation */
   result = funptr(src, dst);
   
final : ;
   /* Finalize the tensor operation */
   result = OcTensorOp_finalize(&info, result);

   /* Set the output pointer if needed */
   if ((result == 0) && (dstptr) && (*dstptr == NULL))
      *dstptr = dst;

   return result;
}



/* --------------------------------------------------------------------- */
OcDType OcTensor_intermediateTypeA(OcDType type1, OcDType type2)
/* --------------------------------------------------------------------- */
{  OcDType dtype;

   if (OcDType_isInteger(type1))
   {  if (OcDType_isInteger(type2))
         dtype = OcDType_getCommonType(type1, type2);
      else if (!OcDType_isComplex(type2))
         dtype = OcDTypeDouble;
      else
         dtype = (type2 == OcDTypeCDouble) ? type2 : OcDTypeDouble;
   }
   else if (!OcDType_isComplex(type2))
   {  if (OcDType_isInteger(type2))
         dtype = type1;
      else if (!OcDType_isComplex(type2))
         dtype = OcDType_getCommonType(type1, type2);
      else
         dtype = (OcDType_size(type2) >= 2*OcDType_size(type1)) ? type2 : type1;
   }
   else
   {  if (OcDType_isInteger(type2))
         dtype = type1;
      else if (!OcDType_isComplex(type2))
         dtype = type1;
      else
         dtype = OcDType_getCommonType(type1, type2);
   }

   return dtype;
}


/* --------------------------------------------------------------------- */
OcDType OcTensor_intermediateTypeB(OcDType type1, OcDType type2)
/* --------------------------------------------------------------------- */
{  OcDType dtype;

   if (OcDType_isInteger(type1) && OcDType_isInteger(type2))
      dtype = OcDTypeDouble;
   else
      dtype = OcTensor_intermediateTypeA(type1, type2);

   return dtype;
}


/* --------------------------------------------------------------------- */
OcDType OcTensor_intermediateTypeC(OcDType type1, OcDType type2)
/* --------------------------------------------------------------------- */
{  OcDType dtype;

   if (OcDType_isInteger(type1))
   {  if (OcDType_isInteger(type2))
         dtype = OcDTypeDouble;
      else if (!OcDType_isComplex(type2))
         dtype = OcDTypeDouble;
      else
         dtype = OcDTypeCDouble;
   }
   else if (!OcDType_isComplex(type2))
   {  if (OcDType_isInteger(type2))
         dtype = type1;
      else if (!OcDType_isComplex(type2))
         dtype = OcDType_getCommonType(type1, type2);
      else
         dtype = OcDType_getCommonType(OcDType_getComplexType(type1), type2);
   }
   else
   {  if (OcDType_isInteger(type2))
         dtype = type1;
      else if (!OcDType_isComplex(type2))
         dtype = type1;
      else
         dtype = OcDType_getCommonType(type1, type2);
   }

   return dtype;
}


/* --------------------------------------------------------------------- */
int OcTensor_negative(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType;

   /* Determine the intermediate or result type */
   castType = ((dst) && (*dst)) ? ((*dst) -> dtype) : src -> dtype;
   if (!OcDType_isSigned(src -> dtype) && !OcDType_isBool(src -> dtype))
   {  castType = OcDType_getSignedType(castType);
   }

   /* Apply the negative function */
   return OcTensor_unary(src, dst, castType, castType, "negative",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_negative));
}


/* --------------------------------------------------------------------- */
int OcTensor_bitwiseNot(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType;

   /* Determine the intermediate or result type */
   castType = src -> dtype;

   /* Apply the bitwise-not function */
   return OcTensor_unary(src, dst, castType, castType, "bitwise NOT",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_bitwiseNot));
}


/* --------------------------------------------------------------------- */
int OcTensor_logicalNot(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType = OcDTypeBool;

   /* Apply the logical-not function */
   return OcTensor_unary(src, dst, castType, castType, "logical NOT",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_logicalNot));
}


/* --------------------------------------------------------------------- */
int OcTensor_conj(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType;

   /* Determine the intermediate or result type */
   castType = ((dst) && (*dst)) ? ((*dst) -> dtype) : src -> dtype;

   /* Deal with non-complex result tensors */
   if (!OcDType_isComplex(castType))
   {  if (dst == NULL)
      {  /* Result is already in src */
         return 0;
      }
      else if (*dst == NULL)
      {  *dst = OcTensor_clone(src);
         return (*dst) ? 0 : -1;
      }
      else
      {  return OcTensor_copy(src, *dst);
      }
   }
   
   /* Apply the conjugate function */
   return OcTensor_unary(src, dst, castType, castType, "conjugate",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_conj));
}


/* --------------------------------------------------------------------- */
int OcTensor_cbrt(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType = ((dst)&&(*dst)) ? (*dst) -> dtype : src -> dtype;

   /* Determine the intermediate or result type */
   if ((dst) && (*dst == NULL))
   {  if (OcDType_isInteger(castType)) castType = OcDTypeDouble;
   }
   else
   {  castType = OcTensor_intermediateTypeA(src -> dtype, castType);
   }
   
   /* Apply the cube root function */
   return OcTensor_unary(src, dst, castType, castType, "cube root",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_cbrt));
}


/* --------------------------------------------------------------------- */
int OcTensor_square(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType = ((dst)&&(*dst)) ? (*dst) -> dtype : src -> dtype;

   /* Apply the square function */
   return OcTensor_unary(src, dst, castType, castType, "square",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_square));
}



/* --------------------------------------------------------------------- */
/* int OcTensor_sin    (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_cos    (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_tan    (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_sinh   (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_cosh   (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_tanh   (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_arcsin (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_arctanh(OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_exp    (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_exp2   (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_exp10  (OcTensor *src, OcTensor **dst)                   */
/* int OcTensor_expm1  (OcTensor *src, OcTensor **dst)                   */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, DESC) \
int OcTensor_ ## OPNAME(OcTensor *src, OcTensor **dst) \
{  OcDType castType = ((dst)&&(*dst)) ? (*dst) -> dtype : src -> dtype; \
   \
   /* Determine the intermediate or result type */ \
   if ((dst) && (*dst == NULL)) \
   {  if (OcDType_isInteger(castType)) castType = OcDTypeDouble; \
   } \
   else \
   {  castType = OcTensor_intermediateTypeA(src -> dtype, castType); \
   } \
   \
   /* Apply the operation */ \
   return OcTensor_unary(src, dst, castType, castType, DESC, \
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_##OPNAME)); \
}

OC_TEMPLATE(sin,     "sine")
OC_TEMPLATE(cos,     "cosine")
OC_TEMPLATE(tan,     "tangent")
OC_TEMPLATE(sinh,    "hyperbolic sine")
OC_TEMPLATE(cosh,    "hyperbolic cosine")
OC_TEMPLATE(tanh,    "hyperbolic tangent")
OC_TEMPLATE(arctan,  "arc tangent")
OC_TEMPLATE(arcsinh, "hyperbolic arc sine")
OC_TEMPLATE(exp,     "exponent")
OC_TEMPLATE(exp2,    "exponent base-2")
OC_TEMPLATE(exp10,   "exponent base-10")
OC_TEMPLATE(expm1,   "exponent minus one")
#undef OC_TEMPLATE


/* --------------------------------------------------------------------- */
int OcTensor_reciprocal(OcTensor *src, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcDType castType = ((dst)&&(*dst)) ? (*dst) -> dtype : src -> dtype;
   char mode = OcTensor_getDefaultMathMode();
   int condition;

   /* Determine the intermediate or result type */
   castType = OcTensor_intermediateTypeA(src -> dtype, castType);

   /* Check for division by zero */
   if ((mode == 'w') || (mode == 'e'))
   {  if ((condition = OcTensor_anyEQZero(src)) < 0) return -1;
      if (condition != 0)
      {  if (mode == 'w')
         {  if (OcWarning_raise(oc_warning_tensor_reciprocal) != 0) return -1;
         }
         else
         {  OcError(-1, "%s", OcWarning_message(oc_warning_tensor_reciprocal));
         }
      }
   }

   /* Apply the reciprocal function */
   return OcTensor_unary(src, dst, castType, castType, "reciprocal",
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_reciprocal));
}


/* --------------------------------------------------------------------- */
/* int OcTensor_arcsin (OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_arccos (OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_arccosh(OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_arctanh(OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_sqrt   (OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_log    (OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_log2   (OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_log10  (OcTensor *src, OcTensor **dst, char mode)        */
/* int OcTensor_log1p  (OcTensor *src, OcTensor **dst, char mode)        */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, CHECK, DESC) \
int OcTensor_ ## OPNAME(OcTensor *src, OcTensor **dst, char mode) \
{  OcDType castType = ((dst)&&(*dst)) ? (*dst) -> dtype : src -> dtype; \
   int condition; \
   \
   /* Determine the intermediate or result type */ \
   if ((dst) && (*dst == NULL)) \
   {  if (OcDType_isInteger(castType)) castType = OcDTypeDouble; \
   } \
   else \
   {  castType = OcTensor_intermediateTypeC(src -> dtype, castType); \
   } \
   \
   /* Check the domain */ \
   if ((!OcDType_isComplex(castType)) && \
       ((mode == 'w') || (mode == 'e') || (mode == 'c'))) \
   {  if ((condition = CHECK(src)) < 0) return -1; \
      if (condition != 0) \
      {  if (mode == 'c') \
         {  castType = OcDType_isFloat(castType) ? OcDType_getComplexType(castType) : OcDTypeCDouble; \
            if ((dst) && (*dst != NULL) && (!OcDType_isComplex((*dst)->dtype))) \
            {  if (OcWarning_raise(oc_warning_tensor_discard_imag2) != 0) return -1; \
            } \
         } \
         else if (mode == 'w') \
         {  if (OcWarning_raise(oc_warning_tensor_ ## OPNAME) != 0) return -1; \
         } \
         else /* Error mode */ \
         {  OcError(-1, "%s", OcWarning_message(oc_warning_tensor_ ## OPNAME)); \
         } \
      } \
   } \
   \
   /* Apply the operation */ \
   return OcTensor_unary(src, dst, castType, castType, DESC, \
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_##OPNAME)); \
}

OC_TEMPLATE(arcsin,  OcTensor_anyGTOneAbs, "arc sine")
OC_TEMPLATE(arccos,  OcTensor_anyGTOneAbs, "arc cosine")
OC_TEMPLATE(arccosh, OcTensor_anyLTOne,    "hyperbolic arc cosine")
OC_TEMPLATE(arctanh, OcTensor_anyGTOneAbs, "hyperbolic arc tangent")
OC_TEMPLATE(sqrt,    OcTensor_anyLTZero,   "square root")
OC_TEMPLATE(log,     OcTensor_anyLTZero,   "logarithm")
OC_TEMPLATE(log2,    OcTensor_anyLTZero,   "logarithm base-2")
OC_TEMPLATE(log10,   OcTensor_anyLTZero,   "logarithm base-10")
OC_TEMPLATE(log1p,   OcTensor_anyLTNegOne, "logarithm one plus")

#undef OC_TEMPLATE


/* --------------------------------------------------------------------- */
/* int OcTensor_fabs(OcTensor *src, OcTensor **dst)                      */
/* int OcTensor_sign(OcTensor *src, OcTensor **dst)                      */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, DESC) \
int OcTensor_ ## OPNAME(OcTensor *src, OcTensor **dst) \
{  OcDType castType = src -> dtype; \
   \
   /* Apply the operation */ \
   return OcTensor_unary(src, dst, castType, castType, DESC, \
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_##OPNAME)); \
}

OC_TEMPLATE(fabs, "fabs")
OC_TEMPLATE(sign, "sign")

#undef OC_TEMPLATE


/* --------------------------------------------------------------------- */
/* int OcTensor_absolute(OcTensor *src, OcTensor **dst)                  */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, DESC) \
int OcTensor_ ## OPNAME(OcTensor *src, OcTensor **dst) \
{  OcDType dtype = src -> dtype; \
   \
   /* Apply the operation */ \
   return OcTensor_unary(src, dst, dtype, OcDType_getBaseType(dtype), DESC, \
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_##OPNAME)); \
}

OC_TEMPLATE(absolute, "absolute")

#undef OC_TEMPLATE


/* --------------------------------------------------------------------- */
/* int OcTensor_ceil (OcTensor *src, OcTensor **dst)                     */
/* int OcTensor_floor(OcTensor *src, OcTensor **dst)                     */
/* int OcTensor_trunc(OcTensor *src, OcTensor **dst)                     */
/* int OcTensor_round(OcTensor *src, OcTensor **dst)                     */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, DESC) \
int OcTensor_ ## OPNAME(OcTensor *src, OcTensor **dst) \
{  OcDType castType = ((dst)&&(*dst)) ? (*dst) -> dtype : src -> dtype; \
   \
   /* Copy integer tensors */ \
   if (OcDType_isInteger(src -> dtype)) \
   {  if (dst == NULL) return 0;\
      if ((*dst) == NULL) \
      {  *dst = OcTensor_replicate(src); \
         return ((*dst) == NULL) ? -1 : 0; \
      } \
      else \
      {  return OcTensor_copy(src, *dst); \
      } \
   } \
   \
   /* Determine the intermediate or result type */ \
   castType = OcTensor_intermediateTypeA(src -> dtype, castType); \
   \
   /* Apply the operation */ \
   return OcTensor_unary(src, dst, castType, castType, DESC, \
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_##OPNAME)); \
}

OC_TEMPLATE(ceil,  "ceil")
OC_TEMPLATE(floor, "floor")
OC_TEMPLATE(trunc, "trunc")
OC_TEMPLATE(round, "round")

#undef OC_TEMPLATE


/* --------------------------------------------------------------------- */
/* int OcTensor_isinf   (OcTensor *src, OcTensor **dst)                  */
/* int OcTensor_isnan   (OcTensor *src, OcTensor **dst)                  */
/* int OcTensor_isfinite(OcTensor *src, OcTensor **dst)                  */
/* int OcTensor_isneginf(OcTensor *src, OcTensor **dst)                  */
/* int OcTensor_isposinf(OcTensor *src, OcTensor **dst)                  */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OPNAME, DESC) \
int OcTensor_ ## OPNAME(OcTensor *src, OcTensor **dst) \
{  \
   /* Apply the operation */ \
   return OcTensor_unary(src, dst, src -> dtype, OcDTypeBool, DESC, \
                         (OcModule *)&oc_module_core, offsetof(OcModuleCore, Tensor_##OPNAME)); \
}

OC_TEMPLATE(isinf,    "isinf")
OC_TEMPLATE(isnan,    "isnan")
OC_TEMPLATE(isfinite, "isfinite")
OC_TEMPLATE(isneginf, "isneginf")
OC_TEMPLATE(isposinf, "isposinf")

#undef OC_TEMPLATE



/* ===================================================================== */
/* Function implementations - Tensor mathematical operations             */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
/* int OcTensor_add     (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_subtract(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SWAP, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   int flagInplace = 0; \
   int result = -1; \
   \
   /* Check for in-place operations */ \
   if ((*dst != NULL) && (OcTensors_match(src1, *dst))) \
   {  flagInplace = 1; \
   } \
   else if ((SWAP) && (*dst != NULL) && (OcTensors_match(src2, *dst))) \
   {  OcTensor *tmp; \
      tmp = src1; src1 = src2; src2 = tmp; \
      flagInplace = 1; \
   } \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (!flagInplace) \
   {  if (OcTensorOp_addTensor(&info, &src1, OC_OP_READ) < 0) goto final; \
   } \
   if ((OcTensorOp_addTensor(&info,  dst,  OC_OP_WRITE) < 0) || \
       (OcTensorOp_addTensor(&info, &src2, OC_OP_READ) < 0)) goto final; \
   \
   /* Determine the common device and data type */ \
   if (OcTensorOp_commonType(&info) != 0) goto final; \
   if (flagInplace) info.device = (*dst) -> device; \
   \
   /* Avoid computations with complex values if the output is real */ \
   if ((OcDType_isComplex(info.dtype)) && (*dst) && (OcTensor_isReal(*dst))) \
   {  OcWarning_raise(oc_warning_tensor_discard_imag); \
      info.dtype = OcDType_getBaseType(info.dtype); \
   } \
   \
   /* Prepare for elementwise operation */ \
   if (OcTensorOp_prepareElemwise(&info) != 0) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(add,      1, "addition")
OC_TEMPLATE(subtract, 0, "subtraction")
#undef OC_TEMPLATE



/* ------------------------------------------------------------------------ */
/* int OcTensor_scale      (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_divide     (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_trueDivide (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_floorDivide(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* ------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SWAP, CHECK, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   char mode = OcTensor_getDefaultMathMode(); \
   int flagInplace = 0; \
   int result = -1; \
   \
   /* Check for in-place operations */ \
   if ((*dst != NULL) && (OcTensors_match(src1, *dst))) \
   {  flagInplace = 1; \
   } \
   else if ((SWAP) && (*dst != NULL) && (OcTensors_match(src2, *dst))) \
   {  OcTensor *tmp; \
      tmp = src1; src1 = src2; src2 = tmp; \
      flagInplace = 1; \
   } \
   \
   /* Domain checks */ \
   if (CHECK) \
   {  int condition; \
      if ((condition = OcTensor_anyEQZero(src2)) < 0) return -1; \
      if (condition != 0) \
      {  if (mode == 'w') \
         {  if (OcWarning_raise(oc_warning_tensor_division_zero) != 0) return -1; \
         } \
         else if (mode == 'e') \
         {  OcError(-1, "%s", OcWarning_message(oc_warning_tensor_division_zero)); \
         } \
      } \
   } \
   else \
   {  (void)mode; /* Suppress possible compile warnings */ \
   } \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (!flagInplace) \
   {  if (OcTensorOp_addTensor(&info, &src1, OC_OP_READ) < 0) goto final; \
   } \
   if ((OcTensorOp_addTensor(&info,  dst,  OC_OP_WRITE) < 0) || \
       (OcTensorOp_addTensor(&info, &src2, OC_OP_READ) < 0)) goto final; \
   \
   /* Determine the common device and data type */ \
   if (OcTensorOp_commonType(&info) != 0) goto final; \
   if (flagInplace) info.device = (*dst) -> device; \
   \
   /* Avoid computations with complex values if the output is real */ \
   if ((OcDType_isComplex(info.dtype)) && (*dst) && (OcTensor_isReal(*dst))) \
   {  if (((src1 == NULL) || (OcTensor_isReal(src1))) || \
          ((src2 == NULL) || (OcTensor_isReal(src2)))) \
      {  OcWarning_raise(oc_warning_tensor_discard_imag); \
         info.dtype = OcDType_getBaseType(info.dtype); \
      } \
   } \
   /* Prepare for elementwise operation */ \
   if (OcTensorOp_prepareElemwise(&info) != 0) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(scale,       1, 0, "scaling"       )
OC_TEMPLATE(divide,      0, 1, "division"      )
OC_TEMPLATE(trueDivide,  0, 1, "true division" )
OC_TEMPLATE(floorDivide, 0, 1, "floor division")
#undef OC_TEMPLATE


/* ----------------------------------------------------------------- */
/* int OcTensor_mod (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_fmod(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* ----------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   char mode = OcTensor_getDefaultMathMode(); \
   int flagInplace = 0; \
   int condition; \
   int result = -1; \
   \
   /* Do not allow complex source data */ \
   if (((src1) && (OcTensor_isComplex(src1))) || \
       ((src2) && (OcTensor_isComplex(src2)))) \
      OcError(-1, "Tensor "#DESC " is not defined for complex data"); \
   \
   /* Check for in-place operations */ \
   if ((*dst != NULL) && (OcTensors_match(src1, *dst))) \
   {  flagInplace = 1; \
   } \
   \
   /* Domain checks */ \
   if ((mode == 'w') || (mode == 'e')) \
   {  if ((condition = OcTensor_anyEQZero(src2)) < 0) return -1; \
      if (condition != 0) \
      {  if (mode == 'w') \
         {  if (OcWarning_raise(oc_warning_tensor_modulo_zero) != 0) return -1; \
         } \
         else \
         {  OcError(-1, "%s", OcWarning_message(oc_warning_tensor_modulo_zero)); \
         } \
      } \
   } \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (!flagInplace) \
   {  if (OcTensorOp_addTensor(&info, &src1, OC_OP_READ) < 0) goto final; \
   } \
   if ((OcTensorOp_addTensor(&info,  dst,  OC_OP_WRITE) < 0) || \
       (OcTensorOp_addTensor(&info, &src2, OC_OP_READ) < 0)) goto final; \
   \
   /* Determine the common device and data type */ \
   if (OcTensorOp_commonType(&info) != 0) goto final; \
   info.dtype  = OcDType_getBaseType(info.dtype); \
   info.device = src1 -> device; \
   \
   /* Prepare for elementwise operation */ \
   if (OcTensorOp_prepareElemwise(&info) != 0) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(mod,  "mod")
OC_TEMPLATE(fmod, "fmod")
#undef OC_TEMPLATE



/* ------------------------------------------------------------------------- */
/* int OcTensor_addScalar       (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_subtractScalar  (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_multiplyScalar  (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_bitwiseAndScalar(OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_bitwiseOrScalar (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_bitwiseXorScalar(OcTensor *src, OcScalar *s, OcTensor **dst) */
/* ------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, PREFIX) \
int OcTensor_##OP(OcTensor *src, OcScalar *s, OcTensor **dst) \
{  OcTensor *tensor; \
   OcDevice *device; \
   int result; \
   \
   /* ----------------------------------------------------------- */ \
   /* Possible optimization: have separate functions implementing */ \
   /* this operation directly, rather than use the generic tensor */ \
   /* version of the operation.                                   */ \
   /* ----------------------------------------------------------- */ \
   \
   /* Determine the device */ \
   device = (*dst) ? (*dst) -> device : src -> device; \
   \
   /* Convert the scalar to a tensor */ \
   tensor = OcTensor_createFromScalar(s, s -> dtype, device, 1); \
   if (tensor == NULL) return -1; \
   \
   /* Call the generic tensor function */ \
   result = OcTensor_##PREFIX(src, tensor, dst); \
   \
   /* Free the intermediate tensor */ \
   OcDecrefTensor(tensor); \
   \
   return result; \
}

OC_TEMPLATE(addScalar,        add)
OC_TEMPLATE(subtractScalar,   subtract)
OC_TEMPLATE(multiplyScalar,   scale)
OC_TEMPLATE(bitwiseAndScalar, bitwiseAnd)
OC_TEMPLATE(bitwiseOrScalar,  bitwiseOr)
OC_TEMPLATE(bitwiseXorScalar, bitwiseXor)
#undef OC_TEMPLATE


/* -------------------------------------------------------------------------- */
/* int OcTensor_divideScalar     (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_trueDivideScalar (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_floorDivideScalar(OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_modScalar        (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_fmodScalar       (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* -------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, WARNING) \
int OcTensor_##OP##Scalar(OcTensor *src, OcScalar *s, OcTensor **dst) \
{  OcTensor *tensor; \
   OcDevice *device; \
   char mode = OcTensor_getDefaultMathMode(); \
   int result; \
   \
   /* ----------------------------------------------------------- */ \
   /* Possible optimization: have separate functions implementing */ \
   /* this operation directly, rather than use the generic tensor */ \
   /* version of the operation.                                   */ \
   /* ----------------------------------------------------------- */ \
   \
   /* Check for division by zero */ \
   if (OcScalar_isZero(s)) \
   {  if (mode == 'w') \
      {  if (OcWarning_raise(WARNING) != 0) return -1; \
      } \
      else if (mode == 'e') \
      {  OcError(-1, "%s", OcWarning_message(WARNING)); \
      } \
   } \
   \
   /* Determine the devicew */ \
   device = src -> device; \
   \
   /* Convert the scalar to a tensor */ \
   tensor = OcTensor_createFromScalar(s, s -> dtype, device, 1); \
   if (tensor == NULL) return -1; \
   \
   /* Call the generic tensor function */ \
   result = OcTensor_##OP(src, tensor, dst); \
   \
   /* Free the intermediate tensor */ \
   OcDecrefTensor(tensor); \
   \
   return result; \
}

OC_TEMPLATE(divide,      oc_warning_tensor_division_zero)
OC_TEMPLATE(trueDivide,  oc_warning_tensor_division_zero)
OC_TEMPLATE(floorDivide, oc_warning_tensor_division_zero)
OC_TEMPLATE(mod,         oc_warning_tensor_modulo_zero  )
OC_TEMPLATE(fmod,        oc_warning_tensor_modulo_zero  )
#undef OC_TEMPLATE


/* ---------------------------------------------------------------------------- */
/* int OcTensor_bitshiftLeftScalar (OcTensor *src, OcScalar *s, OcTensor **dst) */
/* int OcTensor_bitshiftRightScalar(OcTensor *src, OcScalar *s, OcTensor **dst) */
/* ---------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP##Scalar(OcTensor *src, OcScalar *s, OcTensor **dst) \
{  OcTensor *tensor; \
   OcDevice *device; \
   OcScalar scalar; \
   int result; \
   \
   /* ----------------------------------------------------------- */ \
   /* Possible optimization: have separate functions implementing */ \
   /* this operation directly, rather than use the generic tensor */ \
   /* version of the operation.                                   */ \
   /* ----------------------------------------------------------- */ \
   \
   /* Make sure the shift count is integer */ \
   if (!OcDType_isInteger(s -> dtype)) \
      OcError(-1, "Tensor "#DESC " count must be integer"); \
   \
   /* Check the range of the shift count. If it exceeds the 8-bit */ \
   /* signed integer range, zero out the destination.             */ \
   if (!OcScalar_inRange(s, OcDTypeInt8)) \
   {  scalar.dtype = src -> dtype; \
      OcScalar_fromInt64(&scalar,0); \
      if (OcTensor_multiplyScalar(src, &scalar, dst) != 0) \
         OcError(-1, "Error applying the tensor "#DESC " operation"); \
      else return 0; \
   } \
   \
   /* Determine the device */ \
   device = src -> device; \
   \
   /* Convert the scalar to a tensor */ \
   tensor = OcTensor_createFromScalar(s, OcDTypeInt8, device, 1); \
   if (tensor == NULL) return -1; \
   \
   /* Call the generic tensor function */ \
   result = OcTensor_##OP(src, tensor, dst); \
   \
   /* Free the intermediate tensor */ \
   OcDecrefTensor(tensor); \
   \
   return result; \
}

OC_TEMPLATE(bitshiftLeft,  "bitshift left")
OC_TEMPLATE(bitshiftRight, "bitshift right")
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
int OcTensor_powerScalar(OcTensor *src, OcScalar *s, OcTensor **dst, char mode)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor;
   OcDevice *device;
   OcDType dtype;
   int result;

   /* ----------------------------------------------------------- */
   /* Possible optimization: have separate functions implementing */
   /* this operation directly, rather than use the generic tensor */
   /* version of the operation.                                   */
   /* ----------------------------------------------------------- */

   /* Determine the data type */
   dtype = (s -> dtype == OcDTypeUInt64) ? OcDTypeDouble : s -> dtype;
   if (OcDType_isInteger(dtype) && OcDType_isInteger(src -> dtype))
        dtype = OcDTypeInt16;
   else dtype = OcDType_getCommonType(dtype, src -> dtype);

   /* Determine the device */
   device = src -> device;

   /* Convert the scalar to a tensor */
   tensor = OcTensor_createFromScalar(s, dtype, device, 1);
   if (tensor == NULL) return -1;

   /* Call the generic tensor function */
   result = OcTensor_power(src, tensor, dst, mode);

   /* Free the intermediate tensor */
   OcDecrefTensor(tensor);

   return result;
}


/* ----------------------------------------------------------------------- */ \
/* int OcTensor_elemwiseLT(OcTensor *src1, OcTensor *src2, OcTensor **dst) */ \
/* int OcTensor_elemwiseLE(OcTensor *src1, OcTensor *src2, OcTensor **dst) */ \
/* int OcTensor_elemwiseEQ(OcTensor *src1, OcTensor *src2, OcTensor **dst) */ \
/* int OcTensor_elemwiseNE(OcTensor *src1, OcTensor *src2, OcTensor **dst) */ \
/* int OcTensor_elemwiseGE(OcTensor *src1, OcTensor *src2, OcTensor **dst) */ \
/* int OcTensor_elemwiseGT(OcTensor *src1, OcTensor *src2, OcTensor **dst) */ \
/* ----------------------------------------------------------------------- */ \
#define OC_TEMPLATE(OP,NAME) \
int OcTensor_elemwise##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   int idxDst; \
   int result = -1; \
   \
   /* Initialize the operation */ \
   if (OcTensorOp_initialize2(&info, &src1, OC_OP_READ, \
                                     &src2, OC_OP_READ) != 0) goto final; \
   \
   /* Determine the common device and data type */ \
   if (OcTensorOp_commonType(&info) != 0) goto final; \
   if (*dst) info.device = (*dst) -> device; \
   \
   /* Apply the data types to the input tensors */ \
   if (OcTensorOp_applyTypes(&info) != 0) goto final; \
   \
   /* Add the destination tensor */ \
   if ((idxDst = OcTensorOp_addTensor(&info, dst, OC_OP_WRITE)) < 0) goto final; \
   \
   /* Complete the initialization operations */ \
   if ((OcTensorOp_broadcastElemwise(&info) != 0) || \
       (OcTensorOp_allocElemwiseIdx(&info, idxDst, OcDTypeBool) != 0) || \
       (OcTensorOp_alignTensors(&info) != 0) || \
       (OcTensorOp_overlapElemwise(&info) != 0)) goto final; \
   \
   /* Look up the comparison function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_elemwise##OP)) == 0) \
   {  OcErrorMessage("Tensor elementwise " NAME " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* Apply the function */ \
   result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(LT, "less-than"    )
OC_TEMPLATE(LE, "less-equal"   )
OC_TEMPLATE(EQ, "equal"        )
OC_TEMPLATE(NE, "not-equal"    )
OC_TEMPLATE(GE, "greater-equal")
OC_TEMPLATE(GT, "greater-than" )
#undef OC_TEMPLATE



/* ------------------------------------------------------------------------- */
/* int OcTensor_elemwiseMin (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_elemwiseMax (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_elemwiseFMin(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_elemwiseFMax(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* ------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   int flagInplace; \
   int result = -1; \
   \
   /* Check for in-place operations */ \
   flagInplace = ((*dst != NULL) && (OcTensors_match(src1, *dst))); \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (!flagInplace) \
   if (((!flagInplace) && \
        (OcTensorOp_addTensor(&info, &src1, OC_OP_READ ) < 0)) || \
       ((OcTensorOp_addTensor(&info, dst,   OC_OP_WRITE) < 0)) || \
       ((OcTensorOp_addTensor(&info, &src2, OC_OP_READ ) < 0))) goto final; \
   \
   /* Determine the common device and data type */ \
   info.dtype  = OcDType_getCommonType(src1 -> dtype, src2 -> dtype); \
   info.device = src1 -> device; \
   \
   /* Prepare for elementwise operation */ \
   if (OcTensorOp_prepareElemwise(&info) != 0) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(elemwiseMin,  "elementwise minimum")
OC_TEMPLATE(elemwiseMax,  "elementwise maximum")
OC_TEMPLATE(elemwiseFMin, "elementwise minimum")
OC_TEMPLATE(elemwiseFMax, "elementwise maximum")
#undef OC_TEMPLATE



/* ----------------------------------------------------------------------- */
/* int OcTensor_logicalAnd(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_logicalOr (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_logicalXor(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* ----------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   int flagInplace; \
   int result = -1; \
   \
   /* Check for in-place operations */ \
   flagInplace = ((*dst != NULL) && (OcTensors_match(src1, *dst))); \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (((!flagInplace) && \
        (OcTensorOp_addTensor(&info, &src1, OC_OP_READ ) < 0)) || \
       ((OcTensorOp_addTensor(&info, dst,   OC_OP_WRITE) < 0)) || \
       ((OcTensorOp_addTensor(&info, &src2, OC_OP_READ ) < 0))) goto final; \
   \
   /* Determine the common device and data type */ \
   info.dtype  = OcDTypeBool; \
   info.device = src1 -> device; \
   \
   /* Prepare for elementwise operation */ \
   if (OcTensorOp_prepareElemwise(&info) != 0) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(logicalAnd, "logical AND")
OC_TEMPLATE(logicalOr,  "logical OR" )
OC_TEMPLATE(logicalXor, "logical XOR")
#undef OC_TEMPLATE



/* ----------------------------------------------------------------------- */
/* int OcTensor_bitwiseAnd(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_bitwiseOr (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_bitwiseXor(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* ----------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   OcDType dtype, dtypeSrc1, dtypeSrc2, dtypeDst; \
   int idxSrc1, idxSrc2, idxDst; \
   int flagInplace; \
   int result = -1; \
   \
   /* Both sources must be integer */ \
   if (!OcDType_isInteger(src1 -> dtype) || !OcDType_isInteger(src2 -> dtype)) \
     OcError(-1, "Tensor "#DESC " is only supported on integer types"); \
   \
   /* Determine the common data type */ \
   if (OcDType_nbits(src1 -> dtype) >= OcDType_nbits(src2 -> dtype)) \
        dtype = src1 -> dtype; \
   else dtype = src2 -> dtype; \
   \
   /* Check for in-place operations */ \
   flagInplace = ((*dst != NULL) && (OcTensors_match(src1, *dst))); \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (((!flagInplace) &&\
        ((idxSrc1 = OcTensorOp_addTensor(&info, &src1, OC_OP_READ )) < 0)) || \
       (((idxDst  = OcTensorOp_addTensor(&info,  dst,  OC_OP_WRITE)) < 0)) || \
       (((idxSrc2 = OcTensorOp_addTensor(&info, &src2, OC_OP_READ )) < 0))) goto final; \
   \
   /* Ensure number of bits in the integers, ignore signedness */ \
   dtypeSrc1 = src1 -> dtype; \
   dtypeSrc2 = src2 -> dtype; \
   dtypeDst  = (*dst) ? (*dst) -> dtype : OcDTypeNone;  \
   if (OcDType_nbits(dtypeSrc1) != OcDType_nbits(dtype)) dtypeSrc1 = dtype; \
   if (OcDType_nbits(dtypeSrc2) != OcDType_nbits(dtype)) dtypeSrc2 = dtype; \
   if (OcDType_nbits(dtypeDst ) != OcDType_nbits(dtype)) dtypeDst  = dtype;  \
   \
   /* Set the data type for possible allocation of destination tensor */ \
   info.dtype  = dtypeDst; \
   info.device = src1 -> device; \
   \
   /* Make sure the device and data types are correct */ \
   if (((!flagInplace) && \
        (OcTensorOp_ensure(&info, idxSrc1, dtypeSrc1, info.device) != 0)) || \
       ((OcTensorOp_ensure(&info, idxSrc2, dtypeSrc2, info.device) != 0)) || \
       ((OcTensorOp_ensure(&info, idxDst,  dtypeDst,  info.device) != 0))) goto final; \
   \
   /* Remaining elementwise preparations */ \
   if ((OcTensorOp_broadcastElemwise(&info) != 0) || \
       (OcTensorOp_allocElemwise(&info)     != 0) || \
       (OcTensorOp_alignTensors(&info)      != 0) || \
       (OcTensorOp_overlapElemwise(&info)   != 0)) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(bitwiseAnd, "bitwise AND")
OC_TEMPLATE(bitwiseOr,  "bitwise OR" )
OC_TEMPLATE(bitwiseXor, "bitwise XOR")
#undef OC_TEMPLATE



/* -------------------------------------------------------------------------- */
/* int OcTensor_bitshiftLeft (OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* int OcTensor_bitshiftRight(OcTensor *src1, OcTensor *src2, OcTensor **dst) */
/* -------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP(OcTensor *src1, OcTensor *src2, OcTensor **dst) \
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *); \
   OcTensorOp info; \
   OcDType dtype, dtypeSrc, dtypeDst; \
   int idxSrc1, idxSrc2, idxDst; \
   int flagInplace; \
   int result = -1; \
   \
   /* Source #1 must be integer */ \
   if (!OcDType_isInteger(src1 -> dtype)) \
     OcError(-1, "Tensor "#DESC " is only supported on integer input"); \
   \
   /* Determine the common data type */ \
   if ((*dst != NULL) && (OcDType_isInteger((*dst) -> dtype)) && \
       (OcDType_nbits(src1 -> dtype) < OcDType_nbits((*dst) -> dtype))) \
        dtype = (*dst) -> dtype; \
   else dtype = src1 -> dtype; \
   \
   /* Check for in-place operations */ \
   flagInplace = ((*dst != NULL) && (OcTensors_match(src1, *dst))); \
   \
   /* Initialize the tensor operator */ \
   OcTensorOp_initialize(&info); \
   if (((!flagInplace) &&\
        ((idxSrc1 = OcTensorOp_addTensor(&info, &src1, OC_OP_READ )) < 0)) || \
       (((idxDst  = OcTensorOp_addTensor(&info,  dst,  OC_OP_WRITE)) < 0)) || \
       (((idxSrc2 = OcTensorOp_addTensor(&info, &src2, OC_OP_READ )) < 0))) goto final; \
   \
   /* Ensure number of bits in the integers, ignore signedness */ \
   dtypeSrc = src1 -> dtype; \
   dtypeDst = (*dst) ? (*dst) -> dtype : OcDTypeNone;  \
   if (OcDType_nbits(dtypeSrc) != OcDType_nbits(dtype)) dtypeSrc = dtype; \
   if (OcDType_nbits(dtypeDst) != OcDType_nbits(dtype)) dtypeDst = dtype;  \
   \
   /* Set the data type for possible allocation of destination tensor */ \
   info.dtype  = dtypeDst; \
   info.device = src1 -> device; \
   \
   /* Make sure the device and data types are correct */ \
   if (((!flagInplace) && \
        (OcTensorOp_ensure(&info, idxSrc1, dtypeSrc,    info.device) != 0)) || \
       ((OcTensorOp_ensure(&info, idxSrc2, OcDTypeInt8, info.device) != 0)) || \
       ((OcTensorOp_ensure(&info, idxDst,  dtypeDst,    info.device) != 0))) goto final; \
   \
   /* Remaining elementwise preparations */ \
   if ((OcTensorOp_broadcastElemwise(&info) != 0) || \
       (OcTensorOp_allocElemwise(&info)     != 0) || \
       (OcTensorOp_alignTensors(&info)      != 0) || \
       (OcTensorOp_overlapElemwise(&info)   != 0)) goto final; \
   \
   /* Look up the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("Tensor "#DESC " is not supported on device %s", info.device -> name); \
      goto final; \
   } \
   \
   /* --------------------------------------------------------------- */ \
   /* Possible optimization: special function for in-place operation. */ \
   /* --------------------------------------------------------------- */ \
   \
   /* Apply the function */ \
   if (flagInplace) \
        result = funptr(*dst, src2, *dst); \
   else result = funptr(src1, src2, *dst); \
   \
final : \
   return OcTensorOp_finalize(&info, result); \
}

OC_TEMPLATE(bitshiftLeft,  "bitshift left" )
OC_TEMPLATE(bitshiftRight, "bitshift right")
#undef OC_TEMPLATE



/* -------------------------------------------------------------------------- */
int OcTensor_power(OcTensor *src1, OcTensor *src2, OcTensor **dst, char mode)
/* -------------------------------------------------------------------------- */
{  int (*funptr)(OcTensor *, OcTensor *, OcTensor *);
   OcTensorOp info;
   OcDType dtype, dtypeExp;
   int idxSrc1 = 0, idxSrc2 = 0, idxDst = 0;
   int flagInplace;
   int condition;
   int result = -1;

   /* --------------------------------------------------------------- */
   /* Determine the data type - In the current implementation we use  */
   /* a matching exponent when the input is floating-point, and 16-bit*/
   /* signed integer otherwise. It may be better to have the device   */
   /* implementation check the type of integer exponents, and leave   */
   /* it untouched whenever the type is not float.                    */
   /* --------------------------------------------------------------- */
   dtype = (*dst) ? OcDType_getCommonType(src1 -> dtype, (*dst) -> dtype) : src1 -> dtype;
   if (OcDType_isInteger(dtype) && OcDType_isInteger(src2 -> dtype))
   {  dtypeExp = OcDTypeInt16;
   }
   else
   {  dtype = OcDType_getCommonType(dtype, src2 -> dtype);
      dtypeExp = dtype;
   }

   /* Check for in-place operations */
   flagInplace = ((*dst != NULL) && (OcTensors_match(src1, *dst)));

   /* Check the domain */
   if ((!OcDType_isComplex(dtype)) && (mode == 'c') && (dtypeExp != OcDTypeInt16))
   {  /* ------------------------------------------------------------ */
      /* Possible improvement: we currently check if any of the input */
      /* values is negative and if any of the powers is less than one.*/
      /* This criterion can be improved by joinly performing this     */
      /* check elementwise, instead of separately: with the current   */
      /* check power([-1,4],[1,0.5]) would evaluate as true, casting  */
      /* the result to complex unnecessarily. Another check could be  */
      /* added for raising 0 to any negative power. This would apply  */
      /* only to the warning and error modes.                         */
      /* ------------------------------------------------------------ */
      if ((condition = OcTensor_anyLTZero(src1)) < 0) return -1;
      if ((condition != 0) &&
          ((condition = OcTensor_anyLTOne(src2)) < 0)) return -1;

      if (condition != 0)
      {  if (mode == 'c')
         {  dtype = OcDType_isFloat(dtype) ? OcDType_getComplexType(dtype) : OcDTypeCDouble;
            dtypeExp = dtype;
            if ((dst) && (*dst != NULL) && (!OcDType_isComplex((*dst)->dtype)))
            {  if (OcWarning_raise(oc_warning_tensor_discard_imag2) != 0) return -1;
            }
         }
      }
   }

   /* Initialize the tensor operator */
   OcTensorOp_initialize(&info);
   if (((!flagInplace) &&
        ((idxSrc1 = OcTensorOp_addTensor(&info, &src1, OC_OP_READ )) < 0)) ||
       (((idxDst  = OcTensorOp_addTensor(&info,  dst,  OC_OP_WRITE)) < 0)) ||
       (((idxSrc2 = OcTensorOp_addTensor(&info, &src2, OC_OP_READ )) < 0))) goto final;

   /* Set the data type for possible allocation of destination tensor */
   info.dtype  = dtype;
   info.device = src1 -> device;

   /* Make sure the device and data types are correct */
   if (((!flagInplace) &&
        (OcTensorOp_ensure(&info, idxSrc1, dtype,    info.device) != 0)) ||
       ((OcTensorOp_ensure(&info, idxSrc2, dtypeExp, info.device) != 0)) ||
       ((OcTensorOp_ensure(&info, idxDst,  dtype,    info.device) != 0))) goto final;

   /* Remaining elementwise preparations */
   if ((OcTensorOp_broadcastElemwise(&info) != 0) ||
       (OcTensorOp_allocElemwise(&info)     != 0) ||
       (OcTensorOp_alignTensors(&info)      != 0) ||
       (OcTensorOp_overlapElemwise(&info)   != 0)) goto final;

   /* Look up the Tensor_OP function */
   if ((funptr = OC_GET_CORE_FUNCTION(info.device, Tensor_power)) == 0)
   {  OcErrorMessage("Tensor elementwise power is not supported on device %s", info.device -> name);
      goto final;
   }

   /* --------------------------------------------------------------- */
   /* Possible optimization: special function for in-place operation. */
   /* --------------------------------------------------------------- */

   /* Apply the function */
   if (flagInplace)
      result = funptr(*dst, src2, *dst);
   else result = funptr(src1, src2, *dst);

final :
   return OcTensorOp_finalize(&info, result);
}



/* ===================================================================== */
/* Function implementations - Tensor domain reductions                   */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_anyEQZero(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  int result;

   if (OcTensor_all(tensor, &result) != 0) return -1;

   return result ? 0 : 1;
}


/* --------------------------------------------------------------------- */
int OcTensor_anyLTZero(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcScalar scalar;
   int result;

   if (OcDType_isUnsigned(tensor -> dtype)) return 0;

   /* Check the range */
   OcScalar_setZero(&scalar, tensor -> dtype);
   result = OcTensor_allGreaterEqual(tensor, &scalar);
   if (result == -1) return -1; else return (result ? 0 : 1);
}


/* --------------------------------------------------------------------- */
int OcTensor_anyLTOne(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcScalar scalar;
   int result;

   if (tensor -> dtype == OcDTypeBool)
   {  return OcTensor_anyEQZero(tensor);
   }

   /* Check the range */
   OcScalar_setOne(&scalar, tensor -> dtype);
   result = OcTensor_allGreaterEqual(tensor, &scalar);
   if (result == -1) return -1; else return (result ? 0 : 1);
}


/* --------------------------------------------------------------------- */
int OcTensor_anyLTNegOne(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcScalar scalar;
   int result;

   if (OcDType_isUnsigned(tensor -> dtype)) return 0;

   /* Check the range */
   scalar.dtype = tensor -> dtype;
   OcScalar_fromInt64(&scalar, -1);
   result = OcTensor_allGreaterEqual(tensor, &scalar);
   if (result == -1) return -1; else return (result ? 0 : 1);
}


/* --------------------------------------------------------------------- */
int OcTensor_anyGTOneAbs(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcScalar lower, upper, *ptrLower;
   int result;

   if (tensor -> dtype == OcDTypeBool) return 0;

   /* Check the range */
   OcScalar_setOne(&upper, tensor -> dtype);
   if (OcDType_isUnsigned(tensor -> dtype))
   {  ptrLower = NULL;
   }
   else
   {  ptrLower = &lower;
      lower.dtype = tensor -> dtype;
      OcScalar_fromInt64(ptrLower, -1);
   }

   /* Negation of in range [-1,1] */
   result = OcTensor_allInRange(tensor, ptrLower, 1, &upper, 1);
   if (result == -1) return -1; else return (result ? 0 : 1);
}


/* --------------------------------------------------------------------- */
int OcTensor_allLessThan(OcTensor *tensor, OcScalar *value)
/* --------------------------------------------------------------------- */
{
   return OcTensor_allInRange(tensor, NULL, 0, value, 0);
}


/* --------------------------------------------------------------------- */
int OcTensor_allLessEqual(OcTensor *tensor, OcScalar *value)
/* --------------------------------------------------------------------- */
{
   return OcTensor_allInRange(tensor, NULL, 0, value, 1);
}


/* --------------------------------------------------------------------- */
int OcTensor_allGreaterThan(OcTensor *tensor, OcScalar *value)
/* --------------------------------------------------------------------- */
{
   return OcTensor_allInRange(tensor, value, 0, NULL, 0);
}


/* --------------------------------------------------------------------- */
int OcTensor_allGreaterEqual(OcTensor *tensor, OcScalar *value)
/* --------------------------------------------------------------------- */
{
   return OcTensor_allInRange(tensor, value, 1, NULL, 0);
}


/* --------------------------------------------------------------------- */
int OcTensor_allInRange(OcTensor *tensor, OcScalar *lower, int lowerInclusive, OcScalar *upper, int upperInclusive)
/* --------------------------------------------------------------------- */
{  int (*funptr1)(OcTensor *, OcScalar *value, int *result) = 0;
   int (*funptr2)(OcTensor *, OcScalar *lower, OcScalar *upper, int *result) = 0;
   OcTensor  *t1 = NULL, *t2 = NULL;
   OcScalar   lowerRounded, lowerReal, lowerImag, lowerScalar;
   OcScalar   upperRounded, upperReal, upperImag, upperScalar;
   OcScalar  *bound;
   int        result, status = -1;


   /* ------------------------------------------------------- */
   /* Empty lower and empty upper bounds are always satisfied */
   /* ------------------------------------------------------- */

   /* ------------------------------------------------------- */
   /* Process the lower bound                                 */
   /* ------------------------------------------------------- */
   do
   {  /* Check for lower bound */
      if (lower == NULL) break;

      /* Make sure the lower bound is not NaN */
      if (OcScalar_isNaN(lower))
         OcError(-1, "The lower bound cannot be NaN");

      /* Unsigned values are already greater than or equal to zero */
      if ((!OcDType_isSigned(tensor -> dtype)) &&
          ((( lowerInclusive) && (OcScalar_isLTZero(lower))) ||
           ((!lowerInclusive) && (OcScalar_isLEZero(lower)))))
      {  lower = NULL; break;  }

      /* Deal with complex-valued bounds */
      if (OcDType_isComplex(lower -> dtype) && (!OcDType_isComplex(tensor -> dtype)))
      {  /* Change inclusiveness based on imaginary part */
         OcScalar_getImag(lower, &lowerImag);
         if (OcScalar_isGTZero(&lowerImag))
            lowerInclusive = 0;
         else if (OcScalar_isLTZero(&lowerImag))
            lowerInclusive = 1;

         /* Deal with the real part only */
         OcScalar_getReal(lower, &lowerReal);
         lower = &lowerReal;
      }

      /* Domain check for integer types */
      if (OcDType_isInteger(tensor -> dtype))
      {  /* Compute the floor or ceil */
         if (OcDType_isFloat(lower -> dtype))
         {  if (OcScalar_ceil(lower, &lowerRounded) != 0) return -1;
            if (OcScalar_isGT(&lowerRounded, lower)) lowerInclusive = 1;
            lower = &lowerRounded;
         }

         /* Check if the scalar is in range */
         if (!OcScalar_inRange(lower, tensor -> dtype))
         {  /* Check for infeasibility or trivial satisfaction of the lower bound */
            if (OcScalar_isGTZero(lower)) return 0;
            lower = NULL; break;
         }
      }

      /* Convert to the desired data type - for floating-point types  */
      /* make sure that rounding does not change the effective boudns */
      lowerScalar.dtype = tensor -> dtype;
      OcScalar_copy(lower, &lowerScalar);
      if (OcDType_isFloat(tensor -> dtype))
      {  if (OcScalar_isLT(&lowerScalar, lower))
            lowerInclusive = 0;
         else if (OcScalar_isGT(&lowerScalar, lower))
            lowerInclusive = 1;
      }
      lower = &lowerScalar;
   } while (0);


   /* ------------------------------------------------------- */
   /* Process the upper bound                                 */
   /* ------------------------------------------------------- */
   do
   {  /* Check for upper bound */
      if (upper == NULL) break;

      /* Make sure the upper bound is not NaN */
      if (OcScalar_isNaN(upper))
         OcError(-1, "The upper bound cannot be NaN");

      /* Unsigned values are always greater than or equal to zero */
      if ((!OcDType_isSigned(tensor -> dtype)) &&
          ((( upperInclusive) && (OcScalar_isLTZero(upper))) ||
           ((!upperInclusive) && (OcScalar_isLEZero(upper)))))
      {  return 0; /* Infeasible bounds */  }

      /* Deal with complex-valued bounds */
      if (OcDType_isComplex(upper -> dtype) && (!OcDType_isComplex(tensor -> dtype)))
      {  /* Change inclusiveness based on imaginary part */
         OcScalar_getImag(upper, &upperImag);
         if (OcScalar_isGTZero(&upperImag))
            upperInclusive = 1;
         else if (OcScalar_isLTZero(&upperImag))
            upperInclusive = 0;

         /* Deal with the real part only */
         OcScalar_getReal(upper, &upperReal);
         upper = &upperReal;
      }

      /* Domain check for integer types */
      if (OcDType_isInteger(tensor -> dtype))
      {  /* Compute the floor */
         if (OcDType_isFloat(upper -> dtype))
         {  if (OcScalar_floor(upper, &upperRounded) != 0) return -1;
            if (OcScalar_isLT(&upperRounded, upper)) upperInclusive = 1;
            upper = &upperRounded;
         }

         /* Check if the scalar is in range */
         if (!OcScalar_inRange(upper, tensor -> dtype))
         {  /* Check for infeasibility or trivial satisfaction of the upper bound */
            if (OcScalar_isLTZero(upper)) return 0;
            upper = NULL; break;
         }
      }

      /* Convert to the desired data type - for floating-point types  */
      /* make sure that rounding does not change the effective boudns */
      upperScalar.dtype = tensor -> dtype;
      OcScalar_copy(upper, &upperScalar);
      if (OcDType_isFloat(tensor -> dtype))
      {  if (OcScalar_isGT(&upperScalar, upper))
            upperInclusive = 0;
         else if (OcScalar_isLT(&upperScalar, upper))
            upperInclusive = 1;
      }
      upper = &upperScalar;
   } while (0);


   /* ------------------------------------------------------- */
   /* Check for trivial and empty range                       */
   /* ------------------------------------------------------- */
   if ((lower == NULL) && (upper == NULL)) return 1;

   if ((lower != NULL) && (upper != NULL))
   {  if (lowerInclusive && upperInclusive)
      {  /* Required: lower <= upper */
         if (OcScalar_isGT(lower, upper)) return 0;
      }
      else
      {  /* Required: lower < upper */
         if (OcScalar_isGE(lower, upper)) return 0;
      }
   }

   /* ------------------------------------------------------- */
   /* Ensure byte order and remove repeats                    */
   /* ------------------------------------------------------- */
   if (OcTensor_ensureByteOrder(&tensor, &t1) != 0) goto final;
   if ((t2 = OcTensor_removeRepeats(t1)) == NULL) goto final;
   tensor = t2;

   /* Bounds have been determined and converted to desired type */
   if (lower)
   {  if (upper)
      {  /* Upper and lower bounds */
         if (lowerInclusive)
         {  if (upperInclusive)
            { /* Range l <= x <= u */
              if ((funptr2 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allGELE)) == 0)
                 OcErrorMessage("The tensor in range(GE,LE) function is not supported on device %s", tensor -> device -> type -> name);
            }
            else
            { /* Range l <= x < u */
              if ((funptr2 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allGELT)) == 0)
                 OcErrorMessage("The tensor in range(GE,LT) function is not supported on device %s", tensor -> device -> type -> name);
            }
         }
         else
         {  if (upperInclusive)
            {  /* Range l < x <= u */
              if ((funptr2 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allGTLE)) == 0)
                 OcErrorMessage("The tensor in range(GT,LE) function is not supported on device %s", tensor -> device -> type -> name);
            }
            else
            {  /* Range l < x < u */
              if ((funptr2 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allGTLT)) == 0)
                 OcErrorMessage("The tensor in range(GT,LT) function is not supported on device %s", tensor -> device -> type -> name);
            }
         }
      }
      else
      {  /* Lower bound only */
         bound = lower;

         if (lowerInclusive)
         {  /* Range l <= x */
            if ((funptr1 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allGE)) == 0)
               OcErrorMessage("The tensor all greater-equal function is not supported on device %s", tensor -> device -> type -> name);
         }
         else
         {  /* Range l < x */
            if ((funptr1 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allGT)) == 0)
               OcErrorMessage("The tensor all greater-than function is not supported on device %s", tensor -> device -> type -> name);
         }
      }
   }
   else
   {  if (upper)
      {  /* Upper bound only */
         bound = upper;

         if (upperInclusive)
         {  /* Range x <= u */
            if ((funptr1 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allLE)) == 0)
               OcErrorMessage("The tensor all less-equal function is not supported on device %s", tensor -> device -> type -> name);
         }
         else
         {  /* Range x < u */
            if ((funptr1 = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_allLT)) == 0)
               OcErrorMessage("The tensor all less-than function is not supported on device %s", tensor -> device -> type -> name);
         }
      }
   }

   /* Call the appropriate function */
   if (funptr1)
      status = funptr1(tensor, bound, &result);
   else if (funptr2)
      status = funptr2(tensor, lower, upper, &result);
   else
      status = -1;

final :
   OcXDecrefTensor(t1);
   OcXDecrefTensor(t2);
   return (status == 0) ? result : -1;
}


/* ===================================================================== */
/* Function implementations - Tensor global reduction                    */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
/* int OcTensor_any      (OcTensor *tensor, int *result)                 */
/* int OcTensor_all      (OcTensor *tensor, int *result)                 */
/* int OcTensor_allFinite(OcTensor *tensor, int *result)                 */
/* int OcTensor_anyInf   (OcTensor *tensor, int *result)                 */
/* int OcTensor_anyNaN   (OcTensor *tensor, int *result)                 */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC, COND, RESULT) \
int OcTensor_##OP(OcTensor *tensor, int *result) \
{  int (*funptr)(OcTensor *, int *); \
   OcTensor *t; \
   int status = -1; \
   \
   /* Check for trivial result */ \
   if ((COND) && (!OcDType_isFloat(tensor -> dtype))) \
   {  *result = RESULT; \
      return 0; \
   } \
   \
   /* Ensure byte order */ \
   if (OcTensor_ensureByteOrder(&tensor, &t) != 0) return -1; \
   \
   /* Call the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("The tensor "#DESC " function is not supported on device %s", tensor -> device -> type -> name); \
   } \
   else \
   {  status = funptr(t, result); \
   } \
   \
   /* Finalize */ \
   OcDecrefTensor(t); \
   return status; \
}

OC_TEMPLATE(any,       "any",       0, 0)
OC_TEMPLATE(all,       "all",       0, 0)
OC_TEMPLATE(allFinite, "allFinite", 1, 1)
OC_TEMPLATE(anyInf,    "anyInf",    1, 0)
OC_TEMPLATE(anyNaN,    "anyNaN",    1, 0)
#undef OC_TEMPLATE



/* --------------------------------------------------------------------- */
/* int OcTensor_nnz   (OcTensor *tensor, OcUInt64 *result)               */
/* int OcTensor_nnzNaN(OcTensor *tensor, OcUInt64 *result)               */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC) \
int OcTensor_##OP(OcTensor *tensor, OcUInt64 *result) \
{  int (*funptr)(OcTensor *, OcUInt64 *); \
   OcTensor *t1, *t2; \
   OcSize    nRepeat; \
   int status = -1; \
   \
   /* Ensure byte order */ \
   if (OcTensor_ensureByteOrder(&tensor, &t1) != 0) return -1; \
   \
   /* Remove repeated dimensions */ \
   if ((nRepeat = OcTensor_repeatCount(t1)) != 0) \
   {  if ((t2 = OcTensor_removeRepeats(t1)) == NULL) \
      { OcDecrefTensor(t1); return -1; } \
   } \
   else \
   {  t2 = OcIncrefTensor(t1); \
   } \
   \
   /* Call the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("The tensor %s function is not supported on device %s", DESC, tensor -> device -> type -> name); \
   } \
   else \
   {  status = funptr(t2, result); \
   } \
   \
   /* Scale result */ \
   if (nRepeat) (*result) *= (OcUInt64)nRepeat; \
   \
   /* Finalize */ \
   OcDecrefTensor(t1); \
   OcDecrefTensor(t2); \
   return status; \
}

OC_TEMPLATE(nnz,    "nnz")
OC_TEMPLATE(nnzNaN, "nnzNaN")
#undef OC_TEMPLATE



/* --------------------------------------------------------------------- */
/* int OcTensor_sum       (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_prod      (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_sumNaN    (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_prodNaN   (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_sumAbs    (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_sumAbsNaN (OcTensor *tensor, OcScalar *result)           */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC, FINALIZE, INT_TYPE, FLOAT_TYPE) \
int OcTensor_##OP(OcTensor *tensor, OcScalar *result) \
{  int (*funptr)(OcTensor *, OcScalar *); \
   OcTensor *t1 = NULL, *t2 = NULL; \
   OcScalar  intermediate, argument; \
   OcSize    nRepeat; \
   int       status = -1; \
   \
   /* Ensure byte order */ \
   if (OcTensor_ensureByteOrder(&tensor, &t1) != 0) return -1; \
   \
   /* Determine the output data type */ \
   if (OcDType_isFloat(tensor -> dtype)) \
   {  result -> dtype = FLOAT_TYPE(tensor -> dtype); \
   } \
   else if (OcDType_isBool(tensor -> dtype) || OcDType_isUInt(tensor -> dtype)) \
   {  result -> dtype = OcDTypeUInt64; \
   } \
   else \
   {  result -> dtype = INT_TYPE; \
   } \
   \
   /* Remove repeated dimensions */ \
   if ((nRepeat = OcTensor_repeatCount(t1)) != 0) \
   {  t2 = OcTensor_removeRepeats(t1); \
      if (t2 == NULL) goto final; \
   } \
   else \
   {  t2 = OcIncrefTensor(t1); \
   } \
   \
   /* Call the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("The tensor "#DESC " function is not supported on device %s", tensor -> device -> type -> name); \
   } \
   else \
   {  intermediate.dtype = result -> dtype; \
      status = funptr(t2, (nRepeat > 0) ? &intermediate : result); \
      if ((status == 0)  && (nRepeat > 0)) \
      {  argument.dtype = result -> dtype; \
         OcScalar_fromUInt64(&argument, (OcUInt64)nRepeat); \
         FINALIZE(&intermediate, &argument, result); \
      } \
   } \
   \
final : ; \
   /* Finalize */ \
   OcDecrefTensor(t1); \
   OcDecrefTensor(t2); \
   return status; \
}

OC_TEMPLATE(sum,       "sum",       OcScalar_multiply, OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(prod,      "prod",      OcScalar_power,    OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(sumNaN,    "sumNaN",    OcScalar_multiply, OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(prodNaN,   "prodNaN",   OcScalar_power,    OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(sumAbs,    "sumAbs",    OcScalar_multiply, OcDTypeUInt64, OcDType_getBaseType)
OC_TEMPLATE(sumAbsNaN, "sumAbsNaN", OcScalar_multiply, OcDTypeUInt64, OcDType_getBaseType)
#undef OC_TEMPLATE



/* --------------------------------------------------------------------- */
/* int OcTensor_maximum   (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_minimum   (OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_maximumAbs(OcTensor *tensor, OcScalar *result)           */
/* int OcTensor_minimumAbs(OcTensor *tensor, OcScalar *result)           */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC, TYPEFUN) \
int OcTensor_##OP(OcTensor *tensor, OcScalar *result) \
{  int (*funptr)(OcTensor *, OcScalar *); \
   OcTensor *t = NULL; \
   int       status = -1; \
   \
   /* Make sure the tensor is not empty */ \
   if (OcTensor_isEmpty(tensor)) \
      OcError(-1, "The tensor %s function does not apply to empty tensors", #DESC); \
   \
   /* Ensure byte order */ \
   if (OcTensor_ensureByteOrder(&tensor, &t) != 0) return -1; \
   \
   /* Determine the output data type */ \
   result -> dtype = TYPEFUN(tensor -> dtype); \
   \
   /* Call the Tensor_OP function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_##OP)) == 0) \
   {  OcErrorMessage("The tensor "#DESC " function is not supported on device %s", tensor -> device -> type -> name); \
   } \
   else \
   {  status = funptr(t, result); \
   } \
   \
   /* Finalize */ \
   OcDecrefTensor(t); \
   return status; \
}

OC_TEMPLATE(maximum,    "maximum",          OcDType_getType       )
OC_TEMPLATE(minimum,    "minimum",          OcDType_getType       )
OC_TEMPLATE(maximumAbs, "maximum absolute", OcDType_getModulusType)
OC_TEMPLATE(minimumAbs, "minimum absolute", OcDType_getModulusType)
#undef OC_TEMPLATE



/* --------------------------------------------------------------------- */
int OcTensor_norm(OcTensor *tensor, double p, OcScalar *result)
/* --------------------------------------------------------------------- */
{  int (*funptrA)(OcTensor *, OcScalar *) = 0;
   int (*funptrB)(OcTensor *, double, OcScalar *);
   OcTensor *t = NULL;
   int status = -1;

   /* Special cases and validity */
   if (p == 0)
   {  result -> dtype = OcDTypeUInt64;
      return OcTensor_nnz(tensor, (OcUInt64 *)&(result -> value));
   }
   if (p == 1) return OcTensor_sumAbs(tensor, result);
   if ((p > 0) && isinf(p)) return OcTensor_maximumAbs(tensor, result);
   if ((p < 0) || (isnan(p))) OcError(-1, "The vectorized p-norm is defined only for p >= 0");

   /* Ensure byte order */
   if (OcTensor_ensureByteOrder(&tensor, &t) != 0) return -1;

   /* Determine the output data type */
   if (!OcDType_isFloat(tensor -> dtype))
        result -> dtype = OcDTypeDouble;
   else result -> dtype = OcDType_getBaseType(tensor -> dtype);

   /* Call the Tensor_norm or Tensor_norm2 function */
   if (p == 2)
   {  funptrA = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_norm2);
      if (funptrA) status = funptrA(t, result);
   }
   if (funptrA == 0)
   {  if ((funptrB = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_norm)) == 0)
      {  OcErrorMessage("The tensor norm function is not supported on device %s", tensor -> device -> type -> name);
      }
      else
      {  status = funptrB(t, p, result);
      }
   }

   /* Finalize */
   OcDecrefTensor(t);
   return status;
}


/* --------------------------------------------------------------------- */
int OcTensor_normNaN(OcTensor *tensor, double p, OcScalar *result)
/* --------------------------------------------------------------------- */
{  int (*funptrA)(OcTensor *, OcScalar *) = 0;
   int (*funptrB)(OcTensor *, double, OcScalar *);
   OcTensor *t = NULL;
   int status = -1;

   /* Special cases and validity */
   if (p == 0)
   {  result -> dtype = OcDTypeUInt64;
      return OcTensor_nnzNaN(tensor, (OcUInt64 *)&(result -> value));
   }
   if (p == 1) return OcTensor_sumAbsNaN(tensor, result);
   if ((p > 0) && isinf(p)) return OcTensor_maximumAbs(tensor, result);
   if ((p < 0) || (isnan(p))) OcError(-1, "The vectorized p-norm is defined only for p >= 0");

   /* Ensure byte order */
   if (OcTensor_ensureByteOrder(&tensor, &t) != 0) return -1;

   /* Determine the output data type */
   if (!OcDType_isFloat(tensor -> dtype))
        result -> dtype = OcDTypeDouble;
   else result -> dtype = OcDType_getBaseType(tensor -> dtype);

   /* Call the Tensor_norm or Tensor_norm2 function */
   if (p == 2)
   {  funptrA = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_norm2NaN);
      if (funptrA) status = funptrA(t, result);
   }
   if (funptrA == 0)
   {  if ((funptrB = OC_GET_CORE_FUNCTION(tensor -> device, Tensor_normNaN)) == 0)
      {  OcErrorMessage("The tensor normNaN function is not supported on device %s", tensor -> device -> type -> name);
      }
      else
      {  status = funptrB(t, p, result);
      }
   }

   /* Finalize */
   OcDecrefTensor(t);
   return status;
}


/* --------------------------------------------------------------------- */
int OcTensor_norm1(OcTensor *tensor, OcScalar *result)
/* --------------------------------------------------------------------- */
{  return OcTensor_norm(tensor, 1, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_norm2(OcTensor *tensor, OcScalar *result)
/* --------------------------------------------------------------------- */
{  return OcTensor_norm(tensor, 2, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_normInf(OcTensor *tensor, OcScalar *result)
/* --------------------------------------------------------------------- */
{  OcScalar scalar;
   OcScalar_doubleInf(&scalar);
   return OcTensor_norm(tensor, OcScalar_asDouble(&scalar), result);
}


/* ===================================================================== */
/* Function implementations - Tensor single axis reduction               */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_initializeAxisReduction(OcTensor *src, int n, int *axes, int *norm_axes,
                                     int keepdims, OcTensor *dst, OcDType dtype,
                                     OcTensor **effectiveSrc, OcTensor **effectiveDst,
                                     int flagDummyOp)
/* --------------------------------------------------------------------- */
{  OcSize size[OC_TENSOR_MAX_DIMS];
   int    buffer[OC_TENSOR_MAX_DIMS];
   int    flagNew = 1;
   int    result = -1;
   int    a, i, ndims;

   /* Initialize */
   *effectiveSrc = NULL;
   *effectiveDst = NULL;

   /* Make sure the axes are unique, non-empty, and within range */
   if (n == 0) OcError(-1, "Reduction axes list cannot be empty");

   for (i = 0; i < OC_TENSOR_MAX_DIMS; i++) buffer[i] = 0;
   for (i = 0; i < n; i++)
   {  /* Convert negative axes indices */
      a = axes[i];
      if ((a < 0) && (a >= -(src -> ndims))) a += src -> ndims;

      /* Check axis values */
      if ((a < 0) || (a >= src -> ndims) || (a >= OC_TENSOR_MAX_DIMS))
         OcError(-1, "Axis at index %d (%d) is outside the valid range [%d, %d]", i+1, axes[i], -(src -> ndims), (src -> ndims)-1);
      if ((++(buffer[a])) > 1)
         OcError(-1, "Axis index %d appears more than once", a);

      /* Output normalized axis index */
      norm_axes[i] = a;
   }

   /* Determine the output dimensions */
   for (i = 0, ndims = 0; i < src -> ndims; i++)
   {  if (buffer[i] == 0)
      {  size[ndims] = src -> size[i]; ndims++;
      }
      else if (keepdims)
      {  size[ndims] = 1; ndims++;
      }
   }

   /* Check destination tensor dimensions */
   if (dst != NULL)
   {  if (dst -> ndims != ndims)
         OcError(-1, "Mismatch in number of output dimensions expected %d got %d", ndims, dst -> ndims);
      for (i = 0; i < ndims; i++)
      {  if (dst -> size[i] != size[i])
            OcError(-1, "Mismatch in tensor dimension %d: expected %"OC_FORMAT_LU" got "\
                        "%"OC_FORMAT_LU, i+1, (long unsigned)(size[i]), (long unsigned)(dst -> size[i]));
      }
   }

   /* Make sure the source tensor is normalized */
   if (flagDummyOp)
   {  *effectiveSrc = OcIncrefTensor(src);
   }
   else
   {  if (OcTensor_ensureByteOrder(&src, effectiveSrc) != 0) goto final;
   }

   /* Check if a new output tensor is needed */
   if (dst != NULL)
   {  do
      {  /* Make sure the tensor is a valid destination */
         if (!OcTensor_isValidDest(dst, 0)) goto final;

         /* Dummy operations do not require a new tensor */
         if (flagDummyOp) { flagNew = 0; break; }

         /* Make sure the data type and device match */
         if ((dst -> dtype != dtype) || ((dst -> device) != src -> device)) break;

         /* Make sure the tensor is not byte-swapped */
         if (OcTensor_isByteswapped(dst)) break;
 
         /* Check for overlap */
         if (OcTensors_overlap(src, dst)) break;

         /* All checks passed: use existing destination tensor */
         flagNew = 0;
      } while (0);
   }

   /* Allocate the new destination tensor, if needed */
   if (flagNew)
   {  dst = OcTensor_create(ndims, size, NULL, dtype, src -> device);
      if (dst == NULL) goto final; else *effectiveDst = dst;
   }
   else
   {  *effectiveDst = OcIncrefTensor(dst);
   }

   /* Success */
   result = 0;

final : ;
  if (result != 0)
  {  if (*effectiveSrc) { OcDecrefTensor(*effectiveSrc); *effectiveSrc = NULL; }
     if (*effectiveDst) { OcDecrefTensor(*effectiveDst); *effectiveDst = NULL; }
  }

  return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_finalizeAxisReduction(OcTensor *src, OcTensor *dst,
                                   OcTensor **ptrDst, int result)
/* --------------------------------------------------------------------- */
{  
   /* Set the output data */
   if (result == 0)
   {  if (*ptrDst)
      {  result = OcTensor_copy(dst, *ptrDst);
      }
      else
      {  *ptrDst = dst; dst = NULL;
      }
   }

   /* Delete the intermediate tensors */
   if (src) OcDecrefTensor(src);
   if (dst) OcDecrefTensor(dst);

   return result;
}


/* ----------------------------------------------------------------------------------------- */
/* int OcTensor_axisAny      (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisAll      (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisAllFinite(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisAnyInf   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisAnyNaN   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisNnz      (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisNnzNaN   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* ----------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC, DTYPE, FLOAT_ONLY, DEFAULT_VALUE) \
int OcTensor_##OP(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) \
{  int (*funptr)(OcTensor *src, int n, int *axes, OcTensor *dst); \
   OcTensor *effectiveSrc; \
   OcTensor *effectiveDst; \
   OcDType   dtype; \
   int       norm_axes[OC_TENSOR_MAX_DIMS]; \
   int       flagDummy = 0; \
   int       result; \
   \
   /* Look up the function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(src -> device, Tensor_##OP)) == 0) \
   {  OcError(-1, "The tensor %s function is not supported on device %s", DESC, src -> device -> type -> name); \
   } \
   \
   /* Some of the operations are defined only on floating point */ \
   /* tensors and have default output values otherwise.         */ \
   if ((FLOAT_ONLY) && (!OcDType_isFloat(src -> dtype))) \
   {  flagDummy = 1; \
   } \
   \
   /* Determine the output data type */ \
   dtype = DTYPE; \
   \
   /* Initialize the operation */ \
   result = OcTensor_initializeAxisReduction(src, n, axes, norm_axes, keepdims, *dst, dtype, \
                                             &effectiveSrc, &effectiveDst, flagDummy); \
   if (result != 0) return result; \
   \
   if (flagDummy) \
   {  /* Fill the tensor with the default scalar value */ \
      result = OcTensor_fillUInt64(effectiveDst, DEFAULT_VALUE); \
   } \
   else \
   {  /* Call the function */ \
      result = funptr(effectiveSrc, n, norm_axes, effectiveDst); \
   } \
   \
   /* Finalize the operation */ \
   return OcTensor_finalizeAxisReduction(effectiveSrc, effectiveDst, dst, result); \
}

OC_TEMPLATE(axisAny,       "axisAny",       OcDTypeBool,   0, 0)
OC_TEMPLATE(axisAll,       "axisAll",       OcDTypeBool,   0, 0)
OC_TEMPLATE(axisAllFinite, "axisAllFinite", OcDTypeBool,   1, 1)
OC_TEMPLATE(axisAnyInf,    "axisAnyInf",    OcDTypeBool,   1, 0)
OC_TEMPLATE(axisAnyNaN,    "axisAnyNaN",    OcDTypeBool,   1, 0)
OC_TEMPLATE(axisNnz,       "axisNnz",       OcDTypeUInt64, 0, 0)
OC_TEMPLATE(axisNnzNaN,    "axisNnzNaN",    OcDTypeUInt64, 0, 0)
#undef OC_TEMPLATE



/* ----------------------------------------------------------------------------------------- */
/* int OcTensor_axisSum      (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisProd     (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisSumNaN   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisProdNaN  (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisSumAbs   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisSumAbsNaN(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* ----------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, DESC, INT_TYPE, FLOAT_TYPE) \
int OcTensor_##OP(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) \
{  int (*funptr)(OcTensor *src, int n, int *axes, OcTensor *dst); \
   OcTensor *effectiveSrc; \
   OcTensor *effectiveDst; \
   OcDType   dtype; \
   int       norm_axes[OC_TENSOR_MAX_DIMS]; \
   int       result; \
   \
   /* Look up the function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(src -> device, Tensor_##OP)) == 0) \
   {  OcError(-1, "The tensor %s function is not supported on device %s", DESC, src -> device -> type -> name); \
   } \
   \
   /* Determine the output data type */ \
   if (OcDType_isFloat(src -> dtype)) \
   {  dtype = FLOAT_TYPE(src -> dtype); \
   } \
   else if (OcDType_isBool(src -> dtype) || OcDType_isUInt(src -> dtype)) \
   {  dtype = OcDTypeUInt64; \
   } \
   else \
   {  dtype = INT_TYPE; \
   } \
   \
   /* Initialize the operation */ \
   result = OcTensor_initializeAxisReduction(src, n, axes, norm_axes, keepdims, *dst, dtype, \
                                             &effectiveSrc, &effectiveDst, 0); \
   if (result != 0) return result; \
   \
   /* Call the function */ \
   result = funptr(effectiveSrc, n, norm_axes, effectiveDst); \
   \
   /* Finalize the operation */ \
   return OcTensor_finalizeAxisReduction(effectiveSrc, effectiveDst, dst, result); \
}

OC_TEMPLATE(axisSum,       "axisSum",       OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(axisProd,      "axisProd",      OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(axisSumNaN,    "axisSumNaN",    OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(axisProdNaN,   "axisProdNaN",   OcDTypeInt64,  OcDType_getType    )
OC_TEMPLATE(axisSumAbs,    "axisSumAbs",    OcDTypeUInt64, OcDType_getBaseType)
OC_TEMPLATE(axisSumAbsNaN, "axisSumAbsNaN", OcDTypeUInt64, OcDType_getBaseType)
#undef OC_TEMPLATE



/* ------------------------------------------------------------------------------------------ */
/* int OcTensor_axisMaximum   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisMinimum   (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisMaximumAbs(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* int OcTensor_axisMinimumAbs(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) */
/* ------------------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, DESC, DTYPE) \
int OcTensor_##OP(OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst) \
{  int (*funptr)(OcTensor *src, int n, int *axes, OcTensor *dst); \
   OcTensor *effectiveSrc; \
   OcTensor *effectiveDst; \
   OcDType   dtype; \
   OcSize    relem; \
   int       norm_axes[OC_TENSOR_MAX_DIMS], a; \
   int       result, i; \
   \
   /* Look up the function */ \
   if ((funptr = OC_GET_CORE_FUNCTION(src -> device, Tensor_##OP)) == 0) \
   {  OcError(-1, "The tensor %s function is not supported on device %s", DESC, src -> device -> type -> name); \
   } \
   \
   /* Determine the output data type */ \
   dtype = DTYPE(src -> dtype); \
   \
   /* Make sure the reduction is not empty - the case where the number     */ \
   /* of reduction axes is zero is dealt with in the initialization below. */ \
   for (i = 0, relem = 1; i < n; i++) \
   {  a = axes[i]; \
      if ((a >= 0) && (a < src -> ndims)) relem *= src -> size[a]; \
      else if ((a < 0) && (a >= -(src -> ndims))) relem *= src -> size[src -> ndims + a]; \
   } \
   if (relem == 0) OcError(-1, "The number of reduction elements in %s cannot be zero.", DESC); \
   \
   /* Initialize the operation */ \
   result = OcTensor_initializeAxisReduction(src, n, axes, norm_axes, keepdims, *dst, dtype, \
                                             &effectiveSrc, &effectiveDst, 0); \
   if (result != 0) return result; \
   \
   /* Call the function */ \
   result = funptr(effectiveSrc, n, norm_axes, effectiveDst); \
   \
   /* Finalize the operation */ \
   return OcTensor_finalizeAxisReduction(effectiveSrc, effectiveDst, dst, result); \
}

OC_TEMPLATE(axisMinimum,    "axisMinimum",    OcDType_getType       )
OC_TEMPLATE(axisMaximum,    "axisMaximum",    OcDType_getType       )
OC_TEMPLATE(axisMinimumAbs, "axisMinimumAbs", OcDType_getModulusType)
OC_TEMPLATE(axisMaximumAbs, "axisMaximumAbs", OcDType_getModulusType)
#undef OC_TEMPLATE



/* --------------------------------------------------------------------- */
int OcTensor_axisNorm(OcTensor *src, double p, int n,
                      int *axes, int keepdims, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  int (*funptrA)(OcTensor *src, int n, int *axes, OcTensor *dst) = 0;
   int (*funptrB)(OcTensor *src, double p, int n, int *axes, OcTensor *dst) = 0;
   OcTensor *effectiveSrc;
   OcTensor *effectiveDst;
   OcDType   dtype;
   int       norm_axes[OC_TENSOR_MAX_DIMS];
   int       result;

   /* Special cases and validity */
   if (p == 0) return OcTensor_axisNnz(src, n, axes, keepdims, dst);
   if (p == 1) return OcTensor_axisSumAbs(src, n, axes, keepdims, dst);
   if ((p > 0) && isinf(p)) return OcTensor_axisMaximumAbs(src, n, axes, keepdims, dst);
   if ((p < 0) || (isnan(p))) OcError(-1, "The vectorized p-norm is defined only for p >= 0");

   /* Look up the function */
   if (p == 2)
   {  funptrA = OC_GET_CORE_FUNCTION(src -> device, Tensor_axisNorm2);
   }
   if ((p != 2) || (funptrA == 0))
   {  if ((funptrB = OC_GET_CORE_FUNCTION(src -> device, Tensor_axisNorm)) == 0)
      {  OcError(-1, "The tensor axis-norm function is not supported on device %s", src -> device -> type -> name);
      }
   }

   /* Determine the output data type */
   if (!OcDType_isFloat(src -> dtype))
        dtype = OcDTypeDouble;
   else dtype = OcDType_getBaseType(src -> dtype);

   /* Initialize the operation */
   result = OcTensor_initializeAxisReduction(src, n, axes, norm_axes, keepdims, *dst, dtype,
                                             &effectiveSrc, &effectiveDst, 0);
   if (result != 0) return result;

   /* Call the function */
   if (funptrA)
        result = funptrA(effectiveSrc, n, norm_axes, effectiveDst);
   else result = funptrB(effectiveSrc, p, n, norm_axes, effectiveDst);

   /* Finalize the operation */
   return OcTensor_finalizeAxisReduction(effectiveSrc, effectiveDst, dst, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_axisNormNaN(OcTensor *src, double p, int n,
                         int *axes, int keepdims, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  int (*funptrA)(OcTensor *src, int n, int *axes, OcTensor *dst) = 0;
   int (*funptrB)(OcTensor *src, double p, int n, int *axes, OcTensor *dst) = 0;
   OcTensor *effectiveSrc;
   OcTensor *effectiveDst;
   OcDType   dtype;
   int       norm_axes[OC_TENSOR_MAX_DIMS];
   int       result;

   /* Special cases and validity */
   if (p == 0) return OcTensor_axisNnzNaN(src, n, axes, keepdims, dst);
     if (p == 1) return OcTensor_axisSumAbsNaN(src, n, axes, keepdims, dst);
   if ((p > 0) && isinf(p)) return OcTensor_axisMaximumAbs(src, n, axes, keepdims, dst);
   if ((p < 0) || (isnan(p))) OcError(-1, "The vectorized p-norm is defined only for p >= 0");

   /* Look up the function */
   if (p == 2)
   {  funptrA = OC_GET_CORE_FUNCTION(src -> device, Tensor_axisNorm2NaN);
   }
   if ((p != 2) || (funptrA == 0))
   {  if ((funptrB = OC_GET_CORE_FUNCTION(src -> device, Tensor_axisNormNaN)) == 0)
      {  OcError(-1, "The tensor axis-norm function is not supported on device %s", src -> device -> type -> name);
      }
   }

   /* Determine the output data type */
   if (!OcDType_isFloat(src -> dtype))
        dtype = OcDTypeDouble;
   else dtype = OcDType_getBaseType(src -> dtype);

   /* Initialize the operation */
   result = OcTensor_initializeAxisReduction(src, n, axes, norm_axes, keepdims, *dst, dtype,
                                             &effectiveSrc, &effectiveDst, 0);
   if (result != 0) return result;

   /* Call the function */
   if (funptrA)
        result = funptrA(effectiveSrc, n, norm_axes, effectiveDst);
   else result = funptrB(effectiveSrc, p, n, norm_axes, effectiveDst);

   /* Finalize the operation */
   return OcTensor_finalizeAxisReduction(effectiveSrc, effectiveDst, dst, result);
}


/* --------------------------------------------------------------------- */
int OcTensor_axisNorm1(OcTensor *src, int n,
                       int *axes, int keepdims, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  return OcTensor_axisNorm(src, 1, n, axes, keepdims, dst);
}


/* --------------------------------------------------------------------- */
int OcTensor_axisNorm2(OcTensor *src, int n,
                       int *axes, int keepdims, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  return OcTensor_axisNorm(src, 2, n, axes, keepdims, dst);
}


/* --------------------------------------------------------------------- */
int OcTensor_axisNormInf(OcTensor *src, int n,
                         int *axes, int keepdims, OcTensor **dst)
/* --------------------------------------------------------------------- */
{  OcScalar scalar;
   OcScalar_doubleInf(&scalar);
   return OcTensor_axisNorm(src, OcScalar_asDouble(&scalar), n, axes, keepdims, dst);
}



/* ===================================================================== */
/* Function implementations - Tensor multiplication                      */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcTensor_mtimes(OcTensor *src1, OcTensor *src2, OcTensor **dst)
/* --------------------------------------------------------------------- */
{
   return OcTensor_gemm(NULL, src1, 'N', src2, 'N', NULL, dst);
}



/* --------------------------------------------------------------------- */
int OcTensor_gemm(OcScalar *alpha, OcTensor *A, char modeA,
                                   OcTensor *B, char modeB,
                  OcScalar *beta,  OcTensor **ptrC)
/* --------------------------------------------------------------------- */
{  OcTensor *tensorAlpha = NULL;
   OcTensor *tensorBeta = NULL;
   int result = -1;

   /* ----------------------------------------------------------- */
   /* Possible optimization: have special functions for gemm with */
   /* scalars, or allow host values to be used on the device.     */
   /* ----------------------------------------------------------- */

   if (alpha)
   {  tensorAlpha = OcTensor_createFromScalar(alpha, alpha -> dtype, OcCPU, 1);
      if (tensorAlpha == NULL) goto final;
   }

   if (beta)
   {  tensorBeta = OcTensor_createFromScalar(beta, beta -> dtype, OcCPU, 1);
      if (tensorBeta == NULL) goto final;
   }

   /* Call tensor gemm */
   result = OcTensor_bcastgemm(tensorAlpha, A, modeA, B, modeB, tensorBeta, ptrC);

final : ;
   /* Free tensors */
   OcXDecrefTensor(tensorAlpha);
   OcXDecrefTensor(tensorBeta);

   return result;
}


/* --------------------------------------------------------------------- */
void OcTensor_intrnlGemmGetSize(OcTensor *tensor, char mode, OcSize *size, OcIndex *strides)
/* --------------------------------------------------------------------- */
{
   if (tensor == NULL) return ;

   /* Output the effective size and strides */
   if (mode == 'N')
   {  size[0] = (tensor -> ndims >= 1) ? tensor -> size[0] : 1;
      size[1] = (tensor -> ndims >= 2) ? tensor -> size[1] : 1;
      strides[0] = (tensor -> ndims >= 1) ? tensor -> strides[0] : tensor -> elemsize;
      if (size[1] == 1)
         strides[1] = tensor -> elemsize;
      else if (tensor -> ndims >= 2)
         strides[1] = tensor -> strides[1];
      else 
         strides[1] = (strides[0] == tensor -> elemsize) ? size[0] * strides[0] : tensor -> elemsize;
   }
   else
   {  size[1] = (tensor -> ndims >= 1) ? tensor -> size[0] : 1;
      size[0] = (tensor -> ndims >= 2) ? tensor -> size[1] : 1;
      strides[1] = (tensor -> ndims >= 1) ? tensor -> strides[0] : tensor -> elemsize;
      if (size[0] == 1)
         strides[0] = tensor -> elemsize;
      else if (tensor -> ndims >= 2)
         strides[0] = tensor -> strides[1];
      else
         strides[0] = (strides[1] == tensor -> elemsize) ? size[1] * strides[1] : tensor -> elemsize;
   }
}


/* --------------------------------------------------------------------- */
int OcTensor_intrnlGemmCheckReplace(OcTensor *tensor, OcSize *size, OcIndex *strides,
                                    OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  int result = 0;

   /* Check for empty tensors */
   if (tensor == NULL) return 1;

   /* Check device, data type, byte order, and alignment */
   if ((tensor -> device != device) || (tensor -> dtype  != dtype)  ||
       (OcTensor_isByteswapped(tensor)) || (!OcTensor_isAligned(tensor)))
      return 1;

   /* Update the strides */
   if (size[0] == 1)
   {  if (strides[1] == tensor -> elemsize)
           strides[0] = size[1] * strides[1];
      else strides[0] = tensor -> elemsize;
   }
   if (size[1] == 1)
   {  if (strides[0] == tensor -> elemsize)
           strides[1] = size[0] * strides[0];
      else strides[1] = tensor -> elemsize;
   }

   /* Check the strides */
   if (strides[0] == tensor -> elemsize)
   {  if (size[1] == 1) strides[1] = (tensor -> elemsize) * size[0];
      if (strides[1] < strides[0] * size[0]) result = 1;
   }
   else if (strides[1] == tensor -> elemsize)
   {  if (size[0] == 1) strides[0] = (tensor -> elemsize) * size[1];
      if (strides[0] < size[1] * strides[1]) result = 1;
   }
   else
   {  result = 1;
   }

   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_intrnlGemmAlloc(OcTensor *ref, int ndims, OcSize *size,
                                   int order, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------- */
{  OcIndex   strides[OC_TENSOR_MAX_DIMS];
   OcIndex   stride = OcDType_size(dtype);
   OcTensor *result;
   int       j;

   if (order == 0)
   {  strides[0] = stride; stride *= size[0];
      strides[1] = stride; stride *= size[1];
   }
   else
   {  strides[1] = stride; stride *= size[1];
      strides[0] = stride; stride *= size[0];
   }

   /* Set the remaining strides */
   if (ref)
   {  for (j = 2; j < ref -> ndims; j++)
      {  if (ref -> size[j] <= 1)
         {  strides[j] = 0;
         }
         else
         {  strides[j] = stride; stride *= size[j];
         }
      }

      /* Remaining padding */
      for ( ; j < ndims; j++) strides[j] = 0;
   }
   else
   {  for (j = 2; j < ndims; j++)
      {  strides[j] = stride; stride *= size[j];
      }
   }

   /* Create the tensor */
   result = OcTensor_create(ndims, size, strides, dtype, device);
   if (result == NULL) return NULL;

   /* Copy the data */
   if ((ref) && (OcTensor_copy(ref, result) != 0))
   {  OcDecrefTensor(result);
      return NULL;
   }

   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_gemmSupportedOn(OcDevice *device, OcDType dtype)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcDevice *device, OcDType dtype);

   /* --------------------------------- */
   /* Call the device-specific function */
   /* --------------------------------- */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Tensor_gemmSupportedOn)) == 0)
        return 0;
   else return funptr(device, dtype);
}


/* --------------------------------------------------------------------- */
int OcTensor_bcastgemm(OcTensor *alpha, OcTensor *A, char transA,
                                        OcTensor *B, char transB,
                       OcTensor *beta,  OcTensor **ptrC)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcSize M, OcSize N, OcSize K, char transA, char transB,
                 OcTensor *, OcTensor *, OcIndex, OcTensor *, OcIndex,
                 OcTensor *, OcTensor *, OcIndex);
   OcIndex   stridesA[2], stridesB[2], stridesC[2], r1;
   OcIndex   ldA, ldB, ldC;
   OcSize    size[OC_TENSOR_MAX_DIMS], s1, s2;
   OcSize    sizeA[2], sizeB[2], sizeC[2];
   OcSize    M, N, K;
   OcTensor *tensor;
   OcTensor *C = *ptrC;
   OcTensor *prepAlpha = NULL, *prepBeta = NULL;
   OcTensor *prepA = NULL, *prepB = NULL, *prepC = NULL;
   OcDevice *device = NULL, *scalarDevice;
   OcDType   dtype, halftype = OcDTypeNone;
   OcSize    nMultiply;
   int       nComplex, elemsize, orderA, orderB;
   int       scalarA, scalarB;
   int       flagExchange, flagReplace[3];
   char      mode;
   int       i, n, flagVector;
   int       result = -1;

   /* -------------------------------------- */
   /* Check for scaling                      */
   /* -------------------------------------- */
   if ((A == NULL) && (B == NULL) && (alpha == NULL) && (C != NULL))
   {  if (beta == NULL)
           return 0;
      else return OcTensor_scale(C, beta, ptrC);
   }

   /* ------------------------------------------------------- */
   /* Valid modes: 'N' or 'n'   Normal mode                   */
   /*              'T' or 't'   Transpose mode                */
   /*              'C' or 'c'   Conjugate mode                */
   /* ------------------------------------------------------- */

   /* Convert modes to upper case */
   transA = toupper(transA);
   transB = toupper(transB);

   /* Check modes */
   if ((transA != 'N') && (transA != 'T') && (transA != 'C'))
      OcError(-1, "Invalid mode for tensor A");
   if ((transB != 'N') && (transB != 'T') && (transB != 'C'))
      OcError(-1, "Invalid mode for tensor B");

   /* Normalize modes */
   if (!OcDType_isComplex(A -> dtype) && (transA == 'C')) transA = 'T';
   if (!OcDType_isComplex(B -> dtype) && (transB == 'C')) transB = 'T';


   /* -------------------------------------- */
   /* Check for scalar matrix multiplication */
   /* -------------------------------------- */
   scalarA = (A -> nelem == 1) || ((A -> ndims >=3) && (A -> size[0] == 1) && (A -> size[1] == 1));
   scalarB = (B -> nelem == 1) || ((B -> ndims >=3) && (B -> size[0] == 1) && (B -> size[1] == 1));
   if ((alpha == NULL) && (beta == NULL))
   {  
      /* Operation: C <- A * B */
      if (scalarA || scalarB)
      {
         /* Check for transpose and conjugation */
         if (transA == 'T')
         {  if ((!scalarA) && (OcTensor_transpose(&A, &prepA) != 0)) goto final;
         }
         else if (transA == 'C')
         {  if (scalarA)
                 result = OcTensor_conj(A, &prepA);
            else result = OcTensor_ctranspose(&A, &prepA);
            if (result != 0) goto final; 
         }

         if (transB == 'T')
         {  if ((!scalarB) && (OcTensor_transpose(&B, &prepB) != 0)) goto final;
         }
         else if (transB == 'C')
         {  if (scalarB)
                 result = OcTensor_conj(B, &prepB);
            else result = OcTensor_ctranspose(&B, &prepB);
            if (result != 0) goto final; 
         }

         /* Scale the tensor */
         result = OcTensor_scale(prepA ? prepA : A, prepB ? prepB : B, ptrC);

         goto final;
      }
   }


   /* -------------------------------------- */
   /* Evaluate C <- alpha * A * B + beta * C */
   /* -------------------------------------- */
   if (A == NULL) OcError(-1, "Tensor A must be specified");
   if (B == NULL) OcError(-1, "Tensor B must be specified");
   if (C != NULL)
   {  if ((!OcTensor_isValidDest(C, 0)))
         OcError(-1, "Tensor C cannot be used as an output tensor");
   }


   /* ------------------------------------------ */
   /* Check dimensions                           */
   /* ------------------------------------------ */

   OcTensor_intrnlGemmGetSize(A, transA, sizeA, stridesA);
   OcTensor_intrnlGemmGetSize(B, transB, sizeB, stridesB);
   OcTensor_intrnlGemmGetSize(C, 'N',    sizeC, stridesC);
   if (sizeA[1] != sizeB[0]) OcError(-1, "Tensor A and B matrix sizes are incompatible");

   /* Determine the higher-order dimensions */
   n = B -> ndims;
   for (i = 2; i < A -> ndims; i++)
   {  s1 = A -> size[i];
      s2 = (i < n) ? B -> size[i] : 1;
      if (s1 == s2) size[i] = s1;
      else if (s1 == 1) size[i] = s2;
      else if (s2 == 1) size[i] = s1;
      else OcError(-1, "Tensor A and B sizes are incompatible in dimension %d", i+1);
   }
   for ( ; i < n; i++) size[i] = B -> size[i];
   if (A -> ndims > n) n = A -> ndims;
   if (n < 2) n = 2;

   /* Check compatibility of scaling factor alpha */
   if (alpha != NULL)
   {  if (alpha -> ndims > 2)
      {  if ((alpha -> size[0] != 1) || (alpha -> size[1] != 1)) 
            OcError(-1, "First two dimensions of alpha must be 1x1");

         for (i = 2; i < alpha -> ndims; i++)
         {  s1 = alpha -> size[i];
            if (i >= n) { size[i] = s1; continue; } else s2 = size[i];
            if ((s1 == s2) || (s1 == 1)) continue;
            if (s2 == 1) size[i] = s1;
            else OcError(-1, "Scaling factor alpha is incompatible with A and B");
         }
         if (alpha -> ndims > n) n = alpha -> ndims;
      }
      else if (alpha -> nelem != 1)
      {  OcError(-1, "Alpha must be a scalar or scalar tensor");
      }
   }

   /* Check compatibility of scaling factor beta */
   if (beta != NULL)
   {   if (beta -> ndims > 2)
      {  if ((beta -> size[0] != 1) || (beta -> size[1] != 1)) 
            OcError(-1, "First two dimensions of beta must be 1x1");

         for (i = 2; i < beta -> ndims; i++)
         {  s1 = beta -> size[i];
            if (i >= n) { size[i] = s1; continue; } else s2 = size[i];
            if ((s1 == s2) || (s1 == 1)) continue;
            if (s2 == 1) size[i] = s1;
            else OcError(-1, "Scaling factor beta is incompatible with A and B");
         }
         if (beta -> ndims > n) n = beta -> ndims;
      }
      else if (beta -> nelem != 1)
      {  OcError(-1, "Beta must be a scalar or scalar tensor");
      }
   }

   /* Check the size of tensor C */
   size[0] = sizeA[0]; /* Number of rows in Op(A)    */
   size[1] = sizeB[1]; /* Number of columns in Op(B) */
   if (C != NULL)
   {  for (i = 0; i < C -> ndims; i++)
      {  s1 = C -> size[i];
         if (i >= n) { size[i] = s1; continue; } else s2 = size[i];
         if (s1 == s2) continue;
         if (s2 == 1) size[i] = s1;
         else OcError(-1, "Tensor C has an incompatible size at dimension %d", i+1);
      }
      if (C -> ndims >= n) n = C -> ndims;
      else
      {  if (((C -> ndims >= 2)) ||
             ((C -> ndims <= 1) && (size[1] != 1)) ||
             ((C -> ndims == 0) && (size[0] != 1)))
            OcError(-1, "Number of dimensions in C is too small");
      }
   }
   else
   {  if (beta != NULL) OcError(-1, "Beta must be empty when C is unspecified");
   }


   /* Truncate n to suppress compiler array-bounds warning   */
   /* 'array subscript is above array bounds', in loops over */
   /* j in the code below.                                   */
   if (n > OC_TENSOR_MAX_DIMS) n = OC_TENSOR_MAX_DIMS;

   /* Determine the number of matrix-matrix multiplies */
   if (OcShape_nelem(n-2, size+2, &nMultiply) != 0)
      OcError(-1, "Product of tensor dimensions exceeds the maximum");


   /* ----------------------------------------------- */
   /* Check for vector mode                           */
   /* ----------------------------------------------- */
   flagVector = 1;
   if (B -> ndims > 1) flagVector = 0;
   if ((transB == 'T') || (transB == 'C')) flagVector = 0;
   if ((alpha) && (alpha -> ndims > 2)) flagVector = 0;
   if ((beta) && (beta -> ndims > 2)) flagVector = 0;
   if ((C != NULL) && (n <= 2)) flagVector = (C -> ndims <= 1) ? 1 : 0;


   /* ------------------------------------------ */
   /* Determine the device                       */
   /* ------------------------------------------ */
   device = A -> device;

   /* ------------------------------------------ */
   /* Determine the data type                    */
   /* ------------------------------------------ */

   /* Data type of alpha * A * B and beta */
   dtype = OcDType_getCommonType(A -> dtype, B -> dtype);
   if (alpha) dtype = OcDType_getCommonType(dtype, alpha -> dtype);
   if (beta) dtype = OcDType_getCommonType(dtype, beta -> dtype);

   /* Determine the number of complex types - if only one */
   /* of alpha A, and B is complex and C is real we only  */
   /* need to deal with the real part of the product. In  */
   /* case two or three of the tensors are complex we do  */
   /* need to evaluate the complex product to obtain the  */
   /* desired real part.                                  */

   /* ------------------------------------------------------------- */
   /* Possible optimization:                                        */
   /* 1.1  Complex alpha * real A * real B or beta, and empty C     */
   /*      This case can be reduced to real part of C = A * B,      */
   /*      followed by copy real to imaginary and independently     */
   /*      scaling by respectively the real and imaginary parts of  */
   /*      alpha.                                                   */
   /* ------------------------------------------------------------- */
   if ((C != NULL) && (OcDType_isReal(C -> dtype)) && (OcDType_isComplex(dtype)))
   {  nComplex = 0;
      if ((alpha) && (OcDType_isComplex(alpha -> dtype))) nComplex ++;
      if (OcDType_isComplex(A -> dtype)) nComplex ++;
      if (OcDType_isComplex(B -> dtype)) nComplex ++;
      if (nComplex <= 1) dtype = OcDType_getBaseType(dtype);
   }

   /* Check (complex) half-precision support */
   if (dtype == OcDTypeHalf)
   {  if (!OcTensor_gemmSupportedOn(device, dtype))
      {  halftype = dtype;
         dtype = OcDTypeFloat;
      }
   }
   if (dtype == OcDTypeCHalf)
   {  if (!OcTensor_gemmSupportedOn(device, dtype))
      {  halftype = dtype;
         dtype = OcDTypeCFloat;
      }
   }

   /* Get the element size of the data type */
   elemsize = OcDType_size(dtype);


   /* ------------------------------------------ */
   /* Get the device-specific function pointer   */
   /* ------------------------------------------ */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Tensor_gemm)) == 0)
   {  OcErrorMessage("General tensor multiplication is not supported on device %s", device -> name);
      goto final;
   }


   /* ----------------------------------------------- */
   /* Determine the size and stride information for C */
   /* ----------------------------------------------- */
   if (C == NULL)
   {  /* Size and strides for new tensor */
      sizeC[0] = size[0];
      sizeC[1] = size[1];
      stridesC[0] = elemsize;
      stridesC[1] = elemsize * size[1];
   }

   /* ------------------------------------------------------------- */
   /* Ensure alpha and beta have the same data type and device, and */
   /* have the natural byte order. The tensor is not broadcast yet. */
   /* Alpha and beta must either both reside on the selected device */
   /* or both reside on the cpu.                                    */
   /* ------------------------------------------------------------- */

   /* Get the scalar device */
   if (((alpha == NULL) || (alpha -> device == OcCPU)) &&
       ((beta  == NULL) || (beta -> device == OcCPU)))
        scalarDevice = OcCPU;
   else scalarDevice = device; 

   /* Alpha and beta */
   for (i = 0; i < 2; i++)
   {  OcTensor *tensor, **prepTensor;
      OcScalar  scalar;

      if (i == 0)
      {  tensor = alpha; prepTensor = &prepAlpha;  }
      else
      {  tensor = beta;  prepTensor = &prepBeta;   }

      if (tensor == NULL)
      {  scalar.dtype = OcDTypeInt8;
         scalar.value.sInt8 = (i == 0) ? 1 : 0;
         *prepTensor = OcTensor_createFromScalar(&scalar, dtype, scalarDevice, 1);
         if (*prepTensor == NULL) goto final;
      }
      else if ((tensor -> device != scalarDevice) || (tensor -> dtype  != dtype ))
      {  if (OcTensor_checkAutoTypecast(tensor, OcDTypeNone, scalarDevice) != 0) goto final;
         if (OcTensor_ensure(&tensor, dtype, scalarDevice, prepTensor) != 0) goto final;
      }
      else if (OcTensor_isByteswapped(tensor))
      {  if ((*prepTensor = OcTensor_clone(tensor)) == NULL) goto final;
         if (OcTensor_byteswap(*prepTensor) != 0) goto final;
      }
      else
      {  *prepTensor = OcIncrefTensor(tensor);
      }
   }


   /* ------------------------------------------------------------- */
   /* Make sure that A and B are on the correct device, have the    */
   /* correct data type with native byte order, and appropriate     */
   /* strides for the BLAS library. Perform similar checks for C.   */
   /*                                                               */
   /* Possible optimization:                                        */
   /* 2.1 Cast individual matrices of tensors A and B to the new    */
   /*     device or data type, if needed, and overlap communication */
   /*     with matrix multiplication.                               */
   /* 2.2 Deal with the special case where the difference in data   */
   /*     type is due only to dropping the imaginary part of the    */
   /*     tensors. Note that the stride will never be unitary in    */
   /*     this case, which means that the standard BLAS interface   */
   /*     cannot deal with it.                                      */
   /* 2.3 Check whether the strides are allowed and either copy     */
   /*     directly to the desired format or copy on the fly.        */
   /* 2.4 On the CPU we allow strides that are not multiples of the */
   /*     element size. BLAS requires the strides of the matrices to*/
   /*     be multiples, we can still have higher-dimension strides  */
   /*     that are not multiples, thereby leading to unaligned data.*/
   /*     If this is not desirable a check must be added and the    */
   /*     data must be copied if unaligned strides are found.       */
   /* 2.5 In case of repeated multiplies with the same matrix in A, */
   /*     and scalling factors alpha and beta, it may be possible   */
   /*     to combine several matrices in B and C and use a single   */
   /*     large matrix-matrix multiplication instead of several     */
   /*     smaller ones.                                             */
   /* 2.6 In case A and/or B overlaps with C it may be better to    */
   /*     copy the smaller of the two, also depending on whether    */
   /*     tensor needs to be updated regardless of the overlap.     */
   /* 2.7 When matrices in A and B are respectively row and column  */
   /*     major with negative unit stride, we can simply negate both*/
   /*     strides to get unit strides.                              */
   /* ------------------------------------------------------------- */

   /* Check whether tensors need to be replaced - update strides if needed */
   flagReplace[0] = OcTensor_intrnlGemmCheckReplace(A, sizeA, stridesA, dtype, device);
   flagReplace[1] = OcTensor_intrnlGemmCheckReplace(B, sizeB, stridesB, dtype, device);
   flagReplace[2] = OcTensor_intrnlGemmCheckReplace(C, sizeC, stridesC, dtype, device);

   /* For mode 'C' we require that the second stride has elemsize */

   /* Check if we need to work in C^T mode */
   flagExchange = 0;
   if ((flagReplace[2] == 0) && (stridesC[0] != C -> elemsize))
   {  /* Do not allow any additional modifications */
      if (((flagReplace[0]) || (transA != 'C') || (stridesA[0] == A -> elemsize)) &&
          ((flagReplace[1]) || (transB != 'C') || (stridesB[0] == B -> elemsize)))
      {  /* Exchange A and B, change size of C */
         if ((flagReplace[0] == 0) && OcTensors_overlap(A, C)) flagReplace[0] = 1;
         if ((flagReplace[1] == 0) && OcTensors_overlap(B, C)) flagReplace[1] = 1;
         if ((flagReplace[0] == 0) && (flagReplace[1] == 0)) flagExchange = 1;
      }
      else
      {  flagReplace[2] = 1;
      }
   }

   /* Exchange tensors or perform additional checks */
   if (flagExchange)
   {
      /* Exchange and transpose A and B */
      s1 = sizeA[0]; sizeA[0] = sizeB[1]; sizeB[1] = s1;
      s1 = sizeA[1]; sizeA[1] = sizeB[0]; sizeB[0] = s1;
      r1 = stridesA[0]; stridesA[0] = stridesB[1]; stridesB[1] = r1;
      r1 = stridesA[1]; stridesA[1] = stridesB[0]; stridesB[0] = r1;
      tensor = A; A = B; B = tensor;

      /* Exchange the modes to check for 'C' */
      mode = transA; transA = transB; transB = mode;

      /* Transpose C */
      s1 = sizeC[0]; sizeC[0] = sizeC[1]; sizeC[1] = s1;
      r1 = stridesC[0]; stridesC[0] = stridesC[1]; stridesC[1] = r1;
   }
   else
   {  /* Check conjugate */
      if ((transA == 'C') && (stridesA[1] != A -> elemsize)) flagReplace[0] = 1;
      if ((transB == 'C') && (stridesB[1] != B -> elemsize)) flagReplace[1] = 1;

      /* Check for overlap with C */
      if (flagReplace[2] == 0)
      {  if ((flagReplace[0] == 0) && OcTensors_overlap(A, C)) flagReplace[0] = 1;
         if ((flagReplace[1] == 0) && OcTensors_overlap(B, C)) flagReplace[1] = 1;
      }
   }


   /* --------------------------------- */
   /* Create intermediate work tensor C */
   /* --------------------------------- */

   if (flagReplace[2])
   {  /* Check auto device casting */
      if ((C != NULL) && (OcTensor_checkAutoTypecast(C, dtype, device) != 0)) goto final;

      /* The exchange and replace C flags are never set together. */
      /* We can therefore allocate C directly in natural order.   */
      size[0] = sizeC[0]; size[1] = sizeC[1];
      prepC = OcTensor_intrnlGemmAlloc(C, flagVector ? 1 : n, size, 0, dtype, device);

      /* Determine the leading dimension */
      ldC = sizeC[0]; /* The new tensor is column-major contiguous */
   }
   else
   {  prepC = OcIncrefTensor(C);

      /* Determine the leading dimension */
      if (stridesC[1] < stridesC[0])
           ldC = stridesC[0] / elemsize;
      else ldC = stridesC[1] / elemsize;
   }

   /* --------------------------------- */
   /* Create intermediate work tensor B */
   /* --------------------------------- */

   /* Determine the logical order of the first two dimensions in  */
   /* prepB and set first two size dimensions, used when creating */
   /* or broadcasting the tensor (which has to be do according to */
   /* the order in the original tensor.                           */
   orderB = 1;
   if (flagExchange) orderB *= -1;
   if (transB != 'N') orderB *= -1;

   if (orderB == 1)
        size[0] = sizeB[0], size[1] = sizeB[1];
   else size[0] = sizeB[1], size[1] = sizeB[0];

   if (flagReplace[1])
   {
      /* Check auto device casting */
      if (OcTensor_checkAutoTypecast(B, dtype, device) != 0) goto final;

      /* Create the tensor */
      prepB = OcTensor_intrnlGemmAlloc(B, n, size, (orderB == 1) ? 0 : 1, dtype, device);
      if (prepB == NULL) goto final;

      /* Conjugate the entries in B if needed */
      if ((transB == 'C') && (OcTensor_conj(prepB, NULL) != 0)) goto final;
      transB = 'N';

      /* Determine the leading dimension */
      if (orderB == 1)
           ldB = prepB -> size[0];
      else ldB = prepB -> size[1];
   }
   else
   {  /* Broadcast the tensor dimensions */
      if (OcTensor_broadcastTo(&B, n, size, 0, &prepB) != 0) goto final;

      /* Determine the leading dimension */
      if (stridesB[0] > stridesB[1])
      {  ldB = stridesB[0] / elemsize;
         orderB = -1;
      }
      else
      {  ldB = stridesB[1] / elemsize;
         orderB = 1;
      }

      /* Determine the mode */      
      if (orderB == 1)
           transB = 'N';
      else if (transB != 'C') transB = 'T';
   }

   /* --------------------------------- */
   /* Create intermediate work tensor A */
   /* --------------------------------- */

   /* Determine the logical order of the first two dimensions in  */
   /* prepA and set first two size dimensions, used when creating */
   /* or broadcasting the tensor (which has to be do according to */
   /* the order in the original tensor.                           */
   orderA = 1;
   if (flagExchange) orderA *= -1;
   if (transA != 'N') orderA *= -1;

   if (orderA == 1)
        size[0] = sizeA[0], size[1] = sizeA[1];
   else size[0] = sizeA[1], size[1] = sizeA[0];

   if (flagReplace[0])
   {  /* Determine the desired memory layout - when B has a    */
      /* natural memory layout we prefer transposed data in A. */
      orderA *= (B -> strides[1] < B -> strides[0]) ? 1 : -1;

      /* Check auto device casting */
      if (OcTensor_checkAutoTypecast(A, dtype, device) != 0) goto final;

      /* Create the tensor */
      prepA = OcTensor_intrnlGemmAlloc(A, n, size, (orderA == 1) ? 0 : 1, dtype, device);

      /* We need to exchange the first two dimensions if exactly one     */
      /* of flagExchange and (transA != 'N') holds. In order to determine*/
      /* whether prepA has to be transposed or not, we need to multiply  */
      /* the orderA by -1 if the dimensions are reversed. If the result  */
      /* is negative we need to transpose, otherwise we keep the natural */
      /* memory layout. We can obtain the memory ordering from orderA by */
      /* inverting by the condition on the strides of B. It follows that */
      /* the product of the fill and dimension order is negative iff the */
      /* latter condition evaluates to -1, which is equivalent to the    */
      /* condition (B -> strides[1] >= B -> strides[0]).                 */

      /* Conjugate the entries in A if needed */
      if (B -> strides[1] >= B -> strides[0])
      {  if (transA != 'C') transA = 'T';
      }
      else
      {  if ((transA == 'C') && (OcTensor_conj(prepA, NULL) != 0)) goto final;
         transA = 'N';
      }

      /* Determine the leading dimension */
      if (orderA == 1)
           ldA = prepA -> size[0];
      else ldA = prepA -> size[1];
   }
   else
   {  /* Broadcast the tensor dimensions */
      if (OcTensor_broadcastTo(&A, n, size, 0, &prepA) != 0) goto final;

      /* Determine the leading dimension */
      if (stridesA[0] > stridesA[1])
      {  ldA = stridesA[0] / elemsize;
         orderA = -1;
      }
      else
      {  ldA = stridesA[1] / elemsize;
         orderA = 1;
      }

      /* Determine the mode */      
      if (orderA == 1)
           transA = 'N';
      else if (transA != 'C') transA = 'T';
   }

   /* Finalize the size */
   M = sizeA[0];
   N = sizeB[1];
   K = sizeA[1];


   /* ------------------------------------------------------------- */
   /* Possible optimization:                                        */
   /* 3.1 The third dimension can be collapsed whenever its stride  */
   /*     is zero for A, alpha, and beta; and B and C have column   */
   /*     major ordering and the stride of the third dimension is   */
   /*     equal to that of the second multiplied by the size of the */
   /*     second dimension. Note that tensors may first need to be  */
   /*     detached to ensure that the size of the original tensor   */
   /*     is not changed.                                           */
   /* 3.2 Use batched GEMM when available.                          */
   /* ------------------------------------------------------------- */

   /* Add stream synchronization */
   OcTensor_startRead(prepC, prepAlpha);
   OcTensor_startRead(prepC, prepBeta);

   if (scalarDevice == device)
   {  OcTensor_startRead(prepC, prepA);
      OcTensor_startRead(prepC, prepB);
   }
   else
   {  OcTensor_synchronize(prepA);
      OcTensor_synchronize(prepA);
   }

   /* Call the device-specific multiplication routine */
   result = funptr(M, N, K, transA, transB, prepAlpha, prepA, ldA, prepB, ldB, prepBeta, prepC, ldC);
   if (result != 0) goto final;

   /* Add stream synchronization */
   OcTensor_update(prepC);
   OcTensor_finishRead(prepC, prepB);
   OcTensor_finishRead(prepC, prepA);

   if (scalarDevice == device)
   {  OcTensor_finishRead(prepC, prepBeta);
      OcTensor_finishRead(prepC, prepAlpha);
   }

   /* Set the result */
   if (prepC != C)
   {
      if ((C == NULL) && (halftype != OcDTypeNone))
      {  if (OcTensor_ensure(&prepC, halftype, prepC -> device, ptrC) != 0) goto final;
      }
      else if (C != NULL)
      {  if (OcTensor_copy(prepC, C) != 0) goto final;
      }
      else
      {  *ptrC = OcIncrefTensor(prepC);
      }
   }

   /* Success */
   result = 0;

final : ;
   if (prepAlpha) OcDecrefTensor(prepAlpha);
   if (prepBeta)  OcDecrefTensor(prepBeta);
   if (prepA)     OcDecrefTensor(prepA);
   if (prepB)     OcDecrefTensor(prepB);
   if (prepC)     OcDecrefTensor(prepC);

   return result;
}


/* ===================================================================== */
/* Function implementations - Formatting routines                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensor_format(OcTensor *tensor, char **str, const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  OcFormatAnalyze_funptr funptrAnalyze;
   OcFormatOutput_funptr  funptrOutput;
   OcFormat  *format = NULL;
   OcIndex    strides[OC_TENSOR_MAX_DIMS], offset;
   OcSize     size[OC_TENSOR_MAX_DIMS];
   OcSize     index[OC_TENSOR_MAX_DIMS];
   OcSize     j, n, slen;
   int        flagShort, flagCompleteRow, flagNewline;
   char      *s = NULL, *data, *buffer = NULL;
   int        i, k, mode, ndims;
   int        result = -1;

   /* Formatting related variables */
   int        colIndex;
   int        rowIndent = 3;
   int        rowWidth;           /* Maximum width */
   int        rowsPerBlock = 10;  /* Maximum numbers of rows per block */
   int        itemSpacing = 3;
   int        itemsPerRow;
   int        itemsPerBlock;

   /* Make sure the storage is on CPU */
   if (OcTensor_ensureDevice(&tensor, OcCPU, &tensor) != 0) goto final;

   /* Synchronize the tensor storage */
   OcTensor_synchronize(tensor);

   /* Get the number of tensor elemens and dimensions */
   ndims = tensor -> ndims;
   n     = tensor -> nelem;

   /* Copy size and stride information */
   for (i = 0; i < ndims; i++)
   {  size[i]    = tensor -> size[i];
      strides[i] = tensor -> strides[i];
   }
   if (ndims >= 2)
   {  size[0] = tensor -> size[1];      /* Output rows first    */
      size[1] = tensor -> size[0];      /* Output column second */
      strides[0] = tensor -> strides[1];
      strides[1] = tensor -> strides[0];
   }

   /* Determine whether to use full or abbreviated format */
   flagShort = (n > 1000) ? 1 : 0;
   flagCompleteRow = 0; /* Initialize to avoid compiler warning */

   /* Create and initialize a new format structure */
   if ((format = OcFormatCreate(tensor -> dtype)) == NULL)
   {  OcErrorMessage("Could not allocate the formatting information");
      goto final;
   }

   /* Determine the byteswap flag */
   format -> byteswapped = OcTensor_isByteswapped(tensor);

   /* Determine the function handle to use */
   funptrAnalyze = OcFormatAnalyze_function(tensor -> dtype);
   funptrOutput  = OcFormatOutput_function(tensor -> dtype);

   if ((funptrAnalyze == 0) || (funptrOutput == 0))
   {  OcErrorMessage("Could not find formatting analysis and output functions");
      goto final;
   }

   /* Determine upper bounds on the number of items per block */
   rowWidth = oc_format_linewidth;
   itemsPerRow = (int)((rowWidth - rowIndent - 1) / (1 + itemSpacing)) + 1;
   if (itemsPerRow <= 0) itemsPerRow = 1;
   itemsPerBlock = itemsPerRow * rowsPerBlock; 

   /* Get a pointer to the data */
   data = tensor -> storage -> data;

   /* --------------------------- */
   /* Determine the format        */
   /* --------------------------- */
   offset = tensor -> offset;
   for (i = 0; i < ndims; i++) index[i] = 0;
   if ((n > 0) && (ndims > 0)) while (1)
   {
      /* Analyze the data format */
      j = (size[0] > 2*itemsPerBlock) ? itemsPerBlock : size[0];
      for (i = 0; i < j; i++)
      {  funptrAnalyze(data + offset, format, 0);
         offset += strides[0];
      }
      offset -= i * strides[0];
      j = (size[0] > 2*itemsPerBlock) ? size[0] - itemsPerBlock : j;
      offset += j * strides[0];
      for (i = j; i < size[0]; i++)
      {  funptrAnalyze(data + offset, format, 0);
         offset += strides[0];
      }
      offset -= i * strides[0];

      /* Move to the next index */
      for (i = 1; i < ndims; i++)
      {  
         if ((flagShort) && (index[i] == 2) && (size[i] > 6))
         {  offset += (size[i] - 5) * strides[i];
            index[i] = size[i] - 3;
            break;
         }
         if ((++index[i]) < size[i])
         {  offset += strides[i];
            break;
         }
         else
         {  offset -= (size[i]-1) * strides[i];
            index[i] = 0;
         }
      }
      if (i == ndims) break;
   }
   if ((n == 1) && (ndims == 0))
   {  /* Analyze the scalar format */
       funptrAnalyze(OcTensor_data(tensor), format, 0);
   }

   if (n > 0)
   {  /* Finalize formatting */
      OcFormatFinalize(format);

      /* Adjust spacing based on the number of parts */
      if (format -> parts > 1) itemSpacing ++;

      /* Reduce the item spacing by one if only signs   */
      /* appear as the first character of each element. */
      if (format -> flagLeadingSign) itemSpacing --;

      /* Determine the actual number of items per block */
      itemsPerRow = (int)((rowWidth - rowIndent - (format -> width)) / (format -> width + itemSpacing)) + 1;
      if (itemsPerRow <= 0) itemsPerRow = 1;
      itemsPerBlock = itemsPerRow * rowsPerBlock; 

      /* Check if all elements in the last dimension fit the line width */
      if (ndims > 0)
         flagCompleteRow = (itemsPerRow >= size[0]) ? 1 : 0;
   }


   /* --------------------------- */
   /* Format the tensor contents  */
   /* --------------------------- */
   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Output the header */
      if (header != NULL)
      {  k = strlen(header); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", header);
      }

      /* Output the body */
      offset = tensor -> offset; flagNewline = 0;
      for (i = 0; i < ndims; i++) index[i] = 0;
      if ((n > 0) && (ndims > 0)) while (1)
      {
         /* Output the index */
         if ((ndims > 1 + flagCompleteRow) && (!flagCompleteRow || (index[1] == 0)))
         {
            /* Insert an empty line if needed */
            if (flagNewline)
            {  k = format -> newlineWidth; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "\n");
            }

            /* Output the current block index */
            k = 1; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "(");

            for (i = 0; i < ndims; i++)
            {  
               if (i > 0)
               {  k = 1; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, ",");
               }

               if (((i == 0) && (flagCompleteRow == 1)) || (i == 1))
               {  k = 1; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, ":");
               }
               else
               {  j = (i == 0) ? index[1] : index[i];
                  k = OcFormatULongWidth((unsigned long int)j); slen += k;
                  if (mode == 1) s += OcFormatULong(s, k, (unsigned long int)j);
               }
            }
            k = 1 + format -> newlineWidth; slen += k;
            if (mode == 1) s += snprintf(s, k+1, ")\n");
         }
         else if ((ndims == 2) && (flagCompleteRow) && (index[1] == 0))
         {  /* Output a header to differentiate from a vector */
            k = 5 + format -> newlineWidth; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "(:,:)\n");
         }

         /* Output a single `row' of the tensor */
         colIndex = 0;
         for (i = 0; i < size[0]; i++)
         {
            /* Indentation or separation */
            k = (colIndex == 0) ? rowIndent : itemSpacing; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "%*s", k, "");

            /* Format the element */
            k = format -> width; slen += k;
            if (mode == 1)
            {  funptrOutput(data + (offset + i * strides[0]), format, 0, s);
               s += k;
            }

            /* Add a new line */
            if (++colIndex == itemsPerRow)
            {  colIndex = 0;
               k = format -> newlineWidth; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "\n");
             }

            if ((size[0] > 2*itemsPerBlock) && (i == itemsPerBlock-1))
            {
               for (j = 0; j < itemsPerRow; j++)
               {
                  k = (j == 0) ? rowIndent : itemSpacing; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, "%*s", k, "");

                  k = (format -> width + 1) / 2; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, "%*s", k, ":");

                  k = (format -> width) / 2; slen += k;
                  if ((mode == 1) && (k > 0)) s += snprintf(s, k+1, "%*s", k, "");
               }
               k = format -> newlineWidth; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "\n");

               i = size[0] - itemsPerBlock - 1;
            }
         }
         if (colIndex != 0)
         {  k = format -> newlineWidth; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "\n");
         }
         flagNewline = 1;

         /* Move to the next index */
         for (i = 1; i < ndims; i++)
         {  
            if ((flagShort) && (index[i] == 2) && (size[i] > 6))
            {  offset += (size[i] - 5) * strides[i];
               index[i] = size[i] - 3;
               break;
            }
            if ((++index[i]) < size[i])
            {  offset += strides[i];
               break;
            }
            else
            {  offset -= (size[i]-1) * strides[i];
               index[i] = 0;
            }
         }
         if (i == ndims) break;
      } /* While (1) */

      /* Scalar case */
      if ((n == 1) && (ndims == 0))
      {
         /* Format the element */
         k = format -> width; slen += k;
         if (mode == 1)
         {  funptrOutput(OcTensor_data(tensor), format, 0, s);
            s += k;
         }

         /* Add a new line */
         k = format -> newlineWidth; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "\n");
      }

      /* Output the footer */
      if (footer != NULL)
      {  k = strlen(footer); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", footer);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer;
         if (buffer == NULL)
         {  OcErrorMessage("Insufficient memory for output string");
            goto final;
         }
      }

   } /* Mode */

   /* Ensure the string is zero terminated */
   *s = '\0';

   /* Success */
   *str = buffer;
   result = 0;

final :
   /* -------------------------------------------------------- */
   /* Clean up. Note that freeing of the buffer has to be done */
   /* using the regular free function to match its allocation  */
   /* above using malloc.                                      */
   /* -------------------------------------------------------- */

   if (format != NULL) OcFormatFree(format);
   if ((result != 0) && (buffer != NULL)) { free(buffer); }
   OcDecrefTensor(tensor);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensor_formatFooter(OcTensor *tensor, char **str, const char *pre, const char *post)
/* -------------------------------------------------------------------- */
{  char   *s = NULL, *buffer = NULL;
   size_t  slen, n;
   int     i, k, mode;
   int     result = -1;

   /* Get the number of elements */
   n = tensor -> nelem;

   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Pre-string */
      if (pre != NULL)
      {  k = strlen(pre); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", pre);
      }

      /* Scalar or tensor */
      if ((n == 1) && (tensor -> ndims == 0))
      {  /* Scalar */
         k = 7 + strlen(OcDType_name(tensor -> dtype)); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "scalar.%s", OcDType_name(tensor -> dtype));
      }
      else
      {  
         /* Empty */
         if (n == 0)
         {  k = 6; slen += k;
            if (mode == 1) s += snprintf(s, k+1, "empty ");
         }

         /* Tensor */
         k = 7 + strlen(OcDType_name(tensor -> dtype)); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "tensor.%s", OcDType_name(tensor -> dtype));

         /* Dimensions */
         if (tensor -> ndims > 0)
         {  k = 8; slen += k;
            if (mode == 1) s+= snprintf(s, k+1, " of size");

            for (i = 0; i < tensor -> ndims; i++)
            {  k = OcFormatULongWidth((unsigned long int)(tensor -> size[i])) + 1; slen += k;
               if (mode == 1)
               {  s += snprintf(s, 2, "%s", (i == 0) ? " " : "x");
                  s += OcFormatULong(s, k-1, (unsigned long int)(tensor -> size[i]));
               }
            }
         }
      }

      /* Device */
      k = 4 + strlen(tensor -> device -> name); slen += k;
      if (mode == 1) s += snprintf(s, k+1, " on %s", tensor -> device -> name);

      /* Special properties */
      if (OcTensor_isByteswapped(tensor) || OcTensor_isReadOnly(tensor))
      {  int flag = 0;

         /* Opening bracket */
         k = 2; slen += k;
         if (mode == 1) s += snprintf(s, k+1, " (");
      
         /* Byteswapped */
         if (OcTensor_isByteswapped(tensor))
         {  k = 11; slen += k; flag = 1;
            if (mode == 1) s += snprintf(s, k+1, "byteswapped");
         }

         /* Read-only */
         if (OcTensor_isReadOnly(tensor))
         {  if (flag)
            {  k = 2; slen += k;
               if (mode == 1) s += snprintf(s, k+1, ", ");
            }
            k = 9; slen += k; flag = 1;
            if (mode == 1) s += snprintf(s, k+1, "read-only");
         }

         /* Closing bracket */
         k = 1; slen += k;
         if (mode == 1) s += snprintf(s, k+1, ")");
      }

      /* Post-string */
      if (post != NULL)
      {  k = strlen(post); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", post);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer;
         if (buffer == NULL)
         {  OcErrorMessage("Insufficient memory for output string");
            goto final;
         }
      }
   }

   /* Ensure that the string is terminated properly */
   *s = '\0';

   /* Success */
   result = 0;

final :
   /* -------------------------------------------------------- */
   /* Clean up. Note that freeing of the buffer has to be done */
   /* using the regular free function to match its allocation  */
   /* above using malloc.                                      */
   /* -------------------------------------------------------- */
   if ((result != 0) && (buffer != NULL)) { free(buffer); buffer = NULL; }

   *str = buffer;

   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_display(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  char *str = NULL, *footer = NULL;
   int   result;

   /* Sanity check */
   if (tensor == NULL)
   {  printf("<tensor NULL pointer>\n");
      return 0;
   }

   /* Format the footer */
   if (OcTensor_formatFooter(tensor, &footer, "<", ">\n") != 0) return -1;

   /* Format and display the tensor */
   result = OcTensor_format(tensor, &str, NULL, footer);
   if (result == 0)
   {  printf("%s", str);
   }

   /* Deallocate memory */
   if (str) free(str);
   if (footer) free(footer);

   return result;
}


/* --------------------------------------------------------------------- */
void OcTensor_displayShape(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{
   OcShape_display(tensor -> ndims, tensor -> size, tensor -> strides);
}



/* --------------------------------------------------------------------- */
int OcTensor_displayFooter(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  char *footer = NULL;
   int   result;

   /* Format and display the footer */
   result = OcTensor_formatFooter(tensor, &footer, "<", ">\n");
   if ((result == 0) && (footer != NULL))
   {  /* Print and free */
      printf("%s", footer);
      free(footer);
   }

   return result;
}

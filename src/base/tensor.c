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

#include "ocean/base/tensor.h"
#include "ocean/base/shape.h"
#include "ocean/base/types.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <stdlib.h>



/* ===================================================================== */
/* Reference count operations                                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcTensor *OcIncrefTensor(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  /* Returns tensor to allow: tensor = ocIncrefTensor(tensor) */

   if (tensor != NULL) tensor -> refcount ++;

   return tensor;
}


/* -------------------------------------------------------------------- */
void OcDecrefTensor(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{
   if (tensor == NULL) return;

   tensor -> refcount --;
   if (tensor -> refcount == 0)
   {
      /* Decrement the storage reference count */
      OcDecrefStorage(tensor -> storage);

      /* Free the strides and dimensions */
      if (tensor -> size    != NULL) OcFree(tensor -> size);
      if (tensor -> strides != NULL) OcFree(tensor -> strides);

      /* Free the tensor object */
      OcFree(tensor);
   }
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcTensor *OcAllocateTensor(int ndims, OcDType dtype)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor;

   /* Check the data type */
   if (dtype == OcDTypeNone)
      OcError(NULL, "Invalid data type in tensor allocation");

   /* Allocate memory for the tensor structure */
   tensor = (OcTensor *)OcMalloc(sizeof(OcTensor));
   if (tensor == NULL) OcError(NULL, "Insufficient memory to allocate the tensor structure");

   /* Initialize the tensor fields */
   tensor -> ndims       = ndims;
   tensor -> capacity    = 0;
   tensor -> size        = NULL;
   tensor -> strides     = NULL;
   tensor -> storage     = NULL;
   tensor -> device      = NULL;
   tensor -> dtype       = dtype;
   tensor -> elemsize    = OcDType_size(dtype);
   tensor -> nelem       = 0;
   tensor -> blockOffset = 0;
   tensor -> blockExtent = 0;
   tensor -> refcount    = 1;
   tensor -> flags       = 0;

   /* Allocate memory for the strides and dimensions */
   if (ndims == 0) return tensor;
   if (OcTensor_allocDims(tensor, ndims) != 0) goto error;

   /* Success */
   return tensor;

error : ;
   /* Unable to allocate strides and dimensions */
   OcDecrefTensor(tensor);
   return NULL;
}


/* -------------------------------------------------------------------- */
int OcTensor_allocDims(OcTensor *tensor, int ndims)
/* -------------------------------------------------------------------- */
{  OcSize  *size;
   OcIndex *strides;
   int      i;

   /* Check the number of dimensions */
   if ((ndims < 0) || (ndims > OC_TENSOR_MAX_DIMS))
      OcError(-1, "Invalid number of dimensions in tensor allocation");

   if (ndims <= tensor -> capacity) return 0;

   /* Allocate memory for size and strides */
   size    = (OcSize  *)OcMalloc(sizeof(OcSize)  * ndims);
   strides = (OcIndex *)OcMalloc(sizeof(OcIndex) * ndims);
   if ((size == NULL) || (strides == NULL))
   {  if (size) OcFree(size);
      if (strides) OcFree(strides);
      OcError(-1, "Insufficient memory to allocate tensor strides and dimensions");
   }

   /* Copy existing shape information */
   if ((tensor -> size) && (tensor -> strides))
   {  for (i = 0; i < tensor -> ndims; i++)
      {  size[i] = tensor -> size[i];
         strides[i] = tensor -> strides[i];
      }
   }

   /* Replace existing arrays */
   if (tensor -> size) OcFree(tensor -> size);
   if (tensor -> strides) OcFree(tensor -> strides);
   tensor -> size = size;
   tensor -> strides = strides;
   tensor -> capacity = ndims;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensor_updateShape(OcTensor *tensor, int ndims, OcSize *size, OcIndex *strides,
                         int updateFlags, int updateSelfOverlap, int updateExtent)
/* -------------------------------------------------------------------- */
{  OcIndex  stride;
   OcSize   d, nelem = 1;
   int      i;
   int      result = 0;

   /* Make sure all dimensions are nonnegative */
   if (!OcShape_isValidSize(ndims, size))
      OcError(-1, "Negative tensor sizes are not allowed");

   /* Allocate memory for the shape information */
   if (OcTensor_allocDims(tensor, ndims) != 0) return -1;

   /* Update the shape information */
   if (strides == NULL)
   {   /* Copy the dimensions and set the strides */
       stride = tensor -> elemsize;
       for (i = 0; i < ndims; i++)
       {  d = size[i]; nelem *= d;
          tensor -> size[i] = d;
          tensor -> strides[i] = stride;
          stride *= d;
       }
   }
   else
   {   /* Copy the strides and dimensions */
       for (i = 0; i < ndims; i++)
       {  d = size[i]; nelem *= d;
          tensor -> size[i] = d;
          tensor -> strides[i] = strides[i];
       }
   }

   /* Set the number of dimensions and elements */
   tensor -> ndims = ndims;
   tensor -> nelem = nelem;

   /* Update the flags */
   if (updateFlags) OcTensor_updateShapeFlags(tensor, updateSelfOverlap);
   if (updateExtent) result = OcTensor_updateExtent(tensor);

   return result;
}


/* --------------------------------------------------------------------- */
void OcTensor_updateShapeFlags(OcTensor *tensor, int updateSelfOverlap)
/* --------------------------------------------------------------------- */
{  OcTensorFlags flags;
   OcIndex       stride, n;
   OcSize        size;
   int           flagZeros = 0;
   int           flagLinear = 1;
   int           i;

   /* Reset the shape flags (extent information is not included) */
   flags = (tensor -> flags) & ~OC_TENSOR_SHAPE_MASK;
   if (!updateSelfOverlap)
   {  flags |= (tensor -> flags) & (OC_TENSOR_SELF_OVERLAPPING_SET |
                                    OC_TENSOR_SELF_OVERLAPPING);
   }

   /* Check linear and zeros */
   n = tensor -> elemsize;
   for (i = 0; i < tensor -> ndims; i++)
   {  size = tensor -> size[i];
      if (size > 1)
      {  stride = tensor -> strides[i];
         if (stride == 0) flagZeros = 1;
         if (stride != n) flagLinear = 0;
         n *= size;
      }
   }
   if (flagZeros ) flags |= OC_TENSOR_ZERO_STRIDES;
   if (flagLinear) flags |= OC_TENSOR_LINEAR |
                            OC_TENSOR_CONTIGUOUS |
                            OC_TENSOR_CONTIGUOUS_SET |
                            OC_TENSOR_SELF_OVERLAPPING_SET;

   /* Set the tensor flags */
   tensor -> flags = flags;
}


/* --------------------------------------------------------------------- */
int OcTensor_updateExtent(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  int result;

   /* Clear the flag */
   tensor -> flags &= ~OC_TENSOR_EXTENT;

   /* Compute the extent */
   result = OcShape_extent(tensor -> ndims, tensor -> size, tensor -> strides,
                           tensor -> elemsize, &(tensor -> blockOffset),
                           &(tensor -> blockExtent));

   /* Set the flag */
   if (result == 0) tensor -> flags |= OC_TENSOR_EXTENT;

   return result;
}


/* --------------------------------------------------------------------- */
int OcTensor_extent(OcTensor *tensor, OcSize *offset, OcSize *extent)
/* --------------------------------------------------------------------- */
{
   /* Make sure the tensor extent is current */
   if (((tensor -> flags) & OC_TENSOR_EXTENT) == 0)
   {  if (OcTensor_updateExtent(tensor) != 0) return -1;
   }

   /* Set the values */
   *offset = tensor -> blockOffset;
   *extent = tensor -> blockExtent;

   return 0;
}


/* -------------------------------------------------------------------- */
OcTensor *OcTensor_shallowCopy(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcTensor *result;
   int       i;

   /* Create the tensor object */
   result = OcAllocateTensor(tensor -> ndims, tensor -> dtype);
   if (result == NULL) return result;

   /* Copy all fields */
   for (i = 0; i < tensor -> ndims; i++)
   {  result -> size[i]    = tensor -> size[i];
      result -> strides[i] = tensor -> strides[i];
   }

   result -> offset      = tensor -> offset;
   result -> storage     = OcIncrefStorage(tensor -> storage);
   result -> device      = tensor -> device;
   result -> nelem       = tensor -> nelem;
   result -> flags       = tensor -> flags;
   result -> blockOffset = tensor -> blockOffset;
   result -> blockExtent = tensor -> blockExtent;

   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_removeMatchingRepeats(OcTensor *tensor, OcTensor *reference)
/* --------------------------------------------------------------------- */
{  OcTensor    *result;
   OcSize       size[OC_TENSOR_MAX_DIMS];
   OcIndex      strides[OC_TENSOR_MAX_DIMS];
   OcSize       s1, s2;
   int          flagZeroStrides, flagChanged = 0;
   int          ndims;
   int          i, j, k;

   if (!OcTensor_hasZeroStrides(tensor)    ||
       !OcTensor_hasZeroStrides(reference) ||
       (reference -> nelem != tensor -> nelem) ||
       (reference -> nelem == 0))
       return OcIncrefTensor(tensor);

   /* Iterate over blocks of the same size. As the number of elements */
   /* of the tensors match and is nonzero it suffices to compare one  */
   /* of the two dimension indices with the number of dimensions.     */
   i = 0; j = 0; ndims = 0;
   while (i < reference -> ndims)
   {
      s1 = reference -> size[i];
      s2 = tensor -> size[j]; k = 1;
      flagZeroStrides = ((reference -> strides[i] == 0) && (tensor -> strides[i] == 0));

      /* Find a block of matching size -- this is guaranteed to exist */
      /* the total number of elements matches and we therefore do not */
      /* need to check validity of the i and j indices.               */
      while (s1 != s2)
      { 
         if (s1 < s2)
         {  i++; s1 *= reference -> size[i];
            if (reference -> strides[i] != 0) flagZeroStrides = 0;
         }
         else
         {  j++; s2 *= tensor -> size[j]; k++;
            if (tensor -> strides[j] != 0) flagZeroStrides = 0; 
         }
      }

      /* Output the block of k elements unless it has maching zero strides */
      if (!flagZeroStrides)
      {  while (k > 0)
         {  k--;
            size[ndims] = tensor -> size[j-k];
            strides[ndims] = tensor -> strides[j-k];
            ndims ++;
         }
      }
      else
      {  flagChanged = 1;
      }

      /* Increment the indices to start the next block */
      i++; j++;
   }

   /* Create a new or reference-incremented tensor */
   if (flagChanged)
   {  /* Make a shallow copy of the tensor */
      result = OcTensor_shallowCopy(tensor);
      if (result == NULL) return NULL;

      /* Set new shape - update the flags, do not update self-overlap */
      /* information, do not update the extent.                       */
      if (OcTensor_updateShape(result, ndims, size, strides, 1, 0, 0) != 0)
      {  OcDecrefTensor(result);
         return NULL;
      }
   }
   else
   {  result = OcIncrefTensor(tensor);
   }

   return result;
}


/* --------------------------------------------------------------------- */
OcTensor *OcTensor_removeRepeats(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  OcTensor *result;
   int i, j;

   /* Check if the tensor has zero strides */
   if (!OcTensor_hasZeroStrides(tensor))
      return OcIncrefTensor(tensor);

   /* Make a shallow copy */
   result = OcTensor_shallowCopy(tensor);
   if (result == NULL) return NULL;

   /* Remove all zero strides */
   for (i = 0, j = 0; j < tensor -> ndims; j++)
   {  if ((tensor -> strides[j] != 0) || (tensor -> size[j] == 0))
      {  result -> size[i]    = tensor -> size[j];
         result -> strides[i] = tensor -> strides[j];
         i++;
      }
      else
      {  result -> nelem /= tensor -> size[j];
      }
   }
   result -> ndims = i;

   /* Update the flags, do not update self-overlap information */
   OcTensor_updateShapeFlags(result, 0);
   
   return result;
}



/* ===================================================================== */
/* Query and property functions                                          */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensor_isScalar(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{
   /* Empty tensors with zero dimension are not allowed */
   return (tensor -> ndims == 0) ? 1 : 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_isLinear(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{
  return (tensor -> flags & OC_TENSOR_LINEAR) ? 1 : 0;
}


/* --------------------------------------------------------------------- */
int OcTensor_isContiguous(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{
   /* Check for tensors with linear layout */
   if ((tensor -> flags & OC_TENSOR_LINEAR) != 0) return 1;

   /* Return existing result when available */
   if ((tensor -> flags & OC_TENSOR_CONTIGUOUS_SET) != 0)
      return (tensor -> flags & OC_TENSOR_CONTIGUOUS) ? 1 : 0;

   /* Check whether the tensor is contiguous */
   tensor -> flags |= OC_TENSOR_CONTIGUOUS_SET;
   if (OcShape_isContiguous(tensor -> ndims, tensor -> size, tensor -> strides, tensor -> elemsize))
   {  tensor -> flags |= OC_TENSOR_CONTIGUOUS;
      return 1;
   }
   else
   {  return 0;
   }
}


/* --------------------------------------------------------------------- */
int OcTensor_isSelfOverlapping(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{
   /* Use precomputed result when available */
   if (tensor -> flags & OC_TENSOR_SELF_OVERLAPPING_SET)
      return (tensor -> flags & OC_TENSOR_SELF_OVERLAPPING) ? 1 : 0;

   /* Linear and contiguous tensors are not self overlapping */
   if (OcTensor_isContiguous(tensor) ||
       !OcShape_isSelfOverlapping(tensor -> ndims, tensor -> size, tensor -> strides, tensor -> elemsize))
   {  tensor -> flags |= OC_TENSOR_SELF_OVERLAPPING_SET;
      tensor -> flags &= ~OC_TENSOR_SELF_OVERLAPPING;
      return 0;
   }
   else
   {  tensor -> flags |= OC_TENSOR_SELF_OVERLAPPING_SET;
      tensor -> flags |= OC_TENSOR_SELF_OVERLAPPING;
      return 1;
   }
}


/* -------------------------------------------------------------------- */
int OcTensor_isByteswapped(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{
   if (tensor -> flags & OC_TENSOR_BYTESWAPPED)
   {  return (OcStorage_isByteswapped(tensor -> storage) ? 0 : 1);
   }
   else
   {  return OcStorage_isByteswapped(tensor -> storage);
   }
}


/* -------------------------------------------------------------------- */
int OcTensor_isReadOnly(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{
   if (tensor -> flags & OC_TENSOR_READONLY)
   {  return 1;
   }
   else
   {  return OcStorage_isReadOnly(tensor -> storage);
   }
}


/* --------------------------------------------------------------------- */
void OcTensor_setReadOnly(OcTensor *tensor, int readonly)
/* --------------------------------------------------------------------- */
{
   /* Set the read-only flag for the tensor. Note that the tensor */
   /* remains read-only as long as the underlying storage is.     */
   if (readonly)
        tensor -> flags |= OC_TENSOR_READONLY;
   else tensor -> flags &=~OC_TENSOR_READONLY;
}



/* --------------------------------------------------------------------- */
int OcTensor_isAligned(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  char *data = OcTensor_data(tensor);
   int   elemsize = tensor -> elemsize;
   int   i;

   /* Check alignment of the data pointer */
   if (data)
   {  if (((OcUintptr)data) % elemsize != 0) return 0;
   }

   /* Check alignment of the strides */
   for (i = 0; i < tensor -> ndims; i++)
   {  if ((tensor -> size[i] > 1) && (tensor -> strides[i] % elemsize != 0)) return 0;
   }

   return 1;
}


/* --------------------------------------------------------------------- */
int OcTensor_isDetached(OcTensor *tensor, int flagStorage)
/* --------------------------------------------------------------------- */
{
   if (tensor -> refcount != 1) return 0;
   if ((flagStorage) && (!OcStorage_isDetached(tensor -> storage))) return 0;
   return 1;
}


/* --------------------------------------------------------------------- */
int OcTensor_isValidDest(OcTensor *tensor, int allowZeroStrides)
/* --------------------------------------------------------------------- */
{
   if (OcTensor_isReadOnly(tensor))
      OcError(0, "Attempt to modify a read-only tensor");
   if (((!allowZeroStrides) && OcTensor_hasZeroStrides(tensor)) ||
       (OcTensor_isSelfOverlapping(tensor)))
      OcError(0, "Attempt to modify a self-overlapping tensor");
   return 1;
}


/* --------------------------------------------------------------------- */
int OcTensor_hasOrder(OcTensor *tensor, char type)
/* --------------------------------------------------------------------- */
{  OcIndex stride;
   int     i;

   if ((type == 'c') || (type == 'C'))
   {  stride = tensor -> elemsize;
      for (i = tensor -> ndims-1; i >= 0; i--)
      {  if (tensor -> size[i] <= 1) continue;
         if (tensor -> strides[i] != stride) return 0;
         stride *= tensor -> size[i];
      }
      return 1;
   }
   else if ((type == 'f') || (type == 'F'))
   {  return OcTensor_isLinear(tensor);
   }
   else if ((type == 'r') || (type == 'R'))
   {  stride = tensor -> elemsize;
      if (tensor -> ndims == 0) return 1;
      if (tensor -> nelem <= 1) return 1;
      if (tensor -> ndims == 1) return (tensor -> strides[0] == stride) ? 1 : 0;

      /* Two or more dimensions */
      if ((tensor -> size[1] > 1) && (tensor -> strides[1] != stride)) return 0;
      stride *= tensor -> size[1];
      if ((tensor -> size[0] > 1) && (tensor -> strides[0] != stride)) return 0;
      stride *= tensor -> size[0];
      for (i = 2; i < tensor -> ndims; i++)
      {  if (tensor -> size[i] <=1) continue;
         if (tensor -> strides[i] != stride) return 0;
         stride *= tensor -> size[i];
      }
      return 1;
   }
   else
   {  OcError(-1, "Invalid tensor order ('%c')", type);
   }
}


/* --------------------------------------------------------------------- */
int OcTensor_hasValidAlignment(OcTensor *tensor)
/* --------------------------------------------------------------------- */
{  if (tensor -> device -> requiresAlignedData)
   {  return OcTensor_isAligned(tensor);
   }
   else
   {  return 1;
   }
}


/* -------------------------------------------------------------------- */
int OcTensor_hasZeroStrides(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{
   return (tensor -> flags & OC_TENSOR_ZERO_STRIDES) ? 1 : 0;
}


/* -------------------------------------------------------------------- */
OcSize OcTensor_repeatCount(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcSize n = 1;
   int    i;

   if (!OcTensor_hasZeroStrides(tensor)) return 0;

   for (i = 0; i < tensor -> ndims; i++)
   {  if (tensor -> strides[i] == 0)
         n *= tensor -> size[i];
   }

   return n;
}


/* --------------------------------------------------------------------- */
int OcTensors_match(OcTensor *tensor1, OcTensor *tensor2)
/* --------------------------------------------------------------------- */
{
   if (tensor1 == tensor2) return 1;
   if ((tensor1 == NULL) || (tensor2 == NULL)) return 0;
   if ((OcTensor_data(tensor1) == OcTensor_data(tensor2)) &&
       (tensor1 -> dtype == tensor2 -> dtype) &&
       (tensor1 -> device == tensor2 -> device) &&
       (OcTensors_haveIdenticalLayout(tensor1, tensor2)) &&
       (OcTensors_haveSameByteOrder(tensor1, tensor2)))
   {   return 1;
   }

    return 0;
}   


/* --------------------------------------------------------------------- */
int OcTensors_overlap(OcTensor *tensor1, OcTensor *tensor2)
/* --------------------------------------------------------------------- */
{  OcSize   offset1, offset2;
   OcSize   extent1, extent2;
   char    *ptrStart1, *ptrEnd1;
   char    *ptrStart2, *ptrEnd2;

   /* Tensors on different devices to not overlap */
   if (tensor1 -> device != tensor2 -> device) return 0;

   /* Check overlap of storage */
   if (OcStorage_overlap(tensor1 -> storage, tensor2 -> storage) == 0) return 0;

   /* Compute the tensor bounds */
   if (OcTensor_extent(tensor1, &offset1, &extent1) != 0) return -1;
   if (OcTensor_extent(tensor2, &offset2, &extent2) != 0) return -1;

   /* Compute the start and end pointers (the end pointers are not inclusive) */
   ptrStart1 = OcTensor_data(tensor1) - offset1;
   ptrStart2 = OcTensor_data(tensor2) - offset2;
   ptrEnd1   = ptrStart1 + extent1;
   ptrEnd2   = ptrStart2 + extent2;

   if ((ptrStart1 >= ptrEnd2) || (ptrEnd1 <= ptrStart2))
   {  /* The tensor data blocks do not overlap */
      return 0;
   }

   /* Generic computation of tensor overlap */
   return OcShapes_overlap(tensor1 -> ndims, tensor1 -> size, tensor1 -> strides, (OcIndex)(OcTensor_data(tensor1)), tensor1 -> elemsize,
                           tensor2 -> ndims, tensor2 -> size, tensor2 -> strides, (OcIndex)(OcTensor_data(tensor2)), tensor2 -> elemsize);
}


/* --------------------------------------------------------------------- */
int OcTensors_haveSameSize(OcTensor *tensor1, OcTensor *tensor2)
/* --------------------------------------------------------------------- */
{  int i;

   if (tensor1 -> ndims != tensor2 -> ndims) return 0;

   for (i = 0; i < tensor1 -> ndims; i++)
   {  if (tensor1 -> size[i] != tensor2 -> size[i]) return 0;
   }

   return 1;
}


/* -------------------------------------------------------------------- */
int OcTensors_haveSameByteOrder(OcTensor *tensor1, OcTensor *tensor2)
/* -------------------------------------------------------------------- */
{  int result;

   if (tensor1 -> elemsize == 1) return 1;
   if (tensor2 -> elemsize == 1) return 1;

   result  = OcTensor_isByteswapped(tensor1) ? 1 : -1;
   result *= OcTensor_isByteswapped(tensor2) ? 1 : -1;
   if (tensor1 -> device -> endianness !=
       tensor2 -> device -> endianness)
   {  result *= -1;
   }

   return (result == 1) ? 1 : 0;
}


/* --------------------------------------------------------------------- */
int OcTensors_haveIdenticalLayout(OcTensor *tensor1, OcTensor *tensor2)
/* --------------------------------------------------------------------- */
{  int i;

   /* Check if tensor dimensions and strides are identical. We */
   /* allow artibrary stride values when the size is zero.     */
   if (tensor1 -> ndims != tensor2 -> ndims) return 0;
   if (tensor1 -> nelem != tensor2 -> nelem) return 0;

   for (i = 0; i < tensor1 -> ndims; i++)
   {  if (tensor1 -> size[i] != tensor2 -> size[i]) return 0;
      if (tensor1 -> size[i] <= 1) continue;

      if (tensor1 -> strides[i] != tensor2 -> strides[i]) return 0;
   }

   return 1;
}


/* --------------------------------------------------------------------- */
int OcTensors_haveCompatibleLayout(OcTensor *tensor1, OcTensor *tensor2)
/* --------------------------------------------------------------------- */
{  OcSize s1, s2;  
   int i, j;

   /* Check if the tensors match in reduced form */
   if (tensor1 -> nelem != tensor2 -> nelem) return 0;
   if (tensor1 -> nelem <= 1) return 1;
   
   i = tensor1 -> ndims - 1;
   j = tensor2 -> ndims - 1;
   while ((i >= 0) && (j >= 0))
   {  /* Block strides have to match */
      if (tensor1 -> strides[i] != tensor2 -> strides[j]) return 0;

      /* Merge dimensions into blocks */
      s1 = tensor1 -> size[i];
      s2 = tensor2 -> size[j];
      while ((i > 0) && ((tensor1 -> size[i-1] == 1) || (tensor1 -> strides[i-1] == tensor1 -> size[i] * tensor1 -> strides[i])))
      {  i--; s1 *= tensor1 -> size[i];  }
      while ((j > 0) && ((tensor2 -> size[j-1] == 1) || (tensor2 -> strides[j-1] == tensor2 -> size[j] * tensor2 -> strides[j])))
      {  j--; s2 *= tensor2 -> size[j];  }

      /* Block size has to match */
      if (s1 != s2) return 0;

      /* Proceed to the next dimension */
      i--; j --;
   }

   return 1;
}

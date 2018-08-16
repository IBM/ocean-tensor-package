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

#include "ocean/core/interface/index_itf.h"
#include "ocean/core/interface/scalar_itf.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/cpu/device_cpu.h"

#include "ocean/base/format.h"
#include "ocean/base/platform.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <string.h>


/* ===================================================================== */
/* Internal variables                                                    */
/* ===================================================================== */
OcSize __oc_tensor_index_elem_size[] =
   { [OC_IDX_SCALAR]   = sizeof(OcTensorIndexElem_Scalar),
     [OC_IDX_INDICES]  = sizeof(OcTensorIndexElem_Indices),
     [OC_IDX_MASK]     = sizeof(OcTensorIndexElem_Mask),
     [OC_IDX_INSERT]   = sizeof(OcTensorIndexElem_Insert),
     [OC_IDX_ALL]      = sizeof(OcTensorIndexElem_All),
     [OC_IDX_ELLIPSIS] = sizeof(OcTensorIndexElem_Ellipsis),
     [OC_IDX_RANGE]    = sizeof(OcTensorIndexElem_Range),
     [OC_IDX_STEPS]    = sizeof(OcTensorIndexElem_Steps),
     [OC_IDX_OFFSET]   = sizeof(OcTensorIndexElem_Offset)
   };


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

int                OcTensorIndex_addElement         (OcTensorIndex *index, OcTensorIndexElem *elem);
void               OcTensorIndex_deleteElement      (OcTensorIndex *index, int elemIdx);
int                OcTensorIndex_ensureCapacity     (OcTensorIndex *index, int capacity);

/* Tensor index elements */
OcTensorIndexElem *OcTensorIndexElem_alloc          (OcTensorIndexType type);
void               OcTensorIndexElem_free           (OcTensorIndexElem *elem);
OcTensorIndexElem *OcTensorIndexElem_shallowCopy    (OcTensorIndexElem *elem);
int                OcTensorIndexElem_checkSize      (OcTensorIndexElem *elem, int ndims, OcSize *size, OcIndex *strides, int dimOffset);
int                OcTensorIndexElem_setInputSize   (OcTensorIndexElem *elem, int ndims, OcSize *size, OcIndex *strides);

OcTensorIndexElem *OcTensorIndexElem_createScalar   (OcIndex index);
OcTensorIndexElem *OcTensorIndexElem_createAll      (int ndims);
OcTensorIndexElem *OcTensorIndexElem_createSteps    (OcIndex start, OcIndex step, OcSize nelem);
OcTensorIndexElem *OcTensorIndexElem_createOffset   (int ndims, OcSize *size, OcIndex *strides, OcIndex range, OcTensor *tensor);

int                OcTensorIndexElem_indicesToOffset(OcTensorIndexElem **element, OcSize *size, OcIndex *strides, OcTensorIndexElem **result);
int                OcTensorIndexElem_maskToIndices  (OcTensorIndexElem **element, OcTensorIndexElem **result);
int                OcTensorIndexElem_rangeToSteps   (OcTensorIndexElem **element, OcSize n, OcTensorIndexElem **result);
int                OcTensorIndexElem_bind           (OcTensorIndexElem **elem, int dimOffset, OcSize *size,
                                                     OcIndex *strides, OcTensorIndexElem **result);

OcTensorIndexElem *OcIncrefTensorIndexElem(OcTensorIndexElem *elem);
void               OcDecrefTensorIndexElem(OcTensorIndexElem *elem);



/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcTensorIndex *OcTensorIndex_create(void)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index;

   /* Allocate the tensor index */
   index = (OcTensorIndex *)OcMalloc(sizeof(OcTensorIndex));
   if (index == NULL) OcError(NULL, "Error allocating tensor index");

   /* Initialize */
   index -> elem     = NULL;
   index -> capacity = 0;
   index -> n        = 0;
   index -> refcount = 1;

   return index;
}


/* -------------------------------------------------------------------- */
OcTensorIndex *OcTensorIndex_createWithCapacity(int capacity)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *index;

   /* Create a new index */
   index = OcTensorIndex_create();
   if (index == NULL) return NULL;

   /* Set the capacity */
   if (OcTensorIndex_ensureCapacity(index, capacity) != 0)
   {  OcDecrefTensorIndex(index);
      return NULL;
   }

   return index;
}


/* -------------------------------------------------------------------- */
void OcTensorIndex_free(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  int i;

   if (index -> elem)
   {  for (i = 0; i < index -> n; i++)
      {  if (index -> elem[i]) OcDecrefTensorIndexElem(index -> elem[i]);
      }

      OcFree(index -> elem);
   }

   OcFree(index);
}


/* -------------------------------------------------------------------- */
OcTensorIndex *OcIncrefTensorIndex(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{
   if (index) index -> refcount ++;

   return index;
}


/* -------------------------------------------------------------------- */
void OcDecrefTensorIndex(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{
   if (index)
   {  index -> refcount --;
      if (index -> refcount <= 0)
      {  OcTensorIndex_free(index);
      }
   }
}


/* -------------------------------------------------------------------- */
OcTensorIndex *OcTensorIndex_shallowCopy(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *idx;
   int i;

   /* Create a new index */
   idx = OcTensorIndex_createWithCapacity(index -> capacity);
   if (idx == NULL) return NULL;

   /* Copy the elements */
   idx -> n = index -> n;
   for (i = 0; i < idx -> n; i++)
   {  idx -> elem[i] = OcIncrefTensorIndexElem(index -> elem[i]);
   }

   return idx;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_ensureCapacity(OcTensorIndex *index, int capacity)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem **buffer;
   int i, n;

   /* Make sure the tensor has the required capacity */
   if (index -> capacity < capacity)
   {  
      /* Determine the new capacity */
      n = index -> capacity;
      n = (n == 0) ? OC_TENSOR_MAX_DIMS : (2*n);
      if (capacity > n) n = capacity;

     /* Allocate memory for index elements */
      buffer = (OcTensorIndexElem **)OcMalloc(sizeof(OcTensorIndexElem *) * n);
      if (buffer == NULL) OcError(-1, "Error adding tensor index element");

      /* Update the index elements */
      if (index -> elem)
      {  /* Copy the existing elements */
         for (i = 0; i < index -> n; i++)
         {  buffer[i] = index -> elem[i];
         }

         /* Free the existing buffer */
         OcFree(index -> elem);
      }
      index -> elem = buffer;
      index -> capacity = n;      
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addElement(OcTensorIndex *index, OcTensorIndexElem *elem)
/* -------------------------------------------------------------------- */
{  int n = index -> n;

   /* ----------------------------------------------------------- */
   /* This function steals a reference count to the index element */
   /* ----------------------------------------------------------- */

   if (OcTensorIndex_ensureCapacity(index, n + 1) != 0)
   {  OcDecrefTensorIndexElem(elem);
      return -1;
   }

   /* Add the element */
   index -> elem[n] = elem;
   index -> n += 1;

   /* Return the index of the tensor index element */
   return n;
}


/* -------------------------------------------------------------------- */
void OcTensorIndex_deleteElement(OcTensorIndex *index, int elemIdx)
/* -------------------------------------------------------------------- */
{  int i;

   /* Parameter checks */
   if ((index == NULL) || (elemIdx < 0) || (elemIdx >= index -> n))
      return ;

   /* Delete the element */
   if (index -> elem[elemIdx]) OcDecrefTensorIndexElem(index -> elem[elemIdx]);

   /* Shift existing elements */
   index -> n -= 1;
   for (i = elemIdx; i < index -> n; i++)
   {  index -> elem[i] = index -> elem[i+1];
   }
}



/* ==================================================================== */
/* Tensor index elements                                                */
/* ==================================================================== */

/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcTensorIndexElem_alloc(OcTensorIndexType type)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   OcSize size = __oc_tensor_index_elem_size[type];

   /* Allocate the index element */
   elem = (OcTensorIndexElem *)OcMalloc(size * sizeof(char));
   if (elem == NULL) OcError(NULL, "Error allocating tensor index element");

   /* Initialize the index element */
   elem -> type        = type;
   elem -> size        = NULL;
   elem -> strides     = NULL;
   elem -> nInputDims  = 0;
   elem -> nOutputDims = 0;
   elem -> refcount    = 1;

   return elem;   
}


/* -------------------------------------------------------------------- */
void OcTensorIndexElem_free(OcTensorIndexElem *elem)
/* -------------------------------------------------------------------- */
{
   if (elem == NULL) return;
   if (elem -> size) OcFree(elem -> size);
   if (elem -> strides) OcFree(elem -> strides);

   /* Type-specific clean-up */
   switch (elem -> type)
   {
      case OC_IDX_MASK :
         OcXDecrefTensor(((OcTensorIndexElem_Mask *)(elem)) -> tensor);
         break;

      case OC_IDX_INDICES :
         OcXDecrefTensor(((OcTensorIndexElem_Indices *)(elem)) -> tensor);
         break;

      case OC_IDX_OFFSET :
         OcXDecrefTensor(((OcTensorIndexElem_Offset *)(elem)) -> tensor);
         break;

      default : ;
         /* Nothing to clean up */
   }

   /* Free the element */
   OcFree(elem);
}


/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcIncrefTensorIndexElem(OcTensorIndexElem *elem)
/* -------------------------------------------------------------------- */
{
   if (elem) elem -> refcount ++;

   return elem;
}


/* -------------------------------------------------------------------- */
void OcDecrefTensorIndexElem(OcTensorIndexElem *elem)
/* -------------------------------------------------------------------- */
{
   if (elem)
   {  elem -> refcount --;
      if (elem -> refcount <= 0) OcTensorIndexElem_free(elem);
   }
}


/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcTensorIndexElem_shallowCopy(OcTensorIndexElem *elem)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *result;
   int i;

   if (elem == NULL) return NULL;

   /* Allocate memory fo the element */
   result = OcTensorIndexElem_alloc(elem -> type);
   if (result == NULL) return NULL;

   /* Set the size and strides */
   result -> nOutputDims = elem -> nOutputDims;
   if (OcTensorIndexElem_setInputSize(result, elem -> nInputDims, elem -> size, elem -> strides) != 0)
   {  OcDecrefTensorIndexElem(result);
      return result;
   }

   #define OC_COPY_FIELD(T,FIELD) (((T *)result) -> FIELD = ((T *)elem) -> FIELD)

   /* Copy element type */
   switch (elem -> type)
   {  /* ------------------------------ */
      case OC_IDX_SCALAR:
      /* ------------------------------ */
      {  OC_COPY_FIELD(OcTensorIndexElem_Scalar, index);
         break;
      }

      /* ------------------------------ */
      case OC_IDX_INDICES:
      /* ------------------------------ */
      {  for (i = 0; i < elem -> nInputDims; i++)
         {  OC_COPY_FIELD(OcTensorIndexElem_Indices, indexMin[i]);
            OC_COPY_FIELD(OcTensorIndexElem_Indices, indexMax[i]);
         }

         OC_COPY_FIELD(OcTensorIndexElem_Indices, tensor);
         OcIncrefTensor(((OcTensorIndexElem_Indices *)elem) -> tensor);
         break;
      }

      /* ------------------------------ */
      case OC_IDX_MASK:
      /* ------------------------------ */
      {  OC_COPY_FIELD(OcTensorIndexElem_Mask, nnz);
         OC_COPY_FIELD(OcTensorIndexElem_Mask, flagNnz);
         OC_COPY_FIELD(OcTensorIndexElem_Mask, tensor);
         OcIncrefTensor(((OcTensorIndexElem_Mask *)elem) -> tensor);
         break;
      }

      /* ------------------------------ */
      case OC_IDX_OFFSET:
      /* ------------------------------ */
      {  OC_COPY_FIELD(OcTensorIndexElem_Offset, range);
         OC_COPY_FIELD(OcTensorIndexElem_Offset, tensor);
         OcIncrefTensor(((OcTensorIndexElem_Offset *)elem) -> tensor);
         break;
      }


      /* ------------------------------ */
      case OC_IDX_INSERT:
      case OC_IDX_ALL:
      case OC_IDX_ELLIPSIS:
      /* ------------------------------ */
      {  /* Nothing extra to copy */
         break;
      }

      /* ------------------------------ */
      case OC_IDX_RANGE:
      /* ------------------------------ */
      {  OC_COPY_FIELD(OcTensorIndexElem_Range, start);
         OC_COPY_FIELD(OcTensorIndexElem_Range, step);
         OC_COPY_FIELD(OcTensorIndexElem_Range, stop);
         OC_COPY_FIELD(OcTensorIndexElem_Range, flagStart);
         OC_COPY_FIELD(OcTensorIndexElem_Range, flagStop);
         break;
      }

      /* ------------------------------ */
      case OC_IDX_STEPS:
      /* ------------------------------ */
      {  OC_COPY_FIELD(OcTensorIndexElem_Steps, start);
         OC_COPY_FIELD(OcTensorIndexElem_Steps, step);
         OC_COPY_FIELD(OcTensorIndexElem_Steps, nelem);
         break;
      }
   }

   #undef OC_COPY_FIELD

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorIndexElem_checkSize(OcTensorIndexElem *elem, int ndims,
                                OcSize *size, OcIndex *strides, int dimOffset)
/* -------------------------------------------------------------------- */
{  int i;

   if (((elem -> size) || (elem -> strides)) && (elem -> nInputDims != ndims))
      OcError(-1, "Mismatch in number of index dimensions");

   /* Check the size */
   if ((elem -> size) && (size))
   {  for (i = 0; i < ndims; i++)
         if (elem -> size[i] != size[i])
            OcError(-1, "Mismatch in size at dimension %d (expected %"OC_FORMAT_LU" got %"OC_FORMAT_LU")",
                    dimOffset + i, (unsigned long)(elem -> size[i]), (unsigned long)(size[i]));
   }

   /* Check the strides */
   if ((elem -> strides) && (strides))
   {  for (i = 0; i < ndims; i++)
         if (elem -> strides[i] != strides[i])
            OcError(-1, "Mismatch in strides at dimension %d (expected %"OC_FORMAT_LD" got %"OC_FORMAT_LD")",
                    dimOffset + i, (long int)(elem -> strides[i]), (long int)(strides[i]));
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndexElem_setInputSize(OcTensorIndexElem *elem, int ndims,
                                   OcSize *size, OcIndex *strides)
/* -------------------------------------------------------------------- */
{  int result;
   int i;
   
   /* Make sure any existing dimensions are compatible */
   result = OcTensorIndexElem_checkSize(elem, ndims, size, strides, 0);
   if (result != 0)
   {  OcErrorMessage("Internal error: mismatch in new and existing index size");
      return result;
   }

   /* Allocate memory for size information */
   if ((size) && (elem -> size == NULL))
   {  elem -> size = (OcSize *)OcMalloc(sizeof(OcSize) * (ndims > 0 ? ndims : 1));
      if (elem -> size == NULL) OcError(-1, "Insufficient memory to allocate tensor index size");
      for (i = 0; i < ndims; i++) elem -> size[i] = size[i];
   }

   /* Allocate memory for stride information */
   if ((strides) && (elem -> strides == NULL))
   {  elem -> strides = (OcIndex *)OcMalloc(sizeof(OcIndex) * (ndims > 0 ? ndims : 1));
      if (elem -> strides == NULL) OcError(-1, "Insufficient memory to allocate tensor index strides");
      for (i = 0; i < ndims; i++) elem -> strides[i] = strides[i];
   }

   /* Set the number of input dimensions */
   elem -> nInputDims = ndims;

   return 0;
}


/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcTensorIndexElem_createScalar(OcIndex index)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Scalar *elem;

   /* Create a new element */
   elem = (OcTensorIndexElem_Scalar *)OcTensorIndexElem_alloc(OC_IDX_SCALAR);
   if (elem == NULL) return NULL;

   /* Initialize */
   elem -> index = index;

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = 1;
   ((OcTensorIndexElem *)elem) -> nOutputDims = 0;

   return (OcTensorIndexElem *)elem;
}


/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcTensorIndexElem_createAll(int ndims)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_All *elem;

   /* Create a new element */
   elem = (OcTensorIndexElem_All *)OcTensorIndexElem_alloc(OC_IDX_ALL);
   if (elem == NULL) return NULL;

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = ndims;
   ((OcTensorIndexElem *)elem) -> nOutputDims = ndims;

   return (OcTensorIndexElem *)elem;
}


/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcTensorIndexElem_createSteps(OcIndex start, OcIndex step,
                                                 OcSize nelem)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Steps *elem;

   /* Make sure that the step size is non-zero */
   if (step == 0)
      OcError(NULL, "Step size in index cannot be zero");

   /* Create a new element */
   elem = (OcTensorIndexElem_Steps *)OcTensorIndexElem_alloc(OC_IDX_STEPS);
   if (elem == NULL) return NULL;

   /* Initialize */
   elem -> start = start;
   elem -> step  = step;
   elem -> nelem = nelem;

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = 1;
   ((OcTensorIndexElem *)elem) -> nOutputDims = 1;

   return (OcTensorIndexElem *)elem;
}


/* -------------------------------------------------------------------- */
OcTensorIndexElem *OcTensorIndexElem_createOffset(int ndims, OcSize *size,
                                                  OcIndex *strides, OcIndex range,
                                                  OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;

   /* Create a new element */
   elem = OcTensorIndexElem_alloc(OC_IDX_OFFSET);
   if (elem == NULL) return NULL;

   /* Initialize */
   ((OcTensorIndexElem_Offset *)elem) -> tensor = OcIncrefTensor(tensor);
   ((OcTensorIndexElem_Offset *)elem) -> range = range;

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nOutputDims = 1;
   if (OcTensorIndexElem_setInputSize(elem, ndims, size, strides) != 0)
   {  OcDecrefTensorIndexElem(elem);
      return NULL;
   }

   return elem;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addScalar(OcTensorIndex *index, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;

   /* Ensure that the scalar is in range for int64 */
   if (OcScalar_inRange(scalar, OcDTypeInt64) <= 0)
      OcError(-1, "Scalar index out of range");
   if (OcDType_isFloat(scalar -> dtype))
      OcError(-1, "Scalar index must be integer");

   /* Create a new scalar index element */
   elem = OcTensorIndexElem_createScalar((OcIndex)OcScalar_asInt64(scalar));
   if (elem == NULL) return -1;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addTensor(OcTensorIndex *index, OcTensor *tensor)
/* -------------------------------------------------------------------- */
{
   if (OcDType_isBool(tensor -> dtype))
        return OcTensorIndex_addMask(index, tensor);
   else return OcTensorIndex_addIndices(index, tensor);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addMask(OcTensorIndex *index, OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Mask *elem;

   /* Ensure that the tensor is boolean */
   if (!OcDType_isBool(tensor -> dtype))
      OcError(-1, "Index mask must be boolean");

   /* Create a new element */
   elem = (OcTensorIndexElem_Mask *)OcTensorIndexElem_alloc(OC_IDX_MASK);
   if (elem == NULL) return -1;

   /* Initialize */
   elem -> tensor  = OcIncrefTensor(tensor);
   elem -> nnz     = 0;
   elem -> flagNnz = 0;

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = tensor -> ndims;
   ((OcTensorIndexElem *)elem) -> nOutputDims = 1;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, (OcTensorIndexElem *)elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addIndices(OcTensorIndex *index, OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Indices *elem = NULL;
   OcTensor *minIndices = NULL;
   OcTensor *maxIndices = NULL;
   OcScalar  max;
   char     *minData;
   char     *maxData;
   int       axis, status = -1;
   int       i, ndims;

   /* Ensure that the tensor is integer */
   if (!OcDType_isInteger(tensor -> dtype))
      OcError(-1, "Index tensor must be integer (not %s)", OcDType_name(tensor -> dtype));

   /* Make sure that the tensor is two dimensional */
   if (tensor -> ndims > 2)
      OcError(-1, "Index tensor cannot have more than two dimensions");
   if ((tensor -> ndims == 2) && (tensor -> size[0] > OC_TENSOR_MAX_DIMS))
      OcError(-1, "Number of dimensions (%"OC_FORMAT_LU") in the index tensor exceeds the maximum (%d)",
              (unsigned long)(tensor -> size[0]), (int)(OC_TENSOR_MAX_DIMS));

   /* Make sure the index range fits int64 */
   if (tensor -> dtype == OcDTypeUInt64)
   {  if ((OcTensor_maximum(tensor, &max) != 0) || (!OcScalar_inRange(&max, OcDTypeInt64)))
         OcError(-1, "Maximum index value exceeds the indexing data type limit");
   }

   /* Make sure that the tensor is two-dimensional */
   if (tensor -> ndims < 2)
   {  OcSize size[2];
      size[0] = 1;
      size[1] = tensor -> nelem;
      if (OcTensor_reshape(&tensor, 2, size, &tensor) != 0) goto final;
   }
   else
   {  OcIncrefTensor(tensor);
   }

   /* Make sure that the tensor has the correct data type;  */
   /* the ensure data type function creates a new tensor or */
   /* increments the reference count.                       */
   if (OcTensor_ensureDType(&tensor, OcDTypeInt64, NULL) != 0)
   {  OcErrorMessage("Error converting indices to 64-bit integers");
      goto final;
   }

   /* Determine the minimum and maximum elements along axis 1 */
   axis = 1;
   if (OcTensor_axisMinimum(tensor, 1, &axis, 0, &minIndices) != 0) goto final;
   if (OcTensor_axisMaximum(tensor, 1, &axis, 0, &maxIndices) != 0) goto final;
   if (OcTensor_ensureDevice(&minIndices, OcCPU, NULL) != 0) goto final;
   if (OcTensor_ensureDevice(&maxIndices, OcCPU, NULL) != 0) goto final;

   /* Create a new element */
   elem = (OcTensorIndexElem_Indices *)OcTensorIndexElem_alloc(OC_IDX_INDICES);
   if (elem == NULL) goto final;

   /* Initialize */
   elem -> tensor = OcIncrefTensor(tensor);

   /* Copy the minimum and maximum index values */
   ndims = (int)(tensor -> size[0]);
   minData = OcTensor_data(minIndices);
   maxData = OcTensor_data(maxIndices);
   for (i = 0; i < ndims; i++)
   {  elem -> indexMin[i] = *((OcInt64 *)(minData + i * (minIndices -> strides[0])));
      elem -> indexMax[i] = *((OcInt64 *)(maxData + i * (maxIndices -> strides[0])));
   }

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = ndims;
   ((OcTensorIndexElem *)elem) -> nOutputDims = 1;

   /* Add the new index element */
   status = OcTensorIndex_addElement(index, (OcTensorIndexElem *)elem);

final : ;
   OcXDecrefTensor(minIndices);
   OcXDecrefTensor(maxIndices);
   OcXDecrefTensor(tensor);

   return status;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addInsert(OcTensorIndex *index, int ndims)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Insert *elem;

   /* Check the number of dimensions to insert */
   if (ndims < 0) OcError(-1, "Number of dimensions to insert in index cannot be negative");
   if (ndims == 0) return 0; /* Nothing to do */

   /* Create a new element */
   elem = (OcTensorIndexElem_Insert *)OcTensorIndexElem_alloc(OC_IDX_INSERT);
   if (elem == NULL) return -1;

   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = 0;
   ((OcTensorIndexElem *)elem) -> nOutputDims = ndims;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, (OcTensorIndexElem *)elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addAll(OcTensorIndex *index, int ndims)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;

   /* Check the number of dimensions in all */
   if (ndims < 0) OcError(-1, "Number of dimensions in index all cannot be negative");
   if (ndims == 0) return 0; /* Nothing to do */

   /* Create the new index element */
   if ((elem = OcTensorIndexElem_createAll(ndims)) == NULL) return -1;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addEllipsis(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Ellipsis *elem;

   /* Create a new element */
   elem = (OcTensorIndexElem_Ellipsis *)OcTensorIndexElem_alloc(OC_IDX_ELLIPSIS);
   if (elem == NULL) return -1;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, (OcTensorIndexElem *)elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addRange(OcTensorIndex *index, OcScalar *start, OcScalar *stop, OcScalar *step)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Range *elem;
   OcIndex stepValue;

   /* Check for complex numbers */
   if ((start) && OcDType_isComplex(start -> dtype)) OcError(-1, "Start index must be integer");
   if ((step ) && OcDType_isComplex(step  -> dtype)) OcError(-1, "Step size must be integer");
   if ((stop ) && OcDType_isComplex(stop  -> dtype)) OcError(-1, "Stop size must be integer");

   /* Ensure that the parameters are in range for int64 */
   if ((start) && (OcScalar_inRange(start, OcDTypeInt64) <= 0)) OcError(-1, "Start index out of range");
   if ((step ) && (OcScalar_inRange(step, OcDTypeInt64) <= 0))  OcError(-1, "Step size out of range");
   if ((stop ) && (OcScalar_inRange(stop, OcDTypeInt64) <= 0))  OcError(-1, "Stop size out of range");

   /* Make sure that the step size is non-zero */
   stepValue = (step ) ? (OcIndex)OcScalar_asInt64(step)  : 1;
   if (stepValue == 0) OcError(-1, "Step size in index range cannot be zero");

   /* Create a new index element */
   elem = (OcTensorIndexElem_Range *)OcTensorIndexElem_alloc(OC_IDX_RANGE);
   if (elem == NULL) return -1;

   /* Initialize */
   elem -> start = (start) ? (OcIndex)OcScalar_asInt64(start) : 0;
   elem -> stop  = (stop ) ? (OcIndex)OcScalar_asInt64(stop)  : 0;
   elem -> step  = stepValue;
   elem -> flagStart = (start != NULL);
   elem -> flagStop  = (stop  != NULL);


   /* Initialize dimensions */
   ((OcTensorIndexElem *)elem) -> nInputDims  = 1;
   ((OcTensorIndexElem *)elem) -> nOutputDims = 1;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, (OcTensorIndexElem *)elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addSteps(OcTensorIndex *index, OcScalar *start, OcScalar *step, OcScalar *nelem)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   OcIndex _start, _step;
   OcSize  _nelem;

   /* Ensure that the start and step parameters are in range for int64 */
   if (OcScalar_inRange(start, OcDTypeInt64) <= 0) OcError(-1, "Start index out of range");
   if (OcScalar_inRange(step,  OcDTypeInt64) <= 0) OcError(-1, "Step size out of range");
   if (OcScalar_isLTZero(nelem)) OcError(-1, "The number of elements cannot be negative");

   if (OcDType_isFloat(start -> dtype)) OcError(-1, "Start index must be integer");
   if (OcDType_isFloat(step  -> dtype)) OcError(-1, "Step size must be integer");
   if (OcDType_isFloat(nelem -> dtype)) OcError(-1, "Number of elements must be integer");

   /* Initialize */
   _start = (OcIndex)OcScalar_asInt64(start);
   _step  = (OcIndex)OcScalar_asInt64(step);
   _nelem = (OcIndex)OcScalar_asUInt64(nelem);

   /* Create the element */
   elem = OcTensorIndexElem_createSteps(_start, _step, _nelem);
   if (elem == NULL) return -1;

   /* Add the new index element */
   return OcTensorIndex_addElement(index, elem);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_addIndex(OcTensorIndex *index, OcTensorIndex *index2)
/* -------------------------------------------------------------------- */
{  int i, n = index -> n;;

   /* Make sure the elements fit */
   if (OcTensorIndex_ensureCapacity(index, n + index2 -> n) != 0)
   {  return -1;
   }

   /* Add shallow copies of the elements */
   for (i = 0; i < index2 -> n; i++)
      index -> elem[n+i] = OcIncrefTensorIndexElem(index2 -> elem[i]);
   index -> n +=  index2 -> n;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_getNumInputDims(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   int i, n;

   for (i = 0, n = 0; i < index -> n; i++)
   {  elem = index -> elem[i];
      if (elem -> type == OC_IDX_ELLIPSIS) return -1;
      n += elem -> nInputDims;
   }

   return n;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_getNumOutputDims(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   int i,n;

   for (i = 0, n = 0; i < index -> n; i++)
   {  elem = index -> elem[i];
      if (elem -> type == OC_IDX_ELLIPSIS) return -1;
      n += elem -> nOutputDims;
   }

   return n;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_getInputDims(OcTensorIndex *index, OcSize *size, int *ndims)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   int i, j, n;

   /* ------------------------------------------------------ */
   /* Returns the number of input dimensions or a negative   */
   /* number if the input dimensions could not be determined */
   /* ------------------------------------------------------ */

   for (i = 0, n = 0; i < index -> n; i++)
   {  elem = index -> elem[i];
      if ((elem -> nInputDims) == 0) continue;

      /* Special case when index element is a Boolean mask */
      if (elem -> type == OC_IDX_MASK)
      {  OcTensorIndexElem_Mask *mask = (OcTensorIndexElem_Mask *)elem;
         for (j = 0; j < mask -> tensor -> ndims; j++)
         {  size[n] = mask -> tensor -> size[j]; n++;
         }
         continue;
      }

      /* The input size is known only when the size is given */
      /* Ellipsis elements never have a known dimension.     */
      if ((elem -> size == NULL) || (elem -> type == OC_IDX_ELLIPSIS))
      {  *ndims = -1; return 0;  }

      /* Copy the input size */
      for (j = 0; j < elem -> nInputDims; j++)
      {  size[n] = elem -> size[j]; n++;
      }
   }

   *ndims = n;
   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_getInputStrides(OcTensorIndex *index, OcIndex *strides, int *ndims)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   int i, j, n;

   /* ------------------------------------------------------ */
   /* Returns the number of input dimensions or a negative   */
   /* number if the input dimensions could not be determined */
   /* ------------------------------------------------------ */

   for (i = 0, n = 0; i < index -> n; i++)
   {  elem = index -> elem[i];
      if ((elem -> nInputDims) == 0) continue;
      if ((elem -> strides == NULL) || (elem -> type == OC_IDX_ELLIPSIS))
      {  *ndims = -1; return 0;  }

      /* Copy the input size */
      for (j = 0; j < elem -> nInputDims; j++)
      {  strides[n] = elem -> strides[j]; n++;
      }
   }

   *ndims = n;
   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_getOutputDims(OcTensorIndex *index, OcSize *size, int *ndims)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   int i, j, n;

   /* ------------------------------------------------------ */
   /* Returns the number of input dimensions or a negative   */
   /* number if the input dimensions could not be determined */
   /* ------------------------------------------------------ */

   /* Initialize the number of dimensions to -1 */
   *ndims = -1;

   for (i = 0, n = 0; i < index -> n; i++)
   {  elem = index -> elem[i];
      if ((elem -> nOutputDims) == 0) continue;

      switch (elem -> type)
      {  /* ----------------------- */
         case OC_IDX_SCALAR :
         /* ----------------------- */
            break;

         /* ----------------------- */
         case OC_IDX_INDICES :
         /* ----------------------- */
            size[n] = ((OcTensorIndexElem_Indices *)elem) -> tensor -> size[1];
            n++;
            break;

         /* ----------------------- */
         case OC_IDX_MASK :
         /* ----------------------- */
         {  OcTensorIndexElem_Mask *mask = (OcTensorIndexElem_Mask *)elem;

            if (mask -> flagNnz == 0)
            {  OcUInt64 count;
               if (OcTensor_nnz(mask -> tensor, &count) != 0) return -1;
               mask -> nnz     = count;
               mask -> flagNnz = 1;
            }
            size[n] = mask -> nnz;
            n++;
            break;
         }

         /* ----------------------- */
         case OC_IDX_OFFSET :
         /* ----------------------- */
         {  size[n] = ((OcTensorIndexElem_Offset *)elem) -> tensor -> size[0];
            n++;
            break;
         }

         /* ----------------------- */
         case OC_IDX_INSERT :
         /* ----------------------- */
         {  for (j = 0; j < elem -> nOutputDims; j++)
            {  size[n] = 1; n++;
            }
            break;
         }

         /* ----------------------- */
         case OC_IDX_ALL :
         /* ----------------------- */
         {  if (elem -> size == NULL) return 0;
            for (j = 0; j < elem -> nOutputDims; j++)
            {  size[n] = elem -> size[j]; n++;
            }
            break;
         }
            
         /* ----------------------- */
         case OC_IDX_ELLIPSIS :
         /* ----------------------- */
            return -1;

         /* ----------------------- */
         case OC_IDX_RANGE :
         /* ----------------------- */
            /* Even when the range is fully specified it is not    */
            /* possible to determine the number of output elements */
            /* without knowing the size. The reason is that a range*/
            /* such as 2:10 may be truncated when applied to a     */
            /* tensor that has a smaller size.                     */
            if (elem -> size == NULL) return 0;
            size[n] = elem -> size[0]; n++;
            break;

         /* ----------------------- */
         case OC_IDX_STEPS :
         /* ----------------------- */
            size[n] = ((OcTensorIndexElem_Steps *)elem) -> nelem;
            n++;
            break;
      }
   }

   *ndims = n;
   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndexElem_indicesToOffset(OcTensorIndexElem **element, OcSize *size,
                                      OcIndex *strides, OcTensorIndexElem **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Indices *indices = (OcTensorIndexElem_Indices *)(*element);
   OcTensorIndexElem *elem = NULL;
   OcTensor *offsets = NULL;
   OcIndex   range; /* Approximate range in memory for sorting */
   int       ndims = (*element) -> nInputDims;
   int       i, status;

   /* Make sure the element has the correct type */
   if ((*element) -> type != OC_IDX_INDICES)
   {  OcErrorMessage("Internal error: invalid tensor index type");
      goto final;
   }

   /* Determine the offsets */
   status = OcTensor_indexToOffset(indices -> tensor, strides, &offsets);
   if (status != 0) goto final;

   /* Determine the sort stride */
   for (i = 0, range = 0; i < ndims; i++)
   {  if (strides[i] < 0)
           range -= strides[i] * (indices -> indexMax[i]);
      else range += strides[i] * (indices -> indexMax[i]);
   }

   /* Create a new offset tensor index */
   elem = OcTensorIndexElem_createOffset(ndims, size, strides, range, offsets);
   if (elem == NULL) goto final;

final : ;
   /* Free the tensor index element if needed */
   if (result == NULL)
   {  OcDecrefTensorIndexElem(*element);
      result = element;
   }

   /* Free intermediate tensor objects */
   OcXDecrefTensor(offsets);

   /* Set the result */
   *result = elem; /* May be NULL */
   return (elem != NULL) ? 0 : -1;
}


/* -------------------------------------------------------------------- */
int OcTensorIndexElem_maskToIndices(OcTensorIndexElem **element,
                                    OcTensorIndexElem **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Indices *elem = NULL;
   OcTensor *indices = NULL, *mask;
   int i, ndims;
   int status = -1;

   /* Find the non-zero elements */
   mask = ((OcTensorIndexElem_Mask *)(*element)) -> tensor;
   if ((indices = OcTensor_find(mask)) == NULL)
   {  OcErrorMessage("Error converting Boolean index mask to indices");
      goto final;
   }

   /* Create a new element */
   elem = (OcTensorIndexElem_Indices *)OcTensorIndexElem_alloc(OC_IDX_INDICES);
   if (elem == NULL) goto final;

   /* Initialize */
   elem -> tensor = OcIncrefTensor(indices);

   /* Copy the minimum and maximum index values */
   ndims = mask -> ndims;
   for (i = 0; i < ndims; i++)
   {  elem -> indexMin[i] = 0;
      elem -> indexMax[i] = mask -> size[i] - 1;
   }

   /* Set the size */
   ((OcTensorIndexElem *)elem) -> nOutputDims = 1;
   status = OcTensorIndexElem_setInputSize((OcTensorIndexElem *)elem, ndims, mask -> size, NULL);

final : ;
   /* Free the tensor index element if needed */
   if (result == NULL)
   {  OcDecrefTensorIndexElem(*element);
      result = element;
   }

   /* Free intermediate tensors */
   OcXDecrefTensor(indices);

   /* Set the result */
   if ((status != 0) && (elem))
   {  OcDecrefTensorIndexElem((OcTensorIndexElem *)elem);
      elem = NULL;
   }
   *result = (OcTensorIndexElem *)elem;

   return (elem != NULL) ? 0 : -1;
}


/* -------------------------------------------------------------------- */
int OcTensorIndexElem_rangeToSteps(OcTensorIndexElem **element, OcSize n,
                                   OcTensorIndexElem **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem_Range *range = (OcTensorIndexElem_Range *)(*element);
   OcTensorIndexElem *elem = NULL;
   OcIndex start, stop, step, k;
   OcSize nelem;

   /* Determine the step */
   if ((step = range -> step) == 0)
   {  OcErrorMessage("Step size in index range cannot be zero"); goto final; }

   /* Determine the start index */
   if (range -> flagStart)
        start = range -> start;
   else start = (range -> step > 0) ? 0 : n-1;

   /* Determine the stop index */
   if (range -> flagStop)
        stop = range -> stop;
   else stop = (range -> step > 0) ? n : -1;

   /* Deal with negative indices */
   if ((range -> flagStart) && (start < 0)) start += n;
   if ((range -> flagStop ) && (stop  < 0)) stop  += n;

   /* Compute the overlap with the dimension */
   if (step > 0)
   {  if (start < 0)
      {  /* Find smallest k >= 0 such that start + k*step >= 0 */
         k = (-start + step - 1) / step;
         start += k * step;
      }
      if (stop > n) stop = n;
   }
   else
   {  if (start > n)
      {  /* Find smallest k >= 0 such that start + k*step < n */
         k = (start - n + step) / -step;
         start += k * step;
      }
      if (stop < 0) stop = -1;
   }

   /* Compute the number of elements */
   if (step > 0)
   {  nelem = (start >= stop) ? 0 : ((stop - start) + (step - 1)) / step;
   }
   else
   {  nelem = (start <= stop) ? 0 : ((start - stop) - (step + 1)) / -step;
   }

   /* Create a new index element */
   elem = OcTensorIndexElem_createSteps(start, step, nelem);

final : ;
   /* Free the tensor index element if needed */
   if (result == NULL)
   {  OcDecrefTensorIndexElem(*element);
      result = element;
   }

   /* Assign the result */
   *result = elem;

   return (elem != NULL) ? 0 : -1;
}


/* -------------------------------------------------------------------- */
int OcTensorIndexElem_bind(OcTensorIndexElem **element, int dimOffset,
                           OcSize *size, OcIndex *strides, OcTensorIndexElem **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem = NULL;
   int flagSetSize = 0;
   int i, status = -1;

   /* Check existing size */
   if (OcTensorIndexElem_checkSize(*element, (*element)->nInputDims, size, strides, dimOffset) != 0)
      return -1;

   /* Strides cannot be set without size */
   if ((strides != NULL) && (size == NULL))
      OcError(-1, "Tensor index strides cannot be set without giving size");

   /* Check if a new element needs to be added */
   if (((size    == NULL) || ((*element) -> size    != NULL)) &&
       ((strides == NULL) || ((*element) -> strides != NULL)))
   {  if (result) *result = OcIncrefTensorIndexElem(*element);
      return 0;
   }

   /* Update the element */
   switch ((*element) -> type)
   {
      /* ------------------------------ */
      case OC_IDX_SCALAR:
      /* ------------------------------ */
      {  OcIndex index;

         /* Get the scalar index */
         index = ((OcTensorIndexElem_Scalar *)(*element)) -> index;

         /* Check size if needed */
         if ((size) && ((*element) -> size == NULL))
         {  if (index < 0) index += size[0];
            if ((index >= size[0]) || (index < 0))
            {  OcErrorMessage("Scalar tensor index (%"OC_FORMAT_LD") exceeds size (%"OC_FORMAT_LD") at dimension %d",
                              (long int)(((OcTensorIndexElem_Scalar *)(*element)) -> index),
                              size[0], dimOffset);
               goto final;
            }
         }

         /* Create a new scalar index */
         elem = OcTensorIndexElem_createScalar(index);
         flagSetSize = 1;
         break;  /* switch */
      }

      /* ------------------------------ */
      case OC_IDX_INDICES:
      /* ------------------------------ */
      {  OcTensorIndexElem_Indices *idx;

         if ((*element) -> size == NULL)
         {  long int errorIndex;
            int flagCompatible = 1;
            int flagNegative = 0;

            /* Check the dimension ranges */
            idx = (OcTensorIndexElem_Indices *)(*element);
            for (i = 0; i < (*element)->nInputDims; i++)
            {  if (idx -> indexMin[i] < 0)
               {  flagNegative = 1;
                  if ((-(idx -> indexMin[i])) > size[i])
                  {  flagCompatible = 0; errorIndex = idx -> indexMin[i]; break; }
               }
               if (idx -> indexMax[i] >= size[i])
               {  flagCompatible = 0; errorIndex = idx -> indexMax[i]; break; }
            }

            /* Check compatibility */
            if (!flagCompatible)
            {  OcErrorMessage("Tensor index (%"OC_FORMAT_LD") out of range (%"OC_FORMAT_LU") at dimension %d",
                              (long int)errorIndex, (unsigned long)size[i], dimOffset + i);
               goto final;
            }

            /* Create a new element and replace indices if needed */
            elem = OcTensorIndexElem_shallowCopy(*element);
            if (elem == NULL) goto final;

            if (flagNegative)
            {  OcTensor *subtensor;
               OcScalar  scalar;

               /* Detach the indices tensor */
               idx = (OcTensorIndexElem_Indices *)elem;
               if (OcTensor_detach(&(idx -> tensor)) != 0) goto final;

               /* Update indices when the column contains a negative value */
               scalar.dtype = OcDTypeInt64;
               for (i = 0; i < elem -> nInputDims; i++)
               {  if (idx -> indexMin[i] < 0)
                  {  OcScalar_fromInt64(&scalar, (OcInt64)(size[i]));
                     if ((subtensor = OcTensor_slice(idx -> tensor, 0, i, 1)) == NULL) goto final;
                     status = OcTensor_addIfNegative(subtensor, &scalar);
                     OcDecrefTensor(subtensor);
                     if (status == 0) idx -> indexMin[i] = 0; else goto final;
                  }
               }
            }
            flagSetSize = 1;
         }
         else
         {  /* The size was already set */
            elem = OcIncrefTensorIndexElem(*element);
         }

         /* Update the element if we need to set the strides */
         if (strides != NULL)
         {   /* Convert element to offsets */
             if (OcTensorIndexElem_indicesToOffset(&elem, size, strides, NULL) != 0) goto final;
             flagSetSize = 0;
         }

         break; /* switch */
      }

      /* ------------------------------ */
      case OC_IDX_MASK:
      /* ------------------------------ */
      {  OcTensor *mask, *offset;
         OcIndex   range;

         /* Convert the Boolean index mask to indices of offsets */
         mask = ((OcTensorIndexElem_Mask *)(*element)) -> tensor;

         /* Check size if needed */
         if ((*element) -> size == NULL)
         {  int i;
            for (i = 0; i < mask -> ndims; i++)
            {  if (mask -> size[i] != size[i])
               {  OcErrorMessage("Size mismatch at dimension %d (expected %"OC_FORMAT_LU" got %"OC_FORMAT_LU")",
                                 dimOffset + i, (unsigned long)(mask -> size[i]), (unsigned long)(size[i]));
                  goto final;
               }
            }
         }

         if (strides == NULL)
         {  /* Convert mask to indices */
            if (OcTensorIndexElem_maskToIndices(element, &elem) != 0) goto final;
            flagSetSize = 0;
         }
         else
         {  /* Convert mask to offsets */
            if ((offset = OcTensor_maskToOffset(mask, strides)) == NULL)
            {  OcErrorMessage("Error converting Boolean index mask to offsets");
               goto final;
            }

            /* Determine the (approximate) memory range */
            for (i = 0, range = 0; i < mask -> ndims; i++)
            {  if (strides[i] < 0)
                    range -= strides[i] * (mask -> size[i]);
               else range += strides[i] * (mask -> size[i]);
            }

            /* Create an offset element */
            elem = OcTensorIndexElem_createOffset(mask -> ndims, mask -> size, strides, range, offset);
            OcDecrefTensor(offset);
            flagSetSize = 0;
         }
         break; /* switch */
      }

      /* ------------------------------ */
      case OC_IDX_OFFSET:
      case OC_IDX_INSERT:
      case OC_IDX_ELLIPSIS:
      /* ------------------------------ */
      {  elem = OcIncrefTensorIndexElem(*element);
         flagSetSize = 0;
         break; /* switch */
      }

      /* ------------------------------ */
      case OC_IDX_ALL:
      /* ------------------------------ */
      {  /* Create a new all element and set the size */
         elem = OcTensorIndexElem_createAll((*element) -> nInputDims);
         if (elem == NULL) goto final;
         flagSetSize = 1;
         break; /* switch */
      }

      /* ------------------------------ */
      case OC_IDX_RANGE:
      /* ------------------------------ */
      {  status = OcTensorIndexElem_rangeToSteps(element, size[0], &elem);
         if (status != 0) goto final;
         flagSetSize = 1;
         break; /* switch */
      }

      /* ------------------------------ */
      case OC_IDX_STEPS:
      /* ------------------------------ */
      {  OcTensorIndexElem_Steps *steps = (OcTensorIndexElem_Steps *)(*element);
         OcIndex stop = (steps -> start) + (steps -> nelem) * (steps -> step);

         if ((*element) -> size == NULL)
         {  /* Check index range */
            if (((steps -> start) < 0) || (steps -> start >= size[0]))
            {  OcErrorMessage("Invalid start value in index"); goto final;
            }
            if ((stop < 0) || (stop >= size[0]))
            {  OcErrorMessage("Number of elements in index exceeds the tensor dimension"); goto final;
            }
         }
         elem = OcTensorIndexElem_createSteps(steps -> start, steps -> step, steps -> nelem);
         flagSetSize = 1;
         break; /* switch */
      }
   }

   /* Success */
   status = 0;

final : ;
   if ((status == 0) && (flagSetSize))
   {  status = OcTensorIndexElem_setInputSize(elem, (*element)->nInputDims, size, strides);
      if (status != 0) { OcDecrefTensorIndexElem(elem); elem = NULL; }
   }

   /* Assign the index element */
   if (result == NULL)
   {  OcDecrefTensorIndexElem(*element);
      result = element;
   }
   *result = elem;

   return status;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_bind(OcTensorIndex **index, int flagAutoExtend, int ndims,
                       OcSize *size, OcIndex *strides,  OcTensorIndex **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndex *bound;
   int status = -1;
   int dimOffset = 0;
   int idxEllipsis = -1;
   int idxDims;
   int i;

   /* Check for ellipsis elements */
   bound = *index;
   for (i = 0, idxDims = 0; i < bound -> n; i++)
   {  if (bound -> elem[i] -> type == OC_IDX_ELLIPSIS)
      {  if (idxEllipsis != -1)
              OcError(-1, "Multiple ellipsis elements in tensor index");
         else idxEllipsis = i;
      }
      else if (bound -> elem[i] -> type != OC_IDX_INSERT)
      {  idxDims += bound -> elem[i] -> nInputDims;
      }
   }
   if ((idxDims > ndims) || ((idxEllipsis == -1) && (idxDims > ndims)))
      OcError(-1, "Number of index dimensions (%d) exceeds the tensor dimension (%d)", idxDims, ndims);
   if ((idxDims < ndims) && (idxEllipsis == -1) &&
       ((flagAutoExtend == 0) || OcTensorIndex_isBound(bound, 0)))
      OcError(-1, "Mismatch in number of index (%d) and tensor (%d) dimensions", idxDims, ndims);

   /* Make a copy of the index */
   bound = OcTensorIndex_shallowCopy(*index);
   if (bound == NULL) return -1;

   /* Replace or delete the ellipsis element */
   if (idxEllipsis != -1)
   {  if (idxDims == ndims)
      {  OcTensorIndex_deleteElement(bound, idxEllipsis);
      }
      else
      {  OcDecrefTensorIndexElem(bound -> elem[idxEllipsis]);
         bound -> elem[idxEllipsis] = OcTensorIndexElem_createAll(ndims - idxDims);
         if (bound -> elem[idxEllipsis] == NULL) goto final;
      }
   }
   else if (idxDims < ndims)
   {  /* Add implicit all indexing for omitted dimensions */
      if (OcTensorIndex_addAll(bound, ndims - idxDims) < 0) goto final;
   }

   /* Bind each element */
   for (i = 0; i < bound -> n; i++)
   {  ndims = (bound -> elem[i]) -> nInputDims;
      status = OcTensorIndexElem_bind(&(bound -> elem[i]), dimOffset, size, strides, NULL);
      if (status != 0) goto final;
      if (size) size += ndims;
      if (strides) strides += ndims;
      dimOffset += ndims;
   }

   /* Success */
   status = 0;

final : ;
   if (status != 0)
   {  OcDecrefTensorIndex(bound);
      bound = NULL;
   }

   if (result == NULL)
   {  OcDecrefTensorIndex(*index);
      result = index;
   }
   *result = bound; /* Can be NULL on failure */

   return status;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_createView(OcTensorIndex *index, OcTensor *tensor,
                             OcTensorIndexView **viewPtr)
/* -------------------------------------------------------------------- */
{  OcTensorIndexType  type;
   OcTensorIndexView *view;
   OcTensor          *offsets;
   OcIndex            baseOffset;
   OcIndex            strides[OC_TENSOR_MAX_DIMS];
   OcSize             size[OC_TENSOR_MAX_DIMS];
   int                dimInput, dimOutput;
   int                status = -1;
   int                i, j, n;

   /* ---------------------------------------------------- */
   /* The tensor index must be bound with size and strides */
   /* ---------------------------------------------------- */

   /* Check number of output dimensions */
   n = OcTensorIndex_getNumOutputDims(index);
   if (n < 0)
      OcError(-1, "Number of output dimensions could not be resolved");
   if (n > OC_TENSOR_MAX_DIMS)
      OcError(-1, "Number of index dimensions (%d) exceeds the maximum (%d)", n, OC_TENSOR_MAX_DIMS);

   /* Allocate the view structure */
   view = (OcTensorIndexView *)OcMalloc(sizeof(OcTensorIndexView));
   if (view == NULL) OcError(-1, "Error allocating memory for the tensor index view");

   /* Initialize */
   view -> ndims = n;
   view -> view  = NULL;
   view -> offsets = NULL;
   if (n > 0)
   {  view -> offsets = (OcTensor **)OcMalloc(sizeof(OcTensor *) * n);
      if (view -> offsets == NULL) goto final;
      for (i = 0; i < n; i++)
      {  view -> offsets[i] = NULL;
         view -> offsetStrides[i] = 0;
      }
   }

   /* Process the index elements */
   baseOffset = 0;
   dimInput   = 0;
   dimOutput  = 0;
   for (i = 0; i < index -> n; i++)
   {  
      /* Get the index type */
      type = index -> elem[i] -> type;
      switch (type)
      {  /* ------------------------------------------- */
         case OC_IDX_OFFSET :
         /* ------------------------------------------- */
         {  offsets = ((OcTensorIndexElem_Offset *)(index -> elem[i])) -> tensor;
            if (offsets == NULL)
            {  OcErrorMessage("Internal error: empty tensor pointer in index offset");
               goto final;
            }

            if (OcTensor_ensureDevice(&offsets, tensor -> device, &(view -> offsets[dimOutput])) != 0)
            {  OcErrorMessage("Error copying index offsets to device %s", tensor -> device -> name);
               goto final;
            }

            size[dimOutput]    = offsets -> size[0];
            strides[dimOutput] = 0;
            view -> offsetStrides[dimOutput] = ((OcTensorIndexElem_Offset *)(index -> elem[i])) -> range;
            break; /* Case */
         }

         /* ------------------------------------------- */
         case OC_IDX_SCALAR :
         /* ------------------------------------------- */
         {  OcTensorIndexElem_Scalar *elem = (OcTensorIndexElem_Scalar *)(index -> elem[i]);

            baseOffset += tensor -> strides[dimInput] * (elem -> index);
            break; /* Case */
         }

         /* ------------------------------------------- */
         case OC_IDX_ALL    :
         /* ------------------------------------------- */
         {  for (j = 0; j < (index -> elem[i]) -> nInputDims; j++)
            {  size[dimOutput+j]     = tensor -> size[dimInput + j];
               strides[dimOutput+j]  = tensor -> strides[dimInput + j];
            }
            break; /* Case */
         }

         /* ------------------------------------------- */
         case OC_IDX_STEPS  :
         /* ------------------------------------------- */
         {  OcTensorIndexElem_Steps *elem = (OcTensorIndexElem_Steps *)(index -> elem[i]);

            size[dimOutput]     = elem -> nelem;
            strides[dimOutput]  = tensor -> strides[dimInput] * (elem -> step);
            baseOffset += tensor -> strides[dimInput] * (elem -> start);
            break; /* Case */
         }

         /* ------------------------------------------- */
         case OC_IDX_INSERT :
         /* ------------------------------------------- */
         {  for (j = 0; j < index -> elem[i] -> nOutputDims; j++)
            {  size[dimOutput + j]     = 1;
               strides[dimOutput + j]  = 0;
            }
            break; /* Case */
         }

         /* ------------------------------------------- */
         default :
         /* ------------------------------------------- */
            OcErrorMessage("Internal error: unsupported index type");
            goto final;
      }

      dimInput  += index -> elem[i] -> nInputDims;
      dimOutput += index -> elem[i] -> nOutputDims;
   }

   /* Create the view tensor */
   view -> view = OcTensor_createFromStorage(tensor -> storage, n, size, strides,
                                             baseOffset + tensor -> offset, tensor -> dtype);
   if (view -> view == NULL) goto final;


   /* Copy byteswap and read-only flags */
   view -> view -> flags |= tensor -> flags & (OC_TENSOR_BYTESWAPPED | OC_TENSOR_READONLY);

   /* Success */
   status = 0;

final : ;
   /* Return or delete the view */
   if (status == 0) *viewPtr = view;
   else OcTensorIndex_deleteView(view);

   return status;
}

/* -------------------------------------------------------------------- */
void OcTensorIndex_deleteView(OcTensorIndexView *view)
/* -------------------------------------------------------------------- */
{  int i;

   if (view == NULL) return ;

   /* Delete the offset tensors */
   if (view -> offsets)
   {  for (i = 0; i < view -> ndims; i++)
         OcXDecrefTensor(view -> offsets[i]);

      OcFree(view -> offsets);
   }

   /* Delete the view tensor */
   OcXDecrefTensor(view -> view);

   /* Free the structure */
   OcFree(view);
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_detach(OcTensorIndex **index, OcTensorIndex **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem = NULL;
   OcTensorIndex     *idx;
   OcTensor          *tensor, **tensorPtr;
   int                i, status = -1;

   /* Make a copy of the index */
   idx = OcTensorIndex_createWithCapacity((*index) -> n);
   if (idx == NULL) return -1;

   /* Detach each element */
   for (i = 0; i < (*index) -> n; i++)
   {  tensorPtr = NULL;
      switch ((*index) -> elem[i] -> type)
      {
         /* --------------------------------- */
         case OC_IDX_INDICES :
         /* --------------------------------- */
         {  tensor = ((OcTensorIndexElem_Indices *)((*index) -> elem[i])) -> tensor;
            if (!OcTensor_isDetached(tensor, 1))
            {  elem = OcTensorIndexElem_shallowCopy((*index) -> elem[i]);
               tensorPtr = &(((OcTensorIndexElem_Indices *)(elem)) -> tensor);
            }
            break;
         }

         /* --------------------------------- */
         case OC_IDX_MASK :
         /* --------------------------------- */
         {  tensor = ((OcTensorIndexElem_Mask *)((*index) -> elem[i])) -> tensor;
            if (!OcTensor_isDetached(tensor, 1))
            {  elem = OcTensorIndexElem_shallowCopy((*index) -> elem[i]);
               tensorPtr = &(((OcTensorIndexElem_Mask *)(elem)) -> tensor);
            }
            break;
         }

         /* --------------------------------- */
         case OC_IDX_OFFSET :
         /* --------------------------------- */
         {  tensor = ((OcTensorIndexElem_Offset *)((*index) -> elem[i])) -> tensor;
            if (!OcTensor_isDetached(tensor, 1))
            {  elem = OcTensorIndexElem_shallowCopy((*index) -> elem[i]);
               tensorPtr = &(((OcTensorIndexElem_Offset *)(elem)) -> tensor);
            }
            break;
         }

         /* --------------------------------- */
         default : ; /* Empty */
         /* --------------------------------- */
      }

      /* Create a new element or make a shallow copy */
      if (tensorPtr)
      {  if (OcTensor_detach(tensorPtr) != 0)
         {  OcDecrefTensorIndexElem(elem); goto final; }
      }
      else
      {  elem = OcIncrefTensorIndexElem((*index) -> elem[i]);
      }

      /* Add the element */
      if (OcTensorIndex_addElement(idx, elem) < 0) goto final;
   }

   /* Success */
   status = 0;

final : ;
   if (status != 0)
   {  OcDecrefTensorIndex(idx);
      idx = NULL;
   }

   if (result == NULL)
   {  OcDecrefTensorIndex(*index);
      result = index;
   }
   *result = idx; /* Can be NULL on failure */

   return status;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_setDevice(OcTensorIndex **index, OcDevice *device,
                            OcTensorIndex **result)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem = NULL;
   OcTensorIndex     *idx;
   OcTensor          *tensor, **tensorPtr;
   int                i, status = -1;

   /* Make a copy of the index */
   if ((idx = OcTensorIndex_shallowCopy(*index)) == NULL) return -1;

   /* Process all index elements */
   for (i = 0; i < idx -> n; i++)
   {  tensorPtr = NULL;
      switch (idx -> elem[i] -> type)
      {
         /* --------------------------------- */
         case OC_IDX_INDICES :
         /* --------------------------------- */
         {  tensor = ((OcTensorIndexElem_Indices *)(idx -> elem[i])) -> tensor;
            if (tensor -> device != device)
            {  elem = OcTensorIndexElem_shallowCopy(idx -> elem[i]);
               tensorPtr = &(((OcTensorIndexElem_Indices *)(elem)) -> tensor);
            }
            break;
         }

         /* --------------------------------- */
         case OC_IDX_MASK :
         /* --------------------------------- */
         {  tensor = ((OcTensorIndexElem_Mask *)(idx -> elem[i])) -> tensor;
            if (tensor -> device != device)
            {  elem = OcTensorIndexElem_shallowCopy(idx -> elem[i]);
               tensorPtr = &(((OcTensorIndexElem_Mask *)(elem)) -> tensor);
            }
            break;
         }

         /* --------------------------------- */
         case OC_IDX_OFFSET :
         /* --------------------------------- */
         {  tensor = ((OcTensorIndexElem_Offset *)(idx -> elem[i])) -> tensor;
            if (tensor -> device != device)
            {  elem = OcTensorIndexElem_shallowCopy(idx -> elem[i]);
               tensorPtr = &(((OcTensorIndexElem_Offset *)(elem)) -> tensor);
            }
            break;
         }

         /* --------------------------------- */
         default : ; /* Empty */
         /* --------------------------------- */
      }

      /* Update the tensor device if needed */
      if (tensorPtr)
      {  if (OcTensor_ensureDevice(tensorPtr, device, NULL) != 0)
         {  OcDecrefTensorIndexElem(elem); goto final; }

         /* Replace the current element */
         OcDecrefTensorIndexElem(idx -> elem[i]);
         idx -> elem[i] = elem;
      }
   }

   /* Success */
   status = 0;

final : ;
   if (status != 0)
   {  OcDecrefTensorIndex(idx);
      idx = NULL;
   }

   if (result == NULL)
   {  OcDecrefTensorIndex(*index);
      result = index;
   }
   *result = idx; /* Can be NULL on failure */

   return status;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_clear(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcTensorIndexElem *elem;
   int i;

   for (i = 0; i < index -> n; i++)
   {  elem = index -> elem[i];
      if (elem) OcDecrefTensorIndexElem(elem);
      index -> elem[i] = NULL;
   }
   index -> n = 0;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_isScalar(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  int i;

   /* The empty index is not scalar */
   if (index -> n == 0) return 0;

   /* A tensor index is scalar iff all elements are scalar */
   for (i = 0; i < index -> n; i++)
   {  if (index -> elem[i] -> type != OC_IDX_SCALAR) return 0;
   }

   return 1;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_isView(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  OcTensorIndexType type;
   int i;

   /* A tensor index is scalar iff all elements are scalar */
   for (i = 0; i < index -> n; i++)
   {  type = index -> elem[i] -> type;
      if ((type == OC_IDX_OFFSET ) ||
          (type == OC_IDX_MASK   ) ||
          (type == OC_IDX_INDICES)) return 0;          
   }

   return 1;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_isBound(OcTensorIndex *index, int flagStrides)
/* -------------------------------------------------------------------- */
{  OcTensorIndexType type;
   int i, j = 0;

   /* A tensor index is scalar iff all elements are scalar */
   for (i = 0; i < index -> n; i++)
   {  type = index -> elem[i] -> type;
      if (type == OC_IDX_INSERT) continue;
      if (type == OC_IDX_ELLIPSIS) return 0;

      if (index -> elem[i] -> size == NULL) return 0;
      if ((flagStrides) && (index -> elem[i] -> strides == NULL)) return 0;

      /* Increment the number of non-insert dimensions */
      j ++;
   }

   /* Tensors with only None elements are not bound */
   return (j > 0) ? 1 : 0;
}



/* ===================================================================== */
/* Formatting functions                                                  */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorIndex_format(OcTensorIndex *index, char **str,
                         const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  OcTensorIndexType type;
   OcSize  slen;
   char   *s = NULL, *buffer = NULL, *typeStr;
   int     i, j, k, mode;
   int     result = -1;

   /* Format the index */
   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Output the header */
      if (header != NULL)
      {  k = strlen(header)+1; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", header);
      }

      /* Empty */
      if (index -> n == 0)
      {  k = 6; slen += k;
         if (mode == 1) s += snprintf(s, k+1, "empty ");
      }

      /* Index */
      k = 13; slen += k;
      if (mode == 1) s += snprintf(s, k+1, "tensor index ");

      /* Array begin */
      k = 1; slen += k;
      if (mode == 1) s += snprintf(s, k+1, "[");

      /* Index elements */
      for (i = 0; i < index -> n; i++)
      {
         /* Separator */
         if (i > 0)
         {  k = 1; slen += k;
            if (mode == 1) s += snprintf(s, k+1, ",");
         }

         /* Element */
         type = index -> elem[i] -> type;
         typeStr = NULL;
         switch (type)
         {  /* ------------------------------ */
            case OC_IDX_SCALAR :
            /* ------------------------------ */
            {  long int idx = (long int)(((OcTensorIndexElem_Scalar *)(index -> elem[i])) -> index);
               k = OcFormatLongWidth(idx); slen += k;
               if (mode == 1) s += OcFormatLong(s, k, idx);

               break;
            }

            /* ------------------------------ */
            case OC_IDX_INDICES : if (typeStr == NULL) typeStr = "Indices";
            case OC_IDX_MASK    : if (typeStr == NULL) typeStr = "Mask";
            case OC_IDX_OFFSET  : if (typeStr == NULL) typeStr = "Offsets";
            /* ------------------------------ */
            {  long int ndims = (long int)(index -> elem[i] -> nInputDims);

               /* Element type */
               k = strlen(typeStr); slen += k;
               if (mode == 1) s += snprintf(s, k+1, "%s", typeStr);

               /* Number of dimensions */
               k = 1; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "(");
               k = OcFormatLongWidth(ndims); slen += k;
               if (mode == 1) s += OcFormatLong(s, k, ndims);
               k = 2; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "D)");

               break;
            }

            /* ------------------------------ */
            case OC_IDX_INSERT :
            /* ------------------------------ */
            {  for (j = 0; j < index -> elem[i] -> nOutputDims; j++)
               {  if (j > 0)
                  {  k = 1; slen += k;
                     if (mode == 1) s += snprintf(s, k+1, ",");
                  }
                  k = 4; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, "None");
               }
               break;
            }

            /* ------------------------------ */
            case OC_IDX_ALL    :
            /* ------------------------------ */
            {  for (j = 0; j < index -> elem[i] -> nInputDims; j++)
               {  if (j > 0)
                  {  k = 1; slen += k;
                     if (mode == 1) s += snprintf(s, k+1, ",");
                  }
                  k = 1; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, ":");
               }
               break;
            }

            /* ------------------------------ */
            case OC_IDX_ELLIPSIS :
            /* ------------------------------ */
            {  k = 3; slen += 3;
               if (mode == 1) s += snprintf(s, k+1, "...");
               break;
            }

            /* ------------------------------ */
            case OC_IDX_RANGE :
            /* ------------------------------ */
            {  OcTensorIndexElem_Range *elem = (OcTensorIndexElem_Range *)(index -> elem[i]);

               /* Start */
               if (elem -> flagStart)
               {  k = OcFormatLongWidth((long int)(elem -> start)); slen += k;
                  if (mode == 1) s += OcFormatLong(s, k, (long int)(elem -> start));
               }

               /* Separator */
               k = 1; slen += k;
               if (mode == 1) s += snprintf(s, k+1, ":");

               /* Stop */
               if (elem -> flagStop)
               {  k = OcFormatLongWidth((long int)(elem -> stop)); slen += k;
                  if (mode == 1) s += OcFormatLong(s, k, (long int)(elem -> stop));
               }

               /* Separator and step */
               if (elem -> step != 1)
               {  k = 1; slen += k;
                  if (mode == 1) s += snprintf(s, k+1, ":");
                  k = OcFormatLongWidth((long int)(elem -> step)); slen += k;
                  if (mode == 1) s += OcFormatLong(s, k, (long int)(elem -> step));
               }

               break;
            }

            /* ------------------------------ */
            case OC_IDX_STEPS :
            /* ------------------------------ */
            {  OcTensorIndexElem_Steps *elem = (OcTensorIndexElem_Steps *)(index -> elem[i]);

               /* Type */
               k = 6; slen += k;
               if (mode == 1) s += snprintf(s, k+1, "Steps(");

               /* Start */
               k = OcFormatLongWidth((long int)(elem -> start)); slen += k;
               if (mode == 1) s += OcFormatLong(s, k, (long int)(elem -> start));

               /* Separator */
               k = 1; slen += k;
               if (mode == 1) s += snprintf(s, k+1, ",");

               /* Step */
               k = OcFormatLongWidth((long int)(elem -> step)); slen += k;
               if (mode == 1) s += OcFormatLong(s, k, (long int)(elem -> step));

               /* Separator */
               k = 1; slen += k;
               if (mode == 1) s += snprintf(s, k+1, ",");

               /* Number of elements */
               k = OcFormatULongWidth((unsigned long int)(elem -> nelem)); slen += k;
               if (mode == 1) s += OcFormatULong(s, k, (unsigned long int)(elem -> nelem));

               /* Closing bracket */
               k = 1; slen += k;
               if (mode == 1) s += snprintf(s, k+1, ")");

               break;
            }
         } 
      }

      /* Array end */
      k = 1; slen += k;
      if (mode == 1) s += snprintf(s, k+1, "]");

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

final : ;
   /* -------------------------------------------------------- */
   /* Clean up. Note that freeing of the buffer has to be done */
   /* using the regular free function to match its allocation  */
   /* above using malloc.                                      */
   /* -------------------------------------------------------- */
   if ((result != 0) && (buffer != NULL)) { free(buffer); }

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorIndex_display(OcTensorIndex *index)
/* -------------------------------------------------------------------- */
{  char *str = NULL;
   int   result;

   /* Sanity check */
   if (index == NULL)
   {  printf("<tensor-index NULL pointer>\n");
      return 0;
   }

   /* Format and display the index */
   result = OcTensorIndex_format(index, &str, "<", ">");
   if (result == 0)
   {  printf("%s", str);
   }

   /* Deallocate memory */
   if (str) free(str);

   return result;
}

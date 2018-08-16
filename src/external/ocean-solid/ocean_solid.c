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

#include "ocean/external/ocean-solid/ocean_solid.h"
#include "ocean/base/tensor.h"
#include "ocean/base/shape.h"
#include "ocean/base/error.h"


/* -------------------------------------------------------------------- */
int OcSolid_getType(OcDType dtype)
/* -------------------------------------------------------------------- */
{
   switch (dtype)
   {  case OcDTypeBool    : return SD_BOOL;
      case OcDTypeUInt8   : return SD_UINT8;
      case OcDTypeUInt16  : return SD_UINT16;
      case OcDTypeUInt32  : return SD_UINT32;
      case OcDTypeUInt64  : return SD_UINT64;
      case OcDTypeInt8    : return SD_INT8;
      case OcDTypeInt16   : return SD_INT16;
      case OcDTypeInt32   : return SD_INT32;
      case OcDTypeInt64   : return SD_INT64;
      case OcDTypeHalf    : return SD_HALF;
      case OcDTypeFloat   : return SD_FLOAT;
      case OcDTypeDouble  : return SD_DOUBLE;
      case OcDTypeCHalf   : return SD_CHALF;
      case OcDTypeCFloat  : return SD_CFLOAT;
      case OcDTypeCDouble : return SD_CDOUBLE;
      default :
         OcError(-1, "Data type %s is not supported in Solid", OcDType_name(dtype));
   }
}


/* -------------------------------------------------------------------- */
int OcSolid_getScalar(OcScalar *scalar, solid_scalar *value)
/* -------------------------------------------------------------------- */
{
   switch(scalar -> dtype)
   {  /* Real types */
      case OcDTypeBool    : value -> _bool   = scalar -> value.sBool  ; break;
      case OcDTypeUInt8   : value -> _uint8  = scalar -> value.sUInt8 ; break;
      case OcDTypeUInt16  : value -> _uint16 = scalar -> value.sUInt16; break;
      case OcDTypeUInt32  : value -> _uint32 = scalar -> value.sUInt32; break;
      case OcDTypeUInt64  : value -> _uint64 = scalar -> value.sUInt64; break;
      case OcDTypeInt8    : value -> _int8   = scalar -> value.sInt8  ; break;
      case OcDTypeInt16   : value -> _int16  = scalar -> value.sInt16 ; break;
      case OcDTypeInt32   : value -> _int32  = scalar -> value.sInt32 ; break;
      case OcDTypeInt64   : value -> _int64  = scalar -> value.sInt64 ; break;
      case OcDTypeHalf    : value -> _half   = scalar -> value.sHalf  ; break;
      case OcDTypeFloat   : value -> _float  = scalar -> value.sFloat ; break;
      case OcDTypeDouble  : value -> _double = scalar -> value.sDouble; break;

      /* Complex types */
      case OcDTypeCHalf   : value -> _chalf.real   = scalar -> value.sCHalf.real;
                            value -> _chalf.imag   = scalar -> value.sCHalf.imag; break;
      case OcDTypeCFloat  : value -> _cfloat.real  = scalar -> value.sCFloat.real;
                            value -> _cfloat.imag  = scalar -> value.sCFloat.imag; break;
      case OcDTypeCDouble : value -> _cdouble.real = scalar -> value.sCDouble.real;
                            value -> _cdouble.imag = scalar -> value.sCDouble.imag; break;

      default :
         OcError(-1, "Data type %s is not supported in Solid", OcDType_name(scalar -> dtype));
   }

   return 0;
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeElemwise1(OcSolidElemwise1 *config, OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  OcSize   s, size[OC_TENSOR_MAX_DIMS];
   OcIndex  r, strides[OC_TENSOR_MAX_DIMS];
   char    *ptr = OcTensor_data(tensor);
   int      i, j, n = tensor -> ndims;

   /* Deal with empty tensors */
   if (tensor -> nelem == 0)
   {  config -> ndims      = 1;
      config -> ptr        = ptr;
      config -> size[0]    = 0;
      config -> strides[0] = 0;
      return ;
   }

   /* Initialize the size and stride information, remove unitary */
   /* and repeated dimensions, and make sure that the stride     */
   /* values are positive.                                       */
   for (j = 0, i = 0; i < n; i++)
   {  if ((s = tensor -> size[i]) == 1) continue;
      if ((r = tensor -> strides[i]) == 0) continue;
      if (r < 0)
      {  ptr += r * (s-1);
         r *= -1;
      }
      size[j] = s;
      strides[j] = r;
      j ++;
   }
   n = j;

   /* Sort in increasing stride order */
   OcShape_sortStrides(n, size, strides);

   /* Merge dimensions */
   OcShape_mergeBlocks(&n, size, strides);

   /* Finalize and convert size and stride data types */
   config -> ndims = n;
   config -> ptr = ptr;
   for (i = 0; i < n; i++)
   {  config -> size[i] = (solid_size)size[i];
      config -> strides[i] = (solid_index)strides[i];
   }
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeElemwise2(OcSolidElemwise2 *config,
                              OcTensor *tensor1, OcTensor *tensor2)
/* -------------------------------------------------------------------- */
{  OcSize   s, size[OC_TENSOR_MAX_DIMS];
   OcIndex  r1, strides1[OC_TENSOR_MAX_DIMS];
   OcIndex  r2, strides2[OC_TENSOR_MAX_DIMS];
   char    *ptr1 = OcTensor_data(tensor1);
   char    *ptr2 = OcTensor_data(tensor2);
   int      i, j, n = tensor1 -> ndims;

   /* The size of the tensors can be assumed to be the same */

   /* Deal with empty tensors */
   if (tensor1 -> nelem == 0)
   {  config -> ndims       = 1;
      config -> ptr1        = ptr1;
      config -> ptr2        = ptr2;
      config -> size[0]     = 0;
      config -> strides1[0] = 0;
      config -> strides2[0] = 0;
      return ;
   }

   /* Initialize the size and stride information, remove matching */
   /* unitary and repeated dimensions, and ensure that the stride */
   /* values in tensor2 are positive.                             */
   for (j = 0, i = 0; i < n; i++)
   {  if ((s = tensor1 -> size[i]) == 1) continue;

      r1 = tensor1 -> strides[i];
      r2 = tensor2 -> strides[i];
      if ((r1 == 0) && (r2 == 0)) continue;

      if (r2 < 0)
      {  ptr1 += r1 * (s-1);
         ptr2 += r2 * (s-1);
         r2 *= -1;
      }
      size[j] = s;
      strides1[j] = r1;
      strides2[j] = r2;
      j ++;
   }
   n = j;

   /* Sort in increasing stride order */
   OcShape_sortStrides2(n, size, strides2, strides1);

   /* Merge dimensions */
   OcShape_mergeBlocks2(&n, size, strides1, strides2);

   /* Finalize and convert size and stride data types */
   config -> ndims = n;
   config -> ptr1 = ptr1;
   config -> ptr2 = ptr2;
   for (i = 0; i < n; i++)
   {  config -> size[i] = (solid_size)size[i];
      config -> strides1[i] = (solid_index)strides1[i];
      config -> strides2[i] = (solid_index)strides2[i];
   }   
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeElemwise2b(OcSolidElemwise2b *config,
                               OcTensor *tensor1, OcTensor *tensor2)
/* -------------------------------------------------------------------- */
{  OcSize   size1[OC_TENSOR_MAX_DIMS];
   OcSize   size2[OC_TENSOR_MAX_DIMS];
   OcIndex  strides1[OC_TENSOR_MAX_DIMS];
   OcIndex  strides2[OC_TENSOR_MAX_DIMS];
   char    *ptr1 = OcTensor_data(tensor1);
   char    *ptr2 = OcTensor_data(tensor2);
   OcSize   s;
   int      i, n1, n2;
   
   /* Deal with empty tensors */
   if (tensor1 -> nelem == 0)
   {  config -> ndims1      = 1;
      config -> ndims2      = 1;
      config -> ptr1        = ptr1;
      config -> ptr2        = ptr2;
      config -> size1[0]    = 0;
      config -> size2[0]    = 0;
      config -> strides1[0] = 0;
      config -> strides2[0] = 0;
      return ;
   }

   /* Initialize size and stride information for tensor #1 */
   for (n1 = 0, i = 0; i < tensor1 -> ndims; i++)
   {  if ((s = tensor1 -> size[i]) == 1) continue;
      size1[n1] = s;
      strides1[n1] = tensor1 -> strides[i];
      n1 ++;      
   }

   /* Initialize size and stride information for tensor #2 */
   for (n2 = 0, i = 0; i < tensor2 -> ndims; i++)
   {  if ((s = tensor2 -> size[i]) == 1) continue;
      size2[n2] = s;
      strides2[n2] = tensor2 -> strides[i];
      n2 ++;
   }

   /* Merge dimensions */
   OcShape_mergeBlocks(&n1, size1, strides1);
   OcShape_mergeBlocks(&n2, size2, strides2);

   /* Finalize and convert size and stride data types */
   config -> ndims1 = n1;
   config -> ndims2 = n2;
   config -> ptr1 = ptr1;
   config -> ptr2 = ptr2;
   for (i = 0; i < n1; i++)
   {  config -> size1[i] = (solid_size)size1[i];
      config -> strides1[i] = (solid_index)strides1[i];
   }   
   for (i = 0; i < n2; i++)
   {  config -> size2[i] = (solid_size)size2[i];
      config -> strides2[i] = (solid_index)strides2[i];
   }   
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeElemwise3(OcSolidElemwise3 *config, OcTensor *tensor1,
                              OcTensor *tensor2, OcTensor *tensor3)
/* -------------------------------------------------------------------- */
{  OcSize   s, size[OC_TENSOR_MAX_DIMS];
   OcIndex  r1, strides1[OC_TENSOR_MAX_DIMS];
   OcIndex  r2, strides2[OC_TENSOR_MAX_DIMS];
   OcIndex  r3, strides3[OC_TENSOR_MAX_DIMS];
   char    *ptr1 = OcTensor_data(tensor1);
   char    *ptr2 = OcTensor_data(tensor2);
   char    *ptr3 = OcTensor_data(tensor3);
   int      i, j, n = tensor1 -> ndims;

   /* The size of the tensors can be assumed to be the same */

   /* Deal with empty tensors */
   if (tensor1 -> nelem == 0)
   {  config -> ndims       = 1;
      config -> ptr1        = ptr1;
      config -> ptr2        = ptr2;
      config -> ptr3        = ptr3;
      config -> size[0]     = 0;
      config -> strides1[0] = 0;
      config -> strides2[0] = 0;
      config -> strides3[0] = 0;
      return ;
   }

   /* Initialize the size and stride information, remove matching */
   /* unitary and repeated dimensions, and ensure that the stride */
   /* values in tensor3 are positive.                             */
   for (j = 0, i = 0; i < n; i++)
   {  if ((s = tensor1 -> size[i]) == 1) continue;

      r1 = tensor1 -> strides[i];
      r2 = tensor2 -> strides[i];
      r3 = tensor3 -> strides[i];
      if ((r1 == 0) && (r2 == 0) && (r3 == 0)) continue;

      if (r3 < 0)
      {  ptr1 += r1 * (s-1);
         ptr2 += r2 * (s-1);
         ptr3 += r3 * (s-1);
         r3 *= -1;
      }
      size[j] = s;
      strides1[j] = r1;
      strides2[j] = r2;
      strides3[j] = r3;
      j ++;
   }
   n = j;

   /* Sort in increasing stride order */
   OcShape_sortStrides3(n, size, strides3, strides2, strides1);

   /* Merge dimensions */
   OcShape_mergeBlocks3(&n, size, strides1, strides2, strides3);

   /* Finalize and convert size and stride data types */
   config -> ndims = n;
   config -> ptr1 = ptr1;
   config -> ptr2 = ptr2;
   config -> ptr3 = ptr3;
   for (i = 0; i < n; i++)
   {  config -> size[i] = (solid_size)size[i];
      config -> strides1[i] = (solid_index)strides1[i];
      config -> strides2[i] = (solid_index)strides2[i];
      config -> strides3[i] = (solid_index)strides3[i];
   }   
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeReduce(OcSolidReduce *config, OcTensor *tensor, int flagRemoveRepeats)
/* -------------------------------------------------------------------- */
{  OcSize   s, size[OC_TENSOR_MAX_DIMS];
   OcIndex  r, strides[OC_TENSOR_MAX_DIMS];
   char    *ptr = OcTensor_data(tensor);
   int      i, j, n = tensor -> ndims;

   /* Deal with empty tensors */
   if (tensor -> nelem == 0)
   {  config -> ndims      = 1;
      config -> ptr        = ptr;
      config -> size[0]    = 0;
      config -> strides[0] = 0;
      return ;
   }

   /* Initialize the size and stride information, remove unitary */
   /* and repeated dimensions, and make sure that the stride     */
   /* values are positive.                                       */
   for (j = 0, i = 0; i < n; i++)
   {  if ((s = tensor -> size[i]) == 1) continue;
      if (((r = tensor -> strides[i]) == 0) && (flagRemoveRepeats)) continue;
      if (r < 0)
      {  ptr += r * (s-1);
         r *= -1;
      }
      size[j] = s;
      strides[j] = r;
      j ++;
   }
   n = j;

   /* Sort in increasing stride order */
   OcShape_sortStrides(n, size, strides);

   /* Merge dimensions */
   OcShape_mergeBlocks(&n, size, strides);

   /* Finalize and convert size and stride data types */
   config -> ndims = n;
   config -> ptr = ptr;
   for (i = 0; i < n; i++)
   {  config -> size[i] = (solid_size)size[i];
      config -> strides[i] = (solid_index)strides[i];
   }
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeReduceAxis(OcSolidReduceAxis *config,
                               OcTensor *src, OcTensor *dst,
                               int n, int *axes, int flagRemoveRepeats)
/* -------------------------------------------------------------------- */
{  OcSize   size[OC_TENSOR_MAX_DIMS];
   OcSize   rsize[OC_TENSOR_MAX_DIMS];
   OcIndex  strides1[OC_TENSOR_MAX_DIMS];
   OcIndex  strides2[OC_TENSOR_MAX_DIMS];
   OcIndex  rstrides[OC_TENSOR_MAX_DIMS];
   OcIndex  offset1, offset2;
   OcIndex  roffset;
   short    buffer[OC_TENSOR_MAX_DIMS];
   int      i, j, k, keepdims = (src -> ndims == dst -> ndims);

   /* ---------------------------------------------------------- */
   /* The parameters are assumed to be valid, no checks are made */
   /* ---------------------------------------------------------- */

   /* Set the axes indices */
   for (i = 0; i < OC_TENSOR_MAX_DIMS; i++) buffer[i] = 0;
   for (i = 0; i < n; i++) buffer[axes[i]] = 1;

   /* Copy outer dimensions - result may be zero dimensional */
   if (keepdims)
   {  for (i = 0, j = 0; i < src -> ndims; i++)
      {  if ((buffer[i] == 0) && (src -> size[i] != 1))
         {  size[j]     = src -> size[i];
            strides1[j] = src -> strides[i];
            strides2[j] = dst -> strides[i]; j++;
         }
      }
      config -> ndims = j;
   }
   else
   {  for (i = 0, j = 0, k = 0; i < src -> ndims; i++)
      {  if (buffer[i] == 1) continue;
         if (src -> size[i] != 1)
         {  size[j]     = src -> size[i];
            strides1[j] = src -> strides[i];
            strides2[j] = dst -> strides[k]; j++;
         }
         k ++;
      }
      config -> ndims = j;
   }

   /* Copy inner dimensions - result may be zero dimensional */
   for (i = 0, j = 0; i < n; i++)
   {  k = axes[i];
      if ((src -> size[k] != 1) && !(flagRemoveRepeats && (src -> strides[k] == 0)))
      {  rsize[j]    = src -> size[k];
         rstrides[j] = src -> strides[k]; j++;
      }      
   }
   config -> rdims = j;

   /* Sort the outer dimensions on output strides and merge blocks */
   OcShape_normalizeStrides2(config -> ndims, size, strides1, strides2, &offset1, &offset2);
   OcShape_sortStrides2(config -> ndims, size, strides1, strides2);
   OcShape_mergeBlocks2(&(config -> ndims), size, strides1, strides2);

   /* Sort the inner dimensions and merge blocks */
   OcShape_normalizeStrides(config -> rdims, rsize, rstrides, &roffset);
   OcShape_sortStrides(config -> rdims, rsize, rstrides);
   OcShape_mergeBlocks(&(config -> rdims), rsize, rstrides);

   /* Copy the size and stride information */
   for (i = 0; i < config -> ndims; i++)
   {  config -> size[i]     = size[i];
      config -> strides1[i] = strides1[i];
      config -> strides2[i] = strides2[i];
   }
   for (i = 0; i < config -> rdims; i++)
   {  config -> rsize[i]    = rsize[i];
      config -> rstrides[i] = rstrides[i];
   }

   /* Set the pointer values */
   config -> ptr1 = OcTensor_data(src) + offset1 + roffset;
   config -> ptr2 = OcTensor_data(dst) + offset2;
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeIndex1(OcSolidIndex1 *config, OcTensor *view,
                           OcTensor **offsets, OcIndex *offsetStrides)
/* -------------------------------------------------------------------- */
{  solid_index  stride, stride1, stride2;
   solid_size   size;
   solid_int64 *offset;
   char        *ptr;
   int          i, j;

   /* Initialize */
   ptr = (char *)(OcTensor_data(view));
   for (i = 0, j = 0; i < view -> ndims; i++)
   {  /* Check for unitary dimentions*/
      if ((offsets[i] == NULL) && ((view -> size[i]) == 1)) continue;
      config -> size[j] = view -> size[i];

      /* Copy the offset information and determine the stride */
      if (offsets[i])
      {  config -> offsets[j] = (solid_int64 *)(OcTensor_data(offsets[i]));
         stride = offsetStrides[i];
      }
      else
      {  config -> offsets[j] = NULL;
         stride = view -> strides[i];
      }

      /* Ensure positive strides where possible */
      if ((stride < 0) && (config -> offsets[i] == NULL))
      {  ptr += (config -> size[i] - 1) * stride;
         config -> strides[j] = -1 * stride;
      }
      else
      {  config -> strides[j] = stride;
      }

      /* Increase the dimension */
      j ++;
   }
   config -> ptr = ptr;
   config -> ndims = (j < OC_TENSOR_MAX_DIMS) ? j : OC_TENSOR_MAX_DIMS; /* Avoid compile warning */

   /* Sort by strides */
   for (i = 0; i < config -> ndims; i++)
   {  stride1 = config -> strides[i];
      for (j = i+1; j < config -> ndims; j++)
      {  stride2 = config -> strides[j];
         if ((stride2 < stride1) || ((stride2 == stride1) && (config -> offsets[i] == NULL)))
         {  /* Swap the strides */
            config -> strides[j] = stride1;
            config -> strides[i] = stride2;
            stride1 = stride2;

            /* Swap the offsets */
            offset = config -> offsets[i];
            config -> offsets[i] = config -> offsets[j];
            config -> offsets[j] = offset;

            /* Swap the size */
            size = config -> size[i];
            config -> size[i] = config -> size[j];
            config -> size[j] = size;
         }
      }
   }

   /* Merge dimensions with continuous strides */
   for (i = config -> ndims - 1; i > 0; i--)
   {  if (config -> offsets[i]) continue;
      if (config -> offsets[i-1]) continue;

      /* Check if dimensions can be merged */
      if (config -> strides[i] == config -> strides[i-1] * config -> size[i-1])
      {  config -> size[i-1] *= config -> size[i];
         for (j = i; j < view -> ndims - 1; j++)
         {  /* Shift the elements */
            config -> strides[j] = config -> strides[j+1];
            config -> size[j]    = config -> size[j+1];
            config -> offsets[j] = config -> offsets[j+1];
         }
         config -> ndims -= 1;
      }
   }

   /* Check for the zero-dimensional index */
   if (config -> ndims == 0)
   {  config -> ndims      = 1;
      config -> size[0]    = 1;
      config -> strides[0] = 0;
      config -> offsets[0] = NULL;
   }
}


/* -------------------------------------------------------------------- */
void OcSolid_analyzeIndex2(OcSolidIndex2 *config, OcTensor *view,
                           OcTensor **offsets, OcIndex *offsetStrides,
                           int flagSetIndex, OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  solid_index  stride, stride1a, stride1b;
   solid_index *strides1, *strides2;
   solid_size   size;
   solid_int64 *offset;
   char        *ptr1, *ptr2;
   int          i, j;

   /* Initialize */
   for (i = 0, j = 0; i < view -> ndims; i++)
   {  /* Check for unitary dimentions*/
      if ((offsets[i] == NULL) && ((view -> size[i]) == 1)) continue;
      config -> size[j] = view -> size[i];

      /* Copy the offset and stride information */
      if (offsets[i])
      {  config -> offsets[j] = (solid_int64 *)(OcTensor_data(offsets[i]));
         stride = offsetStrides[i];
      }
      else
      {  config -> offsets[j] = NULL;
         stride = view -> strides[i];
      }
      config -> strides1[j] = stride;
      config -> strides2[j] = tensor -> strides[i];

      /* Increase the dimension */
      j ++;
   }
   config -> ndims = j;

   /* Prepare strides and data pointers for sorting */
   if (flagSetIndex)
   {  strides1 = config -> strides1; /* View */
      strides2 = config -> strides2; /* Source tensor */
      ptr1 = (char *)(OcTensor_data(view));
      ptr2 = (char *)(OcTensor_data(tensor));
   }
   else
   {  strides1 = config -> strides2; /* Destination tensor */
      strides2 = config -> strides1; /* View */
      ptr1 = (char *)(OcTensor_data(tensor));
      ptr2 = (char *)(OcTensor_data(view));
   }

   /* Ensure positive strides where possible */
   for (i = 0; i < config -> ndims; i++)
   {  stride = strides1[i];
      if ((stride < 0) && (config -> offsets[i] == NULL))
      {  ptr1 += (config -> size[i] - 1) * stride;
         ptr2 += (config -> size[i] - 1) * strides2[i];
         strides1[i] *= -1;
         strides2[i] *= -1;
      }
   }

   /* Set the data pointers */
   if (flagSetIndex)
   {  config -> ptr1 = ptr1;
      config -> ptr2 = ptr2;
   }
   else
   {  config -> ptr1 = ptr2;
      config -> ptr2 = ptr1;
   }

   /* Sort by strides */
   for (i = 0; i < config -> ndims; i++)
   {  stride1a = strides1[i];
      for (j = i+1; j < config -> ndims; j++)
      {  stride1b = strides1[j];
         if ((stride1b < stride1a) || ((stride1b == stride1a) && (config -> offsets[i] == NULL)))
         {  /* Swap strides1 */
            strides1[j] = stride1a;
            strides1[i] = stride1b;
            stride1a = stride1b;

            /* Swap strides2 */
            stride      = strides2[j];
            strides2[j] = strides2[i];
            strides2[i] = stride;

            /* Swap the offsets */
            offset = config -> offsets[i];
            config -> offsets[i] = config -> offsets[j];
            config -> offsets[j] = offset;

            /* Swap the size */
            size = config -> size[i];
            config -> size[i] = config -> size[j];
            config -> size[j] = size;
         }
      }
   }

   /* Merge dimensions with continuous strides */
   for (i = config -> ndims - 1; i > 0; i--)
   {  if (config -> offsets[i]) continue;
      if (config -> offsets[i-1]) continue;

      /* Check if dimensions can be merged */
      if ((strides1[i] == strides1[i-1] * config -> size[i-1]) &&
          (strides2[i] == strides2[i-1] * config -> size[i-1]))
      {  config -> size[i-1] *= config -> size[i];
         for (j = i; j < view -> ndims - 1; j++)
         {  /* Shift the elements */
            strides1[j] = strides1[j+1];
            strides2[j] = strides2[j+1];
            config -> size[j]    = config -> size[j+1];
            config -> offsets[j] = config -> offsets[j+1];
         }
         config -> ndims -= 1;
      }
   }

   /* Check for the zero-dimensional index */
   if (config -> ndims == 0)
   {  config -> ndims       = 1;
      config -> size[0]     = 1;
      config -> strides1[0] = 0;
      config -> strides2[0] = 0;
      config -> offsets[0] = NULL;
   }
}

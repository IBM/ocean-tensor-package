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

#include "ocean/base/platform.h"
#include "ocean/base/shape.h"
#include "ocean/base/tensor.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <string.h> /* memset */


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct __OcShape_Overlap2State
{  OcIndex min1, max1, size1;    /* Range of memory addresses */
   OcIndex min2, max2, size2;    /* Range of memory addresses */
   OcIndex idxStart1, idxCount1; /* Indices within the current dimension */
   OcIndex idxStart2, idxCount2; /* Indices within the current dimension */
   int     dimIdx1, dimIdx2;     /* Current dimension index */
   struct __OcShape_Overlap2State *next;
} OcShape_Overlap2State;

typedef struct
{  OcSize  size1[OC_TENSOR_MAX_DIMS];     /* Tensor #1 dimensions        */
   OcSize  size2[OC_TENSOR_MAX_DIMS];     /* Tensor #2 dimensions        */
   OcIndex strides1[OC_TENSOR_MAX_DIMS];  /* Tensor #1 strides (> 0)     */
   OcIndex strides2[OC_TENSOR_MAX_DIMS];  /* Tensor #2 strides (> 0)     */
   OcSize  elemsize1, elemsize2;          /* Element sizes               */
   OcIndex offset1, offset2;              /* Offset values               */
   int     ndims1, ndims2;                /* Number of tensor dimensions */

   OcShape_Overlap2State *stack;          /* State stack                 */
   OcShape_Overlap2State *cache;          /* Allocated states            */
} OcShape_Overlap2;



/* ===================================================================== */
/* Function definitions                                                  */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcShape_nelem(int ndims, OcSize *size, OcSize *nelem)
/* --------------------------------------------------------------------- */
{  OcSize  n, d, t;
   int     i;

   /* Determine the number of elements */
   n = 1;
   for (i = 0; i < ndims; i++)
   {  d = size[i];
      if (d == 0) return 0;
      t = n; n *= d;
      if (n / d != t) 
         OcError(-1, "Tensor shape exceeds integer addressing range");
   }

   /* Set the result */
   if (nelem) *nelem = n;

   return 0;
}


/* --------------------------------------------------------------------- */
int OcShape_extent(int ndims, OcSize *size, OcIndex *strides, OcSize elemsize,
                   OcSize *dataOffset, OcSize *dataExtent)
/* --------------------------------------------------------------------- */
{  OcSize  _extent, _x;
   OcIndex _offset = 0;
   OcIndex  s, n;
   OcSize   d;
   int      i, result = 0;

   if (strides != NULL)
   {  _extent = elemsize;
      for (i = 0; i < ndims; i++)
      {
         /* Get the stride and dimension size */
         s = strides[i]; d = size[i];
         if (d == 0) { _offset = 0; _extent = 0; break; }

         if (d > 1)
         {  /* Compute the stride-dimension product */
            d --;
            if (s > 0)
            {  n = ((OcSize)s) * d;
            }
            else
            {  s *= -1; n = ((OcSize)s) * d;
               _offset += n;
            }
            if (n < 0) { result = -1; break; }

            /* Update the extent and check for overflow */
            _x = _extent; _extent += (OcSize)n;
            if (_extent < _x) { result = -1; break; }
         }
      }
   }
   else
   {  _extent = elemsize;
      for (i = 0; i < ndims; i++)
      {  d = size[i]; _x = _extent; _extent *= d;
         if ((d > 0) && (_extent / d != _x)) { result = -1; break; }
      }
   }

   /* Check validity */
   if (result != 0)
      OcError(-1, "Tensor shape exceeds integer addressing range");

   /* Set the return values */
   if (dataOffset != NULL) *dataOffset = _offset;
   if (dataExtent != NULL) *dataExtent = _extent;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcShape_getStrides(int ndims, OcSize *size, int elemsize,
                       OcIndex *strides, char type)
/* -------------------------------------------------------------------- */
{  int i, j;

   if (ndims <= 0) return 0;

   if ((type == 'c') || (type == 'C'))
   {  i = ndims-1;
      strides[i] = elemsize;
      while (i > 0)
      {  strides[i-1] = size[i] * strides[i];
         if ((size[i] > 0) && (strides[i-1] / size[i] != strides[i]))
         {  OcError(-1, "Overflow detected while computing strices");
         }
         i--;
      }
   }
   else if ((type == 'f') || (type == 'F'))
   {  i = 0;
      strides[i] = elemsize;
      while (i < ndims-1)
      {  strides[i+1] = size[i] * strides[i];
         if ((size[i] > 0) && (strides[i+1] / size[i] != strides[i]))
         {  OcError(-1, "Overflow detected while computing strides");
         }
         i++;
      }
   }
   else if ((type == 'r') || (type == 'R'))
   {  /* Similar to Fortran order, but with first two dimensions swapped */
      /* This mode is intended for use in specifying matrices row by row */
      /* followed by transpose to obtain Fortran-style order.            */
      if (ndims == 1)
      {  i = 1;
         strides[0] = elemsize;
      }
      else
      {  i = 2; j = 0;
         strides[0] = elemsize * size[1];
         strides[1] = elemsize;
         if ((size[1] > 0) && (strides[0] / size[1] != elemsize))
            OcError(-1, "Overflow detected while computing strides");
      }      
      while (i < ndims)
      {  strides[i] = size[j] * strides[j];
         if ((size[j] > 0) && (strides[i] / size[j] != strides[j]))
         {  OcError(-1, "Overflow detected while computing strides");
         }
         j = i; i++;
      }
   }
   else
   {  OcError(-1, "Invalid stride type character");
   }
   
   return 0;
}


/* -------------------------------------------------------------------- */
void OcShape_normalizeStrides(int ndims, OcSize *size, OcIndex *strides, OcIndex *offset)
/* -------------------------------------------------------------------- */
{  OcIndex s, o = 0;
   int i;

   /* Reverse all negative strides */
   for (i = 0; i < ndims; i++)
   {  if ((s = strides[i]) < 0)
      {  o += (size[i]-1) * s;
         strides[i] = -s;
      }
   }

   /* Set the offset */
   *offset = o;
}


/* -------------------------------------------------------------------- */
void OcShape_normalizeStrides2(int ndims, OcSize *size, OcIndex *strides1,
                               OcIndex *strides2, OcIndex *offset1, OcIndex *offset2)
/* -------------------------------------------------------------------- */
{  OcIndex s1, s2, o1 = 0, o2 = 0;
   int i;

   /* Reverse all negative strides in strides2 */
   for (i = 0; i < ndims; i++)
   {  if ((s2 = strides2[i]) < 0)
      {  s1 = strides1[i];
         o1 += (size[i]-1) * s1;
         o2 += (size[i]-1) * s2;
         strides1[i] = -s1;
         strides2[i] = -s2;
      }
   }

   /* Set the offsets */
   *offset1 = o1;
   *offset2 = o2;
}


/* -------------------------------------------------------------------- */
void OcShape_sortStrides(int ndims, OcSize *size, OcIndex *strides)
/* -------------------------------------------------------------------- */
{  OcSize  s;
   OcIndex r, rt;
   int     i, j;
   
   /* Sort in increasing stride order. The number of dimensions */
   /* is expected to be small so we sort using insertion sort.  */
   for (i = 1; i < ndims; i++)
   {  s = size[i]; r = strides[i];
      for (j = i; j > 0; j--)
      {  rt = strides[j-1]; if (rt <= r) break;
         strides[j] = rt;
         size[j] = size[j-1];
      }
      size[j]    = s;
      strides[j] = r;
   }
}


/* -------------------------------------------------------------------- */
void OcShape_sortStrides2(int ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2)
/* -------------------------------------------------------------------- */
{  OcSize  s;
   OcIndex r1, r2, rt1, rt2;
   int     i, j;
   
   /* Sort in increasing stride order. The number of dimensions */
   /* is expected to be small so we sort using insertion sort.  */
   for (i = 1; i < ndims; i++)
   {  s  = size[i];
      r1 = strides1[i];
      r2 = strides2[i];
      for (j = i; j > 0; j--)
      {  rt1 = strides1[j-1]; if (rt1 < r1) break;
         rt2 = strides2[j-1]; if ((rt1 == r1) && (rt2 <= r2)) break;
         size[j] = size[j-1];
         strides1[j] = rt1;
         strides2[j] = rt2;
      }
      if (i == j) continue;
      size[j]     = s;
      strides1[j] = r1;
      strides2[j] = r2;
   }
}


/* -------------------------------------------------------------------- */
void OcShape_sortStrides3(int ndims, OcSize *size, OcIndex *strides1,
                          OcIndex *strides2, OcIndex *strides3)
/* -------------------------------------------------------------------- */
{  OcSize  s;
   OcIndex r1, r2, r3, rt1, rt2, rt3;
   int     i, j;
   
   /* Sort in increasing stride order. The number of dimensions */
   /* is expected to be small so we sort using insertion sort.  */
   for (i = 1; i < ndims; i++)
   {  s  = size[i];
      r1 = strides1[i];
      r2 = strides2[i];
      r3 = strides3[i];
      for (j = i; j > 0; j--)
      {  rt1 = strides1[j-1]; if (rt1 < r1) break;
         rt2 = strides2[j-1]; 
         rt3 = strides3[j-1];
         if (rt1 == r1)
         {  if (rt2 <= r2) break;
            if ((rt2 == r2) && (rt3 <= r3)) break;
         }
         size[j] = size[j-1];
         strides1[j] = rt1;
         strides2[j] = rt2;
         strides3[j] = rt3;
      }
      if (i == j) continue;
      size[j]     = s;
      strides1[j] = r1;
      strides2[j] = r2;
      strides3[j] = r3;
   }
}


/* -------------------------------------------------------------------- */
void OcShape_mergeBlocks(int *ndims, OcSize *size, OcIndex *strides)
/* -------------------------------------------------------------------- */
{  int i, j, n = *ndims;

   if (n > 0)
   {  for (j = 0, i = 1; i < n; i++)
      {  if (strides[i] == strides[j] * size[j])
         {  /* Combine */
            size[j] *= size[i];
         }
         else
         {  j ++;
            size[j] = size[i];
            strides[j] = strides[i];
         }
      }
      *ndims = j + 1;
   }
}


/* -------------------------------------------------------------------- */
void OcShape_mergeBlocks2(int *ndims, OcSize *size, OcIndex *strides1, OcIndex *strides2)
/* -------------------------------------------------------------------- */
{  OcSize  s;
   OcIndex r1, r2;
   int     i, j, n = *ndims;

   /* Combine dimensions */
   if (n > 0)
   {  for (j = 0, i = 1; i < n; i++)
      {  s  = size[j];
         r1 = strides1[i];
         r2 = strides2[i];
         if ((r1 == strides1[j] * s) && (r2 == strides2[j] * s))
         {  /* Combine */
            size[j] *= size[i];
         }
         else
         {  j ++;
            size[j] = size[i];
            strides1[j] = r1;
            strides2[j] = r2;
         }
      }
      *ndims = j + 1;
   }
}


/* -------------------------------------------------------------------- */
void OcShape_mergeBlocks3(int *ndims, OcSize *size, OcIndex *strides1,
                          OcIndex *strides2, OcIndex *strides3)
/* -------------------------------------------------------------------- */
{  OcSize  s;
   OcIndex r1, r2, r3;
   int     i, j, n = *ndims;

   /* Combine dimensions */
   if (n > 0)
   {  for (j = 0, i = 1; i < n; i++)
      {  s  = size[j];
         r1 = strides1[i];
         r2 = strides2[i];
         r3 = strides3[i];
         if ((r1 == strides1[j] * s) &&
             (r2 == strides2[j] * s) &&
             (r3 == strides3[j] * s))
         {  /* Combine */
            size[j] *= size[i];
            continue;
         }
         j ++;
         if (j != i)
         {  size[j] = size[i];
            strides1[j] = r1;
            strides2[j] = r2;
            strides3[j] = r3;
         }
      }
      *ndims = j + 1;
   }
}


/* --------------------------------------------------------------------- */
int OcShape_broadcastLeft(int *ndims, OcSize *size, int ndimsRef, OcSize *sizeRef)
/* --------------------------------------------------------------------- */
{  OcSize s1, s2;
   int i,j,k;

   /* Initialize the dimensions */
   i = *ndims;
   j = ndimsRef;

   /* Determine the new number of dimensions */
   *ndims = k = (i >= j) ? i : j;

   /* Merge the data */
   while ((i > 0) && (j > 0))
   {  i--; j--; k--;
      s1 = size[i]; s2 = sizeRef[j];
      if ((s1 == s2) || (s2 == 1))
      {  size[k] = s1;
      }
      else if (s1 == 1)
      {  size[k] = s2;
      }
      else
      {  OcError(-1, "Incompatible sizes in broadcast");
      }
   }
   while (j > 0) { j--; size[j] = sizeRef[j]; }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcShape_broadcastRight(int *ndims, OcSize *size, int ndimsRef, OcSize *sizeRef)
/* --------------------------------------------------------------------- */
{  OcSize s1, s2;
   int i,j,k;

   /* Initialize the dimensions */
   i = *ndims;
   j = ndimsRef;

   /* Determine the new number of dimensions */
   if (i >= j)
        { *ndims = i; k = j; }
   else { *ndims = j; k = i; }

   /* Merge the data */
   for (i = 0; i < k; i ++)
   {  s1 = size[i]; s2 = sizeRef[i];
      if ((s1 == s2) || (s2 == 1))
      {  size[i] = s1;
      }
      else if (s1 == 1)
      {  size[i] = s2;
      }
      else
      {  OcError(-1, "Incompatible sizes in broadcast");
      }
   }
   while (i < j) { size[i] = sizeRef[i]; i++; }

   return 0;
}


/* --------------------------------------------------------------------- */
int OcShape_isValidSize(int ndims, OcSize *size)
/* --------------------------------------------------------------------- */
{  int i;

   /* Check for negative size */
   for (i = 0; i < ndims; i++)
   {  if (!(size[i] >= 0)) return 0;
   }

   return 1;
}


/* --------------------------------------------------------------------- */
int OcShape_isContiguous(int ndims, OcSize *size, OcIndex *strides, int elemsize)
/* --------------------------------------------------------------------- */
{  OcSize  _size[OC_TENSOR_MAX_DIMS];
   OcIndex _strides[OC_TENSOR_MAX_DIMS];
   OcIndex s;
   int     i, j;

   /* Shapes are contiguous if they are linear after appropriately */
   /* permuting the dimensions, removing dimensions with zero      */
   /* strides, and possibly reversing axes.                        */
   if (ndims == 0) return 1;

   /* Copy the strides and take absolute values */
   for (i = 0, j = 0; i < ndims; i++)
   {  if (size[i] == 1) continue;
      if ((s = strides[i]) == 0) continue;
      _size[j] = size[i];
      _strides[j] = (s < 0) ? -1 * s : s;
      j ++;
   }

   /* Sort the strides in increasing order */
   OcShape_sortStrides(j, _size, _strides);

   /* Check if the resulting shape is linear */
   s = elemsize;
   for (i = 0; i < j; i++)
   {  if (_strides[i] != s) return 0;
      s *= _size[i];
   }

   return 1;
}


/* --------------------------------------------------------------------- */
int OcShape_isSelfOverlapping(int ndims, OcSize *size1, OcIndex *strides1, int elemsize)
/* --------------------------------------------------------------------- */
{  OcSize  size[OC_TENSOR_MAX_DIMS], s;
   OcIndex strides[OC_TENSOR_MAX_DIMS], t;
   OcSize  index[OC_TENSOR_MAX_DIMS];
   OcIndex extent;
   char   *data, *ptr;
   int     i, j, result = 0;

   /* ---------------------------------------------------------- */
   /* This function only checks for non-canonical self overlaps. */
   /* That is, it disregards any dimensions with zero strides.   */
   /* ---------------------------------------------------------- */

   /* Copy size and stride information, make sure strides */
   /* are positive and omit singleton dimensions.         */
   for (i = 0, j = 0; i < ndims; i++)
   {  s = size1[i];
      t = strides1[i]; if (t < 0) t *= -1;

      /* Omit singleton dimensions, return when empty */
      if (s <= 0)
      {  if (s == 0) return 0; else continue;
      }

      /* Check for direct overlap of elements */
      if (t < elemsize)
      {  if (t != 0) return 1; else continue;
      }

      /* Copy the information */
      size[j] = s;
      strides[j] = t;
      j ++;
   }
   if ((ndims = j) <= 1) return 0;

   /* Sort the strides and simplify where possible */
   OcShape_sortStrides(ndims, size, strides);
   OcShape_mergeBlocks(&ndims, size, strides);

   /* Remove trailing dimensions with strides that  */
   /* exceed the extent of all previous dimensions. */
   extent = elemsize; j = -1; /* Index of last violation */
   for (i = 0; i < ndims; i++)
   {  if (strides[i] < extent) j = i;
      extent += (size[i] - 1) * strides[i];
   }
   if ((ndims = j+1) <= 1) return 0;

   /* Reduce (halve) the elemsize as much as possible */
   while (elemsize % 2 == 0)
   {  for (i = 0; i < ndims; i++)
      {  if (strides[i] % 2 != 0) break;
      }
      if (i != ndims) break;

      /* Halve elemsize and strides */
      for (i = 0; i < ndims; i++)
      {  strides[i] /= 2;
      }
      elemsize /= 2;
   }

   /* ---------------------------------------------------- */
   /* Working solution: create array and mark all elements */
   /* ---------------------------------------------------- */

   /* Allocate and initialize the array */
   data = (char *)OcMalloc(sizeof(char) * extent);
   if (data == NULL) OcError(-1, "Unable to verify self-overlap");
   memset(data, 0, extent);

   /* Initialize the indices and include correction in the strides */
   for (i = 0; i < ndims; i++) index[i] = 0;
   for (i = ndims-1; i > 0; i--) strides[i] -= size[i-1] * strides[i-1];

   /* Initialize the offset */
   ptr = data; j = 0; s = size[0]; t = strides[0];
   if (elemsize > 1) t -= elemsize;
   while (j < ndims)
   {  /* Loop over the first dimension */
      if (elemsize == 1)
      {  for (i = 0; i < s; i++)
         {  if (*ptr) { result = 1; goto final; }
            *ptr = 1; ptr += t;
         }
      }
      else
      {  for (i = 0; i < s; i++)
         {  for (j = 0; j < elemsize; j++)
            {  if (*ptr) { result = 1; goto final; }
               *ptr = 1; ptr ++;
            }
            ptr += t;
         }
      }

      /* Continue to the next index */
      j = 1;
      while (j < ndims)
      {  ptr += strides[j];
         if ((++index[j]) < size[j]) break;
         index[j] = 0; j++;
      }
   }

final : ;
   if (data) OcFree(data);
   return result;
}


/* --------------------------------------------------------------------- */
int OcShapes_match(int ndims1, OcSize *size1, int ndims2, OcSize *size2)
/* --------------------------------------------------------------------- */
{  int i;

   if (ndims1 != ndims2) return 0;

   for (i = 0; i < ndims1; i++)
   {  if (size1[i] != size2[i]) return 0;
   }

   return 1;
}


/* -------------------------------------------------------------------- */
void OcShape_display(int ndims, OcSize *size, OcIndex *strides)
/* -------------------------------------------------------------------- */
{  int i;

   if (size)
   {  printf("Size = [");
      for (i = 0; i < ndims; i++)
      {  if (i != 0) printf(", ");
         printf("%"OC_FORMAT_LU, (long unsigned)(size[i]));
      }
      printf("%s", (strides) ? "], ": "]\n");
   }

   if (strides)
   {  printf("Strides = [");
      for (i = 0; i < ndims; i++)
      {  if (i != 0) printf(", ");
         printf("%"OC_FORMAT_LD, (long)(strides[i]));
      }
      printf("]\n");
   }
}


/* ===================================================================== */
/* Function definitions - overlap                                        */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
OcShape_Overlap2State *OcShapes_overlapCreateState(OcShape_Overlap2 *info)
/* --------------------------------------------------------------------- */
{  OcShape_Overlap2State *state;

   /* Reuse an existing state */
   if ((info != NULL) && ((state = info -> cache) != NULL))
   {  info -> cache = state -> next;
      return state;
   }

   /* Allocate a new state */
   state = (OcShape_Overlap2State *)OcMalloc(sizeof(OcShape_Overlap2State));
   if (state == NULL) OcError(NULL, "Error allocating state for overlap detection");

   return state;
}


/* --------------------------------------------------------------------- */
int OcShapes_overlapProcessState(OcShape_Overlap2 *info, int *overlap)
/* --------------------------------------------------------------------- */
{  OcShape_Overlap2State *state1, *state2;
   int flagSplit1, flagSplit2;
   
   /* Pop an item from the stack */
   if ((state1 = info -> stack) != NULL)
   {  info -> stack = state1 -> next;
   }
   else
   {  *overlap = 0; return 0;
   }

   /* Check if we can split the state - by prefiltering only the last  */
   /* dimension can have unit size; when a unit size is reached during */
   /* splitting we move to the next dimension.                         */
   flagSplit1 = (state1 -> idxCount1 > 1);
   flagSplit2 = (state1 -> idxCount2 > 1);
   if ((!flagSplit1) && (!flagSplit2))
   {  /* Recycle state1 */
      state1 -> next = info -> cache;
      info -> cache = state1;

      /* Overlap detected */
      *overlap = 1;

      return 0;
   }

   /* Create a new state */
   state2 = OcShapes_overlapCreateState(info);
   if (state2 == NULL) { *overlap = -1; return 0; }

   /* Split the state */
   if ((flagSplit1) && ((state1 -> size1 >= state1 -> size2) || (!flagSplit2)))
   {  
      /* Update the counts and ranges */
      state2 -> idxCount1 = (state1 -> idxCount1) / 2;
      state1 -> idxCount1-= state2 -> idxCount1;
      state2 -> min1 = state1 -> min1 + (info -> strides1[state1 -> dimIdx1] * (state1 -> idxCount1));
      state2 -> max1 = state1 -> max1;
      state1 -> max1-= info -> strides1[state1 -> dimIdx1] * (state2 -> idxCount1);

      /* Check if we should add state 2 */
      if ((state2 -> max1 < state1 -> min2) || (state2 -> min1 > state1 -> max2))
      {  /* Recycle to cache */
         state2 -> next = info -> cache;
         info -> cache = state2;
      }
      else
      {  /* Increase the dimension index if needed */
         state2 -> dimIdx1 = state1 -> dimIdx1;
         state2 -> size1 = (state2 -> max1) - (state2 -> min1);
         if ((state2 -> idxCount1 == 1) && ((state2 -> dimIdx1 + 1) < (info -> ndims1)))
         {  state2 -> dimIdx1  +=1;
            state2 -> idxStart1 = 0;
            state2 -> idxCount1 = info -> size1[state2 -> dimIdx1];
         }
         else
         {  state2 -> idxStart1 = state1 -> idxStart1 + state1 -> idxCount1;
         }

         /* Copy fields for tensor #2 */
         state2 -> dimIdx2   = state1 -> dimIdx2;
         state2 -> idxStart2 = state1 -> idxStart2;
         state2 -> idxCount2 = state1 -> idxCount2;
         state2 -> min2      = state1 -> min2;
         state2 -> max2      = state1 -> max2;
         state2 -> size2     = state1 -> size2;

         /* Add state2 to the stack */
         state2 -> next = info -> stack;
         info -> stack = state2;
      }

      /* Check if we should add state 1 */
      if ((state1 -> max1 < state1 -> min2) || (state1 -> min1 > state1 -> max2))
      {  state1 -> next = info -> cache;
         info -> cache = state1;
      }
      else
      {  /* Increase the dimension index if needed */
         state1 -> size1 = (state1 -> max1) - (state1 -> min1);
         if ((state1 -> idxCount1 == 1) && ((state1 -> dimIdx1 + 1) < info -> ndims1))
         {  state1 -> dimIdx1  += 1;
            state1 -> idxStart1 = 0;
            state1 -> idxCount1 = info -> size1[state1 -> dimIdx1];
         }

         /* Add state 1 to the stack */
         state1 -> next = info -> stack;
         info -> stack = state1;
      }
   }
   else
   {  /* Update the counts and ranges */
      state2 -> idxCount2 = (state1 -> idxCount2) / 2;
      state1 -> idxCount2-= state2 -> idxCount2;
      state2 -> min2 = state1 -> min2 + (info -> strides2[state1 -> dimIdx2] * (state1 -> idxCount2));
      state2 -> max2 = state1 -> max2;
      state1 -> max2-= info -> strides2[state1 -> dimIdx2] * (state2 -> idxCount2);

      /* Check if we should add state 2 */
      if ((state2 -> max2 < state1 -> min1) || (state2 -> min2 > state1 -> max1))
      {  state2 -> next = info -> cache;
         info -> cache = state2;
      }
      else
      {  /* Increase the dimension index if needed */
         state2 -> dimIdx2 = state1 -> dimIdx2;
         state2 -> size2   = (state2 -> max2) - (state2 -> min2);
         if ((state2 -> idxCount2 == 1) && ((state2 -> dimIdx2 + 1) < info -> ndims2))
         {  state2 -> dimIdx2  += 1;
            state2 -> idxStart2 = 0;
            state2 -> idxCount2 = info -> size2[state2 -> dimIdx2];
         }
         else
         {  state2 -> idxStart2 = state1 -> idxStart2 + state1 -> idxCount2;
         }

         /* Copy fields for tensor #1 */
         state2 -> dimIdx1   = state1 -> dimIdx1;
         state2 -> idxCount1 = state1 -> idxCount1;
         state2 -> idxStart1 = state1 -> idxStart1;
         state2 -> min1      = state1 -> min1;
         state2 -> max1      = state1 -> max1;
         state2 -> size1     = state1 -> size1;

         /* Add state2 to the stack */
         state2 -> next = info -> stack;
         info -> stack = state2;
      }

      /* Check if we should add state 1 */
      if ((state1 -> max1 < state1 -> min2) || (state1 -> min1 > state1 -> max2))
      {  state1 -> next = info -> cache;
         info -> cache = state1;
      }
      else
      {  /* Increase the dimension index if needed */
         state1 -> size2 = (state1 -> max2) - (state1 -> min2);
         if ((state1 -> idxCount2 == 1) && ((state1 -> dimIdx2 + 1) < info -> ndims2))
         {  state1 -> dimIdx2  += 1;
            state1 -> idxStart2 = 0;
            state1 -> idxCount2 = info -> size2[state1 -> dimIdx2];
         }

         /* Add state 1 to the stack */
         state1 -> next = info -> stack;
         info -> stack = state1;
      }
   }

   return 1;
}


/* --------------------------------------------------------------------- */
int OcShapes_overlapInitialize(OcShape_Overlap2 *info, int *overlap)
/* --------------------------------------------------------------------- */
{  OcShape_Overlap2State *state;
   int i;

   /* Initialize the state pointers */
   info -> stack = NULL;
   info -> cache = NULL;

   /* Check for empty dimensions */
   if ((info -> ndims1 == 0) || (info -> ndims2 == 0)) { *overlap = 0;  return 1;  }

   /* Check if the offsets match */
   if (info -> offset1 == info -> offset2) { *overlap = 1; return 1; }

   /* Create the initial state */
   if ((state = OcShapes_overlapCreateState(info)) == NULL) { *overlap = -1; return 1; }

   /* Initialize the state */
   state -> dimIdx1   = 0;
   state -> dimIdx2   = 0;
   state -> idxStart1 = 0; 
   state -> idxStart2 = 0;
   state -> idxCount1 = info -> size1[0];
   state -> idxCount2 = info -> size2[0];
   state -> min1      = info -> offset1;
   state -> min2      = info -> offset2;

   /* Determine the state ranges */
   state -> max1 = state -> min1 + info -> elemsize1 - 1;
   for (i = 0; i < info -> ndims1; i++)
   {  state -> max1 += info -> strides1[i] * (info -> size1[i] - 1);
   }
   state -> max2 = state -> min2 + info -> elemsize2 - 1;
   for (i = 0; i < info -> ndims2; i++)
   {  state -> max2 += info -> strides2[i] * (info -> size2[i] - 1);
   }

   /* Determine the size - we omit the additional one since */
   /* the size values are used only to decide which of the  */
   /* two dimensions to split.                              */
   state -> size1 = (state -> max1) - (state -> min1);
   state -> size2 = (state -> max2) - (state -> min2);

   /* Check whether the ranges overlap */
   if ((state -> max1 < state -> min2) || (state -> min1 > state -> max2))
   {  OcFree(state);
      *overlap = 0;
      return 1;
   }

   /* Initialize the stack */
   state -> next = NULL;
   info -> stack = state;

   return 0;
}


/* --------------------------------------------------------------------- */
void OcShapes_overlapFinalize(OcShape_Overlap2 *info)
/* --------------------------------------------------------------------- */
{  OcShape_Overlap2State *state;

   /* Empty the stack */
   while ((state = info -> stack) != NULL)
   {  info -> stack = state -> next;
      OcFree(state);
   }
   
   /* Empty the cache */
   while ((state = info -> cache) != NULL)
   {  info -> cache = state -> next;
      OcFree(state);
   }
}


/* --------------------------------------------------------------------- */
int OcShapes_overlapPrepare(int ndimsSrc, OcSize *sizeSrc, OcIndex *stridesSrc,
                            OcIndex offsetSrc, int elemsizeSrc,
                            int *ndimsDst, OcSize *sizeDst, OcIndex *stridesDst,
                            OcIndex *offsetDst, OcSize *elemsizeDst)
/* --------------------------------------------------------------------- */
{  OcIndex s;
   OcSize  r;
   int i, j, n;
  
   /* Initialize */
   *offsetDst   = offsetSrc;
   *elemsizeDst = elemsizeSrc;

   /* Normalize strides - remove dimensions with zero   */
   /* strides or unit size and convert negative strides */
   for (i = 0, n = 0; i < ndimsSrc; i++)
   {  s = stridesSrc[i];
      r = sizeSrc[i];
      if ((s == 0) || (r == 1)) continue;
      if (r == 0) { *ndimsDst = 0; return 0; }
      if (s < 0)
      {  *offsetDst += (sizeSrc[i]-1) * s;
         s *= -1;
      }
      sizeDst[n] = r;
      stridesDst[n] = s;
      n++;
   }

   /* Simplify */
   if (n > 0)
   {  /* Sort the strides and simplify where possible */
      OcShape_sortStrides(n, sizeDst, stridesDst);
      OcShape_mergeBlocks(&n, sizeDst, stridesDst);
   
      /* Reverse order - sort ascending */
      for (i = 0, j = n-1; i < j; i++, j--)
      {  s = stridesDst[i]; stridesDst[i] = stridesDst[j]; stridesDst[j] = s;
         r = sizeDst[i]; sizeDst[i] = sizeDst[j]; sizeDst[j] = r;
      }

      /* Merge last dimension into elemsize if possible */
      if (stridesDst[n-1] == elemsizeSrc)
      {  *elemsizeDst *= sizeDst[n-1];
         n--;
      }
   }   

   /* Check for scalars */
   if (n == 0)
   {  sizeDst[0] = 1;
      stridesDst[0] = 0;
      n = 1;
   }

   /* Set the dimension */
   *ndimsDst = n;

   return 0;
}


/* --------------------------------------------------------------------- */
int OcShapes_overlap(int ndims1, OcSize *size1, OcIndex *strides1, OcIndex offset1, int elemsize1,
                     int ndims2, OcSize *size2, OcIndex *strides2, OcIndex offset2, int elemsize2)
/* --------------------------------------------------------------------- */
{  OcShape_Overlap2 info;
   int overlap;

   /* Normalize the strides and dimensions */
   OcShapes_overlapPrepare(ndims1, size1, strides1, offset1, elemsize1,
                           &(info.ndims1), info.size1, info.strides1, &(info.offset1), &(info.elemsize1));
   if ((info.ndims1 == 0) || (info.elemsize1 == 0)) return 0; /* Empty tensor does not overlap */

   OcShapes_overlapPrepare(ndims2, size2, strides2, offset2, elemsize2,
                           &(info.ndims2), info.size2, info.strides2, &(info.offset2), &(info.elemsize2));
   if ((info.ndims2 == 0) || (info.elemsize2 == 0)) return 0; /* Empty tensor does not overlap */


   /* Initialize the search */
   if (OcShapes_overlapInitialize(&info, &overlap) != 0) return overlap;

   /* Process the states */
   while (OcShapes_overlapProcessState(&info, &overlap)) ;

   /* Finalize */
   OcShapes_overlapFinalize(&info);

   return overlap;
}

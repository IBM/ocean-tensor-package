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

#ifndef SD_TEMPLATE_FILE
#define SD_TEMPLATE_FILE "core/cpu/template_index.c"

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/core/cpu/apply_elemwise1.h"
#include "solid/core/cpu/apply_elemwise2.h"
#include "solid/core/cpu/index.h"
#include "solid/base/cpu/solid_omp.h"

#include <stdlib.h>


/* ============================================================================== */
/* Structure definitions                                                          */
/* ============================================================================== */

typedef struct
{  int        ndims;
   size_t     nelem;
   size_t     size[SOLID_MAX_TENSOR_DIMS];
   ptrdiff_t  strides[SOLID_MAX_TENSOR_DIMS];
   void      *ptr;
   int        nthreads;
   size_t    *nnz;      /* Number of output elements per thread */
} solid_find_cpu_t;


/* ============================================================================== */
/* Function definition - Find prepare and finalize                                */
/* ============================================================================== */

/* Forward declaration */
void solid_find_finalize_cpu(void *data);

/* -------------------------------------------------------------------- */
int solid_find_allocate_cpu(int ndims, const size_t *size, const ptrdiff_t *strides,
                            void *ptr, void **ptr_data)
/* -------------------------------------------------------------------- */
{  solid_find_cpu_t *data = NULL;
   int i, nthreads;

   /* Check parameters */
   if (ndims > SOLID_MAX_TENSOR_DIMS)
      SOLID_ERROR(-1, "Number of dimensions exceeds the limit");

   /* Determine the number of threads */
   nthreads = solid_omp_get_max_threads();
   if (nthreads < 1) nthreads = 1;

   /* Allocate the internal data structure */
   if ((data = (solid_find_cpu_t *)malloc(sizeof(solid_find_cpu_t))) == NULL) goto error;
   data -> ndims    = ndims;
   data -> nelem    = 1;
   data -> ptr      = ptr;
   data -> nthreads = nthreads;
   data -> nnz      = (size_t *)malloc(sizeof(size_t) * nthreads);
   if (data -> nnz == NULL) goto error;

   /* Copy the size and strides */
   if (ndims == 0)
   {  data -> ndims      = 1;
      data -> size[0]    = 1;
      data -> strides[0] = 0;
   }
   else
   {  for (i = 0; i < ndims; i++)
      {  data -> size[i] = size[i];
         data -> strides[i] = strides[i];
         data -> nelem *= size[i];
      }
   }

   *ptr_data = (void *)data;
   return 0;

error : ;
   solid_find_finalize_cpu(data);
   *ptr_data = NULL;
   SOLID_ERROR(-1, "Error preparing for the find operation");
}


/* -------------------------------------------------------------------- */
int solid_find_initialize_cpu(void *data)
/* -------------------------------------------------------------------- */
{  solid_find_cpu_t *info = (solid_find_cpu_t *)data;

   /* Deal with empty tensors */
   if (info -> nelem == 0)
   {  int i;
      for (i = 0; i < info -> nthreads; i++) info -> nnz[i] = 0;
      return 0;
   }

   /* -------------------------------------------------- */
   /* Single-threaded implementation                     */
   /* -------------------------------------------------- */
   if (info -> nthreads == 1)
   {  size_t      idx[SOLID_MAX_TENSOR_DIMS];
      ptrdiff_t   strides[SOLID_MAX_TENSOR_DIMS];
      size_t      size0, strides0, nelem, index, nnz;
      solid_bool *ptr;
      int         i, ndims;

      ndims = info -> ndims;
      nelem = info -> nelem;
      size0 = info -> size[0];
      strides0 = info -> strides[0];
      for (i = 1; i < ndims; i++)
      {  idx[i] = 0;
         strides[i] = info -> strides[i] - info -> size[i-1] * info -> strides[i-1];
      }

      /* This code requires solid_bool to be one byte */
      ptr = (solid_bool *)(info -> ptr);
      index = 0; nnz = 0;
      while (1)
      {  /* Loop over the first dimension */
         for (i = 0; i < size0; i++)
         {  if (*ptr) nnz ++;
            ptr += strides0;
         }

         /* Check for completion */
         index += size0; if (index >= nelem) break;

         /* Advance the indices */
         i = 1; ptr += strides[i];
         while (++idx[i] == info -> size[i])
         {  idx[i] = 0; i++;
            ptr += strides[i];
         }
      }

      /* Assign the result */
      info -> nnz[0] = nnz;
   }
#if SOLID_ENABLE_OMP
   else
   /* -------------------------------------------------- */
   /* Multi-threaded implementation                      */
   /* -------------------------------------------------- */
   {
      _Pragma("omp parallel num_threads(info -> nthreads)")
      {  int         rank = omp_get_thread_num(); \
         int         nthreads = omp_get_num_threads(); \
         size_t      idx[SOLID_MAX_TENSOR_DIMS];
         size_t      size[SOLID_MAX_TENSOR_DIMS];
         ptrdiff_t   strides[SOLID_MAX_TENSOR_DIMS];
         size_t      offset, index, indexMax, indexStop;
         size_t      size0, strides0, nelem, nnz;
         solid_bool *ptr;
         int         i, ndims;

         /* Initialize */
         ndims = info -> ndims;
         nelem = info -> nelem;
         size0 = info -> size[0];
         strides0 = info -> strides[0];

         /* Compute the offset */
         offset = nelem % nthreads; nelem /= nthreads;
         if (rank < offset)
         {  nelem ++;
            offset = rank * nelem;
         }
         else
         {  offset += rank * nelem;
         }
         index  = offset;
         indexMax = offset + nelem;

         /* Initialize the data pointer and indices */
         ptr = (solid_bool *)(info -> ptr);
         for (i = 0; i < ndims; i++)
         {  size[i] = info -> size[i];
            idx[i] = offset % size[i];
            offset /= size[i];
            ptr += info -> strides[i] * idx[i];
         }
      
         /* Set the strides */
         for (i = 1; i < ndims; i++)
         {  strides[i] = (info -> strides[i]) - size[i-1] * (info -> strides[i-1]);
         }

         /* Loop over the data */ \
         indexStop = index + size0 - idx[0]; nnz = 0;
         if (index < indexMax)
         {  while (1)
            {
               if (indexStop > indexMax) indexStop = indexMax;
               while (index < indexStop)
               {  if (*ptr) nnz++;
                  index ++; ptr += strides0;
               }
               if (index >= indexMax) break;

               i = 1; indexStop = index + size0;
               ptr += strides[i];
               while (++idx[i] == size[i])
               {  idx[i] = 0; i++;
                  ptr += strides[i];
               }
            }
         }

         /* Assign the result */
         info -> nnz[rank] = nnz;
      }
   }
#endif /* SOLID_ENABLE_OMP */

   return 0;
}


/* -------------------------------------------------------------------- */
void solid_find_finalize_cpu(void *data)
/* -------------------------------------------------------------------- */
{  solid_find_cpu_t *info = (solid_find_cpu_t *)data;

   if (info != NULL)
   {  if (info -> nnz) free(info -> nnz);
      free(info);
   }
}


/* ============================================================================== */
/* Generate all data types                                                        */
/* ============================================================================== */
#include "solid/base/generic/generate_all_types.h"
#else


#if (SDTYPE_IS_SIGNED_INT(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(add_if_negative)(int ndims, const size_t *size,
                                    const ptrdiff_t *strides, void *ptr,
                                    solid_scalar scalar)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) value = SOLID_SCALAR_C_VALUE(scalar);

   SOLID_APPLY_ELEMWISE1({ if (*_ptr < 0) *_ptr += value; })

   return 0;
}
#endif



#if (SDTYPE_IS_INT(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(index_to_offset)(int nstrides, solid_int64 *strides,
                                     ptrdiff_t strideReduce, size_t nelem,
                                     ptrdiff_t stride1, void *ptr1,
                                     ptrdiff_t stride2, void *ptr2)
/* -------------------------------------------------------------------- */
{  ptrdiff_t *strides1 = &stride1;
   ptrdiff_t *strides2 = &stride2;
   size_t    *size     = &nelem;
   int        ndims    = 1;

   SOLID_APPLY_ELEMWISE2_TYPES(SDXTYPE, int64, \
                               { int i; \
                                 solid_int64 s = 0; \
                                 for (i = 0; i < nstrides; i++) \
                                 {  s += strides[i] * *((SOLID_C_TYPE(SDXTYPE) *)(((char *)_ptr1) + i * strideReduce)); \
                                 } \
                                 *_ptr2 = s; \
                               });

   return 0;
}
#endif



#if (SDTYPE_IS_BOOL(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(find_prepare)(int ndims, const size_t *size,
                                 const ptrdiff_t *strides, void *ptr,
                                 size_t *nnz, void **ptr_data)
/* -------------------------------------------------------------------- */
{  solid_find_cpu_t *info;
   int i, result;

   /* Allocate the data structure */
   result = solid_find_allocate_cpu(ndims, size, strides, ptr, ptr_data);
   if (result != 0) return result;

   /* Count the number of non-zero indices */
   result = solid_find_initialize_cpu(*ptr_data);
   if (result != 0)
   {  solid_find_finalize_cpu(*ptr_data);
      *ptr_data = NULL;
      return result;
   }

   /* Output the total number of non-zero elements */
   *nnz = 0; info = (solid_find_cpu_t *)(*ptr_data);
   for (i = 0; i < info -> nthreads; i ++)
   {  *nnz += info -> nnz[i];
   }

   return 0;
}
#endif


#if (SDTYPE_IS_BOOL(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(find_apply)(void *data, solid_int64 *output_ptr,
                               const solid_int64 *multiplier)
/* -------------------------------------------------------------------- */
{  solid_find_cpu_t *info = (solid_find_cpu_t *)data;

   if (info -> nelem == 0) return 0;

   /* -------------------------------------------------- */
   /* Single-threaded implementation                     */
   /* -------------------------------------------------- */
   if (info -> nthreads == 1)
   {  size_t       idx[SOLID_MAX_TENSOR_DIMS];
      ptrdiff_t    strides[SOLID_MAX_TENSOR_DIMS];
      size_t       size0, strides0, nelem, index;
      solid_int64  output_value;
      solid_bool  *ptr;
      int          i, j, ndims;

      ndims = info -> ndims;
      nelem = info -> nelem;
      size0 = info -> size[0];
      strides0 = info -> strides[0];
      for (i = 1; i < ndims; i++)
      {  idx[i] = 0;
         strides[i] = info -> strides[i] - info -> size[i-1] * info -> strides[i-1];
      }

      /* This code requires solid_bool to be one byte */
      ptr = (solid_bool *)(info -> ptr);
      index = 0;
      while (1)
      {  /* Loop over the first dimension */
         if (multiplier)
         {  for (i = 0; i < size0; i++)
            {  if (*ptr)
               {  output_value = i * multiplier[0];
                  for (j = 1; j < ndims; j++) output_value += idx[j] * multiplier[j];
                  *output_ptr = output_value;
                  output_ptr += 1;
               }
               ptr += strides0;
            }
         }
         else
         {  for (i = 0; i < size0; i++)
            {  if (*ptr)
               {  output_ptr[0] = i;
                  for (j = 1; j < ndims; j++) output_ptr[j] = idx[j];
                  output_ptr += ndims;
               }
               ptr += strides0;
            }
         }


         /* Check for completion */
         index += size0; if (index >= nelem) break;

         /* Advance the indices */
         i = 1; ptr += strides[i];
         while (++idx[i] == info -> size[i])
         {  idx[i] = 0; i++;
            ptr += strides[i];
         }
      }
   }
#if SOLID_ENABLE_OMP
   else
   /* -------------------------------------------------- */
   /* Multi-threaded implementation                      */
   /* -------------------------------------------------- */
   {
      _Pragma("omp parallel num_threads(info -> nthreads)")
      {  int          rank = omp_get_thread_num(); \
         int          nthreads = omp_get_num_threads(); \
         size_t       idx[SOLID_MAX_TENSOR_DIMS];
         size_t       size[SOLID_MAX_TENSOR_DIMS];
         ptrdiff_t    strides[SOLID_MAX_TENSOR_DIMS];
         size_t       offset, index, indexMax, indexStop;
         size_t       size0, strides0, nelem;
         solid_bool  *ptr;
         solid_int64 *ptr2, output_value;
         int          i, j, ndims;

         /* Initialize */
         ndims = info -> ndims;
         nelem = info -> nelem;
         size0 = info -> size[0];
         strides0 = info -> strides[0];

         /* Compute the offset */
         offset = nelem % nthreads; nelem /= nthreads;
         if (rank < offset)
         {  nelem ++;
            offset = rank * nelem;
         }
         else
         {  offset += rank * nelem;
         }
         index  = offset;
         indexMax = offset + nelem;

         /* Initialize the data pointer and indices */
         ptr = (solid_bool *)(info -> ptr);
         for (i = 0; i < ndims; i++)
         {  size[i] = info -> size[i];
            idx[i] = offset % size[i];
            offset /= size[i];
            ptr += info -> strides[i] * idx[i];
         }

         /* Set the strides */
         for (i = 1; i < ndims; i++)
         {  strides[i] = (info -> strides[i]) - size[i-1] * (info -> strides[i-1]);
         }

         /* Initialize the output data pointer */
         ptr2 = output_ptr; j = (multiplier) ? 1 : ndims;
         for (i = 0; i < rank; i++) ptr2 += j * info -> nnz[i];

         /* Loop over the data */ \
         indexStop = index + size0 - idx[0];
         if (index < indexMax)
         {  while (1)
            {
               if (indexStop > indexMax) indexStop = indexMax;
               while (index < indexStop)
               {  if (*ptr)
                  {  if (multiplier)
                     {  output_value = multiplier[0] * (index % size[0]);
                        for (j = 1; j < ndims; j++) output_value += idx[j] * multiplier[j];
                        *ptr2 = output_value;
                        ptr2 += 1;
                     }
                     else
                     {  ptr2[0] = index % size[0];
                        for (j = 1; j < ndims; j++) ptr2[j] = idx[j];
                        ptr2 += ndims;
                     }
                  }
                  index ++; ptr += strides0;
               }
               if (index >= indexMax) break;

               i = 1; indexStop = index + size0;
               ptr += strides[i];
               while (++idx[i] == size[i])
               {  idx[i] = 0; i++;
                  ptr += strides[i];
               }
            }
         }
      }
   }
#endif /* SOLID_ENABLE_OMP */

   return 0;
}
#endif


#if (SDTYPE_IS_BOOL(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(find_finalize)(void *data)
/* -------------------------------------------------------------------- */
{  solid_find_finalize_cpu(data);
   return 0;
}
#endif


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(get_index)(int ndims, const size_t *size, solid_int64 **offsets,
                              const ptrdiff_t *strides1, void *ptr1,
                              const ptrdiff_t *strides2, void *ptr2)
/* -------------------------------------------------------------------- */
{
   /* Indexing is applied to ptr1 */
   SOLID_INDEX2({ *_ptr2 = *_ptr1; })

   return 0;
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(set_index)(int ndims, const size_t *size, solid_int64 **offsets,
                              const ptrdiff_t *strides1, void *ptr1,
                              const ptrdiff_t *strides2, void *ptr2)
/* -------------------------------------------------------------------- */
{
   /* Indexing is applied to ptr1 */
   SOLID_INDEX2({ *_ptr1 = *_ptr2; })

   return 0;
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(fill_index)(int ndims, const size_t *size,
                               solid_int64 **offsets, const ptrdiff_t *strides,
                               void *ptr, solid_scalar scalar)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) value = SOLID_SCALAR_C_VALUE(scalar);

   SOLID_INDEX1({ *_ptr = value; })

   return 0;
}

#endif

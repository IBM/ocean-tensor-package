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
#define SD_TEMPLATE_FILE "core/gpu/template_index.cu"

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/gpu/dtype_gpu.h"
#include "solid/core/gpu/apply_elemwise1.h"
#include "solid/core/gpu/apply_elemwise2.h"
#include "solid/core/gpu/index1.h"
#include "solid/core/gpu/index2.h"

#include "solid/base/generic/generate_all_types.h"
#else


/* ============================================================================== */
/* Function definition - Add if negative                                          */
/* ============================================================================== */

#if (SDTYPE_IS_SIGNED_INT(SDXTYPE))
/* Create the cuda kernels */
SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, addIfNegative,
                              { SOLID_C_TYPE(SDXTYPE) value; },
                              { if (*_ptr < 0) *_ptr += param.value; })

/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(add_if_negative)(int ndims, const size_t *size,
                                              const ptrdiff_t *strides, void *ptr,
                                              const solid_scalar scalar,
                                              cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(addIfNegative) param;
   int result = 0;

   /* Set user parameters*/
   param.value = SOLID_SCALAR_C_VALUE(scalar);

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLL, addIfNegative, 0, stream, param, result);

   return result;
}
#endif




/* ============================================================================== */
/* Function definition - Index to offset                                          */
/* ============================================================================== */

#if (SDTYPE_IS_INT(SDXTYPE))
/* Create the cuda kernels */
SOLID_KERNELS_ELEMWISE2_FULL(SDXTYPE, int64, 1, NO_UNROLLING, indexToOffset, \
                              { solid_int64 strides[SOLID_MAX_TENSOR_DIMS]; \
                                ptrdiff_t   strideReduce; \
                                int         nstrides; \
                              }, \
                              {  solid_int64 s = 0; \
                                 int i; \
                                 for (i = 0; i < param.nstrides; i++) \
                                 {  s += param.strides[i] * *((SOLID_C_TYPE(SDXTYPE) *)(((char *)_ptr1) + i * param.strideReduce)); \
                                 } \
                                 *_ptr2 = s; \
                              })


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(index_to_offset)(int nstrides, solid_int64 *strides,
                                              ptrdiff_t strideReduce, size_t nelem,
                                              ptrdiff_t stride1, void *ptr1,
                                              ptrdiff_t stride2, void *ptr2,
                                              cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(indexToOffset) param;
   ptrdiff_t *strides1 = &stride1;
   ptrdiff_t *strides2 = &stride2;
   size_t    *size     = &nelem;
   int        ndims    = 1;
   int        result   = 0;
   int        i;

   /* Set user parameters*/
   for (i = 0; i < nstrides; i++) param.strides[i] = strides[i];
   param.strideReduce = strideReduce;
   param.nstrides     = nstrides;

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE2_FULL(SDXTYPE, int64, 1, NO_UNROLLING, indexToOffset, 0, stream, param, result);

   return result;
}
#endif





/* ============================================================================== */
/* Function definitions - Get index                                               */
/* ============================================================================== */

/* Create the cuda kernel */
SOLID_KERNELS_INDEX2(getIndex, { *_ptr2 = *_ptr1; })

/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(get_index)(int ndims, const size_t *size, solid_int64 **offsets,
                                        const ptrdiff_t *strides1, void *ptr1,
                                        const ptrdiff_t *strides2, void *ptr2,
                                        cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  int result = 0;

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_INDEX2(getIndex, stream, result);

   return result;
}


/* ============================================================================== */
/* Function definitions - Set index                                               */
/* ============================================================================== */

/* Create the cuda kernel */
SOLID_KERNELS_INDEX2(setIndex, { *_ptr1 = *_ptr2; })

/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(set_index)(int ndims, const size_t *size, solid_int64 **offsets,
                                        const ptrdiff_t *strides1, void *ptr1,
                                        const ptrdiff_t *strides2, void *ptr2,
                                        cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  int result = 0;

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_INDEX2(setIndex, stream, result);

   return result;
}


/* ============================================================================== */
/* Function definitions - Fill index                                              */
/* ============================================================================== */

/* Create the cuda kernels */
SOLID_KERNELS_INDEX1_PARAM(fill_index,
                           { SOLID_C_TYPE(SDXTYPE) value; },
                           { *_ptr = param.value; })

/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(fill_index)(int ndims, const size_t *size, solid_int64 **offsets,
                                         const ptrdiff_t *strides, void *ptr,
                                         const solid_scalar scalar, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(fill_index) param;
   int result = 0;

   /* Set user parameters*/
   param.value = SOLID_SCALAR_C_VALUE(scalar);

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_INDEX1_PARAM(fill_index, stream, param, result);

   return result;
}


#endif

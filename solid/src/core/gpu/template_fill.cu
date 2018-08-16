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
#define SD_TEMPLATE_FILE "core/gpu/template_fill.cu"

#include "solid.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/gpu/dtype_gpu.h"
#include "solid/core/gpu/apply_elemwise1.h"
#include "solid/core/gpu/apply_elemwise2.h"

#include "solid/base/generic/generate_all_types.h"
#else

/* SDXTYPE must be defined */
#include "solid/core/gpu/unary_ops_gpu.h"


/* Create the cuda kernels */
SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, fill,
                              { SOLID_C_TYPE(SDXTYPE) value; },
                              { *_ptr = param.value; })

/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(fill)(int ndims, const size_t *size,
                                   const ptrdiff_t *strides, void *ptr,
                                   const solid_scalar scalar,
                                   cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(fill) param;
   int result = 0;

   /* Set user parameters*/
   param.value = SOLID_SCALAR_C_VALUE(scalar);

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLL, fill, 0, stream, param, result);

   return result;
}



#if SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE)

/* Create the cuda kernels */
SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, fill_nan,
                              { SOLID_C_TYPE(SDXTYPE) value; },
                              { SOLID_OP_FILLNAN(*_ptr, param.value) })

/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(fill_nan)(int ndims, const size_t *size,
                                       const ptrdiff_t *strides, void *ptr,
                                       const solid_scalar scalar,
                                      cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(fill) param;
   int result = 0;

   /* Set user parameters*/
   param.value = SOLID_SCALAR_C_VALUE(scalar);

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLL, fill, 0, stream, param, result);

   return result;
}

#endif


/* Create the cuda kernels */
SOLID_KERNELS_ELEMWISE2_FULL(SDXTYPE, bool, 1, UNROLL, masked_fill,
                             { SOLID_C_TYPE(SDXTYPE) value; },
                             { if (*_ptr2) *_ptr1 = param.value; })

/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(masked_fill)(int ndims, const size_t *size,
                                          const ptrdiff_t *strides1, void *ptr1,
                                          const ptrdiff_t *strides2, void *ptr2,
                                          const solid_scalar scalar,
                                          cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(masked_fill) param;
   int result = 0;

   /* Set user parameters*/
   param.value = SOLID_SCALAR_C_VALUE(scalar);

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE2_FULL(SDXTYPE, bool, 1, UNROLL, masked_fill, 0, stream, param, result);

   return result;
}

#endif

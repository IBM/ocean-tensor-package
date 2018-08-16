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
#define SD_TEMPLATE_FILE "core/gpu/template_steps.cu"

#include "solid.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/scalar.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/core/gpu/apply_elemwise1.h"
#include "solid/base/gpu/dtype_gpu.h"

#include "solid/base/generic/generate_all_types.h"
#else

/* Create the cuda kernels */
#if SDTYPE_IS_REAL(SDXTYPE)
SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, steps_int64,
                              { solid_int64 offset; solid_int64 step; },
                              { *_ptr = SOLID_FROM_ELEMWORKTYPE(param.offset + _index * param.step); })

SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, steps_double,
                              { solid_double offset; solid_double step; },
                              { *_ptr = SOLID_FROM_ELEMWORKTYPE(param.offset + _index * param.step); })

#else
SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, steps_int64,
                              { solid_int64 offset; solid_int64 step; },
                              { _ptr -> real = SOLID_FROM_ELEMWORKTYPE(param.offset + _index * param.step);
                                _ptr -> imag = SOLID_FROM_ELEMWORKTYPE(0);
                              })

SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, steps_double,
                              { solid_double offset; solid_double step; },
                              { _ptr -> real = SOLID_FROM_ELEMWORKTYPE(param.offset + _index * param.step);
                                _ptr -> imag = SOLID_FROM_ELEMWORKTYPE(0);
                              })

SOLID_KERNELS_ELEMWISE1_PARAM(UNROLL, steps_cdouble,
                              { solid_cdouble offset; solid_cdouble step; },
                              { _ptr -> real = SOLID_FROM_ELEMWORKTYPE(param.offset.real + _index * param.step.real);
                                _ptr -> imag = SOLID_FROM_ELEMWORKTYPE(param.offset.imag + _index * param.step.imag);
                              })
#endif


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(steps_int64)(int ndims, const size_t *size,
                                          const ptrdiff_t *strides, void *ptr,
                                          solid_int64 offset, solid_int64 step,
                                          cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(steps_int64) param;
   int result = 0;

   /* Set user parameters*/
   param.offset = offset;
   param.step   = step;

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLL, steps_int64, 0, stream, param, result);

   return result;
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(steps_double)(int ndims, const size_t *size,
                                           const ptrdiff_t *strides, void *ptr,
                                           solid_double offset, solid_double step,
                                           cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(steps_double) param;
   int result = 0;

   /* Set user parameters*/
   param.offset = offset;
   param.step   = step;

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLL, steps_double, 0, stream, param, result);

   return result;
}


#if SDTYPE_IS_COMPLEX(SDXTYPE)
/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(steps_cdouble)(int ndims, const size_t *size,
                                            const ptrdiff_t *strides, void *ptr,
                                            solid_cdouble offset, solid_cdouble step,
                                            cudaStream_t stream)
/* -------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(steps_cdouble) param;
   int result = 0;

   /* Set user parameters*/
   param.offset = offset;
   param.step   = step;

   /* Set up and launch the appropriate kernel */
   SOLID_LAUNCH_ELEMWISE1_PARAM(UNROLL, steps_cdouble, 0, stream, param, result);

   return result;
}
#endif

#endif

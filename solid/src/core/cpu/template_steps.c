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
#define SD_TEMPLATE_FILE "core/cpu/template_steps.c"

#include "solid/base/generic/scalar.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/core/cpu/apply_elemwise1.h"

#include "solid/base/generic/generate_all_types.h"

#else

/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(steps_int64)(int ndims, const size_t *size,
                                const ptrdiff_t *strides, void *ptr,
                                solid_int64 offset, solid_int64 step)
/* -------------------------------------------------------------------- */
#if SDTYPE_IS_REAL(SDXTYPE)
{
   SOLID_APPLY_ELEMWISE1({ *_ptr = SOLID_FROM_WORKTYPE((SOLID_C_WORKTYPE)(offset + _index * step)); })

   return 0;
}
#else
{  SOLID_C_ELEMTYPE zero = SOLID_FROM_ELEMWORKTYPE(0);

   SOLID_APPLY_ELEMWISE1({ _ptr -> real = SOLID_FROM_ELEMWORKTYPE((SOLID_C_ELEMWORKTYPE)(offset + _index * step)); \
                           _ptr -> imag = zero; })

   return 0;
}
#endif


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(steps_double)(int ndims, const size_t *size,
                                 const ptrdiff_t *strides, void *ptr,
                                 solid_double offset, solid_double step)
/* -------------------------------------------------------------------- */
#if SDTYPE_IS_REAL(SDXTYPE)
{
   SOLID_APPLY_ELEMWISE1({ *_ptr = SOLID_FROM_WORKTYPE((SOLID_C_WORKTYPE)(offset + _index * step)); })

   return 0;
}
#else
{  SOLID_C_ELEMTYPE zero = SOLID_FROM_ELEMWORKTYPE(0);

   SOLID_APPLY_ELEMWISE1({ _ptr -> real = SOLID_FROM_ELEMWORKTYPE((SOLID_C_ELEMWORKTYPE)(offset + _index * step)); \
                           _ptr -> imag = zero; })

   return 0;
}
#endif


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(steps_cdouble)(int ndims, const size_t *size,
                                  const ptrdiff_t *strides, void *ptr,
                                  solid_cdouble offset, solid_cdouble step)
/* -------------------------------------------------------------------- */
#if SDTYPE_IS_REAL(SDXTYPE)
{  solid_double real_offset = offset.real;
   solid_double real_step   = step.real;

   SOLID_APPLY_ELEMWISE1({ *_ptr = SOLID_FROM_WORKTYPE((SOLID_C_WORKTYPE)(real_offset + _index * real_step)); })

   return 0;
}
#else
{
   SOLID_APPLY_ELEMWISE1({ _ptr -> real = SOLID_FROM_ELEMWORKTYPE((SOLID_C_ELEMWORKTYPE)(offset.real + _index * step.real)); \
                           _ptr -> imag = SOLID_FROM_ELEMWORKTYPE((SOLID_C_ELEMWORKTYPE)(offset.imag + _index * step.imag)); })

   return 0;
}
#endif

#endif

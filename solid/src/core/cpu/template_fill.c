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
#define SD_TEMPLATE_FILE "core/cpu/template_fill.c"

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/core/cpu/apply_elemwise1.h"
#include "solid/core/cpu/apply_elemwise2.h"

#include "solid/base/generic/generate_all_types.h"
#else

/* SDXTYPE must be defined */
#include "solid/core/cpu/unary_ops_cpu.h"


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(fill)(int ndims, const size_t *size, const ptrdiff_t *strides,
                         void *ptr, solid_scalar scalar)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) value = SOLID_SCALAR_C_VALUE(scalar);

   SOLID_APPLY_ELEMWISE1({ *_ptr = value; })

   return 0;
}


#if SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE)
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(fill_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                             void *ptr, solid_scalar scalar)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) value = SOLID_SCALAR_C_VALUE(scalar);

   SOLID_APPLY_ELEMWISE1({ SOLID_OP_FILLNAN(*_ptr, value) })

   return 0;
}

#endif


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(masked_fill)(int ndims, const size_t *size, const ptrdiff_t *strides1,
                                void *ptr1, const ptrdiff_t *strides2, void *ptr2,
                                solid_scalar scalar)
/* -------------------------------------------------------------------- */
{  SOLID_C_TYPE(SDXTYPE) value = SOLID_SCALAR_C_VALUE(scalar);

   SOLID_APPLY_ELEMWISE2_TYPES(SDXTYPE, bool, { if (*_ptr2) *_ptr1 = value; })

   return 0;
}

#endif

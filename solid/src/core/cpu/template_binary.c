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
#define SD_TEMPLATE_FILE "core/cpu/template_binary.c"

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/core/cpu/apply_elemwise2.h"
#include "solid/core/cpu/apply_elemwise3.h"

#include "solid/base/generic/generate_all_types.h"
#else 

/* SDXTYPE must be defined */
#include "solid/core/cpu/binary_ops_cpu.h"


/* ============================================================================== */
/* Template definition                                                            */
/* ============================================================================== */

#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2,              */ \
/*                            const ptrdiff_t *strides3, void *ptr3)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2, \
                         const ptrdiff_t *strides3, void *ptr3) \
{ \
   if ((strides3 == NULL) && (ptr3 == NULL)) \
   {  \
      SOLID_APPLY_ELEMWISE2(OP(*_ptr1, *_ptr2, *_ptr1)) \
   } \
   else \
   { \
      SOLID_APPLY_ELEMWISE3(OP(*_ptr1, *_ptr2, *_ptr3)) \
   } \
   \
   return 0; \
}

/* Addition and multiplication */
SD_TEMPLATE(add,          SOLID_OP_ADD)
SD_TEMPLATE(subtract,     SOLID_OP_SUBTRACT)
SD_TEMPLATE(multiply,     SOLID_OP_MULTIPLY)
SD_TEMPLATE(true_divide,  SOLID_OP_TRUE_DIVIDE)



/* ============================================================================== */
/* Template definition - Simplified kernels using only elemwise3                  */
/* ============================================================================== */
#undef SD_TEMPLATE
#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2,              */ \
/*                            const ptrdiff_t *strides3, void *ptr3)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2, \
                         const ptrdiff_t *strides3, void *ptr3) \
{ \
   if ((strides3 == NULL) && (ptr3 == NULL)) \
   {  strides3 = strides1; ptr3 = ptr1; \
   } \
   \
   SOLID_APPLY_ELEMWISE3(OP(*_ptr1, *_ptr2, *_ptr3)) \
   \
   return 0; \
}

#if (!SDTYPE_IS_BOOL(SDXTYPE))
SD_TEMPLATE(min,           SOLID_OP_MIN)
SD_TEMPLATE(max,           SOLID_OP_MAX)
#endif

#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(fmin,          SOLID_OP_FMIN)
SD_TEMPLATE(fmax,          SOLID_OP_FMAX)
#endif

#if (!SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(mod,          SOLID_OP_MOD)
#endif

#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_SIGNED_INT(SDXTYPE))
SD_TEMPLATE(floor_divide, SOLID_OP_FLOOR_DIVIDE)
SD_TEMPLATE(fmod,         SOLID_OP_FMOD)
#endif

#if (SDTYPE_IS_SIGNED_INT(SDXTYPE))
SD_TEMPLATE(divide,       SOLID_OP_DIVIDE)
#endif

/* Logical operations (Boolean type) */
#if (SDTYPE_IS_BOOL(SDXTYPE))
SD_TEMPLATE(logical_and,  SOLID_OP_BITAND)
SD_TEMPLATE(logical_or,   SOLID_OP_BITOR)
SD_TEMPLATE(logical_xor,  SOLID_OP_BITXOR)
#endif

/* Bitwise operations (integer types) */
#if (SDTYPE_IS_INT(SDXTYPE))
SD_TEMPLATE(bitwise_and,  SOLID_OP_BITAND)
SD_TEMPLATE(bitwise_or,   SOLID_OP_BITOR)
SD_TEMPLATE(bitwise_xor,  SOLID_OP_BITXOR)
#endif

#undef SD_TEMPLATE



/* ============================================================================== */
/* Template definition - alias functions                                          */
/* ============================================================================== */
#define SD_TEMPLATE(NAME, REFNAME) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2,              */ \
/*                            const ptrdiff_t *strides3, void *ptr3)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2, \
                         const ptrdiff_t *strides3, void *ptr3) \
{ \
   return SOLID_FUNCTION(REFNAME)(ndims, size, strides1, ptr1, strides2, ptr2, strides3, ptr3); \
}

#if (SDTYPE_IS_BOOL(SDXTYPE))
SD_TEMPLATE(bitwise_and,  logical_and)
SD_TEMPLATE(bitwise_or,   logical_or )
SD_TEMPLATE(bitwise_xor,  logical_xor)
SD_TEMPLATE(min,          logical_and)
SD_TEMPLATE(max,          logical_or )
SD_TEMPLATE(fmin,         logical_and)
SD_TEMPLATE(fmax,         logical_or )
#endif

#if (SDTYPE_IS_INT(SDXTYPE))
SD_TEMPLATE(fmin,         min)
SD_TEMPLATE(fmax,         max)
#endif

#if (!SDTYPE_IS_SIGNED_INT(SDXTYPE))
SD_TEMPLATE(divide,       true_divide)
#endif

#if (SDTYPE_IS_BOOL(SDXTYPE) || SDTYPE_IS_UNSIGNED_INT(SDXTYPE))
SD_TEMPLATE(floor_divide, true_divide)
SD_TEMPLATE(fmod,         mod)
#endif

#undef SD_TEMPLATE



/* ============================================================================== */
/* Template definition - comparison functions                                     */
/* ============================================================================== */
#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2,              */ \
/*                            const ptrdiff_t *strides3, void *ptr3)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2, \
                         const ptrdiff_t *strides3, void *ptr3) \
{ \
   SOLID_APPLY_ELEMWISE3_TYPES(SDXTYPE, SDXTYPE, bool, OP(*_ptr1, *_ptr2, *_ptr3)) \
   \
   return 0; \
}

SD_TEMPLATE(lt, SOLID_OP_LT)
SD_TEMPLATE(le, SOLID_OP_LE)
SD_TEMPLATE(eq, SOLID_OP_EQ)
SD_TEMPLATE(ne, SOLID_OP_NE)
SD_TEMPLATE(ge, SOLID_OP_GE)
SD_TEMPLATE(gt, SOLID_OP_GT)

#undef SD_TEMPLATE



/* ============================================================================== */
/* Template definition - bitshift functions                                       */
/* ============================================================================== */
#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2,              */ \
/*                            const ptrdiff_t *strides3, void *ptr3)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2, \
                         const ptrdiff_t *strides3, void *ptr3) \
{ \
   SOLID_APPLY_ELEMWISE3_TYPES(SDXTYPE, int8, SDXTYPE, OP(*_ptr1, *_ptr2, *_ptr3)) \
   \
   return 0; \
}

#if ((SDTYPE_IS_INT(SDXTYPE)) || (SDTYPE_IS_BOOL(SDXTYPE)))
SD_TEMPLATE(bitshift_right, SOLID_OP_BITSHIFT_RIGHT)
SD_TEMPLATE(bitshift_left,  SOLID_OP_BITSHIFT_LEFT )
#endif

#undef SD_TEMPLATE



/* ============================================================================== */
/* Template definition - power function                                           */
/* ============================================================================== */
#define SD_TEMPLATE(NAME, OP, DTYPE) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2,              */ \
/*                            const ptrdiff_t *strides3, void *ptr3)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2, \
                         const ptrdiff_t *strides3, void *ptr3) \
{ \
   SOLID_APPLY_ELEMWISE3_TYPES(SDXTYPE, DTYPE, SDXTYPE, OP(*_ptr1, *_ptr2, *_ptr3)) \
   \
   return 0; \
}

#if ((SDTYPE_IS_INT(SDXTYPE)) || (SDTYPE_IS_BOOL(SDXTYPE)))
SD_TEMPLATE(power, SOLID_OP_POWER, int16)
#else
SD_TEMPLATE(power, SOLID_OP_POWER, SDXTYPE)
#endif

#undef SD_TEMPLATE


#endif


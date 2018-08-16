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
#define SD_TEMPLATE_FILE "core/cpu/template_unary.c"

#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/core/cpu/apply_elemwise1.h"
#include "solid/core/cpu/apply_elemwise2.h"

#include "solid/base/generic/generate_all_types.h"
#else

/* SDXTYPE must be defined */
#include "solid/core/cpu/unary_ops_cpu.h"

/* ============================================================================== */
/* Template definition                                                            */
/* ============================================================================== */

#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2) \
{ \
   if ((strides2 == NULL) && (ptr2 == NULL)) \
   {  const ptrdiff_t *strides = strides1; \
      void *ptr = ptr1; \
      \
      SOLID_APPLY_ELEMWISE1(OP(*_ptr, *_ptr)) \
   } \
   else \
   { \
      SOLID_APPLY_ELEMWISE2(OP(*_ptr1, *_ptr2)) \
   } \
   \
   return 0; \
}

/* All types */
SD_TEMPLATE(fabs, SOLID_OP_FABS)
SD_TEMPLATE(sign, SOLID_OP_SIGN)

/* All types except Booleand and unsigned int */
#if (!SDTYPE_IS_UNSIGNED_INT(SDXTYPE) && !SDTYPE_IS_BOOL(SDXTYPE))
SD_TEMPLATE(negative, SOLID_OP_NEGATIVE)
#endif

/* Boolean only */
#if (SDTYPE_IS_BOOL(SDXTYPE))
SD_TEMPLATE(logical_not, SOLID_OP_BITNOT)
#endif

/* Integer types */
#if (SDTYPE_IS_INT(SDXTYPE))
SD_TEMPLATE(bitwise_not, SOLID_OP_BITNOT)
#endif

/* Complex types */
#if (SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(conj,     SOLID_OP_CONJ)
#endif

/* Floating point and complex */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(ceil,  SOLID_OP_CEIL)
SD_TEMPLATE(floor, SOLID_OP_FLOOR)
SD_TEMPLATE(trunc, SOLID_OP_TRUNC)
SD_TEMPLATE(round, SOLID_OP_ROUND)
#endif

/* Integer, floating pointer, and complex */
#if (!SDTYPE_MATCHES(SDXTYPE, Bool))
SD_TEMPLATE(sin,        SOLID_OP_SIN)
SD_TEMPLATE(cos,        SOLID_OP_COS)
SD_TEMPLATE(tan,        SOLID_OP_TAN)
SD_TEMPLATE(sinh,       SOLID_OP_SINH)
SD_TEMPLATE(cosh,       SOLID_OP_COSH)
SD_TEMPLATE(tanh,       SOLID_OP_TANH)
SD_TEMPLATE(arcsin,     SOLID_OP_ARCSIN)
SD_TEMPLATE(arccos,     SOLID_OP_ARCCOS)
SD_TEMPLATE(arctan,     SOLID_OP_ARCTAN)
SD_TEMPLATE(arcsinh,    SOLID_OP_ARCSINH)
SD_TEMPLATE(arccosh,    SOLID_OP_ARCCOSH)
SD_TEMPLATE(arctanh,    SOLID_OP_ARCTANH)

SD_TEMPLATE(reciprocal, SOLID_OP_RECIPROCAL)
SD_TEMPLATE(sqrt,       SOLID_OP_SQRT_312)
SD_TEMPLATE(cbrt,       SOLID_OP_CBRT)
SD_TEMPLATE(square,     SOLID_OP_SQUARE)
SD_TEMPLATE(exp,        SOLID_OP_EXP)
SD_TEMPLATE(exp2,       SOLID_OP_EXP2)
SD_TEMPLATE(exp10,      SOLID_OP_EXP10)
SD_TEMPLATE(expm1,      SOLID_OP_EXPM1)
SD_TEMPLATE(log,        SOLID_OP_LOG)
SD_TEMPLATE(log2,       SOLID_OP_LOG2)
SD_TEMPLATE(log10,      SOLID_OP_LOG10)
SD_TEMPLATE(log1p,      SOLID_OP_LOG1P)
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
/*                            const ptrdiff_t *strides2, void *ptr2)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2) \
{ \
   return SOLID_FUNCTION(REFNAME)(ndims, size, strides1, ptr1, strides2, ptr2); \
}

#if (SDTYPE_IS_BOOL(SDXTYPE))
SD_TEMPLATE(negative,    logical_not)
SD_TEMPLATE(bitwise_not, logical_not)
#endif

#if (!SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(absolute, fabs)
#endif

#undef SD_TEMPLATE



/* ============================================================================== */
/* Template definition - Absolute                                                 */
/* ============================================================================== */
#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2) \
{ \
   SOLID_APPLY_ELEMWISE2_TYPES(SDXTYPE, SDTYPE_ELEMTYPE(SDXTYPE), OP(*_ptr1, *_ptr2)) \
   \
   return 0; \
}

/* Complex types */
#if (SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(absolute, SOLID_OP_ABSOLUTE)
#endif

#undef SD_TEMPLATE



/* ============================================================================== */
/* Template definition - Query functions                                          */
/* ============================================================================== */
#define SD_TEMPLATE(NAME, OP) \
\
/* ------------------------------------------------------------------------------ */ \
/* int SOLID_FUNCTION(opname)(int ndims, const size_t *size,                      */ \
/*                            const ptrdiff_t *strides1, void *ptr1,              */ \
/*                            const ptrdiff_t *strides2, void *ptr2)              */ \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(NAME)(int ndims, const size_t *size, \
                         const ptrdiff_t *strides1, void *ptr1, \
                         const ptrdiff_t *strides2, void *ptr2) \
{ \
   SOLID_APPLY_ELEMWISE2_TYPES(SDXTYPE, bool, OP(*_ptr1, *_ptr2)) \
   \
   return 0; \
}

/* Functions for all types */
SD_TEMPLATE(isinf,    SOLID_OP_ISINF)
SD_TEMPLATE(isnan,    SOLID_OP_ISNAN)
SD_TEMPLATE(isfinite, SOLID_OP_ISFINITE)


/* Non-complex types */
#if (!SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(isposinf, SOLID_OP_ISPOSINF)
SD_TEMPLATE(isneginf, SOLID_OP_ISNEGINF)
#endif

#undef SD_TEMPLATE

#endif


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
#define SD_TEMPLATE_FILE "core/cpu/template_reduce_all.c"

#include "solid/base/generic/generate_macros.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/core/cpu/reduce_all.h"
#include "solid_core_cpu.h"

/* Generate functions for all data types */
#include "solid/base/generic/generate_all_types.h"
#else

/* SDXTYPE must be defined */
#include "solid/core/generic/reduce_ops.h"
#include "solid/core/cpu/unary_ops_cpu.h"


/* ==================================================================== */
/* Define the break statement used in the reduce operations             */
/* ==================================================================== */
#ifndef SOLID_OP_REDUCE_BREAK
#define SOLID_OP_REDUCE_BREAK  { _index = _indexMax; break; }  /* Break */
#endif



/* ==================================================================== */
/* Generate all parameter types                                         */
/* ==================================================================== */

/* Instantiate all parameter types */
#define SD_TEMPLATE(TEMPLATE) SOLID_OP_REDUCE_##TEMPLATE
#define SD_TEMPLATE_REDUCE(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, FLAG_FINALIZE, \
                           CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, CODE_FINALIZE) \
        SOLID_PARAM_STRUCT_FLAG(SOLID_FUNCTION_TYPE(NAME,DTYPE1), FLAG_PARAM, PARAM)

SD_TEMPLATE(ALL_LT)
SD_TEMPLATE(ALL_LE)
SD_TEMPLATE(ALL_GT)
SD_TEMPLATE(ALL_GE)
SD_TEMPLATE(ALL_GTLT)
SD_TEMPLATE(ALL_GTLE)
SD_TEMPLATE(ALL_GELT)
SD_TEMPLATE(ALL_GELE)
SD_TEMPLATE(NORM)

#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(NORM_NAN)
#endif

#undef SD_TEMPLATE_REDUCE
#undef SD_TEMPLATE



/* ==================================================================== */
/* Reduction operation functions                                        */
/* ==================================================================== */

/* This macro is called indirectly through SOLID_OP_REDUCE function */
#define SD_TEMPLATE_REDUCE(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, FLAG_FINALIZE, \
                           CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, CODE_FINALIZE) \
   SOLID_REDUCE_ALL_TYPE(DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE) \
   return 0;


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(any)(int ndims, const size_t *size, const ptrdiff_t *strides,
                        void *ptr, solid_bool *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_ANY
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(all)(int ndims, const size_t *size, const ptrdiff_t *strides,
                        void *ptr, solid_bool *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_ALL
}


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(all_finite)(int ndims, const size_t *size, const ptrdiff_t *strides,
                               void *ptr, solid_bool *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_ALL_FINITE
}
#endif


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(any_inf)(int ndims, const size_t *size, const ptrdiff_t *strides,
                            void *ptr, solid_bool *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_ANY_INF
}
#endif


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(any_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                            void *ptr, solid_bool *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_ANY_NAN
}
#endif


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(nnz)(int ndims, const size_t *size, const ptrdiff_t *strides,
                        void *ptr, solid_uint64 *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_NNZ
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(nnz_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                            void *ptr, solid_uint64 *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(nnz)(ndims, size, strides, ptr, result);
#else /* Floating point */
   SOLID_OP_REDUCE_NNZ_NAN
#endif
}


#define SD_TEMPLATE(OP, OP_MACRO) \
/* -------------------------------------------------------------------- */ \
int SOLID_FUNCTION(all_##OP)(int ndims, const size_t *size, const ptrdiff_t *strides, \
                             void *ptr, solid_scalar *bound, solid_bool *result) \
/* -------------------------------------------------------------------- */ \
{  SOLID_PARAM(all_##OP) param; \
   solid_funptr_cpu_scalar_copy funptr; \
   \
   /* Copy the scalar */ \
   funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(SDXTYPE)][SDTYPE_INDEX(SOLID_WORKTYPE(SDXTYPE))]; \
   if (funptr == 0) SOLID_ERROR(-1, "Error converting scalar bound"); \
   funptr(bound, &(param.bound)); \
   \
   /* Apply the reduction */ \
   SOLID_OP_REDUCE_ALL_##OP_MACRO \
}

SD_TEMPLATE(lt, LT)
SD_TEMPLATE(le, LE)
SD_TEMPLATE(gt, GT)
SD_TEMPLATE(ge, GE)
#undef SD_TEMPLATE



#define SD_TEMPLATE(OP, OP_MACRO) \
/* -------------------------------------------------------------------- */ \
int SOLID_FUNCTION(all_##OP)(int ndims, const size_t *size, const ptrdiff_t *strides, \
                             void *ptr, solid_scalar *lower, solid_scalar *upper, solid_bool *result) \
/* -------------------------------------------------------------------- */ \
{  SOLID_PARAM(all_##OP) param; \
   solid_funptr_cpu_scalar_copy funptr; \
   \
   /* Copy the scalars */ \
   funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(SDXTYPE)][SDTYPE_INDEX(SOLID_WORKTYPE(SDXTYPE))];\
   if (funptr == 0) SOLID_ERROR(-1, "Error converting scalar bound"); \
   funptr(lower, &(param.lower)); \
   funptr(upper, &(param.upper)); \
   \
   /* Apply the reduction */ \
   SOLID_OP_REDUCE_ALL_##OP_MACRO \
}

SD_TEMPLATE(gtlt, GTLT)
SD_TEMPLATE(gtle, GTLE)
SD_TEMPLATE(gelt, GELT)
SD_TEMPLATE(gele, GELE)
#undef SD_TEMPLATE


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(sum)(int ndims, const size_t *size,
                        const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_SUM
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(prod)(int ndims, const size_t *size,
                         const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_PROD
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(sum_nan)(int ndims, const size_t *size,
                            const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(sum)(ndims, size, strides, ptr, result);
#else /* Floating point */
   SOLID_OP_REDUCE_SUM_NAN
#endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(prod_nan)(int ndims, const size_t *size,
                             const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(prod)(ndims, size, strides, ptr, result);
#else /* Floating point */
   SOLID_OP_REDUCE_PROD_NAN
#endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(sum_abs)(int ndims, const size_t *size,
                            const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(sum)(ndims, size, strides, ptr, result);
#else /* Floating point or signed integer */
   SOLID_OP_REDUCE_SUM_ABS
#endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(sum_abs_nan)(int ndims, const size_t *size,
                                const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(sum)(ndims, size, strides, ptr, result);
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   return SOLID_FUNCTION(sum_abs)(ndims, size, strides, ptr, result);
#else /* Floating point */
   SOLID_OP_REDUCE_SUM_ABS_NAN
#endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(maximum)(int ndims, const size_t *size,
                            const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (solid_empty_tensor(ndims, size))
         SOLID_ERROR(-1, "The tensor maximum funtion is undefined on empty tensors");

   SOLID_OP_REDUCE_MAXIMUM
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(minimum)(int ndims, const size_t *size,
                            const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
  /* Make sure the tensor is not empty */
   if (solid_empty_tensor(ndims, size))
      SOLID_ERROR(-1, "The tensor minimum function is undefined on empty tensors");

   SOLID_OP_REDUCE_MINIMUM
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(maximum_abs)(int ndims, const size_t *size,
                                const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(maximum)(ndims, size, strides, ptr, result);
#else
   /* Floating pointer of signed integer */
   if (solid_empty_tensor(ndims, size))
      SOLID_ERROR(-1, "The tensor absolute maximum function is undefined on empty tensors");

   SOLID_OP_REDUCE_MAXIMUM_ABS
#endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(minimum_abs)(int ndims, const size_t *size,
                                const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(minimum)(ndims, size, strides, ptr, result);
#else
   /* Floating pointer of signed integer */
   if (solid_empty_tensor(ndims, size))
      SOLID_ERROR(-1, "The tensor absolute minimum function is undefined on empty tensors");

   SOLID_OP_REDUCE_MINIMUM_ABS
#endif
}



/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(norm)(int ndims, const size_t *size, const ptrdiff_t *strides,
                         void *ptr, double p, void *result)
/* -------------------------------------------------------------------- */
{  SOLID_PARAM(norm) param;
   double pinv = 1 / p;
   double phalf = p / 2;

   /* Set the parameters */
   #if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   {  /* Parameters have type double */
      param.phalf = phalf;
      param.pinv = pinv;
   }
   #else
   {  solid_funptr_cpu_scalar_copy funptr;
      funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_ELEMWORKTYPE(SDXTYPE))];
      if (funptr == 0) SOLID_ERROR(-1, "Error converting norm parameter");
      funptr(&phalf, &(param.phalf));
      funptr(&pinv, &(param.pinv));
   }
   #endif

   /* Apply the reduction */
   SOLID_OP_REDUCE_NORM
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(norm_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                             void *ptr, double p, void *result)
/* -------------------------------------------------------------------- */
{
   #if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   {  return SOLID_FUNCTION(norm)(ndims, size, strides, ptr, p, result);
   }
   #else /* Floating point */
   {  SOLID_PARAM(norm_nan) param;
      double pinv = 1 / p;
      double phalf = p / 2;

      /* Set the parameters */
      solid_funptr_cpu_scalar_copy funptr;
      funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_ELEMWORKTYPE(SDXTYPE))];
      if (funptr == 0) SOLID_ERROR(-1, "Error converting norm parameter");
      funptr(&phalf, &(param.phalf));
      funptr(&pinv, &(param.pinv));

      /* Apply the reduction */
      SOLID_OP_REDUCE_NORM_NAN
   }
   #endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(norm2)(int ndims, const size_t *size,
                          const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
   SOLID_OP_REDUCE_NORM2
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(norm2_nan)(int ndims, const size_t *size,
                              const ptrdiff_t *strides, void *ptr, void *result)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_FUNCTION(norm2)(ndims, size, strides, ptr, result);
#else /* Floating point */
   SOLID_OP_REDUCE_NORM2_NAN
#endif
}


/* Undefined the reduction template */
#undef SD_TEMPLATE_REDUCE

#endif

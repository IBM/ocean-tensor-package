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
#define SD_TEMPLATE_FILE "core/gpu/template_reduce_all.cu"

#include "solid.h"
#include "solid_gpu.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/gpu/dtype_gpu.h"
#include "solid/core/gpu/reduce_all.h"

/* Scalar copy */
#include "solid_core_cpu.h"

#include "solid/base/generic/generate_all_types.h"
#else

/* SDXTYPE must be defined */
#include "solid/core/generic/reduce_ops.h"
#include "solid/core/gpu/unary_ops_gpu.h"


/* ======================================================================== */
/* Create the kernels, parameter types, and launch functions                */
/* ======================================================================== */

/* Define the break statement */
#define SOLID_OP_REDUCE_BREAK  /* Empty */

/* Create the kernels and launch functions */
#define SD_TEMPLATE(TEMPLATE) SOLID_OP_REDUCE_##TEMPLATE

/* This macro is called indirectly through SOLID_OP_REDUCE function */
#define SD_TEMPLATE_REDUCE(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, FLAG_FINALIZE, \
                           CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, CODE_FINALIZE) \
   SOLID_CREATE_REDUCE_ALL(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, \
                           CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE)

SD_TEMPLATE(ANY     )
SD_TEMPLATE(ALL     )
SD_TEMPLATE(NNZ     )
SD_TEMPLATE(ALL_LT  )
SD_TEMPLATE(ALL_LE  )
SD_TEMPLATE(ALL_GT  )
SD_TEMPLATE(ALL_GE  )
SD_TEMPLATE(ALL_GTLT)
SD_TEMPLATE(ALL_GTLE)
SD_TEMPLATE(ALL_GELT)
SD_TEMPLATE(ALL_GELE)
SD_TEMPLATE(SUM     )
SD_TEMPLATE(PROD    )
SD_TEMPLATE(MAXIMUM )
SD_TEMPLATE(MINIMUM )
SD_TEMPLATE(NORM    )
SD_TEMPLATE(NORM2   )

#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(ALL_FINITE )
SD_TEMPLATE(ANY_INF    )
SD_TEMPLATE(ANY_NAN    )
SD_TEMPLATE(NNZ_NAN    )
SD_TEMPLATE(SUM_NAN    )
SD_TEMPLATE(PROD_NAN   )
SD_TEMPLATE(SUM_ABS_NAN)
SD_TEMPLATE(NORM_NAN   )
SD_TEMPLATE(NORM2_NAN  )
#endif

#if (SDTYPE_IS_SIGNED_INT(SDXTYPE) || SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(SUM_ABS    )
SD_TEMPLATE(MAXIMUM_ABS)
SD_TEMPLATE(MINIMUM_ABS)
#endif

#undef SD_TEMPLATE_REDUCE
#undef SD_TEMPLATE
#undef SOLID_OP_REDUCE_BREAK


/* ======================================================================== */
/* Create the API functions */
/* ======================================================================== */

/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(any)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                   void *ptr, solid_bool *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(any);
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(all)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                   void *ptr, solid_bool *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(all);
}


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(all_finite)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                         void *ptr, solid_bool *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(all_finite);
}
#endif


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(any_inf)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                      void *ptr, solid_bool *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(any_inf);
}
#endif

#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(any_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                      void *ptr, solid_bool *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */ \
   return SOLID_CALL_REDUCE_ALL(any_nan); \
}
#endif


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(nnz)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                  void *ptr, solid_uint64 *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(nnz);
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(nnz_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                      void *ptr, solid_uint64 *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(nnz_nan);
#else
   return SOLID_FUNCTION(nnz)(ndims, size, strides, ptr, result, buffer, stream);
#endif
}


#define SD_TEMPLATE(OP) \
/* ------------------------------------------------------------------------ */ \
SOLID_API int SOLID_FUNCTION(OP)(int ndims, const size_t *size, const ptrdiff_t *strides, \
                                 void *ptr, solid_scalar *bound, solid_bool *result, \
                                 void *buffer, cudaStream_t stream) \
/* ----------------------------------------------------------------------- */ \
{  SOLID_KERNEL_PARAM(OP) param; \
   solid_funptr_cpu_scalar_copy funptr; \
   \
   /* Copy the scalar */ \
   funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(SDXTYPE)][SDTYPE_INDEX(SOLID_WORKTYPE(SDXTYPE))]; \
   if (funptr == 0) SOLID_ERROR(-1, "Error converting scalar bound"); \
   funptr(bound, &(param.bound)); \
   \
   /* Call the reduce operation with default parameters */ \
   return SOLID_CALL_REDUCE_ALL_PARAM(OP, &param); \
}

SD_TEMPLATE(all_lt)
SD_TEMPLATE(all_le)
SD_TEMPLATE(all_gt)
SD_TEMPLATE(all_ge)
#undef SD_TEMPLATE


#define SD_TEMPLATE(OP) \
/* ------------------------------------------------------------------------ */ \
SOLID_API int SOLID_FUNCTION(OP)(int ndims, const size_t *size, const ptrdiff_t *strides,\
                                 void *ptr, solid_scalar *lower, solid_scalar *upper, \
                                 solid_bool *result, void *buffer, cudaStream_t stream) \
/* ----------------------------------------------------------------------- */ \
{  SOLID_KERNEL_PARAM(OP) param; \
   solid_funptr_cpu_scalar_copy funptr; \
   \
   /* Copy the scalar */ \
   funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(SDXTYPE)][SDTYPE_INDEX(SOLID_WORKTYPE(SDXTYPE))];\
   if (funptr == 0) SOLID_ERROR(-1, "Error converting scalar bound"); \
   funptr(lower, &(param.lower)); \
   funptr(lower, &(param.upper)); \
   \
   /* Call the reduce operation with default parameters */ \
   return SOLID_CALL_REDUCE_ALL_PARAM(OP, &param); \
}

SD_TEMPLATE(all_gtlt)
SD_TEMPLATE(all_gtle)
SD_TEMPLATE(all_gelt)
SD_TEMPLATE(all_gele)
#undef SD_TEMPLATE


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(sum)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                  void *ptr, void *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(sum);
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(prod)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                   void *ptr, void *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(prod);
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(sum_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                      void *ptr, void *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(sum_nan);
#else
   return SOLID_FUNCTION(sum)(ndims, size, strides, ptr, result, buffer, stream);
#endif
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(prod_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                       void *ptr, void *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(prod_nan);
#else
   return SOLID_FUNCTION(prod)(ndims, size, strides, ptr, result, buffer, stream);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(sum_abs)(int ndims, const size_t *size,
                                      const ptrdiff_t *strides, void *ptr,
                                      void *result, void *buffer, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(sum);
#else /* Floating point or signed integer */
   return SOLID_CALL_REDUCE_ALL(sum_abs);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(sum_abs_nan)(int ndims, const size_t *size,
                                          const ptrdiff_t *strides, void *ptr,
                                          void *result, void *buffer, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(sum);
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   return SOLID_CALL_REDUCE_ALL(sum_abs);
#else /* Floating point */
   return SOLID_CALL_REDUCE_ALL(sum_abs_nan);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(maximum)(int ndims, const size_t *size,
                                      const ptrdiff_t *strides, void *ptr,
                                      void *result, void *buffer, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (solid_empty_tensor(ndims, size))
         SOLID_ERROR(-1, "The tensor maximum funtion is undefined on empty tensors");

   return SOLID_CALL_REDUCE_ALL(maximum);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(minimum)(int ndims, const size_t *size,
                                      const ptrdiff_t *strides, void *ptr,
                                      void *result, void *buffer, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (solid_empty_tensor(ndims, size))
      SOLID_ERROR(-1, "The tensor minimum function is undefined on empty tensors");

   return SOLID_CALL_REDUCE_ALL(minimum);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(maximum_abs)(int ndims, const size_t *size,
                                      const ptrdiff_t *strides, void *ptr,
                                      void *result, void *buffer, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (solid_empty_tensor(ndims, size))
      SOLID_ERROR(-1, "The tensor maximum absolute function is undefined on empty tensors");

#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(maximum);
#else /* Floating pointer of signed integer */
   return SOLID_CALL_REDUCE_ALL(maximum_abs);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(minimum_abs)(int ndims, const size_t *size,
                                      const ptrdiff_t *strides, void *ptr,
                                      void *result, void *buffer, cudaStream_t stream)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (solid_empty_tensor(ndims, size))
      SOLID_ERROR(-1, "The tensor minumum absolute function is undefined on empty tensors");

#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(minimum);
#else /* Floating pointer of signed integer */
   return SOLID_CALL_REDUCE_ALL(minimum_abs);
#endif
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(norm)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                   void *ptr, double p, void *result,
                                   void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(norm) param;
   solid_funptr_cpu_scalar_copy funptr;
   double pinv = 1 / p;
   double phalf = p / 2;

   /* Determine the copy function */
   #if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   {  funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_WORKTYPE(double))];
   }
   #else
   {  funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_ELEMWORKTYPE(SDXTYPE))];
   }
   #endif

   /* Set the parameters */
   if (funptr == 0) SOLID_ERROR(-1, "Error converting the norm parameters");
   funptr(&phalf, &(param.phalf));
   funptr(&pinv, &(param.pinv));

   /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL_PARAM(norm, &param);
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(norm_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                       void *ptr, double p, void *result,
                                       void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{
   #if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   {  SOLID_KERNEL_PARAM(norm_nan) param;
      solid_funptr_cpu_scalar_copy funptr;
      double pinv = 1 / p;
      double phalf = p / 2;

      /* Set the parameters */
      funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_ELEMWORKTYPE(SDXTYPE))];
      if (funptr == 0) SOLID_ERROR(-1, "Error converting the norm parameters");
      funptr(&phalf, &(param.phalf));
      funptr(&pinv, &(param.pinv));

      /* Call the reduce operation with default parameters */
      return SOLID_CALL_REDUCE_ALL_PARAM(norm_nan, &param);
   }
   #else
   {
      return SOLID_FUNCTION(norm)(ndims, size, strides, ptr, p, result, buffer, stream);
   }
   #endif
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(norm2)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                    void *ptr, void *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_ALL(norm2);
}


/* ------------------------------------------------------------------------ */
SOLID_API int SOLID_FUNCTION(norm2_nan)(int ndims, const size_t *size, const ptrdiff_t *strides,
                                        void *ptr, void *result, void *buffer, cudaStream_t stream)
/* ----------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_ALL(norm2_nan);
#else
   return SOLID_CALL_REDUCE_ALL(norm2);
#endif
}


#endif

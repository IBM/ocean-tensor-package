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
#define SD_TEMPLATE_FILE "core/cpu/template_reduce_axis.c"

#include "solid/base/generic/generate_macros.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/cpu/dtype_cpu.h"
#include "solid/core/cpu/reduce_axis.h"
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
#define SOLID_OP_REDUCE_BREAK  { k2 = relem; break; }  /* Break */
#endif


/* ==================================================================== */
/* Create parameter types and kernel                                    */
/* ==================================================================== */
#define SD_TEMPLATE(TEMPLATE) SOLID_OP_REDUCE_##TEMPLATE
#define SD_TEMPLATE_REDUCE(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, FLAG_FINALIZE, \
                           CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, CODE_FINALIZE) \
   SOLID_CREATE_REDUCE_AXIS(axis_##NAME, DTYPE1, DTYPE2, CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, \
                            FLAG_FINALIZE, CODE_FINALIZE, FLAG_PARAM, PARAM)

/* All types */
SD_TEMPLATE(ANY)
SD_TEMPLATE(ALL)
SD_TEMPLATE(NNZ)
SD_TEMPLATE(SUM)
SD_TEMPLATE(PROD)
SD_TEMPLATE(MINIMUM)
SD_TEMPLATE(MAXIMUM)
SD_TEMPLATE(NORM)
SD_TEMPLATE(NORM2)

/* Floating-point only */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(ALL_FINITE)
SD_TEMPLATE(ANY_INF)
SD_TEMPLATE(ANY_NAN)
SD_TEMPLATE(NNZ_NAN)
SD_TEMPLATE(SUM_NAN)
SD_TEMPLATE(PROD_NAN)
SD_TEMPLATE(SUM_ABS_NAN)
SD_TEMPLATE(NORM_NAN)
SD_TEMPLATE(NORM2_NAN)
#endif

/* Signed types only */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE) || SDTYPE_IS_SIGNED_INT(SDXTYPE))
SD_TEMPLATE(SUM_ABS)
SD_TEMPLATE(MINIMUM_ABS)
SD_TEMPLATE(MAXIMUM_ABS)
#endif

#undef SD_TEMPLATE_REDUCE
#undef SD_TEMPLATE



/* ==================================================================== */
/* Define the API                                                       */
/* ==================================================================== */

#define SD_TEMPLATE(OP) \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(OP)(int ndims, const size_t *size, \
                       const ptrdiff_t *strides1, void *ptr1, \
                       const ptrdiff_t *strides2, void *ptr2, \
                       int rdims, const size_t *rsize, const ptrdiff_t *rstrides) \
/* ------------------------------------------------------------------------------ */ \
{ \
   SOLID_CALL_REDUCE_AXIS(OP); \
   \
   return 0; \
}

/* All types */
SD_TEMPLATE(axis_any        )
SD_TEMPLATE(axis_all        )
SD_TEMPLATE(axis_nnz        )
SD_TEMPLATE(axis_sum        )
SD_TEMPLATE(axis_prod       )
SD_TEMPLATE(axis_norm2      )

/* Floating-point only */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
SD_TEMPLATE(axis_all_finite )
SD_TEMPLATE(axis_any_inf    )
SD_TEMPLATE(axis_any_nan    )
#endif
#undef SD_TEMPLATE



/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_nnz_nan)(int ndims, const size_t *size,
                                 const ptrdiff_t *strides1, void *ptr1,
                                 const ptrdiff_t *strides2, void *ptr2,
                                 int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   SOLID_CALL_REDUCE_AXIS(axis_nnz_nan);
   return 0;
#else
   return SOLID_FUNCTION(axis_nnz)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#endif
}


/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_sum_nan)(int ndims, const size_t *size,
                                 const ptrdiff_t *strides1, void *ptr1,
                                 const ptrdiff_t *strides2, void *ptr2,
                                 int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   SOLID_CALL_REDUCE_AXIS(axis_sum_nan);
   return 0;
#else
   return SOLID_FUNCTION(axis_sum)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#endif
}


/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_prod_nan)(int ndims, const size_t *size,
                                  const ptrdiff_t *strides1, void *ptr1,
                                  const ptrdiff_t *strides2, void *ptr2,
                                  int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   SOLID_CALL_REDUCE_AXIS(axis_prod_nan);
   return 0;
#else
   return SOLID_FUNCTION(axis_prod)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#endif
}


/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_sum_abs)(int ndims, const size_t *size,
                                 const ptrdiff_t *strides1, void *ptr1,
                                 const ptrdiff_t *strides2, void *ptr2,
                                 int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE) || SDTYPE_IS_SIGNED_INT(SDXTYPE))
   SOLID_CALL_REDUCE_AXIS(axis_sum_abs);
   return 0;
#else
   return SOLID_FUNCTION(axis_sum)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#endif
}


/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_sum_abs_nan)(int ndims, const size_t *size,
                                     const ptrdiff_t *strides1, void *ptr1,
                                     const ptrdiff_t *strides2, void *ptr2,
                                     int rdims, const size_t *rsize,  const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   SOLID_CALL_REDUCE_AXIS(axis_sum_abs_nan);
   return 0;
#elif (SDTYPE_IS_SIGNED_INT(SDXTYPE))
   return SOLID_FUNCTION(axis_sum_abs)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#else
   return SOLID_FUNCTION(axis_sum)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#endif
}



#define SD_TEMPLATE(OP, DESC) \
/* ------------------------------------------------------------------------------ */ \
int SOLID_FUNCTION(OP)(int ndims, const size_t *size, \
                       const ptrdiff_t *strides1, void *ptr1, \
                       const ptrdiff_t *strides2, void *ptr2, \
                       int rdims, const size_t *rsize, const ptrdiff_t *rstrides) \
/* ------------------------------------------------------------------------------ */ \
{  size_t relem; \
   int i; \
   \
   /* Make sure the reduction size is not zero */ \
   for (i = 0, relem = 1; i < rdims; i++) relem *= rsize[i]; \
   if (relem == 0) \
      SOLID_ERROR(-1, "The number of reduction elements in "#DESC" cannot be zero"); \
   \
   SOLID_CALL_REDUCE_AXIS(OP); \
   \
   return 0; \
}

SD_TEMPLATE(axis_minimum, "axis minimum")
SD_TEMPLATE(axis_maximum, "axis maximum")
#undef SD_TEMPLATE



/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_minimum_abs)(int ndims, const size_t *size,
                                     const ptrdiff_t *strides1, void *ptr1,
                                     const ptrdiff_t *strides2, void *ptr2,
                                     int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
   #if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE) || SDTYPE_IS_SIGNED_INT(SDXTYPE))
   {  size_t relem;
      int i;

     /* Make sure the reduction size is not zero */
     for (i = 0, relem = 1; i < rdims; i++) relem *= rsize[i];
     if (relem == 0)
         SOLID_ERROR(-1, "The number of reduction elements in axis minimum absolute cannot be zero");

     SOLID_CALL_REDUCE_AXIS(axis_minimum_abs);

     return 0;
   }
   #else
   {
      return SOLID_FUNCTION(axis_minimum)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
   }
   #endif
}


/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_maximum_abs)(int ndims, const size_t *size,
                                     const ptrdiff_t *strides1, void *ptr1,
                                     const ptrdiff_t *strides2, void *ptr2,
                                     int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
   #if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE) || SDTYPE_IS_SIGNED_INT(SDXTYPE))
   {  size_t relem;
      int i;

     /* Make sure the reduction size is not zero */
     for (i = 0, relem = 1; i < rdims; i++) relem *= rsize[i];
     if (relem == 0)
         SOLID_ERROR(-1, "The number of reduction elements in axis maximum absolute cannot be zero");

     SOLID_CALL_REDUCE_AXIS(axis_maximum_abs);

     return 0;
   }
   #else
   {
      return SOLID_FUNCTION(axis_maximum)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
   }
   #endif
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(axis_norm)(int ndims, const size_t *size, double p,
                             const ptrdiff_t *strides1, void *ptr1,
                             const ptrdiff_t *strides2, void *ptr2,
                             int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* -------------------------------------------------------------------- */
{  SOLID_PARAM(axis_norm) param;
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
   SOLID_CALL_REDUCE_AXIS_PARAM(axis_norm, &param);

   return 0;
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(axis_norm_nan)(int ndims, const size_t *size, double p,
                                  const ptrdiff_t *strides1, void *ptr1,
                                  const ptrdiff_t *strides2, void *ptr2,
                                  int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* -------------------------------------------------------------------- */
{
   #if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   {  SOLID_PARAM(axis_norm_nan) param;
      double pinv = 1 / p;
      double phalf = p / 2;

      /* Set the parameters */
      solid_funptr_cpu_scalar_copy funptr;
      funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_ELEMWORKTYPE(SDXTYPE))];
      if (funptr == 0) SOLID_ERROR(-1, "Error converting norm parameter");
      funptr(&phalf, &(param.phalf));
      funptr(&pinv, &(param.pinv));

      /* Apply the reduction */
      SOLID_CALL_REDUCE_AXIS_PARAM(axis_norm_nan, &param);
      return 0;
   }
   #else
   {
      return SOLID_FUNCTION(axis_norm)(ndims, size, p, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
   }
   #endif
}


/* ------------------------------------------------------------------------------ */
int SOLID_FUNCTION(axis_norm2_nan)(int ndims, const size_t *size,
                                   const ptrdiff_t *strides1, void *ptr1,
                                   const ptrdiff_t *strides2, void *ptr2,
                                   int rdims, const size_t *rsize, const ptrdiff_t *rstrides)
/* ------------------------------------------------------------------------------ */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   SOLID_CALL_REDUCE_AXIS(axis_norm2_nan);
   return 0;
#else
   return SOLID_FUNCTION(axis_norm2)(ndims, size, strides1, ptr1, strides2, ptr2, rdims, rsize, rstrides);
#endif
}

#endif

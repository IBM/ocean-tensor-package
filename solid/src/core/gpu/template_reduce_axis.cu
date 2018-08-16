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
#define SD_TEMPLATE_FILE "core/gpu/template_reduce_axis.cu"

#include "solid.h"
#include "solid_gpu.h"
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/generic/dtype_assign.h"
#include "solid/base/gpu/dtype_gpu.h"
#include "solid/core/gpu/reduce_axis.h"

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
   SOLID_CREATE_REDUCE_AXIS(axis_##NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, CODE_INIT, \
                            CODE_ACCUMULATE, CODE_REDUCE, FLAG_FINALIZE, CODE_FINALIZE)

SD_TEMPLATE(ANY)
SD_TEMPLATE(ALL)
SD_TEMPLATE(NNZ)
SD_TEMPLATE(SUM)
SD_TEMPLATE(PROD)
SD_TEMPLATE(MAXIMUM)
SD_TEMPLATE(MINIMUM)
SD_TEMPLATE(NORM)
SD_TEMPLATE(NORM2)

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

/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_any)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_any);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_all)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_all);
}


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_all_finite)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_all_finite);
}
#endif


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_any_inf)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_any_inf);
}
#endif


#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_any_nan)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_any_nan);
}
#endif


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_nnz)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_nnz);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_nnz_nan)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_nnz_nan);
#else
   return SOLID_FUNCTION(axis_nnz)(config);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_sum)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_sum);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_prod)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_prod);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_sum_nan)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_sum_nan);
#else
   return SOLID_FUNCTION(axis_sum)(config);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_prod_nan)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_prod_nan);
#else
   return SOLID_FUNCTION(axis_prod)(config);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_sum_abs)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_sum);
#else /* Floating point or signed integer */
   return SOLID_CALL_REDUCE_AXIS(axis_sum_abs);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_sum_abs_nan)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_sum);
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   return SOLID_CALL_REDUCE_AXIS(axis_sum_abs);
#else /* Floating point */
   return SOLID_CALL_REDUCE_AXIS(axis_sum_abs_nan);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_maximum)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (config -> data.relem == 0)
      SOLID_ERROR(-1, "The tensor maximum function is undefined on empty data");

   return SOLID_CALL_REDUCE_AXIS(axis_maximum);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_minimum)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (config -> data.relem == 0)
      SOLID_ERROR(-1, "The tensor minimum function is undefined on empty data");

   return SOLID_CALL_REDUCE_AXIS(axis_minimum);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_maximum_abs)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (config -> data.relem == 0)
      SOLID_ERROR(-1, "The tensor maximum absolute function is undefined on empty data");

#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_maximum);
#else /* Floating pointer of signed integer */
   return SOLID_CALL_REDUCE_AXIS(axis_maximum_abs);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_minimum_abs)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{
   /* Make sure the tensor is not empty */
   if (config -> data.relem == 0)
      SOLID_ERROR(-1, "The tensor minimum absolute function is undefined on empty data");

#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_minimum);
#else /* Floating pointer of signed integer */
   return SOLID_CALL_REDUCE_AXIS(axis_minimum_abs);
#endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_norm)(solid_gpu_reduce_axis_config *config, double p)
/* -------------------------------------------------------------------- */
{  SOLID_KERNEL_PARAM(axis_norm) param;
   double pinv = 1 / p;
   double phalf = p / 2;

   /* Set the parameters */
   #if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   {  /* Parameters have type double */
      param.phalf = phalf;
      param.pinv  = pinv;
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
   return SOLID_CALL_REDUCE_AXIS_PARAM(axis_norm, &param);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_norm_nan)(solid_gpu_reduce_axis_config *config, double p)
/* -------------------------------------------------------------------- */
{
   #if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   {  SOLID_KERNEL_PARAM(axis_norm_nan) param;
      solid_funptr_cpu_scalar_copy funptr;
      double pinv = 1 / p;
      double phalf = p / 2;

      /* Set the parameters */
      funptr = solid_cpu_scalar_copy[SDTYPE_INDEX(double)][SDTYPE_INDEX(SOLID_ELEMWORKTYPE(SDXTYPE))];
      if (funptr == 0) SOLID_ERROR(-1, "Error converting norm parameter");
      funptr(&phalf, &(param.phalf));
      funptr(&pinv, &(param.pinv));

      /* Apply the reduction */
      return SOLID_CALL_REDUCE_AXIS_PARAM(axis_norm_nan, &param);
   }
   #else
   {
      return SOLID_FUNCTION(axis_norm)(config, p);
   }
   #endif
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_norm2)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
   return SOLID_CALL_REDUCE_AXIS(axis_norm2);
}


/* -------------------------------------------------------------------- */
SOLID_API int SOLID_FUNCTION(axis_norm2_nan)(solid_gpu_reduce_axis_config *config)
/* -------------------------------------------------------------------- */
{  /* Call the reduce operation with default parameters */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
   return SOLID_CALL_REDUCE_AXIS(axis_norm2_nan);
#else /* Integer or Boolean */
   return SOLID_CALL_REDUCE_AXIS(axis_norm2);
#endif
}


#endif

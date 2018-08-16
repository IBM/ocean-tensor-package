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

#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/cpu/op/tensor_reduce_all_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorCPU_any       (OcTensor *tensor, int *result);
OC_API int OcTensorCPU_all       (OcTensor *tensor, int *result);
OC_API int OcTensorCPU_allFinite (OcTensor *tensor, int *result);
OC_API int OcTensorCPU_anyInf    (OcTensor *tensor, int *result);
OC_API int OcTensorCPU_anyNaN    (OcTensor *tensor, int *result);
OC_API int OcTensorCPU_nnz       (OcTensor *tensor, OcUInt64 *result);
OC_API int OcTensorCPU_nnzNaN    (OcTensor *tensor, OcUInt64 *result);
OC_API int OcTensorCPU_allLT     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorCPU_allLE     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorCPU_allGT     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorCPU_allGE     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorCPU_allGTLT   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorCPU_allGTLE   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorCPU_allGELT   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorCPU_allGELE   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorCPU_sum       (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_prod      (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_sumNaN    (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_prodNaN   (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_sumAbs    (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_sumAbsNaN (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_maximum   (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_minimum   (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_maximumAbs(OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_minimumAbs(OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_norm2     (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_norm2NaN  (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorCPU_norm      (OcTensor *tensor, double p, OcScalar *result);
OC_API int OcTensorCPU_normNaN   (OcTensor *tensor, double p, OcScalar *result);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeReduceAllOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Reduction operations */
   module -> Tensor_any        = OcTensorCPU_any;
   module -> Tensor_all        = OcTensorCPU_all;
   module -> Tensor_allFinite  = OcTensorCPU_allFinite;
   module -> Tensor_anyInf     = OcTensorCPU_anyInf;
   module -> Tensor_anyNaN     = OcTensorCPU_anyNaN;
   module -> Tensor_nnz        = OcTensorCPU_nnz;
   module -> Tensor_nnzNaN     = OcTensorCPU_nnzNaN;

   module -> Tensor_allLT      = OcTensorCPU_allLT;
   module -> Tensor_allLE      = OcTensorCPU_allLE;
   module -> Tensor_allGT      = OcTensorCPU_allGT;
   module -> Tensor_allGE      = OcTensorCPU_allGE;
   module -> Tensor_allGTLT    = OcTensorCPU_allGTLT;
   module -> Tensor_allGTLE    = OcTensorCPU_allGTLE;
   module -> Tensor_allGELT    = OcTensorCPU_allGELT;
   module -> Tensor_allGELE    = OcTensorCPU_allGELE;

   module -> Tensor_sum        = OcTensorCPU_sum;
   module -> Tensor_prod       = OcTensorCPU_prod;
   module -> Tensor_sumNaN     = OcTensorCPU_sumNaN;
   module -> Tensor_prodNaN    = OcTensorCPU_prodNaN;
   module -> Tensor_sumAbs     = OcTensorCPU_sumAbs;
   module -> Tensor_sumAbsNaN  = OcTensorCPU_sumAbsNaN;

   module -> Tensor_maximum    = OcTensorCPU_maximum;
   module -> Tensor_minimum    = OcTensorCPU_minimum;
   module -> Tensor_maximumAbs = OcTensorCPU_maximumAbs;
   module -> Tensor_minimumAbs = OcTensorCPU_minimumAbs;

   module -> Tensor_norm       = OcTensorCPU_norm;
   module -> Tensor_normNaN    = OcTensorCPU_normNaN;
   module -> Tensor_norm2      = OcTensorCPU_norm2;
   module -> Tensor_norm2NaN   = OcTensorCPU_norm2NaN;
}


/* ===================================================================== */
/* Function implementations - global reductions                          */
/* ===================================================================== */

/* ------------------------------------------------------------------------------ */
/* int OcTensorCPU_any      (OcTensor *tensor, int *result)                       */
/* int OcTensorCPU_all      (OcTensor *tensor, int *result)                       */
/* int OcTensorCPU_allFinite(OcTensor *tensor, int *result)                       */
/* int OcTensorCPU_anyInf   (OcTensor *tensor, int *result)                       */
/* int OcTensorCPU_anyNaN   (OcTensor *tensor, int *result)                       */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *tensor, int *result) \
{  OcSolidReduce               config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   solid_bool                  value; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(#DESC, solid_cpu_##SOLID_OP, funptr, tensor -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - remove replications */ \
   OcSolid_analyzeReduce(&config, tensor, 1); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, config.strides, config.ptr, &value); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (int)value; \
   \
   return status; \
}

OC_TEMPLATE(any,       any,        "any"      )
OC_TEMPLATE(all,       all,        "all"      )
OC_TEMPLATE(allFinite, all_finite, "allFinite") /* Only needs to be implemented for floating-point types */
OC_TEMPLATE(anyInf,    any_inf,    "anyInf"   ) /* Only needs to be implemented for floating-point types */
OC_TEMPLATE(anyNaN,    any_nan,    "anyNaN"   ) /* Only needs to be implemented for floating-point types */
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------ */
/* int OcTensorCPU_allLT(OcTensor *tensor, OcScalar *bound, int *result)          */
/* int OcTensorCPU_allLE(OcTensor *tensor, OcScalar *bound, int *result)          */
/* int OcTensorCPU_allGT(OcTensor *tensor, OcScalar *bound, int *result)          */
/* int OcTensorCPU_allGE(OcTensor *tensor, OcScalar *bound, int *result)          */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *tensor, OcScalar *bound, int *result) \
{  OcSolidReduce               config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   solid_scalar                bound_solid; \
   solid_bool                  result_solid; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(#DESC, solid_cpu_##SOLID_OP, funptr, tensor -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - remove replications */ \
   OcSolid_analyzeReduce(&config, tensor, 1); \
   \
   /* Convert the scalar */ \
   if (OcSolid_getScalar(bound, &bound_solid) != 0) return -1; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, config.strides, config.ptr, &bound_solid, &result_solid); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (int)result_solid; \
   \
   return status; \
}

OC_TEMPLATE(allLT, all_lt, "less than")
OC_TEMPLATE(allLE, all_le, "less than or equal")
OC_TEMPLATE(allGT, all_gt, "greater than")
OC_TEMPLATE(allGE, all_ge, "greater than or equal")
#undef OC_TEMPLATE


/* ---------------------------------------------------------------------------------------- */
/* int OcTensorCPU_allGTLT(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* int OcTensorCPU_allGTLE(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* int OcTensorCPU_allGELT(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* int OcTensorCPU_allGELE(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* ---------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) \
{  OcSolidReduce               config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   solid_scalar                bound_lower; \
   solid_scalar                bound_upper; \
   solid_bool                  result_solid; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(#DESC, solid_cpu_##SOLID_OP, funptr, tensor -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - remove replications */ \
   OcSolid_analyzeReduce(&config, tensor, 1); \
   \
   /* Convert the scalars */ \
   if (OcSolid_getScalar(lower, &bound_lower) != 0) return -1; \
   if (OcSolid_getScalar(upper, &bound_upper) != 0) return -1; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, config.strides, config.ptr, &bound_lower, &bound_upper, &result_solid); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (int)result_solid; \
   \
   return status; \
}

OC_TEMPLATE(allGTLT, all_gtlt, "in range (lower,upper)")
OC_TEMPLATE(allGTLE, all_gtle, "in range (lower,upper]")
OC_TEMPLATE(allGELT, all_gelt, "in range [lower,upper)")
OC_TEMPLATE(allGELE, all_gele, "in range [lower,upper]")
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------ */
/* int OcTensorCPU_nnz   (OcTensor *tensor, OcUInt64 *result)                     */
/* int OcTensorCPU_nnzNaN(OcTensor *tensor, OcUInt64 *result)                     */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *tensor, OcUInt64 *result) \
{  OcSolidReduce         config; \
   solid_funptr_cpu_nnz  funptr; \
   solid_uint64          value; \
   int                   status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, tensor -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcSolid_analyzeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, config.strides, config.ptr, &value); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (OcUInt64)value; \
   \
   return status; \
}

OC_TEMPLATE(nnz,    nnz,     "nnz"   )
OC_TEMPLATE(nnzNaN, nnz_nan, "nnzNaN")
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------ */
/* int OcTensorCPU_sum       (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_prod      (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_sumNaN    (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_prodNaN   (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_sumAbs    (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_sumAbsNaN (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_maximum   (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_minimum   (OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_maximumAbs(OcTensor *tensor, OcScalar *result)                 */
/* int OcTensorCPU_minimumAbs(OcTensor *tensor, OcScalar *result)                 */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *tensor, OcScalar *result) \
{  OcSolidReduce               config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(#DESC, solid_cpu_##SOLID_OP, funptr, tensor -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcSolid_analyzeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, config.strides, config.ptr, (void *)&(result->value)); \
   \
   return status; \
}

OC_TEMPLATE(sum,        sum,         "sum"       )
OC_TEMPLATE(prod,       prod,        "prod"      )
OC_TEMPLATE(sumNaN,     sum_nan,     "sumNaN"    )
OC_TEMPLATE(prodNaN,    prod_nan,    "prodNaN"   )
OC_TEMPLATE(sumAbs,     sum_abs,     "sumAbs"    )
OC_TEMPLATE(sumAbsNaN,  sum_abs_nan, "sumAbsNaN" )
OC_TEMPLATE(maximum,    maximum,     "maximum"   )
OC_TEMPLATE(minimum,    minimum,     "minimum"   )
OC_TEMPLATE(maximumAbs, maximum_abs, "maximumAbs")
OC_TEMPLATE(minimumAbs, minimum_abs, "minimumAbs")
OC_TEMPLATE(norm2,      norm2,       "norm2"     )
OC_TEMPLATE(norm2NaN,   norm2_nan,   "norm2NaN"  )
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------ */
/* int OcTensorCPU_norm   (OcTensor *tensor, double p, OcScalar *result)          */
/* int OcTensorCPU_normNaN(OcTensor *tensor, double p, OcScalar *result)          */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *tensor, double p, OcScalar *result) \
{  OcSolidReduce               config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, tensor -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcSolid_analyzeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, config.strides, config.ptr, p, &(result -> value)); \
   \
   return status; \
}

OC_TEMPLATE(norm,    norm,     "norm"   )
OC_TEMPLATE(normNaN, norm_nan, "normNaN")
#undef OC_TEMPLATE

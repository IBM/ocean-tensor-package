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
#include "ocean/core/cpu/op/tensor_reduce_axis_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorCPU_axisAny       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisAll       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisAllFinite (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisAnyInf    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisAnyNaN    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisNnz       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisNnzNaN    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisSum       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisProd      (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisSumNaN    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisProdNaN   (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisSumAbs    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisSumAbsNaN (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisMinimum   (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisMaximum   (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisMinimumAbs(OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisMaximumAbs(OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisNorm2     (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisNorm2NaN  (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisNorm      (OcTensor *src, double p, int n, int *axes, OcTensor *dst);
OC_API int OcTensorCPU_axisNormNaN   (OcTensor *src, double p, int n, int *axes, OcTensor *dst);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeReduceAxisOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Axis-wise reductions */
   module -> Tensor_axisAny        = OcTensorCPU_axisAny;
   module -> Tensor_axisAll        = OcTensorCPU_axisAll;
   module -> Tensor_axisAllFinite  = OcTensorCPU_axisAllFinite;
   module -> Tensor_axisAnyInf     = OcTensorCPU_axisAnyInf;
   module -> Tensor_axisAnyNaN     = OcTensorCPU_axisAnyNaN;
   module -> Tensor_axisNnz        = OcTensorCPU_axisNnz;
   module -> Tensor_axisNnzNaN     = OcTensorCPU_axisNnzNaN;

   module -> Tensor_axisSum        = OcTensorCPU_axisSum;
   module -> Tensor_axisProd       = OcTensorCPU_axisProd;
   module -> Tensor_axisSumNaN     = OcTensorCPU_axisSumNaN;
   module -> Tensor_axisProdNaN    = OcTensorCPU_axisProdNaN;
   module -> Tensor_axisSumAbs     = OcTensorCPU_axisSumAbs;
   module -> Tensor_axisSumAbsNaN  = OcTensorCPU_axisSumAbsNaN;

   module -> Tensor_axisMinimum    = OcTensorCPU_axisMinimum;
   module -> Tensor_axisMaximum    = OcTensorCPU_axisMaximum;
   module -> Tensor_axisMinimumAbs = OcTensorCPU_axisMinimumAbs;
   module -> Tensor_axisMaximumAbs = OcTensorCPU_axisMaximumAbs;

   module -> Tensor_axisNorm       = OcTensorCPU_axisNorm;
   module -> Tensor_axisNormNaN    = OcTensorCPU_axisNormNaN;
   module -> Tensor_axisNorm2      = OcTensorCPU_axisNorm2;
   module -> Tensor_axisNorm2NaN   = OcTensorCPU_axisNorm2NaN;
}


/* ===================================================================== */
/* Function implementations - axis reductions                            */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
/* int OcTensorCPU_OP(OcTensor *src, int n, int *axes, OcTensor *dst)    */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC, REMOVE_REPEATS) \
int OcTensorCPU_##OP(OcTensor *src, int n, int *axes, OcTensor *dst) \
{  OcSolidReduceAxis           config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(#DESC, solid_cpu_##SOLID_OP, funptr, src -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation  */ \
   OcSolid_analyzeReduceAxis(&config, src, dst, n, axes, REMOVE_REPEATS); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, \
                 config.strides1, config.ptr1, config.strides2, config.ptr2, \
                 config.rdims, config.rsize, config.rstrides); \
   \
   return status; \
}

OC_TEMPLATE(axisAny,        axis_any,         "axisAny",        1)
OC_TEMPLATE(axisAll,        axis_all,         "axisAll",        1)
OC_TEMPLATE(axisAllFinite,  axis_all_finite,  "axisAllFinite",  1)
OC_TEMPLATE(axisAnyInf,     axis_any_inf,     "axisAnyInf",     1)
OC_TEMPLATE(axisAnyNaN,     axis_any_nan,     "axisAnyNaN",     1)
OC_TEMPLATE(axisNnz,        axis_nnz,         "axisNnz",        0)
OC_TEMPLATE(axisNnzNaN,     axis_nnz_nan,     "axisNnzNaN",     0)
OC_TEMPLATE(axisSum,        axis_sum,         "axisSum",        0)
OC_TEMPLATE(axisProd,       axis_prod,        "axisProd",       0)
OC_TEMPLATE(axisSumNaN,     axis_sum_nan,     "axisSumNaN",     0)
OC_TEMPLATE(axisProdNaN,    axis_prod_nan,    "axisProdNaN",    0)
OC_TEMPLATE(axisSumAbs,     axis_sum_abs,     "axisSumAbs",     0)
OC_TEMPLATE(axisSumAbsNaN,  axis_sum_abs_nan, "axisSumAbsNaN",  0)
OC_TEMPLATE(axisMinimum,    axis_minimum,     "axisMinimum",    1)
OC_TEMPLATE(axisMaximum,    axis_maximum,     "axisMaximum",    1)
OC_TEMPLATE(axisMinimumAbs, axis_minimum_abs, "axisMinimumAbs", 1)
OC_TEMPLATE(axisMaximumAbs, axis_maximum_abs, "axisMaximumAbs", 1)
OC_TEMPLATE(axisNorm2,      axis_norm2,       "axisNorm2",      0)
OC_TEMPLATE(axisNorm2NaN,   axis_norm2_nan,   "axisNorm2NaN",   0)
#undef OC_TEMPLATE


/* ------------------------------------------------------------------------------------- */
/* int OcTensorCPU_axisNorm   (OcTensor *src, double p, int n, int *axes, OcTensor *dst) */
/* int OcTensorCPU_axisNormNaN(OcTensor *src, double p, int n, int *axes, OcTensor *dst) */
/* ------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorCPU_##OP(OcTensor *src, double p, int n, int *axes, OcTensor *dst) \
{  OcSolidReduceAxis           config; \
   solid_funptr_cpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, src -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation  */ \
   OcSolid_analyzeReduceAxis(&config, src, dst, n, axes, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, config.ndims, config.size, p, \
                 config.strides1, config.ptr1, config.strides2, config.ptr2, \
                 config.rdims, config.rsize, config.rstrides); \
   \
   return status; \
}

OC_TEMPLATE(axisNorm,    axis_norm,     "axisNorm"   )
OC_TEMPLATE(axisNormNaN, axis_norm_nan, "axisNormNaN")
#undef OC_TEMPLATE

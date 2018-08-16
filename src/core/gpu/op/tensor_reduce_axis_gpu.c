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

#include "ocean/core/gpu/op/tensor_reduce_axis_gpu.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/core/gpu/cuda.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_gpu.h"
#include "solid_core_gpu.h"


/* ===================================================================== */
/* Local data types                                                      */
/* ===================================================================== */
typedef struct
{  solid_gpu_reduce_axis_config solidConfig;
   OcSolidReduceAxis            data;
   OcStorage                   *buffer;
   OcTensor                    *src;
   OcTensor                    *dst;
} OcTensorGPU_ReduceAxis;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_axisAny       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisAll       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisAllFinite (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisAnyInf    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisAnyNaN    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisNnz       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisNnzNaN    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisSum       (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisProd      (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisSumNaN    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisProdNaN   (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisSumAbs    (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisSumAbsNaN (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisMinimum   (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisMaximum   (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisMinimumAbs(OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisMaximumAbs(OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisNorm2     (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisNorm2NaN  (OcTensor *src, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisNorm      (OcTensor *src, double p, int n, int *axes, OcTensor *dst);
OC_API int OcTensorGPU_axisNormNaN   (OcTensor *src, double p, int n, int *axes, OcTensor *dst);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeReduceAxisOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Axis-wise reductions */
   module -> Tensor_axisAny        = OcTensorGPU_axisAny;
   module -> Tensor_axisAll        = OcTensorGPU_axisAll;
   module -> Tensor_axisAllFinite  = OcTensorGPU_axisAllFinite;
   module -> Tensor_axisAnyInf     = OcTensorGPU_axisAnyInf;
   module -> Tensor_axisAnyNaN     = OcTensorGPU_axisAnyNaN;
   module -> Tensor_axisNnz        = OcTensorGPU_axisNnz;
   module -> Tensor_axisNnzNaN     = OcTensorGPU_axisNnzNaN;

   module -> Tensor_axisSum        = OcTensorGPU_axisSum;
   module -> Tensor_axisProd       = OcTensorGPU_axisProd;
   module -> Tensor_axisSumNaN     = OcTensorGPU_axisSumNaN;
   module -> Tensor_axisProdNaN    = OcTensorGPU_axisProdNaN;
   module -> Tensor_axisSumAbs     = OcTensorGPU_axisSumAbs;
   module -> Tensor_axisSumAbsNaN  = OcTensorGPU_axisSumAbsNaN;

   module -> Tensor_axisMinimum    = OcTensorGPU_axisMinimum;
   module -> Tensor_axisMaximum    = OcTensorGPU_axisMaximum;
   module -> Tensor_axisMinimumAbs = OcTensorGPU_axisMinimumAbs;
   module -> Tensor_axisMaximumAbs = OcTensorGPU_axisMaximumAbs;

   module -> Tensor_axisNorm       = OcTensorGPU_axisNorm;
   module -> Tensor_axisNormNaN    = OcTensorGPU_axisNormNaN;
   module -> Tensor_axisNorm2      = OcTensorGPU_axisNorm2;
   module -> Tensor_axisNorm2NaN   = OcTensorGPU_axisNorm2NaN;
}


/* ===================================================================== */
/* Internal functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorGPU_initializeReduceAxis(OcTensorGPU_ReduceAxis *config,
                                     OcTensor *src, OcTensor *dst,
                                     int n, int *axes, int flagRemoveRepeats)
/* -------------------------------------------------------------------- */
{  OcSolidReduceAxis *data = &(config -> data);
   int    solidTypeIndex, solidElemsize;
   size_t bufferSize;

   /* Initialize */
   config -> buffer     = NULL;
   config -> src        = NULL;
   config -> dst        = NULL;

   /* Actvate the device */
   if (OcCuda_setDevice(dst -> device -> index) != 0) return -1;

   /* Analyze the operation - generic */
   OcSolid_analyzeReduceAxis(data, src, dst, n, axes, flagRemoveRepeats);

   /* Analyze the operation - Solid core GPU */
   if ((solidTypeIndex = OcSolid_getType(dst -> dtype)) == -1) return -1;
   solidElemsize = solid_gpu_worktype_size[solidTypeIndex];
   if (solid_gpu_reduce_axis_prepare(&(config -> solidConfig), solidElemsize, data -> ndims,
                                     data -> size, data -> strides1, data -> ptr1,
                                     data -> strides2, data -> ptr2, data -> rdims,
                                     data -> rsize, data -> rstrides,
                                     OcTensorGPU_cudaStream(dst)) != 0) return -1;

   /* Set the tensor pointers (borrowed reference) */
   config -> src = src;
   config -> dst = dst;

   /* Synchronize */
   if (OcTensor_startRead(config -> dst, config -> src) != 0) return -1;

   /* Allocate the intermediate buffer */
   bufferSize = config -> solidConfig.bufferSize;
   if (bufferSize > 0)
   {  config -> buffer = OcStorage_createTemporary(bufferSize, OcDTypeInt8, dst -> device);
      if (config -> buffer == NULL) return -1;

      /* Add a write dependency to the buffer */
      if (OcStorage_startWrite(config -> dst -> storage, config -> buffer) != 0)
      {  OcDecrefStorage(config -> buffer); 
         config -> buffer = NULL;
         return -1;
      }

      /* Set the buffer data */
      config -> solidConfig.data.ptrBuffer = config -> buffer -> data;
   }
   else
   {  config -> buffer = NULL;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_finalizeReduceAxis(OcTensorGPU_ReduceAxis *config, int result)
/* -------------------------------------------------------------------- */
{  
   /* Record event to indicate completion of the action */
   if (OcTensor_update(config -> dst) != 0) result = -1;

   /* Synchronization and finalization */
   if (config -> buffer)
   {  if (OcStorage_finishWrite(config -> dst -> storage, config -> buffer) != 0) result = -1;
      OcDecrefStorage(config -> buffer);
   }

   /* Synchronization with the source */
   if (OcTensor_finishRead(config -> dst, config -> src) != 0) result = -1; \

   return result;
}


/* ===================================================================== */
/* Function implementations - axis reductions                            */
/* ===================================================================== */

/* ------------------------------------------------------------------------------ */
/* int OcTensorGPU_axisAny       (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisAll       (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisAllFinite (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisAnyInf    (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisAnyNaN    (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisNnz       (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisNnzNaN    (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisSum       (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisProd      (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisSumNaN    (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisProdNaN   (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisSumAbs    (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisSumAbsNaN (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisMaximum   (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisMinimum   (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisMaximumAbs(OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisMinimumAbs(OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisNorm2     (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisNorm2NaN  (OcTensor *src, int n, int *axes, OcTensor *dst) */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *src, int n, int *axes, OcTensor *dst) \
{  OcTensorGPU_ReduceAxis      config; \
   solid_funptr_gpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, src -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Initialize the reduction */ \
   if (OcTensorGPU_initializeReduceAxis(&config, src, dst, n, axes, 0) != 0) return -1; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, &(config.solidConfig)); \
   \
   /* Finalize the reduction */ \
   return OcTensorGPU_finalizeReduceAxis(&config, status); \
}

OC_TEMPLATE(axisAny,        axis_any,         "axisAny"       )
OC_TEMPLATE(axisAll,        axis_all,         "axisAll"       )
OC_TEMPLATE(axisAllFinite,  axis_all_finite,  "axisAllFinite" )
OC_TEMPLATE(axisAnyInf,     axis_any_inf,     "axisAnyInf"    )
OC_TEMPLATE(axisAnyNaN,     axis_any_nan,     "axisAnyNaN"    )
OC_TEMPLATE(axisNnz,        axis_nnz,         "axisNnz"       )
OC_TEMPLATE(axisNnzNaN,     axis_nnz_nan,     "axisNnzNaN"    )
OC_TEMPLATE(axisSum,        axis_sum,         "axisSum"       )
OC_TEMPLATE(axisProd,       axis_prod,        "axisProd"      )
OC_TEMPLATE(axisSumNaN,     axis_sum_nan,     "axisSumNaN"    )
OC_TEMPLATE(axisProdNaN,    axis_prod_nan,    "axisProdNaN"   )
OC_TEMPLATE(axisSumAbs,     axis_sum_abs,     "axisSumAbs"    )
OC_TEMPLATE(axisSumAbsNaN,  axis_sum_abs_nan, "axisSumAbsNaN" )
OC_TEMPLATE(axisMaximum,    axis_maximum,     "axisMaximum"   )
OC_TEMPLATE(axisMinimum,    axis_minimum,     "axisMinimum"   )
OC_TEMPLATE(axisMaximumAbs, axis_maximum_abs, "axisMaximumAbs")
OC_TEMPLATE(axisMinimumAbs, axis_minimum_abs, "axisMinimumAbs")
OC_TEMPLATE(axisNorm2,      axis_norm2,       "axisNorm2"     )
OC_TEMPLATE(axisNorm2NaN,   axis_norm2_nan,   "axisNorm2NaN"  )
#undef OC_TEMPLATE



/* ------------------------------------------------------------------------------------- */
/* int OcTensorGPU_axisNorm   (OcTensor *src, double p, int n, int *axes, OcTensor *dst) */
/* int OcTensorGPU_axisNormNaN(OcTensor *src, double p, int n, int *axes, OcTensor *dst) */
/* ------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *src, double p, int n, int *axes, OcTensor *dst) \
{  OcTensorGPU_ReduceAxis      config; \
   solid_funptr_gpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, src -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Initialize the reduction */ \
   if (OcTensorGPU_initializeReduceAxis(&config, src, dst, n, axes, 0) != 0) return -1; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, &(config.solidConfig), p); \
   \
   /* Finalize the reduction */ \
   return OcTensorGPU_finalizeReduceAxis(&config, status); \
}

OC_TEMPLATE(axisNorm,    axis_norm,     "axisNorm"   )
OC_TEMPLATE(axisNormNaN, axis_norm_nan, "axisNormNaN")
#undef OC_TEMPLATE

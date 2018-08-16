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

#include "ocean/core/gpu/op/tensor_reduce_all_gpu.h"
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
{  OcSolidReduce  data;
   OcStorage     *buffer;
   OcTensor      *tensor;
   void          *bufferData;
} OcTensorGPU_ReduceAll;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_any       (OcTensor *tensor, int *result);
OC_API int OcTensorGPU_all       (OcTensor *tensor, int *result);
OC_API int OcTensorGPU_allFinite (OcTensor *tensor, int *result);
OC_API int OcTensorGPU_anyInf    (OcTensor *tensor, int *result);
OC_API int OcTensorGPU_anyNaN    (OcTensor *tensor, int *result);
OC_API int OcTensorGPU_nnz       (OcTensor *tensor, OcUInt64 *result);
OC_API int OcTensorGPU_nnzNaN    (OcTensor *tensor, OcUInt64 *result);
OC_API int OcTensorGPU_allLT     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorGPU_allLE     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorGPU_allGT     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorGPU_allGE     (OcTensor *tensor, OcScalar *bound, int *result);
OC_API int OcTensorGPU_allGTLT   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorGPU_allGTLE   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorGPU_allGELT   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorGPU_allGELE   (OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result);
OC_API int OcTensorGPU_sum       (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_prod      (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_sumNaN    (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_prodNaN   (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_sumAbs    (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_sumAbsNaN (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_maximum   (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_minimum   (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_maximumAbs(OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_minimumAbs(OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_norm2     (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_norm2NaN  (OcTensor *tensor, OcScalar *result);
OC_API int OcTensorGPU_norm      (OcTensor *tensor, double p, OcScalar *result);
OC_API int OcTensorGPU_normNaN   (OcTensor *tensor, double p, OcScalar *result);



/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeReduceAllOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Reduction operations */
   module -> Tensor_any        = OcTensorGPU_any;
   module -> Tensor_all        = OcTensorGPU_all;
   module -> Tensor_allFinite  = OcTensorGPU_allFinite;
   module -> Tensor_anyInf     = OcTensorGPU_anyInf;
   module -> Tensor_anyNaN     = OcTensorGPU_anyNaN;
   module -> Tensor_nnz        = OcTensorGPU_nnz;
   module -> Tensor_nnzNaN     = OcTensorGPU_nnzNaN;

   module -> Tensor_allLT      = OcTensorGPU_allLT;
   module -> Tensor_allLE      = OcTensorGPU_allLE;
   module -> Tensor_allGT      = OcTensorGPU_allGT;
   module -> Tensor_allGE      = OcTensorGPU_allGE;
   module -> Tensor_allGTLT    = OcTensorGPU_allGTLT;
   module -> Tensor_allGTLE    = OcTensorGPU_allGTLE;
   module -> Tensor_allGELT    = OcTensorGPU_allGELT;
   module -> Tensor_allGELE    = OcTensorGPU_allGELE;

   module -> Tensor_sum        = OcTensorGPU_sum;
   module -> Tensor_prod       = OcTensorGPU_prod;
   module -> Tensor_sumNaN     = OcTensorGPU_sumNaN;
   module -> Tensor_prodNaN    = OcTensorGPU_prodNaN;
   module -> Tensor_sumAbs     = OcTensorGPU_sumAbs;
   module -> Tensor_sumAbsNaN  = OcTensorGPU_sumAbsNaN;

   module -> Tensor_maximum    = OcTensorGPU_maximum;
   module -> Tensor_minimum    = OcTensorGPU_minimum;
   module -> Tensor_maximumAbs = OcTensorGPU_maximumAbs;
   module -> Tensor_minimumAbs = OcTensorGPU_minimumAbs;

   module -> Tensor_norm       = OcTensorGPU_norm;
   module -> Tensor_normNaN    = OcTensorGPU_normNaN;
   module -> Tensor_norm2      = OcTensorGPU_norm2;
   module -> Tensor_norm2NaN   = OcTensorGPU_norm2NaN;
}



/* ===================================================================== */
/* Internal functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorGPU_intrnlInitializeReduce(OcTensorGPU_ReduceAll *config,
                                       OcTensor *tensor, int flagRemoveRepeats)
/* -------------------------------------------------------------------- */
{  size_t bufferSize;

   /* Initialize */
   config -> buffer     = NULL;
   config -> tensor     = NULL;
   config -> bufferData = NULL;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

    /* Analyze the operation */
   OcSolid_analyzeReduce(&(config -> data), tensor, flagRemoveRepeats);

   /* Set the tensor pointer (borrowed reference) */
   config -> tensor = tensor;

   /* Determine the required buffer size */
   if (solid_gpu_reduce_all_buffer_size(tensor -> nelem, tensor -> device -> index, &bufferSize) != 0)
   {  OC_SOLID_ERRMSG(); return -1;  }

   /* Allocate the intermediate buffer */
   if (bufferSize > 0)
   {  config -> buffer = OcStorage_createTemporary(bufferSize, OcDTypeInt8, tensor -> device);
      if (config -> buffer == NULL) return -1;

      /* Add a write dependency on the buffer */
      if (OcStorage_startWrite(tensor -> storage, config -> buffer) != 0)
      {  OcDecrefStorage(config -> buffer); 
         config -> buffer = NULL;
         return -1;
      }
      else
      {  config -> bufferData = (void *)(config -> buffer -> data);
      }
   }
   else
   {  config -> buffer = NULL;
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_intrnlFinalizeReduce(OcTensorGPU_ReduceAll *config, int result)
/* -------------------------------------------------------------------- */
{  
   /* Synchronization and finalization */
   if (config -> buffer)
   {  if (OcStorage_update(config -> buffer) != 0) result = -1;
      if (OcStorage_finishWrite(config -> tensor -> storage, config -> buffer) != 0) result = -1;

      OcDecrefStorage(config -> buffer);
   }

   return result;
}



/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
/* int OcTensorGPU_any       (OcTensor *tensor, int *result)            */
/* int OcTensorGPU_all       (OcTensor *tensor, int *result)            */
/* int OcTensorGPU_allFinite (OcTensor *tensor, int *result)            */
/* int OcTensorGPU_anyInf    (OcTensor *tensor, int *result)            */
/* int OcTensorGPU_anyNaN    (OcTensor *tensor, int *result)            */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *tensor, int *result) \
{  OcTensorGPU_ReduceAll        config; \
   solid_funptr_gpu_##SOLID_OP  funptr; \
   solid_bool                   value; \
   int                          status; \
   \
   /* Look up the function pointer */  \
   OC_SOLID_FUNPTR(#DESC, solid_gpu_##SOLID_OP, funptr, tensor -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - remove replications */ \
   OcTensorGPU_intrnlInitializeReduce(&config, tensor, 1); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 config.data.ndims, config.data.size, config.data.strides, config.data.ptr, \
                 &value, (void *)(config.bufferData), \
                 OcTensorGPU_cudaStream(config.tensor)); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (int)value; \
   \
   /* Finalize the reduction - includes update synchronization */ \
   return OcTensorGPU_intrnlFinalizeReduce(&config, status); \
}

OC_TEMPLATE(any,       any,        "any"      )
OC_TEMPLATE(all,       all,        "all"      )
OC_TEMPLATE(allFinite, all_finite, "allFinite") /* Only needs to be implemented for floating-point types */
OC_TEMPLATE(anyInf,    any_inf,    "anyInf"   ) /* Only needs to be implemented for floating-point types */
OC_TEMPLATE(anyNaN,    any_nan,    "anyNaN"   ) /* Only needs to be implemented for floating-point types */
#undef OC_TEMPLATE



/* -------------------------------------------------------------------- */
/* int OcTensorGPU_nnz   (OcTensor *tensor, OcUInt64 *result)           */
/* int OcTensorGPU_nnzNaN(OcTensor *tensor, OcUInt64 *result)           */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *tensor, OcUInt64 *result) \
{  OcTensorGPU_ReduceAll       config; \
   solid_funptr_gpu_##SOLID_OP funptr; \
   solid_uint64                value; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, tensor -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcTensorGPU_intrnlInitializeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 config.data.ndims, config.data.size, config.data.strides, config.data.ptr, \
                 &value, (void *)(config.bufferData), \
                 OcTensorGPU_cudaStream(config.tensor)); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (OcUInt64)value; \
   \
   /* Finalize the reduction - includes update synchronization */ \
   return OcTensorGPU_intrnlFinalizeReduce(&config, status); \
}

OC_TEMPLATE(nnz,    nnz,     "nnz"   )
OC_TEMPLATE(nnzNaN, nnz_nan, "nnzNaN")
#undef OC_TEMPLATE



/* --------------------------------------------------------------------- */
/* int OcTensorGPU_allLT(OcTensor *tensor, OcScalar *bound, int *result) */
/* int OcTensorGPU_allLE(OcTensor *tensor, OcScalar *bound, int *result) */
/* int OcTensorGPU_allGT(OcTensor *tensor, OcScalar *bound, int *result) */
/* int OcTensorGPU_allGE(OcTensor *tensor, OcScalar *bound, int *result) */
/* --------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *tensor, OcScalar *bound, int *result) \
{  OcTensorGPU_ReduceAll       config; \
   solid_funptr_gpu_##SOLID_OP funptr; \
   solid_scalar                bound_solid; \
   solid_bool                  result_solid; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, tensor -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Convert the scalar */ \
   if (OcSolid_getScalar(bound, &bound_solid) != 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcTensorGPU_intrnlInitializeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 config.data.ndims, config.data.size, config.data.strides, config.data.ptr, \
                 &bound_solid, &result_solid, (void *)(config.bufferData), \
                 OcTensorGPU_cudaStream(config.tensor)); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (int)result_solid; \
   \
   /* Finalize the reduction - includes update synchronization */ \
   return OcTensorGPU_intrnlFinalizeReduce(&config, status); \
}

OC_TEMPLATE(allLT, all_lt, "less than")
OC_TEMPLATE(allLE, all_le, "less than or equal")
OC_TEMPLATE(allGT, all_gt, "greater than")
OC_TEMPLATE(allGE, all_ge, "greater than or equal")
#undef OC_TEMPLATE



/* ---------------------------------------------------------------------------------------- */
/* int OcTensorGPU_allGTLT(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* int OcTensorGPU_allGTLE(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* int OcTensorGPU_allGELT(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* int OcTensorGPU_allGELE(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) */
/* ---------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *tensor, OcScalar *lower, OcScalar *upper, int *result) \
{  OcTensorGPU_ReduceAll       config; \
   solid_funptr_gpu_##SOLID_OP funptr; \
   solid_scalar                lower_solid, upper_solid; \
   solid_bool                  result_solid; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, tensor -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Convert the scalars */ \
   if (OcSolid_getScalar(lower, &upper_solid) != 0) return -1; \
   if (OcSolid_getScalar(upper, &upper_solid) != 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcTensorGPU_intrnlInitializeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 config.data.ndims, config.data.size, config.data.strides, config.data.ptr, \
                 &lower_solid, &upper_solid, &result_solid, (void *)(config.bufferData), \
                 OcTensorGPU_cudaStream(config.tensor)); \
   \
   /* Set the result value */ \
   if (status == 0) *result = (int)result_solid; \
   \
   /* Finalize the reduction - includes update synchronization */ \
   return OcTensorGPU_intrnlFinalizeReduce(&config, status); \
}

OC_TEMPLATE(allGTLT, all_gtlt, "in range (lower,upper)")
OC_TEMPLATE(allGTLE, all_gtle, "in range (lower,upper]")
OC_TEMPLATE(allGELT, all_gelt, "in range [lower,upper)")
OC_TEMPLATE(allGELE, all_gele, "in range [lower,upper]")
#undef OC_TEMPLATE



/* -------------------------------------------------------------------- */
/* int OcTensorGPU_sum       (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_prod      (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_sumNaN    (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_prodNaN   (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_sumAbs    (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_sumAbsNaN (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_maximum   (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_minimum   (OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_maximumAbs(OcTensor *tensor, OcScalar *result)       */
/* int OcTensorGPU_minimumAbs(OcTensor *tensor, OcScalar *result)       */
/* -------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *tensor, OcScalar *result) \
{  OcTensorGPU_ReduceAll        config; \
   solid_funptr_gpu_##SOLID_OP  funptr; \
   int                          status; \
   \
   /* Look up the function pointer */  \
   OC_SOLID_FUNPTR(#DESC, solid_gpu_##SOLID_OP, funptr, tensor -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcTensorGPU_intrnlInitializeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 config.data.ndims, config.data.size, config.data.strides, config.data.ptr, \
                 (void *)&(result->value), (void *)(config.bufferData), \
                 OcTensorGPU_cudaStream(config.tensor)); \
   \
   /* Finalize the reduction - includes update synchronization */ \
   return OcTensorGPU_intrnlFinalizeReduce(&config, status); \
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
/* int OcTensorGPU_norm   (OcTensor *tensor, double p, OcScalar *result)          */
/* int OcTensorGPU_normNaN(OcTensor *tensor, double p, OcScalar *result)          */
/* ------------------------------------------------------------------------------ */
#define OC_TEMPLATE(OP, SOLID_OP, DESC) \
int OcTensorGPU_##OP(OcTensor *tensor, double p, OcScalar *result) \
{  OcTensorGPU_ReduceAll       config; \
   solid_funptr_gpu_##SOLID_OP funptr; \
   int                         status; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, tensor -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation - do not remove replications */ \
   OcTensorGPU_intrnlInitializeReduce(&config, tensor, 0); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(status, funptr, \
                 config.data.ndims, config.data.size, config.data.strides, config.data.ptr, \
                 p, (void *)&(result -> value), (void *)(config.bufferData), \
                 OcTensorGPU_cudaStream(config.tensor)); \
   \
   /* Finalize the reduction - includes update synchronization */ \
   return OcTensorGPU_intrnlFinalizeReduce(&config, status); \
}

OC_TEMPLATE(norm,    norm,     "norm"   )
OC_TEMPLATE(normNaN, norm_nan, "normNaN")
#undef OC_TEMPLATE

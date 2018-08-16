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

#include "ocean/core/gpu/op/tensor_binary_gpu.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/core/gpu/cuda.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_gpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* Function declaration template */
#define OC_TEMPLATE(NAME, X, Y, Z) \
OC_API int OcTensorGPU_ ## NAME (OcTensor *src1, OcTensor *src2, OcTensor *dst);
#include "ocean/core/generic/generate_tensor_binary.h"
#undef OC_TEMPLATE


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeBinaryOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the module functions */
   #define OC_TEMPLATE(NAME, X, Y, Z) \
   module -> Tensor_##NAME = OcTensorGPU_##NAME;
   #include "ocean/core/generic/generate_tensor_binary.h"
   #undef OC_TEMPLATE
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

#define OC_TEMPLATE(OP, SOLID_OP, CHECK, DESC) \
/* -------------------------------------------------------------------- */ \
/* int OcTensorGPU_<OP>(OcTensor *src1, OcTensor *src2, OcTensor *dst)  */ \
/* -------------------------------------------------------------------- */ \
int OcTensorGPU_##OP(OcTensor *src1, OcTensor *src2, OcTensor *dst) \
{  solid_funptr_gpu_##SOLID_OP funptr; \
   OcSolidElemwise3 config; \
   int result; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_gpu_##SOLID_OP, funptr, src1 -> dtype, "GPU"); \
   if (funptr == 0) return -1; \
   \
   /* Actvate the device */ \
   if (OcCuda_setDevice(dst -> device -> index) != 0) return -1; \
   \
   /* Analyze the operation */ \
   OcSolid_analyzeElemwise3(&config, src1, src2, dst); \
   \
   /* Synchronize */ \
   if (OcTensor_startRead(dst, src1) != 0) return -1; \
   if (OcTensor_startRead(dst, src2) != 0) return -1; \
   \
   /* Call the function */ \
   OC_SOLID_CALL(result, funptr, \
                 config.ndims, config.size, config.strides1, config.ptr1, \
                 config.strides2, config.ptr2, config.strides3, config.ptr3, \
                 OcTensorGPU_cudaStream(dst)); \
   \
   /* Synchronize */ \
   if (result != 0) return result; \
   if (OcTensor_update(dst) != 0) return -1; \
   if (OcTensor_finishRead(dst, src2) != 0) return -1; \
   if (OcTensor_finishRead(dst, src1) != 0) return -1; \
   \
   return 0; \
}

#include "ocean/core/generic/generate_tensor_binary.h"
#undef OC_TEMPLATE

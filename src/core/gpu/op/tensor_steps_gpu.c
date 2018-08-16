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

#include "ocean/core/gpu/op/tensor_steps_gpu.h"
#include "ocean/core/gpu/tensor_gpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_gpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_stepsInt64  (OcTensor *tensor, OcInt64 offset, OcInt64 step);
OC_API int OcTensorGPU_stepsDouble (OcTensor *tensor, OcDouble offset, OcDouble step);
OC_API int OcTensorGPU_stepsCDouble(OcTensor *tensor, OcCDouble offset, OcCDouble step);

/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeStepOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor steps */
   module -> Tensor_stepsInt64   = OcTensorGPU_stepsInt64;
   module -> Tensor_stepsDouble  = OcTensorGPU_stepsDouble;
   module -> Tensor_stepsCDouble = OcTensorGPU_stepsCDouble;
}


/* ===================================================================== */
/* Function implementations - step operations                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorGPU_stepsInt64(OcTensor *tensor, OcInt64 offset, OcInt64 step)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_steps_int64 funptr;
   OcSolidElemwise1 config;
   int result;

   /* Look up the function pointer */
   OC_SOLID_FUNPTR("stepsInt64", solid_gpu_steps_int64, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr,
                 (solid_int64)offset, (solid_int64)step,
                 OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result != 0) return result;
   if (OcTensor_update(tensor) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_stepsDouble(OcTensor *tensor, OcDouble offset, OcDouble step)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_steps_double funptr;
   OcSolidElemwise1 config;
   int result;

   /* Look up the function pointer */
   OC_SOLID_FUNPTR("stepsDouble", solid_gpu_steps_double, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr,
                 (solid_double)offset, (solid_double)step,
                 OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result != 0) return result;
   if (OcTensor_update(tensor) != 0) return -1;

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_stepsCDouble(OcTensor *tensor, OcCDouble offset, OcCDouble step)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_steps_cdouble funptr;
   OcSolidElemwise1 config;
   solid_cdouble s_offset, s_step;
   int result;

   /* Special case for non-complex tensors */
   if (!OcDType_isComplex(tensor -> dtype))
   {  return OcTensorGPU_stepsDouble(tensor, offset.real, step.real);
   }

   /* Look up the function pointer */
   OC_SOLID_FUNPTR("stepsCDouble", solid_gpu_steps_cdouble, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Convert the offset and step */
   s_offset.real = offset.real;
   s_offset.imag = offset.imag;
   s_step.real   = step.real;
   s_step.imag   = step.imag;

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr,
                 s_offset, s_step,
                 OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result != 0) return result;
   if (OcTensor_update(tensor) != 0) return -1;

   return result;
}

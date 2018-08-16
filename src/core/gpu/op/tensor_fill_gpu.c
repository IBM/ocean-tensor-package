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

#include "ocean/core/gpu/op/tensor_fill_gpu.h"
#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/core/gpu/cuda.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_gpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorGPU_fill   (OcTensor *tensor, OcScalar *scalar);
OC_API int OcTensorGPU_fillNaN(OcTensor *tensor, OcScalar *scalar);
OC_API int OcTensorGPU_maskedFill(OcTensor *tensor, OcTensor *mask, OcScalar *scalar);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorGPU_initializeFillOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor fill */
   module -> Tensor_fill       = OcTensorGPU_fill;
   module -> Tensor_fillNaN    = OcTensorGPU_fillNaN;
   module -> Tensor_maskedFill = OcTensorGPU_maskedFill;
}


/* ===================================================================== */
/* Function implementations - fill operations                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorGPU_fill(OcTensor *tensor, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcSolidElemwise1      config;
   solid_funptr_gpu_fill funptr;
   solid_scalar          value;
   int                   result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("fill", solid_gpu_fill, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr, value,
                 OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result == 0) result = OcTensor_update(tensor);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_fillNaN(OcTensor *tensor, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  OcSolidElemwise1          config;
   solid_funptr_gpu_fill_nan funptr;
   solid_scalar              value;
   int                       result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("fillNaN", solid_gpu_fill_nan, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr, value,
                 OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result == 0) result = OcTensor_update(tensor);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorGPU_maskedFill(OcTensor *tensor, OcTensor *mask, OcScalar *scalar)
/* -------------------------------------------------------------------- */
{  solid_funptr_gpu_masked_fill funptr;
   solid_scalar      value;
   OcSolidElemwise2  config;
   int               result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("maskedFill", solid_gpu_masked_fill, funptr, tensor -> dtype, "GPU");
   if (funptr == 0) return -1;

   /* Actvate the device */
   if (OcCuda_setDevice(tensor -> device -> index) != 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise2(&config, tensor, mask);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides1, config.ptr1,
                 config.strides2, config.ptr2, value, OcTensorGPU_cudaStream(tensor));

   /* Synchronization */
   if (result == 0) result = OcTensor_update(tensor);

   return result;
}

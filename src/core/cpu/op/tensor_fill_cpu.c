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

#include "ocean/core/cpu/op/tensor_fill_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorCPU_fill      (OcTensor *tensor, OcScalar *scalar);
OC_API int OcTensorCPU_fillNaN   (OcTensor *tensor, OcScalar *scalar);
OC_API int OcTensorCPU_maskedFill(OcTensor *tensor, OcTensor *mask, OcScalar *scalar);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeFillOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor fill */
   module -> Tensor_fill       = OcTensorCPU_fill;
   module -> Tensor_fillNaN    = OcTensorCPU_fillNaN;
   module -> Tensor_maskedFill = OcTensorCPU_maskedFill;
}


/* ===================================================================== */
/* Function implementations - fill operations                            */
/* ===================================================================== */

/* ------------------------------------------------------------------------------ */
int OcTensorCPU_fill(OcTensor *tensor, OcScalar *scalar)
/* ------------------------------------------------------------------------------ */
{  OcSolidElemwise1      config;
   solid_funptr_cpu_fill funptr;
   solid_scalar          value;
   int                   result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("fill", solid_cpu_fill, funptr, tensor -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.strides, config.ptr, value);

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_fillNaN(OcTensor *tensor, OcScalar *scalar)
/* ------------------------------------------------------------------------------ */
{  OcSolidElemwise1          config;
   solid_funptr_cpu_fill_nan funptr;
   solid_scalar              value;
   int                       result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("fillNaN", solid_cpu_fill_nan, funptr, tensor -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.strides, config.ptr, value);

   return result;
}


/* ------------------------------------------------------------------------------ */
int OcTensorCPU_maskedFill(OcTensor *tensor, OcTensor *mask, OcScalar *scalar)
/* ------------------------------------------------------------------------------ */
{  solid_funptr_cpu_masked_fill funptr;
   solid_scalar      value;
   OcSolidElemwise2  config;
   int               result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("maskedFill", solid_cpu_masked_fill, funptr, tensor -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Convert the scalar */
   if (OcSolid_getScalar(scalar, &value) != 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise2(&config, tensor, mask);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides1, config.ptr1,
                 config.strides2, config.ptr2, value);

   return result;
}

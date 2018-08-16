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

#include "ocean/core/cpu/op/tensor_steps_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

OC_API int OcTensorCPU_stepsInt64  (OcTensor *tensor, OcInt64 offset, OcInt64 step);
OC_API int OcTensorCPU_stepsDouble (OcTensor *tensor, OcDouble offset, OcDouble step);
OC_API int OcTensorCPU_stepsCDouble(OcTensor *tensor, OcCDouble offset, OcCDouble step);


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeStepOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor steps */   
   module -> Tensor_stepsInt64   = OcTensorCPU_stepsInt64;
   module -> Tensor_stepsDouble  = OcTensorCPU_stepsDouble;
   module -> Tensor_stepsCDouble = OcTensorCPU_stepsCDouble;
}


/* ===================================================================== */
/* Function implementations - step functions                             */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorCPU_stepsInt64(OcTensor *tensor, OcInt64 offset, OcInt64 step)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_steps_int64 funptr;
   OcSolidElemwise1 config;
   int result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("stepsInt64", solid_cpu_steps_int64, funptr, tensor -> dtype, "CPU");

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr, offset, step);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorCPU_stepsDouble(OcTensor *tensor, OcDouble offset, OcDouble step)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_steps_double funptr;
   OcSolidElemwise1 config;
   int result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("stepsDouble", solid_cpu_steps_double, funptr, tensor -> dtype, "CPU");

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr, offset, step);

   return result;
}


/* -------------------------------------------------------------------- */
int OcTensorCPU_stepsCDouble(OcTensor *tensor, OcCDouble offset, OcCDouble step)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_steps_cdouble funptr;
   solid_cdouble s_offset;
   solid_cdouble s_step;
   OcSolidElemwise1 config;
   int result;

   /* Special case for non-complex tensors */
   if (!OcDType_isComplex(tensor -> dtype))
   {  return OcTensorCPU_stepsDouble(tensor, offset.real, step.real);
   }

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("stepsCDouble", solid_cpu_steps_cdouble, funptr, tensor -> dtype, "CPU");

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Convert the complex scalar */
   s_offset.real = offset.real;
   s_offset.imag = offset.imag;
   s_step.real   = step.real;
   s_step.imag   = step.imag;

   /* Call the function */
   OC_SOLID_CALL(result, funptr,
                 config.ndims, config.size, config.strides, config.ptr, s_offset, s_step);

   return result;
}

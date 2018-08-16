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

#include "ocean/core/cpu/op/tensor_byteswap_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeByteswapOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Tensor byteswap */
   module -> Tensor_byteswapNoFlag = OcTensorCPU_byteswapNoFlag;
}


/* ===================================================================== */
/* Function implementations - byteswap operations                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcTensorCPU_byteswapNoFlag(OcTensor *tensor)
/* -------------------------------------------------------------------- */
{  solid_funptr_cpu_byteswap funptr;
   OcSolidElemwise1 config;
   int result;

   /* Look up the function pointer */ 
   OC_SOLID_FUNPTR("byteswap", solid_cpu_byteswap, funptr, tensor -> dtype, "CPU");
   if (funptr == 0) return -1;

   /* Analyze the operation */
   OcSolid_analyzeElemwise1(&config, tensor);

   /* Call the function */
   OC_SOLID_CALL(result, funptr, config.ndims, config.size, config.strides, config.ptr);

   return result;
}

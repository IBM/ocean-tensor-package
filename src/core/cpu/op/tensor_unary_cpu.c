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

#include "ocean/core/cpu/op/tensor_unary_cpu.h"

/* Solid library */
#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_core_cpu.h"


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

#define OC_TEMPLATE(NAME, X, Y, Z) \
OC_API int OcTensorCPU_ ## NAME (OcTensor *src, OcTensor *dst);
#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE



/* ===================================================================== */
/* Register functions                                                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcTensorCPU_initializeUnaryOps(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the module functions */
   #define OC_TEMPLATE(NAME, X, Y, Z) \
   module -> Tensor_##NAME = OcTensorCPU_##NAME;
   #include "ocean/core/generic/generate_tensor_unary.h"
   #undef OC_TEMPLATE
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

#define OC_TEMPLATE(OP, SOLID_OP, CHECK, DESC) \
/* -------------------------------------------------------------------- */ \
/* int OcTensorCPU_<OP>(OcTensor *src, OcTensor *dst)  */ \
/* -------------------------------------------------------------------- */ \
int OcTensorCPU_##OP(OcTensor *src, OcTensor *dst) \
{  solid_funptr_cpu_##SOLID_OP funptr; \
   OcSolidElemwise2 config; \
   int result; \
   \
   /* Look up the function pointer */ \
   OC_SOLID_FUNPTR(DESC, solid_cpu_##SOLID_OP, funptr, src -> dtype, "CPU"); \
   if (funptr == 0) return -1; \
   \
   /* Analyze the operation */ \
   OcSolid_analyzeElemwise2(&config, src, dst); \
   \
   /* Call the function */ \
   OC_SOLID_CALL(result, funptr, \
                 config.ndims, config.size, \
                 config.strides1, config.ptr1, \
                 config.strides2, config.ptr2); \
   \
   return result; \
}
#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE

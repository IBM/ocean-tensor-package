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
#include "ocean/core/cpu/tensor_cpu.h"
#include "ocean/core/cpu/op/tensor_copy_cpu.h"
#include "ocean/core/cpu/op/tensor_fill_cpu.h"
#include "ocean/core/cpu/op/tensor_byteswap_cpu.h"
#include "ocean/core/cpu/op/tensor_steps_cpu.h"
#include "ocean/core/cpu/op/tensor_index_cpu.h"
#include "ocean/core/cpu/op/tensor_unary_cpu.h"
#include "ocean/core/cpu/op/tensor_binary_cpu.h"
#include "ocean/core/cpu/op/tensor_reduce_all_cpu.h"
#include "ocean/core/cpu/op/tensor_reduce_axis_cpu.h"
#include "ocean/core/cpu/op/tensor_blas_cpu.h"


/* ===================================================================== */
/* Function registration                                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcRegisterTensorCPU(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Initialize look-up tables */
   OcTensorCPU_initializeCopyOps(module);
   OcTensorCPU_initializeByteswapOps(module);
   OcTensorCPU_initializeFillOps(module);
   OcTensorCPU_initializeIndexOps(module);
   OcTensorCPU_initializeUnaryOps(module);
   OcTensorCPU_initializeBinaryOps(module);
   OcTensorCPU_initializeStepOps(module);
   OcTensorCPU_initializeBlasOps(module);
   OcTensorCPU_initializeReduceAllOps(module);
   OcTensorCPU_initializeReduceAxisOps(module);
}

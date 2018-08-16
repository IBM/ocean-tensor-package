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

#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean/core/gpu/op/tensor_copy_gpu.h"
#include "ocean/core/gpu/op/tensor_fill_gpu.h"
#include "ocean/core/gpu/op/tensor_steps_gpu.h"
#include "ocean/core/gpu/op/tensor_index_gpu.h"
#include "ocean/core/gpu/op/tensor_unary_gpu.h"
#include "ocean/core/gpu/op/tensor_binary_gpu.h"
#include "ocean/core/gpu/op/tensor_reduce_all_gpu.h"
#include "ocean/core/gpu/op/tensor_reduce_axis_gpu.h"
#include "ocean/core/gpu/op/tensor_blas_gpu.h"


/* ===================================================================== */
/* Function registration                                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcRegisterTensorGPU(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Initialize look-up tables */
   OcTensorGPU_initializeCopyOps(module);
   OcTensorGPU_initializeFillOps(module);
   OcTensorGPU_initializeStepOps(module);
   OcTensorGPU_initializeIndexOps(module);
   OcTensorGPU_initializeUnaryOps(module);
   OcTensorGPU_initializeBinaryOps(module);
   OcTensorGPU_initializeBlasOps(module);
   OcTensorGPU_initializeReduceAllOps(module);
   OcTensorGPU_initializeReduceAxisOps(module);

   return 0;
}

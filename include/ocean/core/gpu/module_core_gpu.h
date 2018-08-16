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

#ifndef __OC_MODULE_CORE_GPU_H__
#define __OC_MODULE_CORE_GPU_H__

#include "ocean/core/interface/module_core.h"
#include "ocean/base/api.h"
#include "cublas_v2.h"

/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  OcModuleCore_Context HEAD;
   int                  handleInitialized;
   cublasHandle_t       handle;
} OcModuleCoreGPU_Context;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int OcInitModuleCore_GPU(void);
OC_API int OcModuleCoreGPU_cublasHandle(OcDevice *device, cublasHandle_t *handle);

#endif

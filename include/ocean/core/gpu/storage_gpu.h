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

#ifndef __OC_STORAGE_GPU_H__
#define __OC_STORAGE_GPU_H__

#include "ocean/base/api.h"
#include "ocean/base/storage.h"
#include "ocean/core/interface/module_core.h"
#include "ocean/core/gpu/device_gpu.h"
#include "ocean/core/gpu/cuda.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  OcStorage HEAD;

   /* Reserved for device-specific storage information */
} OcStorageGPU;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Internal function to register storage functions in the core module */
OC_API int OcRegisterStorageGPU(OcModuleCore *module);


/* Functions for internal usage */
OC_API int OcBufferGPU_copy             (OcStorage *srcStorage, void *srcData,
                                         OcStorage *dstStorage, void *dstData, OcDType dtype, OcSize n);
OC_API int OcBufferGPU_copyFromCPU      (OcStorage *srcStorage, void *srcData,
                                         OcStorage *dstStorage, void *dstData, OcDType dtype, OcSize n);
OC_API int OcBufferGPU_copyToCPU        (OcStorage *srcStorage, void *srcData,
                                         OcStorage *dstStorage, void *dstData, OcDType dtype, OcSize n);
OC_API int OcBufferGPU_copyHostToStorage(OcStorage *storage, OcIndex offset, OcSize nbytes, void *ptr);
OC_API int OcBufferGPU_copyStorageToHost(OcStorage *storage, OcIndex offset, OcSize nbytes, void *ptr);
OC_API int OcBufferGPU_zero             (OcStorage *storage, void *data, OcDType dtype, OcSize n);


/* Macros */
#define OcStorageGPU_cudaStream(STORAGE)  (((OcStreamGPU *)(((OcStorage *)(STORAGE)) -> stream)) -> cudaStream)

#endif

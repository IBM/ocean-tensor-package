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

#ifndef __OC_STORAGE_CPU_H__
#define __OC_STORAGE_CPU_H__

#include "ocean/core/interface/module_core.h"
#include "ocean/base/storage.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  OcStorage HEAD;

   /* Devices may support references to existing storage. In this case   */
   /* the storage does not own the data and therefore needs a way to     */
   /* deallocate the memory when it is freed. This is done by setting    */
   /* the free function and providing addition user data in the freeData */
   /* field.                                                             */
   void (*referenceFree)(void *reference, void *data);
   void  *reference;

} OcStorageCPU;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Internal function to register storage functions in the core module */
OC_API void OcRegisterStorageCPU(OcModuleCore *module);

/* Functions for internal usage */
OC_API int OcBufferCPU_copy(OcStorage *srcStorage, void *srcData,
                            OcStorage *dstStorage, void *dstData,
                            OcDType dtype, OcSize n);
OC_API int OcBufferCPU_zero(OcStorage *storage, void *data, OcDType dtype, OcSize n);
OC_API int OcBufferCPU_copyHostToStorage(OcStorage *storage, OcIndex offset, OcSize nbytes, void *ptr);
OC_API int OcBufferCPU_copyStorageToHost(OcStorage *storage, OcIndex offset, OcSize nbytes, void *ptr);

#endif

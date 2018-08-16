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

#ifndef __OC_CUDA_H__
#define __OC_CUDA_H__

#include "ocean/base/api.h"

#include <cuda.h>
#include <cuda_runtime_api.h>


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int  OcInitCuda(void);

/* Setting the device */
OC_API int OcCuda_setDevice(int deviceID);

/* Cuda status */
OC_API int  OcCuda_setStatus(cudaError_t status);
OC_API void OcCuda_clearStatus(void);
OC_API int  OcCuda_checkStatus(void);

#endif

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

#include "ocean/core/gpu/cuda.h"
#include "ocean.h"


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

static int          oc_cuda_initialized = 0;
static cudaError_t  oc_cuda_status = cudaSuccess;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

static void OcFinalizeCuda(void);


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcInitCuda(void)
/* -------------------------------------------------------------------- */
{
   oc_cuda_initialized = 1;

   return OcFinalizeAddHandler(OcFinalizeCuda, "Finalize CUDA");
}


/* -------------------------------------------------------------------- */
void OcFinalizeCuda(void)
/* -------------------------------------------------------------------- */
{
   if (oc_cuda_initialized)
   {
      /* Unset the initialized flag */
      oc_cuda_initialized = 0;
   }
}


/* ===================================================================== */
/* Function implementations - device functions                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcCuda_setDevice(int deviceID)
/* -------------------------------------------------------------------- */
{
   if (cudaSetDevice(deviceID) != cudaSuccess)
   {  cudaGetLastError();
      OcError(-1, "Error activating device GPU%d", deviceID);
   }

   return 0;
}


/* ===================================================================== */
/* Function implementations - status                                     */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcCuda_setStatus(cudaError_t status)
/* -------------------------------------------------------------------- */
{
   if (oc_cuda_status == cudaSuccess)
   {  oc_cuda_status = status;
   }

   if (status != cudaSuccess)
   {  cudaGetLastError();
      return -1;
   }
   else
   {  return 0;
   }
}


/* -------------------------------------------------------------------- */
void OcCuda_clearStatus(void)
/* -------------------------------------------------------------------- */
{
   oc_cuda_status = cudaSuccess;
}
 

/* -------------------------------------------------------------------- */
int OcCuda_checkStatus(void)
/* -------------------------------------------------------------------- */
{
   if (oc_cuda_status == cudaSuccess) return 0;

   OcErrorMessage("Cuda error: %s", cudaGetErrorString(oc_cuda_status));
   cudaGetLastError();
   oc_cuda_status = cudaSuccess;

   return -1;
}

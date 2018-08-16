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

#include "ocean.h"
#include "ocean/core/ocean_gpu.h"
#include "ocean/core/gpu/device_gpu.h"
#include "ocean/core/gpu/module_core_gpu.h"

/* Initialization flag */
static int oc_init_gpu_flag = 0;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

void OcFinalizeGPU(void);


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcInitGPU(void)
/* -------------------------------------------------------------------- */
{
   /* Check initialization status */
   if (oc_init_gpu_flag != 0) return 0;

   /* Initialize cuda */
   if (OcInitCuda() != 0) return -1;

   /* Initialize the GPU devices */
   if (OcInitDevicesGPU() != 0) return -1;

   /* Initialize default modules */
   if (OcInitModuleCore_GPU() != 0) return -1;

   /* Successfully initialized */
   oc_init_gpu_flag = 1;

   /* Register finalization function */
   return OcFinalizeAddHandler(OcFinalizeGPU, "Finalize Ocean GPU");
}


/* -------------------------------------------------------------------- */
void OcFinalizeGPU(void)
/* -------------------------------------------------------------------- */
{
   /* Check if finalization is needed */
   if (oc_init_gpu_flag == 0) return;

   /* Successfully finalized */
   oc_init_gpu_flag = 0;
}

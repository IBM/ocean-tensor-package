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

#include "ocean/core/ocean.h"
#include "ocean/base/ocean.h"
#include "ocean/base/malloc.h"
#include "ocean/base/warning.h"
#include "ocean/base/error.h"

#include "ocean/core/cpu/device_cpu.h"
#include "ocean/core/cpu/module_core_cpu.h"

#include <stdlib.h>


/* Initialization flag */
static int oc_init_flag = 0;


/* -------------------------------------------------------------------- */
int OcInit(void)
/* -------------------------------------------------------------------- */
{
   /* Check initialization status */
   if (oc_init_flag != 0) return 0;

   /* Call the internal initialize function */
   if (OcInitializeIntrnl() != 0) return -1;

   /* Initialize device information */
   if (OcInitDevices() != 0) return -1;

   /* Initialize the CPU device */
   if (OcInitDevicesCPU() != 0) return -1;

   /* Initialize default modules */
   if (OcInitModuleCore_CPU() != 0) return -1;

   /* Successfully initialized */
   oc_init_flag = 1;

   return 0;
}


/* -------------------------------------------------------------------- */
void OcFinalize(void)
/* -------------------------------------------------------------------- */
{
   /* The OcFinalize function cleans up interal data structures.  */
   /* When Ocean is used as a library in languages with automatic */
   /* garbage collection (such as Python), it is possible that    */
   /* references to Ocean objects still remain. Those, as well as */
   /* internally linked objects, will be cleaned up when freed    */
   /* by automatic garbage collection.                            */

   /* Check if finalization is needed */
   if (oc_init_flag == 0) return ;

   /* Call the internal finalize function */
   OcFinalizeIntrnl();

   /* Successfully finalized */
   oc_init_flag = 0;
}

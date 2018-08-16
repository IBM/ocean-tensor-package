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

#include "ocean/core/interface/module_core.h"
#include "ocean/core/cpu/module_core_cpu.h"
#include "ocean/core/cpu/storage_cpu.h"
#include "ocean/core/cpu/tensor_cpu.h"
#include "ocean/core/cpu/device_info.h"
#include "ocean/base/scalar.h"
#include "ocean/base/error.h"

#include <stdio.h>
#include <stdlib.h>


static OcModuleCore oc_module_core_cpu = {{0}};


/* ===================================================================== */
/*                    Declaration of module functions                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static void initializeFunctions(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set all the function pointers here; it would also have been */
   /* possible to initialize them directly in the definition of   */
   /* oc_module_dummy_gpu above, but in that case the function    */
   /* ordering must be carefully followed. Moreover if the device */
   /* implementation of the module only implements certain values */
   /* they would need to be padded with many 0 values for the     */
   /* remaining function pointers. Explicitly specifying the      */
   /* function pointers here is clearer and also avoids any       */
   /* dependency on the order in which the functions are defined  */
   /* in the module interface.                                    */

   OcRegisterDeviceCPU(module);
   OcRegisterStorageCPU(module);
   OcRegisterTensorCPU(module);
}


/* ===================================================================== */
/*               Module context operations and finalization              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int initializeContext(OcDevice *device, OcDeviceModule *deviceModule, void **context)
/* -------------------------------------------------------------------- */
{  OcModuleCoreCPU_Context *ctx;

   /* Create and assign the context to make sure that partial */
   /* initialization is cleaned up in case of failure.        */
   *context = (void *)OcModuleCore_createContext(device, sizeof(OcModuleCoreCPU_Context));
   ctx = (OcModuleCoreCPU_Context *)(*context);
   if (ctx == NULL) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
static void finalizeContext(OcDevice *device, OcDeviceModule *deviceModule, void *context)
/* -------------------------------------------------------------------- */
{
   /* Free the context */
   OcModuleCore_freeContext((OcModuleCore_Context *)context);
}


/* -------------------------------------------------------------------- */
static void finalizeModule(OcDeviceModule *deviceModule)
/* -------------------------------------------------------------------- */
{
   /* This function is called when the module is finalized */
}


/* ===================================================================== */
/*                          Module initialization                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcInitModuleCore_CPU(void)
/* -------------------------------------------------------------------- */
{  OcModuleCore *module = &oc_module_core_cpu;

   /* Initialize the core module interface */
   if (OcModuleCore_initialize() != 0) return -1;

   /* Module management functions */
   module -> HEAD.module            = &oc_module_core;  /* Registration will incref */
   module -> HEAD.initializeContext = initializeContext;
   module -> HEAD.finalizeContext   = finalizeContext;
   module -> HEAD.finalizeModule    = finalizeModule;

   /* Set module function pointers */
   initializeFunctions(module);

   /* Register the module */
   return OcRegisterModule("CPU", (OcDeviceModule *)module, sizeof(*module));
}

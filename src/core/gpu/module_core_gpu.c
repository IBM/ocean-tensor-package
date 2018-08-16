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

#include "ocean/core/gpu/module_core_gpu.h"
#include "ocean/core/gpu/device_gpu.h"
#include "ocean/core/gpu/device_info.h"
#include "ocean/core/gpu/storage_gpu.h"
#include "ocean/core/gpu/tensor_gpu.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include "ocean/external/ocean-solid/ocean_solid.h"
#include "solid_gpu.h"

#include <stdio.h>
#include <stdlib.h>


static OcModuleCore oc_module_core_gpu = {{0}};


/* ===================================================================== */
/*                    Declaration of module functions                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int initializeFunctions(OcModuleCore *module)
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

   if ((OcRegisterDevicesGPU(module) != 0) ||
       (OcRegisterStorageGPU(module) != 0) ||
       (OcRegisterTensorGPU(module)  != 0)) return -1;

   /* Initialize the solid environment */
   if (solid_gpu_initialize() != 0)
   {  OC_SOLID_ERRMSG(); return -1; }

   return 0;
}


/* ===================================================================== */
/*               Module context operations and finalization              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int initializeContext(OcDevice *device, OcDeviceModule *deviceModule, void **context)
/* -------------------------------------------------------------------- */
{  OcModuleCoreGPU_Context *ctx;

   /* Create and assign the context to make sure that partial */
   /* initialization is cleaned up in case of failure.        */
   *context = (void *)OcModuleCore_createContext(device, sizeof(OcModuleCoreGPU_Context));
   ctx = (OcModuleCoreGPU_Context *)(*context);
   if (ctx == NULL) return -1;

   /* Initialize the context */
   ctx -> handleInitialized = 0;

   /* Activate the device */
   if (OcCuda_setDevice(device -> index) != 0) return -1;

   /* Create the cuBLAS context */
   if (cublasCreate(&(ctx -> handle)) != CUBLAS_STATUS_SUCCESS) return -1;
   ctx -> handleInitialized = 1;

   return 0;
}


/* -------------------------------------------------------------------- */
static void finalizeContext(OcDevice *device, OcDeviceModule *deviceModule, void *context)
/* -------------------------------------------------------------------- */
{  OcModuleCoreGPU_Context *ctx;

   /* This function is called when removing the device instance */
   if ((ctx = (OcModuleCoreGPU_Context *)context) == NULL) return ;

   /* Free dynamic allocates within the context here */
   if (ctx -> handleInitialized) cublasDestroy(ctx -> handle);

   /* Free the context */
   OcModuleCore_freeContext((OcModuleCore_Context *)context);
}


/* -------------------------------------------------------------------- */
static void finalizeModule(OcDeviceModule *deviceModule)
/* -------------------------------------------------------------------- */
{
   /* This function is called when the module is finalized */

   /* Finalize the solid environment */
   solid_gpu_finalize();
}


/* ===================================================================== */
/*                         Module initialization                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcInitModuleCore_GPU(void)
/* -------------------------------------------------------------------- */
{  OcModuleCore *module = &oc_module_core_gpu;

   /* Initialize the core module interface */
   if (OcModuleCore_initialize() != 0) return -1;

   /* Module management functions */
   module -> HEAD.module            = &oc_module_core;  /* Registration will incref */
   module -> HEAD.initializeContext = initializeContext;
   module -> HEAD.finalizeContext   = finalizeContext;
   module -> HEAD.finalizeModule    = finalizeModule;

   /* Set module function pointers */
   if (initializeFunctions(module) != 0) return -1;

   /* Register the module */
   return OcRegisterModule("GPU", (OcDeviceModule *)module, sizeof(*module));
}


/* ===================================================================== */
/*                         Access to the context                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcModuleCoreGPU_cublasHandle(OcDevice *device, cublasHandle_t *handle)
/* -------------------------------------------------------------------- */
{  OcModuleCoreGPU_Context *context;

   /* Make sure the given device is a GPU */
   if (!OcDeviceGPU_isInstance(device))
      OcError(-1, "CuBlas handles are only available for GPU devices");

   /* Get the context information for the given device instance */
   context = (OcModuleCoreGPU_Context *)OC_GET_DEVICE_CONTEXT(device, oc_module_core);
   if (context == NULL) OcError(-1, "Unable to access the device context while getting cuBlas handle");

   /* Set the result */
   if (handle) *handle = context -> handle;
   return 0;
}

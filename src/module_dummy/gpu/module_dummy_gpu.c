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

#include "ocean/module_dummy/module_dummy_gpu.h"

#include "ocean.h"
#include "ocean_gpu.h"

#include <stdio.h>
#include <stdlib.h>


static OcModuleDummy oc_module_dummy_gpu = {{0}};
static int           oc_module_dummy_gpu_init = 0;

/* Define the module context type; this information will be */
/* available for each device instance of the type for which */
/* this function provides the module implementation (GPU in */
/* this case).                                              */

typedef struct
{  int counter;
} ModuleContext;


/* ===================================================================== */
/*                    Declaration of module functions                    */
/* ===================================================================== */

static int HelloWorld_GPU(OcDevice *device);


/* -------------------------------------------------------------------- */
static void initializeFunctions(OcModuleDummy *module)
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

   module -> HelloWorld = HelloWorld_GPU;
}


/* ===================================================================== */
/*               Module context operations and finalization              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int initializeContext(OcDevice *device, OcDeviceModule *deviceModule, void **context)
/* -------------------------------------------------------------------- */
{  ModuleContext *ctx;

   /* Create the context */
   ctx = (ModuleContext *)malloc(sizeof(ModuleContext));
   if (ctx == NULL) OcError(-1, "Error allocating device context");

   /* Initialize the context */
   ctx -> counter = 0;

   /* Assign the context */
   *context = (void *)ctx;

   return 0;
}


/* -------------------------------------------------------------------- */
static void finalizeContext(OcDevice *device, OcDeviceModule *deviceModule, void *context)
/* -------------------------------------------------------------------- */
{
   /* This function is called when removing the device instance */
   if (context == NULL) return ;

   /* Free dynamic allocates within the context here */

   /* Free the context */
   free(context);
}


/* -------------------------------------------------------------------- */
static void finalizeModule(OcDeviceModule *deviceModule)
/* -------------------------------------------------------------------- */
{
   /* This function is called when the module is finalized */
}


/* ===================================================================== */
/*                         Module initialization                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcInitModule_Dummy_GPU(void)
/* -------------------------------------------------------------------- */
{  OcModuleDummy *module = &oc_module_dummy_gpu;
   int result;

   /* Check whether module was already initialized */
   if (oc_module_dummy_gpu_init != 0) return 0;

   /* Module management functions */
   module -> HEAD.module            = &oc_module_dummy;    /* Registration will incref */
   module -> HEAD.initializeContext = initializeContext;
   module -> HEAD.finalizeContext   = finalizeContext;
   module -> HEAD.finalizeModule    = finalizeModule;

   /* Set module function pointers */
   initializeFunctions(module);

   /* Register the module - make sure to get the device name correct */
   result = OcRegisterModule("GPU", (OcDeviceModule *)module, sizeof(*module));
   if (result == 0) oc_module_dummy_gpu_init = 1;

   return result;
}


/* ===================================================================== */
/*                   Implementation of module routines                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int HelloWorld_GPU(OcDevice *device)
/* -------------------------------------------------------------------- */
{  ModuleContext *context;

   /* Get the context information for the given device instance */
   context = (ModuleContext *)OC_GET_DEVICE_CONTEXT(device, oc_module_dummy);
   if (context == NULL) OcError(-1, "Invalid context for %s", __FUNCTION__);

   /* Perform the desired operation and update the context if needed */
   context -> counter ++;

   /* The module interface uses the device parameter to find the */
   /* correct module implementation, so here it is guaranteed    */
   /* that device is of type OcDeviceGPU.                        */
   printf("Hello World (%d) from GPU device %s\n",
          context -> counter,
          device -> name);

   return 0;
}

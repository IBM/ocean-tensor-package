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

#include "ocean/module_dummy/module_dummy_cpu.h"

#include <stdio.h>
#include <stdlib.h>

static OcModuleDummy oc_module_dummy_cpu = {{0}};
static int oc_module_dummy_cpu_init = 0;


/* Example of a module without context information */


/* ===================================================================== */
/*                    Declaration of module functions                    */
/* ===================================================================== */

static int HelloWorld_CPU(OcDevice *device);


/* -------------------------------------------------------------------- */
static void initializeFunctions(OcModuleDummy *module)
/* -------------------------------------------------------------------- */
{
   /* Set all the function pointers here; it would also have been */
   /* possible to initialize them directly in the definition of   */
   /* oc_module_dummy_cpu above, but in that case the function    */
   /* ordering must be carefully followed. Moreover if the device */
   /* implementation of the module only implements certain values */
   /* they would need to be padded with many 0 values for the     */
   /* remaining function pointers. Explicitly specifying the      */
   /* function pointers here is clearer and also avoids any       */
   /* dependency on the order in which the functions are defined  */
   /* in the module interface.                                    */

   module -> HelloWorld = HelloWorld_CPU;
}


/* ===================================================================== */
/*                          Module finalization                          */
/* ===================================================================== */

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
int OcInitModule_Dummy_CPU(void)
/* -------------------------------------------------------------------- */
{  OcModuleDummy *module = &oc_module_dummy_cpu;
   int result;

   /* Check whether module was already initialized */
   if (oc_module_dummy_cpu_init != 0) return 0;

   /* Module management functions */
   module -> HEAD.module             = &oc_module_dummy;  /* Registration will incref */
   module -> HEAD.initializeContext  = NULL;
   module -> HEAD.finalizeContext    = NULL;
   module -> HEAD.finalizeModule     = finalizeModule;

   /* Set module function pointers */
   initializeFunctions(module);

   /* Register the module - make sure to get the device name correct */
   result = OcRegisterModule("CPU", (OcDeviceModule *)module, sizeof(*module));
   if (result == 0) oc_module_dummy_cpu_init = 1;

   return result;
}


/* ===================================================================== */
/*                   Implementation of module routines                   */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int HelloWorld_CPU(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   /* The module interface uses the device parameter to find the */
   /* correct module implementation, so here it is guaranteed    */
   /* that device is of type OcDeviceCPU.                        */
   printf("Hello World from the CPU device (%s)\n", device -> name);

   return 0;
}

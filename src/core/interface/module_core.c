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
#include "ocean/core/interface/device_itf.h"
#include "ocean/core/interface/scalar_itf.h"
#include "ocean/core/interface/tensor_itf.h"
#include "ocean.h"


/* ===================================================================== */
/* Global module structure                                               */
/* ===================================================================== */

OcModule oc_module_core = OC_INIT_STATIC_MODULE("core", OcModuleCore);
int      oc_module_core_initialized = 0;


/* ===================================================================== */
/*                          Module initialization                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcModuleCore_initialize(void)
/* -------------------------------------------------------------------- */
{
   if (oc_module_core_initialized) return 0;

   /* Initialize the tensor interface */
   if (OcModuleCore_initializeScalarItf() != 0) return -1;
   if (OcModuleCore_initializeTensorItf() != 0) return -1;

   oc_module_core_initialized = 1;
   return 0;
}


/* -------------------------------------------------------------------- */
OcModuleCore_Context *OcModuleCore_createContext(OcDevice *device, size_t size)
/* -------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;
   int i, j;

   /* Allocate the context */
   if ((ctx = (OcModuleCore_Context *)OcMalloc(size)) == NULL)
      OcError(NULL, "Error allocating the core module context for device %s", device -> name);

   /* Initialize scalar pointers */
   ctx -> scalarCount = 8; /* MUST match definition size */
   for (i = 0; i < OC_DTYPE_COUNT; i++)
   {  ctx -> scalarIndex[i] = 0;
      for (j = 0; j < ctx -> scalarCount; j++)
      {  ctx -> scalarList[i][j] = NULL;
      }
   }

   /* Initialize the context */
   ctx -> bufferCount   = 3;
   ctx -> bufferIndex   = 0;
   ctx -> bufferMaxSize = 0; /* No limit */
   ctx -> bufferList    = (OcStorage **)OcMalloc(sizeof(OcStorage *) * ctx -> bufferCount);
   if (ctx -> bufferList == NULL)
   {  OcFree(ctx);
      OcError(NULL, "Error initializing buffer storage for device %s", device -> name);
   }
   for (i = 0; i < ctx -> bufferCount; i++) ctx -> bufferList[i] = NULL;

   return ctx;
}


/* -------------------------------------------------------------------- */
void OcModuleCore_freeContext(OcModuleCore_Context *context)
/* -------------------------------------------------------------------- */
{  int i, j;

   if (context == NULL) return;
   
   /* Finalize the dynamically allocated fields */
   if (context -> bufferList)
   {  for (i = 0; i < context -> bufferCount; i++)
      {  OcXDecrefStorage(context -> bufferList[i]);
      }
      OcFree(context -> bufferList);
   }

   /* Finalize all temporary scalars */
   for (i = 0; i < OC_DTYPE_COUNT; i++)
   {  for (j = 0; j < context -> scalarCount; j++)
      {  OcXDecrefTensor(context -> scalarList[i][j]);
      }
   }

   /* Free the context */
   OcFree(context);
}


/* -------------------------------------------------------------------- */
OcModuleCore_Context *OcModuleCore_getContext(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;

   ctx = (OcModuleCore_Context *)(OC_GET_DEVICE_CONTEXT(device, oc_module_core));
   if (ctx == NULL)
      OcError(NULL, "Error retrieving core module context information for device %s", device -> name);
   return ctx;
}

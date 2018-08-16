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
#include "ocean/core/cpu/device_cpu.h"
#include "ocean/core/cpu/module_core_cpu.h"

#include "ocean/base/ocean.h"
#include "ocean/base/types.h"
#include "ocean/base/malloc.h"
#include "ocean/base/warning.h"
#include "ocean/base/error.h"

#include <stdlib.h>


/* Structure for list of finalize handle functions */
typedef struct __OcFinalizeHandler
{   void (*funptr)(void);
    const char *desc;
    struct __OcFinalizeHandler *next;
} OcFinalizeHandler;

static OcFinalizeHandler *oc_finalize_handlers = NULL;
static OcSize             oc_ocean_references = 0;



/* -------------------------------------------------------------------- */
int OcInitializeIntrnl(void)
/* -------------------------------------------------------------------- */
{
   /* Initialize memory allocation */
   OcMallocInit();

   /* Increment Ocean */
   OcIncrefOcean();

   return 0;
}


/* -------------------------------------------------------------------- */
void OcFinalizeIntrnl(void)
/* -------------------------------------------------------------------- */
{  OcFinalizeHandler *handler;

   /* Finalize all parts */
   while ((handler = oc_finalize_handlers) != NULL)
   {
      /* Call the finalize function */
      if (handler -> funptr) handler -> funptr();

      /* Get the next function */
      oc_finalize_handlers = handler -> next;

      /* Free the current handler */
      OcFree(handler);
   }

   /* Decrement Ocean */
   OcDecrefOcean();
}


/* -------------------------------------------------------------------- */
void OcShutdownIntrnl(void)
/* -------------------------------------------------------------------- */
{
   /* Finalize warning information */
   OcWarning_finalize();

   /* Finalize memory allocation */
   OcMallocFinalize();
}


/* -------------------------------------------------------------------- */
void OcIncrefOcean(void)
/* -------------------------------------------------------------------- */
{
   oc_ocean_references ++;
}


/* -------------------------------------------------------------------- */
void OcDecrefOcean(void)
/* -------------------------------------------------------------------- */
{
   if (oc_ocean_references <= 1)
   {  oc_ocean_references = 0;
      OcShutdownIntrnl();
   }
   else
   {  oc_ocean_references --;
   }
}

/* -------------------------------------------------------------------- */
int OcFinalizeAddHandler(void (*funptr)(void), const char *desc)
/* -------------------------------------------------------------------- */
{  OcFinalizeHandler *handler;

   /* Allocate handler structure */
   handler = (OcFinalizeHandler *)OcMalloc(sizeof(OcFinalizeHandler));
   if (handler == NULL)
   {  /* Call the handler and return an error */
      if (funptr) funptr();
      OcError(-1, "Error allocating memory for finalize handler");
   }

   /* Initialize handler */
   handler -> funptr = funptr;
   handler -> desc   = desc;

   /* Add the handler */
   handler -> next = oc_finalize_handlers;
   oc_finalize_handlers = handler;

   return 0;
}

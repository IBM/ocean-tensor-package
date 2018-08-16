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

#include "ocean/core/cpu/device_cpu.h"
#include "ocean/core/cpu/device_info.h"
#include "ocean/base/arch.h"
#include "ocean.h"

#include <string.h>
#include <stdlib.h>


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

static int OcDeviceCPU_formatInfo(OcDevice *device, char **str, const char *header, const char *footer);


/* ===================================================================== */
/* Function implemenations                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcRegisterDeviceCPU(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the core module function pointers */
   module -> Device_formatInfo = OcDeviceCPU_formatInfo;
}


/* -------------------------------------------------------------------- */
static int OcDeviceCPU_formatInfo(OcDevice *device, char **str,
                                  const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  const char *processor = OC_ARCH_NAME;
   char       *s = NULL, *buffer = NULL;
   int         k, mode, newlineWidth;
   size_t      slen, n;

   /* Newline width */
   newlineWidth = strlen("\n");

   for (mode = 0; mode < 2; mode ++)
   {  slen = 0;

      /* Header */
      if (header != NULL)
      {  k = strlen(header); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", header);
      }

      /* Device */
      n = strlen(processor);
      k = 10 + (n > 0) * (3 + n) + ((footer) ? newlineWidth : 0);
      if (mode == 1) s += snprintf(s, k+1, "Device CPU%s%s%s", (n > 0) ? " - " : "", (n > 0) ? processor : "", (footer) ? "\n" : "");

      /* Footer */
      if (footer != NULL)
      {  k = strlen(footer); slen += k;
         if (mode == 1) s += snprintf(s, k+1, "%s", footer);
      }

      /* Allocate memory for the string */
      if (mode == 0)
      {
         /* ------------------------------------------------------------- */
         /* Allocate the memory for the string. We use a regular malloc   */
         /* here instead of OcMalloc to ensure that the library can be    */
         /* recompiled with new memory allocation routines without having */
         /* to recompile any language bindings.                           */
         /* ------------------------------------------------------------- */
         buffer = (char *)malloc(sizeof(char) * (slen + 1));
         s = buffer; *str = buffer;
         if (buffer == NULL) OcError(-1, "Insufficient memory for output string");
      }
   }  
   
   /* Ensure that the string is terminated properly */
   *s = '\0';

   return 0;
}


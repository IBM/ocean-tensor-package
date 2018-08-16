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
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <stdlib.h>
#include <string.h>



/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcDevice_supportsTensorByteswap(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcModuleCore *module;

   /* Look up the module */
   module = (OcModuleCore *)(OC_GET_MODULE(oc_module_core, device));

   return ((module == NULL) || (module -> Tensor_byteswapNoFlag == NULL)) ? 0 : 1;
}


/* -------------------------------------------------------------------- */
int OcDevice_getMaxBufferSize(OcDevice *device, OcSize *maxBufferSize)
/* -------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;

   /* Get the core module device context */
   if ((ctx = OcModuleCore_getContext(device)) == NULL) return -1;

   /* Retrieve the maximum buffer size; */
   *maxBufferSize = ctx -> bufferMaxSize;
   return 0;
}


/* -------------------------------------------------------------------- */
int OcDevice_setMaxBufferSize(OcDevice *device, OcSize maxBufferSize)
/* -------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;
   OcStorage *buffer;
   int i;

   /* Get the core module device context */
   if ((ctx = OcModuleCore_getContext(device)) == NULL) return -1;

   /* Set the maximum buffer size */
   if (maxBufferSize < 0) maxBufferSize = 0;
   ctx -> bufferMaxSize = maxBufferSize;
   if ((maxBufferSize > 0) && (ctx -> bufferList != NULL))
   {  for (i = 0; i < ctx -> bufferCount; i++)
      {  buffer = ctx -> bufferList[i];
         if ((buffer) && (buffer -> capacity > maxBufferSize))
         {  OcDecrefStorage(buffer);
            ctx -> bufferList[i] = NULL;
         }
      }
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int OcDevice_getBufferCount(OcDevice *device, int *count)
/* -------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;

   /* Get the core module device context */
   if ((ctx = OcModuleCore_getContext(device)) == NULL) return -1;

   /* Retrieve the number of buffers */
   *count = ctx -> bufferCount;
   return 0;
}


/* -------------------------------------------------------------------- */
int OcDevice_setBufferCount(OcDevice *device, int count)
/* -------------------------------------------------------------------- */
{  OcModuleCore_Context *ctx;
   OcStorage **bufferList;
   int i;

   /* Get the core module device context */
   if ((ctx = OcModuleCore_getContext(device)) == NULL) return -1;

   if (count < 0) count = 0;
   if (count > ctx -> bufferCount)
   {  bufferList = (OcStorage **)OcMalloc(sizeof(OcStorage *) * count);
      if (bufferList== NULL) OcError(-1, "Error allocating buffer list for device %s", device -> name);

      /* Copy existing buffers */
      for (i = 0; i < ctx -> bufferCount; i++) bufferList[i] = ctx -> bufferList[i];

      /* Reset the remaining buffers */
      for ( ; i < count; i++) bufferList[i] = NULL;
      
      /* Replace the buffer array */
      if (ctx -> bufferList) OcFree(ctx -> bufferList);
      ctx -> bufferList = bufferList;

      /* Update the buffer index */
      ctx -> bufferIndex = ctx -> bufferCount;
   }
   else
   {  /* Decref buffers that exceed the new count */
      for (i = count; i < ctx -> bufferCount; i++)
      {  OcXDecrefStorage(ctx -> bufferList[i]);
      }

      /* Update the buffer index */
      if (ctx -> bufferIndex >= count) ctx -> bufferIndex = 0;
   }

   /* Update the buffer count */
   ctx -> bufferCount = count;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcDevice_format(OcDevice *device, char **str, const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  size_t  n;

   n = 11; /* "<device '*'>" */
   n += strlen(device -> name);
   if (header) n += strlen(header);
   if (footer) n += strlen(footer);
   
   /* Allocate memory for the output string */
   *str = (char *)malloc(sizeof(char) * (n+1));
   if (*str == NULL) OcError(-1, "Insufficient memory for output string");

   /* Generate the string */
   sprintf(*str, "%s<device '%s'>%s", (header == NULL) ? "" : header,
                                      device -> name,
                                      (footer == NULL) ? "" : footer);

   return 0;
}


/* -------------------------------------------------------------------- */
int OcDevice_display(OcDevice *device)
/* -------------------------------------------------------------------- */
{  char *str = NULL;
   int   result;

   /* Format and display the device */
   result = OcDevice_format(device, &str, NULL, NULL);
   if (result == 0)
   {  printf("%s", str);
   }

   /* Deallocate memory */
   if (str) free(str);

   return result;
}


/* -------------------------------------------------------------------- */
int OcDevice_formatInfo(OcDevice *device, char **str, const char *header, const char *footer)
/* -------------------------------------------------------------------- */
{  int (*funptr)(OcDevice *device, char **str, const char *header, const char *footer);

   /* Look up the DeviceFormatInfo function */
   if ((funptr = OC_GET_CORE_FUNCTION(device, Device_formatInfo)) == 0)
   {
      /* Allocate memory */
      *str = (char *)malloc(sizeof(char) * (strlen(device -> name) + 8));
      if (*str == NULL) OcError(-1, "Insufficient memory for output string");

      /* Return a copy of the name */
      sprintf(*str, "Device %s", device -> name);
      return 0;
   }

   /* Call the function */
   return funptr(device, str, header, footer);
}


/* -------------------------------------------------------------------- */
int OcDevice_displayInfo(OcDevice *device)
/* -------------------------------------------------------------------- */
{  char *str = NULL;
   int   result;

   /* Format and display the device information */
   result = OcDevice_formatInfo(device, &str, NULL, NULL);
   if (result == 0)
   {  printf("%s", str);
   }

   /* Deallocate memory */
   if (str) free(str);

   return result;
}

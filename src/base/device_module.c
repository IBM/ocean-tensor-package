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
#include "ocean/base/device_module.h"
#include "ocean/base/device.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <string.h>



/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

/* Number of registered modules */
static int oc_device_module_count  = 0;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

int  OcDeviceTypePreallocateLUT(OcDeviceType *deviceType, size_t size);
void OcDeviceTypePrepareModule(OcDeviceType *deviceType, OcModule *module);
int  OcDevicePreallocateContext(OcDevice *device, int elements);
int  OcDeviceInitializeContext(OcDevice *device, OcDeviceModule *deviceModule);
void OcDeviceFinalizeContext(OcDevice *device, OcDeviceModule *deviceModule);



/* ===================================================================== */
/* Function implementations - Device module                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcDeviceModuleInitialize(void)
/* -------------------------------------------------------------------- */
{
   /* Initialize device modules */
   oc_device_module_count  = 0;

   return 0;
}


/* -------------------------------------------------------------------- */
void OcDeviceModuleFinalize(void)
/* -------------------------------------------------------------------- */
{
}



/* ===================================================================== */
/* Function implementations - Device type initialization                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcDeviceTypeInitializeModules(OcDeviceType *deviceType)
/* -------------------------------------------------------------------- */
{  OcDeviceType *template;
   OcDeviceModule *deviceModule;
   size_t          offset = 0;

   /* We use any existing device type as an example for the look-up */
   /* table initialization. Since modules will register a device    */
   /* type if needed, we can return immediately if no device types  */
   /* are available.                                                */
   if (OcDeviceTypeCount() == 0) return 0;

   /* Get the template device type */
   template = OcDeviceTypeByIndex(0);

   /* Allocate the LUT buffer */
   if (template -> lutSize == 0) return 0;
   if (OcDeviceTypePreallocateLUT(deviceType, template -> lutSize) != 0)
      OcError(-1, "Error allocating memory for device type look-up table");

   /* Initialize the LUT */
   while ((deviceModule = OcDeviceTypeNextModule(template, &offset)))
   {  /* Prepare an empty module */
      OcDeviceTypePrepareModule(deviceType, deviceModule -> module);
   }

   /* Update the look-up table size */
   deviceType -> lutSize = template -> lutSize;

   return 0;
}


/* -------------------------------------------------------------------- */
void OcDeviceTypeFinalizeModules(OcDeviceType *deviceType)
/* -------------------------------------------------------------------- */
{  OcDeviceModule *deviceModule;
   size_t          offset = 0;

   /* Finalize all device modules */
   while ((deviceModule = OcDeviceTypeNextModule(deviceType, &offset)))
   {  if ((deviceModule -> available) &&
          (deviceModule -> finalizeModule != 0))
      {  deviceModule -> finalizeModule(deviceModule);
      }
   }

   /* Free the look-up table buffer */
   if (deviceType -> lutBuffer)
   {  OcFreeAligned(deviceType -> lutBuffer);
      deviceType -> lutBuffer = NULL;
   }
}



/* ===================================================================== */
/* Function implementations - Device initialization                      */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcDeviceInitializeModules(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcDeviceModule *deviceModule;
   size_t          offset = 0, n;

   /* Determine the initial context buffer size */
   for (n = 32; n < oc_device_module_count; n*=2) ;

   /* Allocate the context buffer */
   if (OcDevicePreallocateContext(device, n) != 0) return -1;

   /* Initialize the context for each of the device modules */
   while ((deviceModule = OcDeviceTypeNextModule(device -> type, &offset)))
   {  if (OcDeviceInitializeContext(device, deviceModule) != 0)
      {  return -1;
      }
   }

   return 0;
}


/* -------------------------------------------------------------------- */
void OcDeviceFinalizeModules(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcDeviceModule *deviceModule;
   size_t          offset = 0;
  
    /* Clean up all context variables */
   while ((deviceModule = OcDeviceTypeNextModule(device -> type, &offset)))
   {  OcDeviceFinalizeContext(device, deviceModule);
   }
}



/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcDeviceTypePreallocateLUT(OcDeviceType *deviceType, size_t size)
/* -------------------------------------------------------------------- */
{  char   *buffer;
   size_t  n;


   /* Return immediately if sufficient memory is available */
   if (deviceType -> lutSize + size <= deviceType -> lutCapacity)
      return 0;

   /* Determine the new capacity */
   n = (deviceType -> lutCapacity == 0) ? 1 : deviceType -> lutCapacity;
   while (n < deviceType -> lutSize + size) n *= 2;

   /* Allocate memory */
   buffer = (char *)OcMallocAligned(sizeof(char) * n);
   if (buffer == NULL) OcError(-1, "Error allocating memory for the LUT");

   /* Copy the existing data and zero pad the remainder */
   memcpy(buffer, deviceType -> lutBuffer, sizeof(char) * deviceType -> lutSize);
   memset(buffer + deviceType -> lutSize, 0, sizeof(char) * (n - deviceType -> lutSize));

   /* Free the existing buffer */
   if (deviceType -> lutBuffer) OcFreeAligned(deviceType -> lutBuffer);

   /* Update the buffer information */
   deviceType -> lutBuffer = buffer;
   deviceType -> lutCapacity = n;

   return 0;
}


/* -------------------------------------------------------------------- */
void OcDeviceTypePrepareModule(OcDeviceType *deviceType, OcModule *module)
/* -------------------------------------------------------------------- */
{  OcDeviceModule *deviceModule;

   /* Get the device module */
   deviceModule = (OcDeviceModule *)(deviceType -> lutBuffer + module -> offset);

   /* Initialize basic fields */
   deviceModule -> module    = OcIncrefModule(module);
   deviceModule -> data      = 0;
   deviceModule -> available = 0;

   /* Initialize function pointers */
   deviceModule -> initializeContext  = 0;
   deviceModule -> finalizeContext    = 0;
   deviceModule -> finalizeModule     = 0;
}


/* -------------------------------------------------------------------- */
int OcDevicePreallocateContext(OcDevice *device, int elements)
/* -------------------------------------------------------------------- */
{  void  **buffer;
   size_t  n;

   /* Return immediately if sufficient memory is available */
   if (device -> contextSize + elements <= device -> contextCapacity)
      return 0;

   /* Determine the new capacity */
   n = (device -> contextCapacity == 0) ? 1 : device -> contextCapacity;
   while (n < device -> contextSize + elements) n *= 2;

   /* Allocate memory */
   buffer = (void **)OcMalloc(sizeof(void *) * n);
   if (buffer == NULL) OcError(-1, "Error allocating context buffer for device %s", device -> name);

   /* Copy the existing data and zero pad the remainder */
   memcpy(buffer, device -> contextBuffer, sizeof(void *) * device -> contextSize);
   memset(buffer + device -> contextSize, 0, sizeof(void *) * (n - device -> contextSize));

   /* Free the existing buffer */
   if (device -> contextBuffer) OcFree(device -> contextBuffer);

   /* Update the buffer information */
   device -> contextBuffer = buffer;
   device -> contextCapacity = n;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcDeviceInitializeContext(OcDevice *device, OcDeviceModule *deviceModule)
/* -------------------------------------------------------------------- */
{  void *context = NULL;

   if (deviceModule -> initializeContext != 0)
   {  if (deviceModule -> initializeContext(device, deviceModule, &context) == 0)
         device -> contextBuffer[deviceModule -> module -> index] = context;
      else
         OcError(-1, "Error initializing the context for module %s on device %s", deviceModule->module->name, device->name);
   }

   return 0;
}


/* -------------------------------------------------------------------- */
void OcDeviceFinalizeContext(OcDevice *device, OcDeviceModule *deviceModule)
/* -------------------------------------------------------------------- */
{  void *context;

   if (deviceModule -> finalizeContext != 0)
   {  context = device -> contextBuffer[deviceModule -> module -> index];

      /* Call the finalize context function, even when context is NULL */
      deviceModule -> finalizeContext(device, deviceModule, context);
   }
}


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcDeviceModule *OcDeviceTypeNextModule(OcDeviceType *deviceType, size_t *offset)
/* -------------------------------------------------------------------- */
{  OcDeviceModule *module;

   if (*offset >= deviceType -> lutSize) return NULL;

   module   = (OcDeviceModule *)(deviceType -> lutBuffer + *offset);
   *offset += module -> module -> blockSize;

   /* Return a borrowed reference */
   return module;
}


/* -------------------------------------------------------------------- */
int OcRegisterModule(const char *deviceTypeStr, OcDeviceModule *info, size_t size)
/* -------------------------------------------------------------------- */
{  OcDeviceType   *type, *deviceType;
   OcModule       *module;
   OcDevice       *device;
   OcDeviceModule *deviceModule;
   int             deviceTypeCount;
   int             deviceCount;
   int             i;

   /* Extract the module information */
   module = info -> module;
   if (module == NULL) OcError(-1, "Module field cannot be NULL");

   /* Ensure the size of the modules matches; a mismatch would indicate */
   /* that the interface and module implementation were compiled with   */
   /* different versions of the code.                                   */
   if (size != module -> size) OcError(-1, "Mismatch in module interface and implementation size - recompile needed");

   /* Determine the padded block size */
   if ((module -> size % OC_MEMORY_BYTE_ALIGNMENT) == 0)
        module -> blockSize = module -> size;
   else module -> blockSize = module -> size + (OC_MEMORY_BYTE_ALIGNMENT - (module -> size % OC_MEMORY_BYTE_ALIGNMENT));

   /* Get the device type */
   type = OcCreateDeviceType(deviceTypeStr);
   if (type == NULL) OcError(-1, "Could not create entry for device type %s", deviceTypeStr);

    /* Get device and device type counts */
    deviceTypeCount = OcDeviceTypeCount();
    deviceCount     = OcDeviceCount();

   /* Create module for all device types */
   if (module -> initialized == 0)
   {  
      /* Pre-allocate space in the LUT buffers */
      for (i = 0; i < deviceTypeCount; i++)
      {  if (OcDeviceTypePreallocateLUT(OcDeviceTypeByIndex(i), module -> blockSize) != 0)
         {  OcError(-1, "Error allocating look-up table for device type %s", OcDeviceTypeByIndex(i) -> name);
         }
      }

      /* Allocate context buffers for all devices */
      for (i = 0; i < deviceCount; i++)
      {  if (OcDevicePreallocateContext(OcDeviceByIndex(i), 1) != 0)
         {  OcError(-1, "Error allocating context buffer for device %s", OcDeviceByIndex(i) -> name);
         }
      }

      /* Initialize the module */
      module -> offset      = type -> lutSize;
      module -> index       = oc_device_module_count;
      module -> initialized = 1;

      /* Increment the contextSize for all devices */
      for (i = 0; i < deviceCount; i++)
      {  OcDeviceByIndex(i) -> contextSize ++;
      }

      /* Increase the number of modules */
      oc_device_module_count ++;

      /* Initialize the module on all device types and update the LUT size */
      for (i = 0; i < deviceTypeCount; i++)
      {  deviceType = OcDeviceTypeByIndex(i);
         OcDeviceTypePrepareModule(deviceType, module);
         deviceType -> lutSize += module -> blockSize;
      }
   }

   /* Get the device module structure */
   deviceModule = (OcDeviceModule *)(type -> lutBuffer + module -> offset);

   /* Activate the module for the given device type */
   if (deviceModule -> available == 0)
   {
      /* Copy the generic OcDeviceModule part to ensure the functions for */
      /* context initialization and finalization are set. This operation  */
      /* also overwrites the module field, but as the value of the pointer*/
      /* is the same as before, this poses no problem.                    */
      memcpy((void *)deviceModule, (void *)info, sizeof(OcDeviceModule));

      /* Allocate the context for all instances of the given device type. */
      /* If this step fails for any of the devices we reset the context   */
      /* for all instances and return without setting the available flag. */
      /* in the device module.                                            */
      for (i = 0; i < deviceCount; i++)
      {  device = OcDeviceByIndex(i);
         if (device -> type == type)
         {  if (OcDeviceInitializeContext(device, deviceModule) != 0)
            {  for ( ; i >= 0; i--)
               {  device = OcDeviceByIndex(i);
                  if (device -> type == type)
                  {  OcDeviceFinalizeContext(device, deviceModule);
                  }
               }
               return -1;
            }
         }
      }

      /* Copy the entire look-up table for the current device type */
      memcpy((void *)deviceModule, (void *)info, module -> size * sizeof(char));

      /* Set the available flag for the device module */
      deviceModule -> available = 1;
   }

   return 0;
}

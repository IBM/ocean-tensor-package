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

#include "ocean/base/device.h"
#include "ocean/base/device_module.h"
#include "ocean/base/ocean.h"
#include "ocean/base/malloc.h"
#include "ocean/base/error.h"

#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <ctype.h>


/* ===================================================================== */
/* Local variables                                                       */
/* ===================================================================== */

/* Initialization flag */
static int            oc_device_initialized   = 0;

/* List of device types */
static OcDeviceType **oc_device_types         = NULL;
static int            oc_device_type_count    = 0;
static int            oc_device_type_capacity = 0;

/* List of device instances */
static OcDevice     **oc_devices              = NULL;
static int            oc_device_count         = 0;
static int            oc_device_capacity      = 0;

/* Default device */
static OcDevice      *oc_default_device       = NULL;


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

void OcFinalizeDevices(void);
void OcDeallocateDevice(OcDevice *device);


/* ===================================================================== */
/* Function implementations                                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcInitDevices(void)
/* -------------------------------------------------------------------- */
{
   /* Make sure we are initialized only once */
   if (oc_device_initialized)
      OcError(-1, "Device functions were already initialized");

   /* Initialize the device modules */
   if (OcDeviceModuleInitialize() != 0) return -1;

   /* Initialize device types */
   oc_device_types         = NULL;
   oc_device_type_count    = 0;
   oc_device_type_capacity = 0;

   /* Initialize device instances */
   oc_devices              = NULL;
   oc_device_count         = 0;
   oc_device_capacity      = 0;

   /* Successfully initialized */
   oc_device_initialized = 1;

   /* Register finalization function */
   if (OcFinalizeAddHandler(OcFinalizeDevices, "Finalize devices") != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
void OcFinalizeDevices(void)
/* -------------------------------------------------------------------- */
{  int i;

   /* Set the default device to NULL */
   OcDevice_setDefault(NULL);

   /* Finalize can be called even if the module was not initialized. */
   if (oc_device_initialized == 0) return ;
   oc_device_initialized = 0;

   /* Decrement the counter for all device instances */
   for (i = 0; i < oc_device_count; i++)
   {  OcDecrefDevice(oc_devices[i]);
   }

   /* Decrement the counter for all device types */
   for (i = 0; i < oc_device_type_count; i++)
   {  OcDecrefDeviceType(oc_device_types[i]);
   }

   /* Clean up the arrays */
   if (oc_devices) { OcFree(oc_devices); oc_devices = NULL; }
   if (oc_device_types) { OcFree(oc_device_types); oc_device_types = NULL; }

    /* Clean up the device modules */
    OcDeviceModuleFinalize();
}



/* ===================================================================== */
/* Function implementations - Reference counting                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcDeviceType *OcIncrefDeviceType(OcDeviceType *deviceType)
/* -------------------------------------------------------------------- */
{  /* Returns deviceType to allow: type = ocIncrefDeviceType(type) */

   if (deviceType != NULL) deviceType -> refcount ++;
   return deviceType;
}


/* -------------------------------------------------------------------- */
void OcDecrefDeviceType(OcDeviceType *deviceType)
/* -------------------------------------------------------------------- */
{  
   if (deviceType == NULL) return;

   deviceType -> refcount --;
   if (deviceType -> refcount == 0)
   {
      /* Finalize all device modules and the look-up table */
      OcDeviceTypeFinalizeModules(deviceType);

      /* Free the device type fields */
      if (deviceType -> name) { OcFree(deviceType -> name); deviceType -> name = NULL; }

      /* Free the device type structure */
      if (deviceType -> devices) OcFree(deviceType -> devices);
      OcFree(deviceType);

      /* Decrement Ocean: device types are among the last objects */
      /* to be deleted. When Ocean is used from languages with    */
      /* automatic garbage collection it is possible for devices  */
      /* and device types to remain until after OcFinalize has    */
      /* been called. To ensure that Ocean is correctly shut down */
      /* after all objects have been deleted we increment and     */
      /* decrement the Ocean counter.                             */
      OcDecrefOcean();
   }
}


/* -------------------------------------------------------------------- */
OcDevice *OcIncrefDevice(OcDevice *device)
/* -------------------------------------------------------------------- */
{  /* Returns device to allow: device = ocIncrefDevice(device) */

   if (device != NULL) device -> refcount ++;
   return device;
}


/* -------------------------------------------------------------------- */
void OcDecrefDevice(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   if (device == NULL) return;

   device -> refcount --;
   if (device -> refcount <= 1)
   {  OcStream *stream = device -> defaultStream;

      /* The default stream also contains a reference to  */
      /* the device. Make sure that the device gets freed */
      /* if there are no other references to the default  */
      /* stream.                                          */
      if ((stream) && (stream -> refcount == 1))
      {  /* Decrement the stream count, which will in turn*/
         /* decrement the device count again and finalize */
         /* the device. We reset the default device to    */
         /* avoid errors on re-entry.                     */
         device -> defaultStream = NULL;
         OcDecrefStream(stream);
      }
      else if (device -> refcount == 0)
      {  /* Deallocate the device */
         OcDeallocateDevice(device);
      }
   }
}


/* -------------------------------------------------------------------- */
OcStream *OcIncrefStream(OcStream *stream)
/* -------------------------------------------------------------------- */
{  /* Returns stream to allow: stream = ocIncrefStorage(stream) */

   /* Increment the reference count */
   if (stream != NULL) stream -> refcount ++;

   return stream;
}


/* -------------------------------------------------------------------- */
void OcDecrefStream(OcStream *stream)
/* -------------------------------------------------------------------- */
{
   if (stream == NULL) return;

   /* Decrement the reference count and finalize if needed */
   if ((--(stream -> refcount)) == 0)
   {
      OcStream_synchronize(stream);
      stream -> device -> delete_stream(stream);
   }
}


/* ===================================================================== */
/* Function implementations - Device type functions                      */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcAppendDeviceType(OcDeviceType *deviceType)
/* -------------------------------------------------------------------- */
{  OcDeviceType **list;
   int            i,n;

   /* Increase the device type list capacity if needed */
   if (oc_device_type_count == oc_device_type_capacity)
   {
      /* Determine the new capacity of the list */
      n = (oc_device_type_capacity == 0) ? 1 : 2 * oc_device_type_capacity;

      /* Allocate the new list */
      list = (OcDeviceType **)OcMalloc(sizeof(OcDeviceType *) * n);
      if (list == NULL) OcError(-1, "Error allocating memory for the device type list");

      /* Copy existing items and pad with NULL values */
      for (i = 0; i < oc_device_type_count; i++)
      {  list[i] = oc_device_types[i];
      }
      for ( ; i < n; i++)
      {  list[i] = NULL;
      }

      /* Free the existing list and set the new one */
      if (oc_device_types) OcFree(oc_device_types);
      oc_device_types         = list;
      oc_device_type_capacity = n;
   }

   /* Add the device type */
   oc_device_types[oc_device_type_count] = deviceType;
   oc_device_type_count ++;

   return 0;
}


/* -------------------------------------------------------------------- */
OcDeviceType *OcAllocateDeviceType(const char *name)
/* -------------------------------------------------------------------- */
{  OcDeviceType *result;
   char         *ptr;

   /* Parameter checks */
   if (name == NULL) OcError(NULL, "Device type name cannot be NULL");

   /* Allocate memory for the structure */
   result = (OcDeviceType *)OcMalloc(sizeof(OcDeviceType));
   if (result == NULL) OcError(NULL, "Insufficient memory to allocate device type '%s'", name);

   /* Initialize the structure */
   result -> name        = NULL;
   result -> refcount    = 1;
   result -> devices     = NULL;
   result -> deviceCount = 0;
   result -> lutBuffer   = NULL;
   result -> lutSize     = 0;
   result -> lutCapacity = 0;

   /* Copy the name */
   result -> name = (char *)OcMalloc(sizeof(char) * (strlen(name) + 1));
   if (result -> name == NULL)
   {  OcDecrefDeviceType(result);
      OcError(NULL, "Insufficient memory to allocate device name '%s'", name);
   }
   else
   {  strcpy(result -> name, name);
      ptr = result -> name;
      while (*ptr) { *ptr = (char)toupper((int)(*ptr)); ptr ++; }
   }

   /* Increment Ocean: device types are among the last objects */
   /* to be deleted. When Ocean is used from languages with    */
   /* automatic garbage collection it is possible for devices  */
   /* and device types to remain until after OcFinalize has    */
   /* been called. To ensure that Ocean is correctly shut down */
   /* after all objects have been deleted we increment and     */
   /* decrement the Ocean counter.                             */
   OcIncrefOcean();

   return result;
}


/* -------------------------------------------------------------------- */
OcDeviceType *OcCreateDeviceType(const char *name)
/* -------------------------------------------------------------------- */
{  OcDeviceType *result;

   /* Check if the name exists */
   result = OcDeviceTypeByName(name);
   if (result != NULL) return result;

   /* Allocate the device type structure */
   result = OcAllocateDeviceType(name);
   if (result == NULL) return NULL;

   /* Initialize the device-type modules */
   if (OcDeviceTypeInitializeModules(result) != 0)
   {  OcDecrefDeviceType(result);
      return NULL;
   }

   /* Add the device type to the list */
   if (OcAppendDeviceType(result) != 0)
   {  OcDecrefDeviceType(result);
      return NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
OcDeviceType *OcDeviceTypeByName(const char *name)
/* -------------------------------------------------------------------- */
{  int i;

   /* Return an existing device type when available */
   for (i = 0; i < oc_device_type_count; i++)
   {  if (strcasecmp(oc_device_types[i] -> name, name) == 0)
      {  return oc_device_types[i];
      }
   }

   return NULL;
}


/* -------------------------------------------------------------------- */
int OcDeviceTypeCount(void)
/* -------------------------------------------------------------------- */
{
   return oc_device_type_count;
}


/* -------------------------------------------------------------------- */
OcDeviceType *OcDeviceTypeByIndex(int index)
/* -------------------------------------------------------------------- */
{
   if ((index < 0) || (index >= oc_device_type_count)) return NULL;

   return oc_device_types[index];
}


/* -------------------------------------------------------------------- */
int OcDeviceType_deviceCount(OcDeviceType *deviceType)
/* -------------------------------------------------------------------- */
{
   return deviceType -> deviceCount;
}


/* -------------------------------------------------------------------- */
OcDevice *OcDeviceType_getDevice(OcDeviceType *deviceType, int index)
/* -------------------------------------------------------------------- */
{
   if ((index < 0) || (index >= deviceType -> deviceCount)) return NULL;

   return deviceType -> devices[index];
}



/* ===================================================================== */
/* Function implementations - Device functions                           */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcDevice *OcAllocateDevice(const char *name, size_t size)
/* -------------------------------------------------------------------- */
{  OcDevice *result;

   /* Allocate memory for the structure */
   if (size < sizeof(OcDevice)) size = sizeof(OcDevice);
   result = (OcDevice *)OcMalloc(size);
   if (result == NULL) OcError(NULL, "Insufficient memory to allocate device instance");

   /* Copy the name */
   result -> name = (char *)OcMalloc(sizeof(char) * (strlen(name) + 1));
   if (result -> name == NULL)
   {  OcDecrefDevice(result);
      OcError(NULL, "Insufficient memory to allocate device name '%s'", name);
   }
   else
   {  strcpy(result -> name, name);
   }

   /* Initialize the structure */
   result -> type            = NULL;
   result -> index           = 0;
   result -> refcount        = 1;
   result -> contextBuffer   = NULL;
   result -> contextSize     = 0;
   result -> contextCapacity = 0;
   result -> defaultStream   = NULL;
   result -> finalize        = 0;

   return result;   
}


/* -------------------------------------------------------------------- */
void OcDeallocateDevice(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcDeviceType *deviceType = NULL;

   /* Clean up all context variables */
   OcDeviceFinalizeModules(device);

   /* Deregister from the device type */
   if ((deviceType = device -> type) != NULL)
   {  if ((deviceType -> devices) &&
          (deviceType -> deviceCount > device -> index))
      {  deviceType -> devices[device -> index] = NULL;
      }
   }

   /* Free default stream */
   if (device -> defaultStream) OcDecrefStream(device -> defaultStream);

   /* Free device-specific information */
   if (device -> finalize) device -> finalize(device);

   /* Free the device fields */
   if (device -> name         ) { OcFree(device -> name); device -> name = NULL; }
   if (device -> contextBuffer) { OcFree(device -> contextBuffer); device -> contextBuffer = NULL; }

   /* Free the device structure */
   OcFree(device);

   /* Decrement the device type; this is done at the end */
   /* to ensure proper shutdown of Ocean.                */
   if (deviceType)  OcDecrefDeviceType(deviceType);
}


/* -------------------------------------------------------------------- */
OcDevice *OcCreateDevice(OcDeviceType *type, int index, const char *name, size_t size)
/* -------------------------------------------------------------------- */
{  OcDevice *result;

   /* Create a new device structure */
   result = OcAllocateDevice(name, size);
   if (result == NULL) return NULL;

   /* Set the device type, index, and default free function */
   result -> type  = OcIncrefDeviceType(type);
   result -> index = index;

   /* Initialize the device-specific module contexts */
   if (OcDeviceInitializeModules(result) != 0)
   {  OcDecrefDevice(result);
      return NULL;
   }

   return result;
}


/* -------------------------------------------------------------------- */
int OcAppendDevice(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcDevice **list;
   int        i,n;

   /* Add the device to the list of all devices, as well as to the */
   /* list of device instances of the associated device type.      */

   /* Increase the device list capacity if needed */
   if (oc_device_count == oc_device_capacity)
   {
      /* Determine the new capacity of the list */
      n = (oc_device_capacity == 0) ? 1 : 2 * oc_device_capacity;

      /* Allocate the new list */
      list = (OcDevice **)OcMalloc(sizeof(OcDevice *) * n);
      if (list == NULL) OcError(-1, "Error allocating memory for the device list");

      /* Copy existing items and pad with NULL values */
      for (i = 0; i < oc_device_count; i++)
      {  list[i] = oc_devices[i];
      }
      for ( ; i < n; i++)
      {  list[i] = NULL;
      }

      /* Free the existing list and set the new one */
      if (oc_devices) OcFree(oc_devices);
      oc_devices = list;
      oc_device_capacity = n;
   }

   /* Increase the device type capacity if needed */
   if (device -> index >= device -> type -> deviceCount)
   {
      /* Determine the new capacity of the list */
      n = (device -> index) + 1;

      /* Allocate the new list */
      list = (OcDevice **)OcMalloc(sizeof(OcDevice *) * n);
      if (list == NULL) OcError(-1, "Error allocating memory for the device list");

      /* Copy existing items and pad with NULL values */
      for (i = 0; i < device -> type -> deviceCount; i++)
      {  list[i] = device -> type -> devices[i];
      }
      for ( ; i < n; i++)
      {  list[i] = NULL;
      }
   
      /* Free the existing list and set the new one */
      if (device -> type -> devices) OcFree(device -> type -> devices);
      device -> type -> devices = list;
      device -> type -> deviceCount = n;
   }

   /* Add to the list of all devices */
   oc_devices[oc_device_count] = device;
   oc_device_count ++;

   /* Add to the list of device instances */
   device -> type -> devices[device -> index] = device;

   return 0;
}


/* -------------------------------------------------------------------- */
OcDevice *OcDevice_getCommonDevice(OcDevice *device1, OcDevice *device2)
/* -------------------------------------------------------------------- */
{
   /* Return the first device encountered */
   return device1 ? device1 : device2;
}


/* ===================================================================== */
/* Function implementations - Streams and events                         */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
OcStream *OcDevice_createStream(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   return device -> create_stream(device);
}


/* -------------------------------------------------------------------- */
OcStream *OcDevice_getDefaultStream(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   return device -> defaultStream;
}


/* -------------------------------------------------------------------- */
int OcDevice_setDefaultStream(OcDevice *device, OcStream *stream)
/* -------------------------------------------------------------------- */
{
   /* Make sure the stream has the same scheduler */
   if ((stream != NULL) && (stream -> device != device))
      OcError(-1, "Mismatch between device and stream");

   if (device -> defaultStream) OcDecrefStream(device -> defaultStream);

   device -> defaultStream = OcIncrefStream(stream);

   return 0;
}


/* -------------------------------------------------------------------- */
int OcStream_synchronize(OcStream *stream)
/* -------------------------------------------------------------------- */
{  OcDevice *device = stream -> device;

   if (device -> sync_stream)
        return device -> sync_stream(stream);
   else return 0;
}


/* -------------------------------------------------------------------- */
int OcStream_waitFor(OcStream *stream, OcEvent *event)
/* -------------------------------------------------------------------- */
{
   if (event == NULL) return 0;

   return stream -> device -> wait_event(stream, event);
}


/* -------------------------------------------------------------------- */
int OcDevice_createEvent(OcDevice *device, OcEvent **event)
/* -------------------------------------------------------------------- */
{
   if (device -> create_event == 0)
   {  *event = NULL;
      return 0;
   }

   return device -> create_event(device, event);
}


/* -------------------------------------------------------------------- */
void OcDevice_freeEvent(OcDevice *device, OcEvent *event)
/* -------------------------------------------------------------------- */
{
   if (event) device -> delete_event(device, event);
}


/* -------------------------------------------------------------------- */
int OcEvent_synchronize(OcDevice *device, OcEvent *event)
/* -------------------------------------------------------------------- */
{
   if (event == NULL) return 0;
   return device -> sync_event(event);
}


/* -------------------------------------------------------------------- */
int OcEvent_record(OcEvent *event, OcStream *stream)
/* -------------------------------------------------------------------- */
{
   if (event == NULL) return 0;
   return stream -> device -> record_event(event, stream);
}



/* ===================================================================== */
/* Function implementations - Miscellaneous                              */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcDevice_setDefault(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   /* Set the default device */
   if (oc_default_device) OcDecrefDevice(oc_default_device);
   if (device)
        oc_default_device = OcIncrefDevice(device);
   else oc_default_device = NULL;
}


/* -------------------------------------------------------------------- */
OcDevice *OcDevice_getDefault(void)
/* -------------------------------------------------------------------- */
{
   return oc_default_device;
}


/* -------------------------------------------------------------------- */
OcDevice *OcDevice_applyDefault(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   if (device != NULL) return device;
   if (oc_default_device != NULL) return oc_default_device;
   OcError(NULL, "A device must be specified if no default is set");
}


/* -------------------------------------------------------------------- */
int OcDeviceCount(void)
/* -------------------------------------------------------------------- */
{
   return oc_device_count;
}


/* -------------------------------------------------------------------- */
OcDevice *OcDeviceByIndex(int index)
/* -------------------------------------------------------------------- */
{
   if ((index < 0) || (index >= oc_device_count))
      OcError(NULL, "Device index out of bounds");

   return oc_devices[index];
}


/* -------------------------------------------------------------------- */
int OcRegisterDevice(OcDevice *device)
/* -------------------------------------------------------------------- */
{  int i;

   /* Check if device instance exists */
   for (i = 0; i < oc_device_count; i++)
   {  if ((oc_devices[i] -> type  == device -> type) &&
          (oc_devices[i] -> index == device -> index))
      {  OcDecrefDevice(device);
         OcError(-1, "Device instance %d of type %s already exists", device->index, device->type->name);
      }
   }

   /* Add device to list */
   if (OcAppendDevice(device) != 0)
   {  OcDecrefDevice(device);
      return -1;
   }

   return 0;
}

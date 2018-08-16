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

#ifndef __OC_DEVICE_H__
#define __OC_DEVICE_H__

#include <stddef.h>

#include "ocean/base/types.h"
#include "ocean/base/module.h"
#include "ocean/base/api.h"

/* Device types are identified by their name ('CPU','GPU', etcetera)  */
/* rather than a device type ID to make sure that new device types    */
/* can be added by outside modules wihout changing any of the core    */
/* code. The OcDeviceType defined below can be instantiated for       */
/* device types for which no device instances have yet been created,  */
/* or will not be created, thereby allowing a module to register all  */
/* function pointers for all supported devices, regardless of whether */
/* these devices will be instantiated or not.                         */

/* Modules are added to all device types. When a new device type is   */
/* added a copy of any existing module is added with all function     */
/* pointers set to zero and with a flag indicated that the module has */
/* not yet been initialized. When a new device instance is added, an  */
/* the setup function for the given device type is called for all     */
/* initialized modules for the corresponding device type. This allows */
/* the modules to set up a device-instance specific context.          */

/* Given that the number of functions per module is relatively small, */
/* adding a module to each device type should not provide too much    */
/* overhead. It avoids having to store offsets in look-up tables for  */
/* each combination of module and device type, which would also       */
/* require a two-stage look up of functions, rather than a single     */
/* step.                                                              */

/* Addition of devices and modules is done predominantly at start up, */
/* so we prefer simplicify of the implementation here rather than     */
/* focus on high performance of this part of the code.                */


/* ===================================================================== */
/* Forward type declarations                                             */
/* ===================================================================== */

struct __OcDevice;
struct __OcStream;
struct __OcEvent;


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct __OcDeviceType
{  char        *name;                  /* Name of the device type            */
   long int     refcount;              /* Reference count                    */

   struct __OcDevice **devices;        /* Device instance, not ref. counted  */
   int          deviceCount;           /* Size of the device instance list   */

   char        *lutBuffer;             /* Buffer for module look-up tables   */
   size_t       lutSize;               /* Current size of the look-up buffer */
   size_t       lutCapacity;           /* Capacity of the look-up table      */
} OcDeviceType;


typedef struct __OcStream
{  struct __OcDevice *device;          /* Parent device        */
   struct __OcStream *next;            /* Next element in list */
   long int  refcount;                 /* Reference count      */
} OcStream;


typedef struct __OcEvent
{  struct __OcEvent *next;             /* Next element in list */
} OcEvent;


typedef struct __OcDevice
{  OcDeviceType *type;                 /* Device type                       */
   int           index;                /* Device index                      */
   char         *name;                 /* Device instance name              */
   int           endianness;           /* 0 = little endian, 1 = big endian */
   int           requiresAlignedData;  /* Requires aligned data             */
   long int      refcount;             /* Reference count                   */

   void        **contextBuffer;        /* Device specific contexts          */
   size_t        contextSize;          /* Current size of the context data  */
   size_t        contextCapacity;      /* Capacity of the context buffer    */
   OcStream     *defaultStream;        /* Default stream for the device     */

   /* Stream and event related functions */
   OcStream *(*create_stream)(struct __OcDevice *device);
   int       (*create_event )(struct __OcDevice *device, OcEvent **event);
   void      (*delete_stream)(OcStream *stream);
   void      (*delete_event )(struct __OcDevice *device, OcEvent *event);
   int       (*sync_stream  )(OcStream *stream);
   int       (*sync_event   )(OcEvent *event);
   int       (*record_event )(OcEvent *event, OcStream *stream);
   int       (*wait_event   )(OcStream *stream, OcEvent *event);

   /* Device-specific finalize function */
   void     (*finalize      )(struct __OcDevice *device); 
} OcDevice;



/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Functions to manipulate reference counts */
OC_API OcDeviceType *OcIncrefDeviceType(OcDeviceType *deviceType);
OC_API void          OcDecrefDeviceType(OcDeviceType *deviceType);
OC_API OcDevice     *OcIncrefDevice(OcDevice *device);
OC_API void          OcDecrefDevice(OcDevice *device);
OC_API OcStream     *OcIncrefStream(OcStream *stream);
OC_API void          OcDecrefStream(OcStream *stream);

/* Device type functions */
OC_API int           OcDeviceType_deviceCount(OcDeviceType *deviceType);
OC_API OcDevice     *OcDeviceType_getDevice(OcDeviceType *deviceType, int index);

/* Device functions */
OC_API OcDevice     *OcDevice_getCommonDevice(OcDevice *device1, OcDevice *device2);

/* Stream functions */
OC_API OcStream     *OcDevice_createStream(OcDevice *device);
OC_API OcStream     *OcDevice_getDefaultStream(OcDevice *device);
OC_API int           OcDevice_setDefaultStream(OcDevice *device, OcStream *stream);
OC_API int           OcStream_synchronize(OcStream *stream);
OC_API int           OcStream_waitFor(OcStream *stream, OcEvent *event);

/* Event functions */
OC_API int           OcDevice_createEvent(OcDevice *device, OcEvent **event);
OC_API void          OcDevice_freeEvent(OcDevice *device, OcEvent *event);
OC_API int           OcEvent_synchronize(OcDevice *device, OcEvent *event);
OC_API int           OcEvent_record(OcEvent *event, OcStream *stream);

/* Default device */
OC_API void          OcDevice_setDefault(OcDevice *device);
OC_API OcDevice     *OcDevice_getDefault(void);
OC_API OcDevice     *OcDevice_applyDefault(OcDevice *device);

/* Function to query devices instances and device types globally */
OC_API int           OcDeviceTypeCount(void);
OC_API OcDeviceType *OcDeviceTypeByIndex(int index);
OC_API OcDeviceType *OcDeviceTypeByName(const char *name);
OC_API int           OcDeviceCount(void);
OC_API OcDevice     *OcDeviceByIndex(int index);


/* ===================================================================== */
/* Function declarations - Internal use only                             */
/* ===================================================================== */

/* Function for device structure allocation; the size field can be used */
/* when instantiated subclasses of the standard device structure.       */
OC_API OcDevice *OcCreateDevice(OcDeviceType *type, int index, const char *name, size_t size);

/* Function to create a new device type; when the device type already   */
/* exists, the existing type is returned.                               */
OC_API OcDeviceType *OcCreateDeviceType(const char *name);

/* Function to register a device; the reference to the device is stolen */
/* and decremented in case of failure.                                  */
OC_API int OcRegisterDevice(OcDevice *device);

/* Function for initializing device information */
OC_API int  OcInitDevices(void);

#endif

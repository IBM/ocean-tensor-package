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

#ifndef __OC_DEVICE_MODULE_H__
#define __OC_DEVICE_MODULE_H__

#include "ocean/base/device.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct __OcDeviceModule
{  OcModule *module;
   void     *data;        /* Device-type specific user data (optional)            */
   int       available;   /* Flag to indicate whether the module is available for */
                          /* When set, this means that all standard and module    */
                          /* specific function pointers have been set.            */

   /* Generic device module methods */
   int    (*initializeContext )(OcDevice *device, struct __OcDeviceModule *deviceModule, void **context);
   void   (*finalizeContext   )(OcDevice *device, struct __OcDeviceModule *deviceModule, void *context);
   void   (*finalizeModule    )(struct __OcDeviceModule *deviceModule);
} OcDeviceModule;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Module registration */
OC_API int OcRegisterModule(const char *deviceType, OcDeviceModule *info, size_t size);

/* Function to query device type modules */
OC_API OcDeviceModule *OcDeviceTypeNextModule(OcDeviceType *deviceType, size_t *offset);


/* ===================================================================== */
/* Function declarations - Internal use only                             */
/* ===================================================================== */

/* Internal function used by device.c */
OC_API int  OcDeviceModuleInitialize(void);
OC_API void OcDeviceModuleFinalize(void);

/* Internal functions used by device.c */
OC_API int  OcDeviceTypeInitializeModules(OcDeviceType *deviceType);
OC_API void OcDeviceTypeFinalizeModules(OcDeviceType *deviceType);
OC_API int  OcDeviceInitializeModules(OcDevice *device);
OC_API void OcDeviceFinalizeModules(OcDevice *device);

#endif

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

#ifndef __OC_MODULE_CORE_ITF_DEVICE_H__
#define __OC_MODULE_CORE_ITF_DEVICE_H__

#include "ocean/base/device.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int OcDevice_supportsTensorByteswap(OcDevice *device);
OC_API int OcDevice_getMaxBufferSize      (OcDevice *device, OcSize *maxBufferSize);
OC_API int OcDevice_setMaxBufferSize      (OcDevice *device, OcSize maxBufferSize);
OC_API int OcDevice_getBufferCount        (OcDevice *device, int *count);
OC_API int OcDevice_setBufferCount        (OcDevice *device, int count);

OC_API int OcDevice_format                (OcDevice *device, char **str, const char *header, const char *footer);
OC_API int OcDevice_formatInfo            (OcDevice *device, char **str, const char *header, const char *footer);
OC_API int OcDevice_display               (OcDevice *device);
OC_API int OcDevice_displayInfo           (OcDevice *device);

#endif

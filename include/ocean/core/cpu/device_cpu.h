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

#ifndef __OC_DEVICE_CPU_H__
#define __OC_DEVICE_CPU_H__

#include "ocean/core/interface/module_core.h"
#include "ocean/base/device.h"
#include "ocean/base/api.h"


/* ===================================================================== */
/* Structure definition for the CPU device                               */
/* ===================================================================== */

typedef struct
{   OcDevice HEAD;

    /* CPU-specific data */
   OcStream *streamBuffer;  /* Recycled streams         */
   int       streamCount;   /* Streams in stream buffer */

} OcDeviceCPU;


/* ===================================================================== */
/* Declaration of the global CPU device                                  */
/* ===================================================================== */

extern OcDevice *OcCPU;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int OcInitDevicesCPU(void);

#endif

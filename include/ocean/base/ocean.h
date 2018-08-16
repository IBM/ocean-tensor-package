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

#ifndef __OC_BASE_OCEAN_H__
#define __OC_BASE_OCEAN_H__

#include "ocean/base/api.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int  OcFinalizeAddHandler(void (*funptr)(void), const char *desc);

/* Internal usage only */
OC_API int  OcInitializeIntrnl(void); /* Called once from OcInit              */
OC_API void OcFinalizeIntrnl(void);   /* Called once from OcFinalize          */
OC_API void OcIncrefOcean(void);      /* Called when device types are added   */
OC_API void OcDecrefOcean(void);      /* Called when device types are deleted */

#endif

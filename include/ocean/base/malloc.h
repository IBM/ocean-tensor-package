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

#ifndef __OC_MALLOC_H__
#define __OC_MALLOC_H__

#include "ocean/base/api.h"

#include <stdlib.h>

#define OC_MEMORY_BYTE_ALIGNMENT  64

/* Uncomment the following define to enable debug versions of memory */
/* allocation. Note that changing this flag requires recompilation   */
/* of the entire Ocean library.                                      */
/* #define OC_MALLOC_DEBUG */


/* ===================================================================== */
/* Definition of the OcMalloc and OcFree macros                          */
/* ===================================================================== */

#ifdef OC_MALLOC_DEBUG
   #define OcMalloc(size)     OcMallocDebug(__FILE__,__LINE__, (size))
   #define OcFree(ptr)        OcFreeDebug(__FILE__, __LINE__, (ptr))
   #define OcMallocInit       OcMallocDebugInit
   #define OcMallocFinalize   OcMallocDebugFinalize
#else
   #define OcMalloc           malloc
   #define OcFree             free
   #define OcMallocInit()
   #define OcMallocFinalize()
#endif


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

#ifdef OC_MALLOC_DEBUG
OC_API void  OcMallocDebugInit    (void);
OC_API void  OcMallocDebugFinalize(void);
OC_API void *OcMallocDebug        (const char *file, size_t line, size_t size);
OC_API void  OcFreeDebug          (const char *file, size_t line, void *ptr);
#endif

OC_API void *OcMallocAligned      (size_t size);
OC_API void  OcFreeAligned        (void *ptr);

#endif

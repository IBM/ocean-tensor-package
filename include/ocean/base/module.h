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

#ifndef __OC_MODULE_H__
#define __OC_MODULE_H__

#include "ocean/base/api.h"

#include <stddef.h>


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  size_t    size;             /* Size of the corresponding OcDeviceModule-based type */
   char     *name;             /* Name of the module */

   int       index;            /* Index of the module in order of addition. This value is */
                               /* also used to index the context information table in the */
                               /* device instances.                                       */

   size_t    offset;           /* Offset within the device type look-up table             */
   size_t    blockSize;        /* Block size within the look-up table, includes padding;  */
                               /* this field is initialized on module registration.       */
   int       initialized;      /* Flag to indicate whether offset and index have been set */

   long int  refcount;
} OcModule;


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

/* Marco for use in static declarations, for example:                 */
/* OcModule oc_module = OC_INIT_STATIC_MODULE('dummy',OcModuleDummy)  */
#define OC_INIT_STATIC_MODULE(name,type) {sizeof(type), name, 0, 0, 0, 0, 1}

/* Macro for looking up modules. If the module is not registered the  */
/* result will be a NULL pointer.                                     */
#define OC_GET_MODULE(module,device) \
   ((module).initialized ? ((device) -> type -> lutBuffer + (module).offset) : NULL)

/* Macro for looking up module functions. The macro returns zero if   */
/* the macros is not registered.                                      */
#define OC_GET_FUNCTION(mtype,module,device,function) \
   ((module).initialized ? ((mtype *)((device) -> type -> lutBuffer + (module).offset)) -> function : 0)

/* Macro for looking up module context on a given device. We do not   */
/* check whether the module has been initialized because this macro   */
/* is mostly intended for use within implementation of the module     */
/* functions, which can only be reached if the module was registered. */
#define OC_GET_DEVICE_CONTEXT(device,module) ((module).initialized ? ((device) -> contextBuffer[(module).index]) : NULL)


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Functions to manipulate module reference counts */
OC_API OcModule *OcIncrefModule(OcModule *module);
OC_API void      OcDecrefModule(OcModule *module);

#endif

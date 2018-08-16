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

#ifndef __OC_MODULE_DUMMY_H__
#define __OC_MODULE_DUMMY_H__

#include "ocean.h"


/* Macro for module function access */
#define OC_GET_DUMMY_FUNCTION(device, function) OC_GET_FUNCTION(OcModuleDummy, oc_module_dummy, device, function)


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

OC_API int OcModuleDummy_HelloWorld(OcDevice *device);


/* ===================================================================== */
/* Definition of the module structure                                    */
/* ===================================================================== */

typedef struct
{  OcDeviceModule HEAD;       /* This must always be the first field */

   /* Module functions and variables */
   int (*HelloWorld)(OcDevice *);
} OcModuleDummy;


/* ===================================================================== */
/* Definition of the module for implementations                          */
/* ===================================================================== */

extern OcModule oc_module_dummy;


#endif

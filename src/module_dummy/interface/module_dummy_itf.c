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

#include "ocean/module_dummy/module_dummy.h"


OcModule oc_module_dummy = OC_INIT_STATIC_MODULE("dummy", OcModuleDummy);


/* ===================================================================== */
/* Implementation of the genereric interface functions                   */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
int OcModuleDummy_HelloWorld(OcDevice *device)
/* --------------------------------------------------------------------- */
{  int (*funptr)(OcDevice *); /* Function pointer */

   /* Look up the function */
   if ((funptr = OC_GET_DUMMY_FUNCTION(device, HelloWorld)) == 0)
   {  OcError(-1, "Module dummy function helloWorld is not supported on device %s", device -> type -> name);
   }

   /* Call the function */
   return funptr(device);
}

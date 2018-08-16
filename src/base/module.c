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

#include "ocean/base/module.h"
#include "ocean/base/malloc.h"

#include <stdlib.h>


/* -------------------------------------------------------------------- */
OcModule *OcIncrefModule(OcModule *module)
/* -------------------------------------------------------------------- */
{  /* Returns module to allow: module = ocIncrefDevice(module) */

   if (module != NULL) module -> refcount ++;
   return module;
}


/* -------------------------------------------------------------------- */
void OcDecrefModule(OcModule *module)
/* -------------------------------------------------------------------- */
{
   if (module == NULL) return;

   module -> refcount --;
   if (module -> refcount == 0)
   {  /* Free the module fields */
      if (module -> name) { OcFree(module -> name); module -> name = NULL; }

      /* Free the module structure */
      OcFree(module);
   }
}

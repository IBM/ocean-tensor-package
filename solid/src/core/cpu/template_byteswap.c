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

#ifndef SD_TEMPLATE_FILE
#define SD_TEMPLATE_FILE "core/cpu/template_byteswap.c"

#include "solid/base/generic/dtype_assign.h"
#include "solid/core/cpu/apply_elemwise1.h"

#include "solid/base/generic/generate_all_types.h"
#else

/* -------------------------------------------------------------------- */
int SOLID_FUNCTION(byteswap)(int ndims, const size_t *size,
                             const ptrdiff_t *strides, void *ptr)
/* -------------------------------------------------------------------- */
{
   SOLID_APPLY_ELEMWISE1(SOLID_BYTESWAP(SDXTYPE, _ptr, _ptr))

   return 0;
}

#endif

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
#define SD_TEMPLATE_FILE "core/cpu/template_copy.c"

#include "solid/base/generic/dtype_assign.h"
#include "solid/core/cpu/apply_elemwise2.h"
#include "solid/core/cpu/apply_elemwise2b.h"

#include "solid/base/generic/generate_all_types2.h"

#else

/* -------------------------------------------------------------------- */
int SOLID_FUNCTION2(copy_regular)(int ndims, const size_t *size,
                                  const ptrdiff_t *strides1, void *ptr1,
                                  const ptrdiff_t *strides2, void *ptr2)
/* -------------------------------------------------------------------- */
{

   SOLID_APPLY_ELEMWISE2_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE2), \
                               SOLID_ASSIGN(SDXTYPE, SDXTYPE2, _ptr1, _ptr2), \
                               ndims, size, ptr1, strides1, ptr2, strides2, -1)

   return 0;
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION2(copy)(int ndims1, const size_t *size1, const ptrdiff_t *strides1, void *ptr1,
                          int ndims2, const size_t *size2, const ptrdiff_t *strides2, void *ptr2)
/* -------------------------------------------------------------------- */
{  int regular = 1, i;

   /* Check whether the tensor sizes match */
   if (ndims1 == ndims2)
   {  if (size1 != size2)
      {  for (i = 0; i < ndims1; i++)
         {  if (size1[i] != size2[i])
            {  regular = 0;
               break;
            }
         }
      }
   }
   else if ((ndims1 != 0) && (ndims2 != 0))
   {  regular = 0;
   }

   /* Call regular or full copy */
   if (regular)
   {  return SOLID_FUNCTION2(copy_regular)(ndims1, size1, strides1, ptr1, strides2, ptr2);
   }
   else
   {  SOLID_APPLY_ELEMWISE2B_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE2), \
                                   SOLID_ASSIGN(SDXTYPE, SDXTYPE2, _ptr1, _ptr2), \
                                   ndims1, size1, strides1, ptr1, ndims2, size2, strides2, ptr2, -1)
   }

   return 0;
}


/* -------------------------------------------------------------------- */
int SOLID_FUNCTION2(byteswap_copy)(int ndims1, const size_t *size1, const ptrdiff_t *strides1, void *ptr1,
                                   int ndims2, const size_t *size2, const ptrdiff_t *strides2, void *ptr2,
                                   int nthreads)
/* -------------------------------------------------------------------- */
{
   SOLID_APPLY_ELEMWISE2B_CTYPE(SOLID_C_TYPE(SDXTYPE), SOLID_C_TYPE(SDXTYPE2), \
                                SOLID_BYTESWAP_ASSIGN(SDXTYPE, SDXTYPE2, _ptr1, _ptr2), \
                                ndims1, size1, strides1, ptr1, ndims2, size2, strides2, ptr2, -1)

   return 0;
}
#endif

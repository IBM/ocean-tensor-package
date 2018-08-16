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

#include "solid/core/gpu/apply_elemwise.h"
#include "solid/core/gpu/index2.h"
#include "solid_gpu.h"


/* ------------------------------------------------------------------------ */
int solid_gpu_index2_analyze(int ndims, const size_t *size, solid_int64 **offsets,
                             const ptrdiff_t *strides1, void *ptr1,
                             const ptrdiff_t *strides2, void *ptr2,
                             int elemsize, solid_index2_data *data,
                             solid_gpu_config *config)
/* ------------------------------------------------------------------------ */
{  size_t       s, nelem;
   int          result = 0;
   int          i, j;

   /* Check the number of dimensions */
   if (ndims > SOLID_MAX_TENSOR_DIMS)
      SOLID_ERROR(-1, "Number of dimensions exceeds the maximum");

   /* Initialize the data structure */
   nelem = 1;
   for (i = 0, j = 0; i < ndims; i++)
   {  s = size[i];
      if ((s == 1) && (offsets[i] == NULL)) continue;

      nelem *= s;
      data -> size[j]     = s;
      data -> offsets[j]  = offsets[i];
      data -> strides1[j] = strides1[i];
      data -> strides2[j] = strides2[i];
      j ++;
   }

   /* Update the number of dimensions */
   ndims = j;
   if (ndims == 0)
   {  ndims = 1;
      data -> size[0]     = 1;
      data -> offsets[0]  = NULL;
      data -> strides1[0] = 0;
      data -> strides2[0] = 0;
   }

   /* Finalize the data structure */
   data -> ptr1  = (char *)ptr1;
   data -> ptr2  = (char *)ptr2;
   data -> nelem = nelem;
   data -> ndims = ndims;

   /* ------------------------------------------------------- */
   /* Determine the number of threads and blocks              */
   /* ------------------------------------------------------- */
   if (config)
   {  int threads_per_block;

      /* Set the number of threads per block */
      threads_per_block = 512;

      /* Determine the configuration */
      result = solid_gpu_config_elemwise(config, nelem, threads_per_block);
   }

   return result;
}

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
#include "solid_gpu.h"

#include <limits.h>


/* ------------------------------------------------------------------------ */
int solid_gpu_config_elemwise(solid_gpu_config *config, size_t nelem, int threads_per_block)
/* ------------------------------------------------------------------------ */
{  int      device;
   int      blocks_per_multiprocessor;
   int      multiprocessor_count;
   long int threads, blocks;
   int      result = 0;

   if (cudaGetDevice(&device) == cudaSuccess)
   {
      if (nelem <= threads_per_block)
      {  threads_per_block = nelem;
         blocks = 1;
      }
      else
      {  multiprocessor_count = solid_gpu_multiprocessor_count(device);
         blocks_per_multiprocessor = solid_gpu_max_threads_per_multiprocessor(device) / threads_per_block;
         threads = nelem;
         blocks = (threads + (threads_per_block - 1)) / (threads_per_block);
         if (blocks > blocks_per_multiprocessor * multiprocessor_count)
            blocks = blocks_per_multiprocessor * multiprocessor_count;
      }
   }
   else
   {  /* Use invalid number of blocks and threads */
      SOLID_ERROR_MESSAGE("Error getting device information");
      result = -1;
      blocks = -1;
      threads_per_block = -1;
   }

   /* Update the configuration */
   config -> blocks.x  = blocks;
   config -> blocks.y  = 1;
   config -> blocks.z  = 1;
   config -> threads.x = threads_per_block;
   config -> threads.y = 1;
   config -> threads.z = 1;

   return result;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_large_indexing(int ndims, const size_t *size, const ptrdiff_t *strides, int elemsize)
/* ------------------------------------------------------------------------ */
{  ptrdiff_t    min_range = 0;
   ptrdiff_t    max_range = elemsize;
   int          flag_large;
   int          i;

   /* Determine range of memory addresses */
   for (i = 0; i < ndims; i++)
   {  
      if (size[i] == 0)
      {  min_range = 0;
         max_range = 0;
         break;
      }
      else
      {  if (strides[i] >= 0)
         {  max_range += (size[i]-1) * strides[i];
         }
         else
         {  min_range -= (size[i]-1) * strides[i];
         }
      }
   }

   /* Check range */
   if ((min_range <= INT_MIN) || (max_range >= INT_MAX))
        flag_large = 1;
   else flag_large = 0;

   return flag_large;
}

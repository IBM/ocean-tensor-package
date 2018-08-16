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

#include "solid/core/gpu/reduce_all.h"

#include "solid_gpu.h"


/* ------------------------------------------------------------------------ */
int solid_gpu_reduce_all_analyze(int ndims, const size_t *size, const ptrdiff_t *strides,
                                 void *ptr, void *buffer,
                                 solid_reduce_all_data *data, solid_gpu_config *config)
/* ------------------------------------------------------------------------ */
{  size_t      nelem;
   size_t      size_zero;
   ptrdiff_t   strides_zero;
   int         device;
   int         i;

   /* ------------------------------------------------------- */
   /* Determine data parameters                               */
   /* ------------------------------------------------------- */

   /* Initialize the size for scalar */
   if (ndims == 0)
   {  nelem        = 1;
      ndims        = 1;
      size_zero    = 1;
      strides_zero = 0;
      size         = &size_zero;
      strides      = &strides_zero;
   }
   else
   {  /* Determine the number of elements */
      for (nelem = 1, i = 0; i < ndims; i++) nelem *= size[i];
   }

   /* Copy the tensor layout */
   for (i = 0; i < ndims; i++)
   {  data -> size[i]    = size[i];
      data -> strides[i] = strides[i];
   }
   data -> ndims     = ndims;
   data -> nelem     = nelem;
   data -> ptrData   = (char *)ptr;
   data -> ptrBuffer = (char *)buffer;
   data -> ptrOutput = (char *)buffer;

   /* ------------------------------------------------------- */
   /* Determine the number of threads and blocks              */
   /* ------------------------------------------------------- */
   if (config)
   {  if ((solid_gpu_get_current_device(&device) != 0) ||
          (solid_gpu_reduce_all_config(nelem, device, config) != 0))
      { return -1; }
   }


   return 0;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_reduce_all_config(size_t nelem, int device, solid_gpu_config *config)
/* ------------------------------------------------------------------------ */
{  solid_gpu_properties *info;
   size_t blockSize;
   size_t blocksPerMultiprocessor;
   size_t blocks, blocksMax;

   /* Get the device properties */
   if ((info = solid_gpu_get_device_properties(device)) == NULL) return -1;

   /* Maximum blocksize of 256 or 512 */
   blockSize = (info -> max_threads_per_multiprocessor % 512 == 0) ? 512 : 256;

   /* Determine the maximum number of blocks per multiprocessor */
   blocksPerMultiprocessor = (info -> max_threads_per_multiprocessor) / blockSize;

   /* Determine the number of blocks */
   blocksMax = blocksPerMultiprocessor * (info -> multiprocessor_count);
   blocks = (nelem + blockSize - 1) / blockSize;
   if (blocks > blocksMax) blocks = blocksMax;

   /* Deal with small problem sizes */
   while ((2*nelem < blockSize) && (blockSize > 32))
      blockSize /= 2;

   /* Update the configuration */
   config -> blocks.x  = blocks;
   config -> threads.x = blockSize;

   return 0;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_reduce_all_buffer_size(size_t nelem, int device, size_t *size)
/* ------------------------------------------------------------------------ */
{  solid_gpu_config config;

   /* Get the configuration */
   if (solid_gpu_reduce_all_config(nelem, device, &config) != 0) return -1;

   /* Set the number of blocks multiplied by the maximum data */
   /* size (complex double).                                  */
   *size = config.blocks.x * 2 * 8;

   return 0;
}

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

#include "solid/core/gpu/reduce_axis.h"
#include "solid_gpu.h"


/* ------------------------------------------------------------------------ */
int solid_gpu_reduce_axis_prepare(solid_gpu_reduce_axis_config *config, int elemsize,
                                  int ndims, const size_t *size, const ptrdiff_t *strides1,
                                  void *ptr1, const ptrdiff_t *strides2, void *ptr2,
                                  int rdims, const size_t *rsize, const ptrdiff_t *rstrides,
                                  cudaStream_t stream)
/* ------------------------------------------------------------------------ */
{  solid_gpu_properties *prop;
   size_t nelem, relem, N, M;
   int i;

   /* Check the dimensions */
   if ((ndims > SOLID_MAX_TENSOR_DIMS) || (rdims > SOLID_MAX_TENSOR_DIMS))
      SOLID_ERROR(-1, "Maximum number of dimensions exceeded");

   /* Copy the outer dimensions */
   if (ndims == 0)
   {  config -> data.size[0]     = 1;
      config -> data.strides1[0] = 0;
      config -> data.strides2[0] = 0;
      config -> data.ndims       = 1;
      config -> data.nelem       = 1;
   }
   else
   {  for (i = 0, nelem = 1; i < ndims; i++)
      {  config -> data.size[i]     = size[i];
         config -> data.strides1[i] = strides1[i];
         config -> data.strides2[i] = strides2[i];
         nelem *= size[i];
      }
      config -> data.ndims = ndims;
      config -> data.nelem = nelem;
   }

   /* Copy the inner dimensions */
   if (rdims == 0)
   {  config -> data.rsize[0]    = 1;
      config -> data.rstrides[0] = 0;
      config -> data.rdims       = 1;
      config -> data.relem       = 1;
   }
   else
   {  for (i = 0, relem = 1; i < rdims; i++)
      {  config -> data.rsize[i]    = rsize[i];
         config -> data.rstrides[i] = rstrides[i];
         relem *= rsize[i];
      }
      config -> data.rdims = rdims;
      config -> data.relem = relem;
   }

   /* Set the pointer information */
   config -> data.ptr1 = (char *)ptr1;
   config -> data.ptr2 = (char *)ptr2;
   config -> data.ptrBuffer = NULL;

   /* Determine the mode and the buffer size */
   config -> bufferSize = 0;
   config -> mode = 1;

   /* Initialize the cuda parameters */
   config -> gridSize.x  = 1;
   config -> gridSize.y  = 1;
   config -> gridSize.z  = 1;
   config -> blockSize.x = 1;
   config -> blockSize.y = 1;
   config -> blockSize.z = 1;
   config -> stream      = stream;

   /* Get device properties */
   prop = solid_gpu_get_current_device_properties();
   if (prop == NULL) return -1;

   /* Determine the mode */
   if ((config -> data.relem < 32) ||
       (config -> data.nelem >= 2 * (prop -> multiprocessor_count) * (prop -> max_threads_per_multiprocessor)))
   {  /* There are enough output elements for each thread to run at least */
      /* two entire reductions, or the reduction size is so small that it */
      /* cannot be done at the warp level.                                */
      config -> mode = 1;
   }
   else if ((config -> data.relem <= 2*32) ||
            (config -> data.nelem >= 2 * (prop -> multiprocessor_count) * (prop -> max_threads_per_multiprocessor) / 32))
   {  /* The reduction size is not large enough for the reduction to be  */
      /* done efficiently at the block level, or there are enough output */
      /* elements for the reductions to be run on the warp level.        */
      config -> mode = 2;
   }
   else
   {  config -> mode = 3;
   }


   /* ----------------------- */
   /*  Configuration mode #3  */
   /* ----------------------- */
   if (config -> mode == 3)
   {
      /* Initial configuration */
      config -> blockSize.x = 32;
      config -> blockSize.y = 1;
      config -> gridSize.x = 1;
      config -> gridSize.y = 1;

      /* Increase the number of blocks */
      while (1)
      {  /* Check if we can increase the number of blocks */
         N = config -> gridSize.y + 1;
         if (N > (prop -> max_gridsize[1])) break;

         /* Make sure each thread has something to work on */
         if ((N * (config -> blockSize.x * config -> blockSize.y)) > config -> data.relem) break;

         /* Determine the number of blocks per multiprocessor */
         M = (N + (prop -> multiprocessor_count) - 1) / (prop -> multiprocessor_count);
         if (M > (prop -> max_blocks_per_multiprocessor)) break;

         /* Check if the maximum number of resident blocks is exceeded */
         if (M > (prop -> max_blocks_per_multiprocessor)) break;

         /* The total amount of shared memory per multiprocessor cannot be exceeded */
         if ((M * (config -> blockSize.x * config -> blockSize.y * elemsize)) >
             (prop -> max_shared_mem_per_multiprocessor)) break;

         /* Increment the number of blocks */
         config -> gridSize.y += 1;
      }

      /* Determine the maximum number of blocks per multi-processor */
      M = (config -> gridSize.y + (prop -> multiprocessor_count) - 1) / (prop -> multiprocessor_count);

      /* Number of threads per block */
      while (1)
      {  /* Check if we can increase the number of threads */
         N = config -> blockSize.y + 1;
         if (N > (prop -> max_blocksize[1])) break;

         /* We cannot exceed the maximum number of threads per block */
         N *= 32;
         if ((N > prop -> max_threads_per_block) ||
             (N > prop -> max_threads_per_multiprocessor) ||
             (N > config -> data.relem)) break;

         /* Check if there are sufficiently many elements in the reduction */
         if (N * config -> gridSize.y > config -> data.relem) break;

         /* The amount of shared memory per block cannot be exceeded */
         N *= elemsize;
         if (N > (prop -> max_shared_mem_per_threadblock)) break;

         /* The total amount of shared memory per multiprocessor cannot be exceeded */
         if ((M * N) > (prop -> max_shared_mem_per_multiprocessor)) break;

         config -> blockSize.y += 1;
      }

      /* Increase the number of reduction blocks */
      config -> gridSize.x = 1;
      while (1)
      {  /* Check if we can increase the number of blocks */
         N = (config -> gridSize.x) + 1;
         if (N > prop -> max_gridsize[0]) break;

         /* Check if the number of reductions is sufficiently large */
         if (N > config -> data.nelem) break;

         /* Determine the number of blocks per multiprocessor */
         N *= (config -> gridSize.y);
         M = (N + (prop -> multiprocessor_count) - 1) / (prop -> multiprocessor_count);
         if (M > (prop -> max_blocks_per_multiprocessor)) break;

         /* Check if the maximum number of resident blocks is exceeded */
         if (M > (prop -> max_blocks_per_multiprocessor)) break;

         /* The total amount of shared memory per multiprocessor cannot be exceeded */
         if ((M * (config -> blockSize.x * config -> blockSize.y * elemsize)) >
             (prop -> max_shared_mem_per_multiprocessor)) break;

         /* Increment the number of blocks */
         config -> gridSize.x += 1;
      }

      /* Set the shared memory and buffer size */
      config -> sharedMem  = (config -> blockSize.x * config -> blockSize.y) * elemsize;
      config -> bufferSize = (config -> blockSize.y * config -> gridSize.x * config -> gridSize.y) * elemsize;


      /* ------------------------------- */
      /*  Final reduction configuration  */
      /* ------------------------------- */
      config -> sharedMemFinalize   = 0;
      config -> blockSizeFinalize.y = 1;
      config -> blockSizeFinalize.z = 1;
      config -> gridSizeFinalize.y  = 1;
      config -> gridSizeFinalize.z  = 1;

      /* Determine the number of output elements per reduction */
      N = config -> blockSize.y * config -> gridSize.y;
      
      /* Determine the number of threads (needs not be a multiple of 32) */
      config -> blockSizeFinalize.x = N;
      if (config -> blockSizeFinalize.x > prop -> max_threads_per_block)
         config -> blockSizeFinalize.x = prop -> max_threads_per_block;
      if (config -> blockSizeFinalize.x > prop -> max_blocksize[0])
         config -> blockSizeFinalize.x = prop -> max_blocksize[0];

      /* Determine the maximum number of blocks per multi-processor */
      M = (config -> blockSizeFinalize.x + (prop -> max_threads_per_multiprocessor) - 1) / (prop -> max_threads_per_multiprocessor);
      if (M > prop -> max_blocks_per_multiprocessor) M = prop -> max_blocks_per_multiprocessor;

      /* Determine the number of blocks */
      config -> gridSizeFinalize.x  = M * prop -> multiprocessor_count;
      if (config -> gridSizeFinalize.x > config -> data.nelem) config -> gridSizeFinalize.x = config -> data.nelem;
   }


   /* ----------------------- */
   /*  Configuration mode #2  */
   /* ----------------------- */
   if (config -> mode == 2)
   {  config -> gridSize.x = prop -> multiprocessor_count * (prop -> max_blocks_per_multiprocessor);
      config -> blockSize.x = 32;
      config -> blockSize.y = 1;

      /* The number of blocks should not exceed the number of elements */
      if (config -> gridSize.x > config -> data.nelem)
      {  config -> gridSize.x = config -> data.nelem;
      }

      while (1)
      {  /* Check if we can double the number of warps N -- each warp takes care of one reduction */
         N = 2 * config -> blockSize.y;

         /* We want to maintain the maximum number of blocks per multi-processor */
         if ((2*N * (prop -> multiprocessor_count) * (prop -> max_blocks_per_multiprocessor)) > (config -> data.nelem)) break;

         /* We want at least two elements for each thread in the warp */
         N *= 32;
         if ((2*N) > (config -> data.relem)) break;

         /* We cannot exceed the maximum number of threads per block */
         if (N > (prop -> max_threads_per_block)) break;

          /* The amount of shared memory per block cannot be exceeded */
         N *= elemsize;
         if (N > (prop -> max_shared_mem_per_threadblock)) break;

         /* The amount of shared memory per multi-processor cannot be exceeded */
         if ((N * (prop -> max_blocks_per_multiprocessor)) > (prop -> max_shared_mem_per_multiprocessor)) break;

         /* Double the number of threads */
         config -> blockSize.y *= 2;
      }

      /* Make sure the configuration is valid */
      config -> bufferSize = 0;
      config -> sharedMem  = (config -> blockSize.x * config -> blockSize.y) * elemsize;
      if (((config -> sharedMem) > (prop -> max_shared_mem_per_threadblock)) ||
          (((config -> sharedMem) * (prop -> max_blocks_per_multiprocessor)) > (prop -> max_shared_mem_per_multiprocessor)))
      {  /* Insufficient shared memory to perform the reduction at the warp level */
         config -> mode = 1;
      }
   }


   /* ----------------------- */
   /*  Configuration mode #1  */
   /* ----------------------- */
   if (config -> mode == 1)
   {  config -> data.relem /= config -> data.rsize[0]; /* Kernel has explicit for-loop over inner reduction dimension */
      config -> bufferSize  = 0;
      config -> sharedMem   = 0;

      config -> blockSize.x = 32;
      while(1)
      {
         config -> gridSize.x = (config -> data.nelem + config -> blockSize.x - 1) / (config -> blockSize.x);
         if (config -> gridSize.x <= (prop -> multiprocessor_count) * 4)
         {  break;
         }

         if (2 * config -> blockSize.x > prop -> max_threads_per_block)
         {  if (config -> gridSize.x > (prop -> multiprocessor_count) * (prop -> max_blocks_per_multiprocessor))
            {  config -> gridSize.x = (prop -> multiprocessor_count) * (prop -> max_blocks_per_multiprocessor);
            }
            break;
         }
         else
         {  config -> blockSize.x *= 2;
         }
      }
   }

   return 0;
}

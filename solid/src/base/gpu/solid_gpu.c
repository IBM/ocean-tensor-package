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

#include "solid_gpu.h"
#include "solid/base/gpu/solid_cuda.h"

#include <stdlib.h>

static solid_gpu_properties *_solid_gpu_device_prop  = NULL;
static int                   _solid_gpu_device_count = 0;
static int                   _solid_gpu_initialized  = 0;


/* ------------------------------------------------------------------------ */
int solid_gpu_initialize(void)
/* ------------------------------------------------------------------------ */
{  struct cudaDeviceProp prop;
   solid_gpu_properties *info;
   cudaError_t status;
   int         device_count, device;
   int         result = 0;

   /* Increment the initialized count */
   _solid_gpu_initialized ++;
   if (_solid_gpu_initialized == 1)
   {  /* Reset the cuda error status */
      cudaGetLastError();

      /* Determine the number of devices */
      status = cudaGetDeviceCount(&device_count);
      if (status != cudaSuccess)
      {
         /* Ignore the no device and insufficient driver errors */
         if ((status != cudaErrorNoDevice) && (status != cudaErrorInsufficientDriver))
         {  SOLID_ERROR(-1, "Error determining the number of GPU devices (%s)", cudaGetErrorString(status));
         }
         else
         {  device_count = 0;
         }
      }

      /* Allocate the property information */
      if (device_count == 0) goto final;
      _solid_gpu_device_prop = (solid_gpu_properties *)malloc(sizeof(solid_gpu_properties) * device_count);
      if (_solid_gpu_device_prop == NULL) { result = -1; goto final; }

      /* Initialize the device information */
      for (device = 0; device < device_count; device++)
      {  status = cudaGetDeviceProperties(&prop, device);
         if (status != cudaSuccess) { result = -1; goto final; }

         /* Get a pointer to the device property information */
         info = &(_solid_gpu_device_prop[device]);

         /* Copy the relevant properties */
         info -> major                          = prop.major;
         info -> minor                          = prop.minor;
         info -> multiprocessor_count           = prop.multiProcessorCount;
         info -> max_threads_per_block          = prop.maxThreadsPerBlock;
         info -> max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
         info -> max_shared_mem_per_threadblock = 48 * 1024;
         info -> max_gridsize[0]                = prop.maxGridSize[0];
         info -> max_gridsize[1]                = prop.maxGridSize[1];
         info -> max_gridsize[2]                = prop.maxGridSize[2];
         info -> max_blocksize[0]               = prop.maxThreadsDim[0];
         info -> max_blocksize[1]               = prop.maxThreadsDim[1];
         info -> max_blocksize[2]               = prop.maxThreadsDim[2];

         #if CUDA_VERSION >= 6050
            info -> max_shared_mem_per_multiprocessor = prop.sharedMemPerMultiprocessor;
         #else
            if (prop.major == 1)
               info -> max_shared_mem_per_multiprocessor    = 16 * 1024;  /* Compute capability 1.x       */
            else if (prop.major == 2)
               info -> max_shared_mem_per_multiprocessor    = 48 * 1024;  /* Compute capability 2.x       */
            else if (prop.major == 3)
            {  if (prop.minor <= 5)
                  info -> max_shared_mem_per_multiprocessor = 48 * 1024;  /* Compute capability 3.0 - 3.5 */
               else
                  info -> max_shared_mem_per_multiprocessor = 112 * 1024; /* Compute capability 3.6 - 3.x */
            }
            else if (prop.major == 5)
            {  if (prop.minor < 2)
                  info -> max_shared_mem_per_multiprocessor = 64 * 1024;  /* Compute capability 5.0 - 5.1 */
               else if (prop.minor < 3)
                  info -> max_shared_mem_per_multiprocessor = 96 * 1024;  /* Compute capability 5.2       */
               else
                  info -> max_shared_mem_per_multiprocessor = 96 * 1024;  /* Compute capability 5.3 - 5.x */
            }
            else if (prop.major == 6)
            {  if (prop.minor == 0)
                  info -> max_shared_mem_per_multiprocessor = 96 * 1024;  /* Compute capability 6.0       */
               else if (prop.minor == 1)
                  info -> max_shared_mem_per_multiprocessor = 96 * 1024;  /* Compute capability 6.1       */
               else if (prop.minor == 2)
                  info -> max_shared_mem_per_multiprocessor = 96 * 1024;  /* Compute capability 6.2       */
            }
         #endif


         /* Properties derived from compute capability */
         if (prop.major == 1)
         {  if (prop.minor <= 1)
            {  info -> max_blocks_per_multiprocessor = 8;    /* Compute capability 1.0 - 1.1 */
               info -> max_warps_per_multiprocessor  = 24;
            }
            else
            {  info -> max_blocks_per_multiprocessor = 8;    /* Compute capability 1.2 - 1.x */
               info -> max_warps_per_multiprocessor  = 32;
            }
         }
         else if (prop.major == 2)
         {  if (prop.minor >= 0)
            {  info -> max_blocks_per_multiprocessor = 8;    /* Compute capability 2.x       */
               info -> max_warps_per_multiprocessor  = 48;
            }
         }
         else if (prop.major == 3)
         {  if (prop.minor >= 0)
            {  info -> max_blocks_per_multiprocessor = 16;   /* Compute capability 3.x       */
               info -> max_warps_per_multiprocessor  = 64;
            }
         }
         else if (prop.major >= 5)
         {  if (prop.minor >= 0)
            {  info -> max_blocks_per_multiprocessor = 32;   /* Compute capability 5.x - y.x */
               info -> max_warps_per_multiprocessor  = 64;
            }
         }

         /* Derived properties */
         info -> max_resident_blocks_per_device = (info -> max_blocks_per_multiprocessor) * (info -> multiprocessor_count);
      }

      /* Set the number of devices */
      _solid_gpu_device_count = device_count;
   }

final :
   if (result != 0) solid_gpu_finalize();
   return result;
}


/* ------------------------------------------------------------------------ */
void solid_gpu_finalize(void)
/* ------------------------------------------------------------------------ */
{  _solid_gpu_initialized --;
   if (_solid_gpu_initialized != 0) return;

   /* Free the device property information */
   if (_solid_gpu_device_prop != NULL)
   {  free(_solid_gpu_device_prop);
      _solid_gpu_device_prop = NULL;
   }

   /* Reset the device count */
   _solid_gpu_device_count = 0;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_check_status(void)
/* ------------------------------------------------------------------------ */
{  cudaError_t status;

   if ((status = cudaGetLastError()) != cudaSuccess)
      SOLID_ERROR(-1, "Cuda error: %s", cudaGetErrorString(status));

   return 0;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_get_current_device(int *device)
/* ------------------------------------------------------------------------ */
{
   if (cudaGetDevice(device) != cudaSuccess)
      SOLID_ERROR(-1, "Error getting the current device");

   return 0;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_multiprocessor_count(int device)
/* ------------------------------------------------------------------------ */
{  if ((device < 0) || (device >= _solid_gpu_device_count))
      SOLID_ERROR(-1, "Device index out of bounds");

   return _solid_gpu_device_prop[device].multiprocessor_count;
}


/* ------------------------------------------------------------------------ */
int solid_gpu_max_threads_per_multiprocessor(int device)
/* ------------------------------------------------------------------------ */
{  if ((device < 0) || (device >= _solid_gpu_device_count)) return -1;

   return _solid_gpu_device_prop[device].max_threads_per_multiprocessor;
}


/* ------------------------------------------------------------------------ */
solid_gpu_properties *solid_gpu_get_device_properties(int device)
/* ------------------------------------------------------------------------ */
{  if ((device < 0) || (device >= _solid_gpu_device_count))
      SOLID_ERROR(NULL, "Error getting device properties: invalid device index");

   return &(_solid_gpu_device_prop[device]);
}


/* ------------------------------------------------------------------------ */
solid_gpu_properties *solid_gpu_get_current_device_properties(void)
/* ------------------------------------------------------------------------ */
{  int device;

   if (cudaGetDevice(&device) != cudaSuccess)
      SOLID_ERROR(NULL, "Error getting device information");

   return solid_gpu_get_device_properties(device);
}

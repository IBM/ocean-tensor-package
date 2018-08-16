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

#include "ocean/core/gpu/device_gpu.h"
#include "ocean/core/gpu/storage_gpu.h"
#include "ocean/core/gpu/cuda.h"

#include "ocean/core/cpu/storage_cpu.h"
#include "ocean/base/error.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

static OcStorage *OcStorageGPU_create     (OcSize nelem, OcDType dtype, OcDevice *device, OcStream *stream);
static void       OcStorageGPU_finalize   (OcStorage *storage);


/* ===================================================================== */
/* Function registration                                                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcRegisterStorageGPU(OcModuleCore *module)
/* -------------------------------------------------------------------- */
{
   /* Set the function pointers */
   module -> Storage_create         = OcStorageGPU_create;

   module -> Buffer_copy            = OcBufferGPU_copy;
   module -> Buffer_copyFromCPU     = OcBufferGPU_copyFromCPU;
   module -> Buffer_copyToCPU       = OcBufferGPU_copyToCPU;
   module -> Buffer_copyHostStorage = OcBufferGPU_copyHostToStorage;
   module -> Buffer_copyStorageHost = OcBufferGPU_copyStorageToHost;
   module -> Buffer_zero            = OcBufferGPU_zero;

   return 0;
}



/* ===================================================================== */
/* Function implemenations                                               */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static OcStorage *OcStorageGPU_create(OcSize nelem, OcDType dtype,
                                      OcDevice *device, OcStream *stream)
/* -------------------------------------------------------------------- */
{  OcStorage    *storage = NULL;
   OcSize        size;
   cudaError_t   status;

   /* Allocate storage and initialize basic fields */
   storage = OcAllocateStorage(device, dtype, stream, sizeof(OcStorageGPU));
   if (storage == NULL) goto error;

   /* Initialize fields */
   storage -> flags   |= OC_STORAGE_OWNER;
   storage -> finalize = OcStorageGPU_finalize;
   storage -> data     = NULL;

   /* Allocate storage memory */
   size = nelem * (storage -> elemsize);
   storage -> size     = size;
   storage -> capacity = size;
   storage -> nelem    = nelem;

   if (nelem > 0)
   {  
      /* Activate the device */
      if (OcCuda_setDevice(device -> index) != 0) goto error;

      /* Allocate memory */
      status = cudaMalloc((void **)&(storage -> data), size);
      if (status != cudaSuccess)
      {  cudaGetLastError();
         OcErrorMessage("Error allocating %s storage of size %lu on %s",
                        OcDType_name(dtype), (unsigned long)nelem, device -> name);
         goto error;
      }
   }

   return (OcStorage *)storage;

error:
   OcDecrefStorage((OcStorage *)storage);
   return NULL;
}


/* -------------------------------------------------------------------- */
static void OcStorageGPU_finalize(OcStorage *storage)
/* -------------------------------------------------------------------- */
{
   /* Free storage memory */
   if (storage -> data)
   {  if (OcStorage_isOwner(storage))
      {  cudaFree((void *)(storage -> data));
      }
   }
}


/* -------------------------------------------------------------------- */
int OcBufferGPU_copy(OcStorage *srcStorage, void *srcData,
                     OcStorage *dstStorage, void *dstData,
                     OcDType dtype, OcSize n)
/* -------------------------------------------------------------------- */
{  OcDevice    *srcDevice, *dstDevice;
   cudaError_t  status;

   /* Check for empty buffers */
   if (n == 0) return 0;

   /* Get the devices */
   srcDevice = srcStorage -> stream -> device;
   dstDevice = dstStorage -> stream -> device;

   /* Synchronization */
   if (OcStorage_startRead(dstStorage, srcStorage) != 0) return -1;

   /* Copy the data */
   if (srcDevice != dstDevice)
   {  /* Between devices */
      status = cudaMemcpyPeerAsync(dstData, dstDevice -> index,
                                   srcData, srcDevice -> index,
                                   OcDType_size(dtype) * n,
                                   OcStorageGPU_cudaStream(dstStorage));
   }
   else
   {  /* Within the device */
      status = cudaSetDevice(srcDevice -> index);
      if (status == cudaSuccess)
      {  status = cudaMemcpyAsync(dstData, srcData, OcDType_size(dtype) * n,
                                  cudaMemcpyDeviceToDevice,
                                  OcStorageGPU_cudaStream(dstStorage));
      }
   }

   /* Check for errors and synchronize */
   if (status != cudaSuccess) { OcCuda_setStatus(status); return -1; }
   if (OcStorage_update(dstStorage) != 0) return -1;
   if (OcStorage_finishRead(dstStorage, srcStorage) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcBufferGPU_copyFromCPU(OcStorage *srcStorage, void *srcData,
                            OcStorage *dstStorage, void *dstData,
                            OcDType dtype, OcSize n)
/* -------------------------------------------------------------------- */
{  OcDevice    *dstDevice;
   cudaError_t  status;

   /* Get the destination device */
   dstDevice = dstStorage -> stream -> device;

   /* Synchronization */
   if (OcStorage_synchronize(dstStorage) != 0) return -1;

   /* Synchronous data copy */
   status = cudaSetDevice(dstDevice -> index);
   if (status == cudaSuccess)
   {  status = cudaMemcpy(dstData, srcData, OcDType_size(dtype) * n, cudaMemcpyHostToDevice);
   }

   /* Check for errors and synchronize */
   if (status != cudaSuccess) { OcCuda_setStatus(status); return -1; }
   if (OcStorage_update(dstStorage) != 0) return -1;

   return 0;
}



/* -------------------------------------------------------------------- */
int OcBufferGPU_copyToCPU(OcStorage *srcStorage, void *srcData,
                          OcStorage *dstStorage, void *dstData,
                          OcDType dtype, OcSize n)
/* -------------------------------------------------------------------- */
{  OcDevice    *srcDevice;
   cudaError_t  status;

   /* Get the source device */
   srcDevice = srcStorage -> stream -> device;

   /* Synchronization */
   if (OcStorage_synchronize(srcStorage) != 0) return -1;

   /* Synchronous data copy */
   status = cudaSetDevice(srcDevice -> index);
   if (status == cudaSuccess)
   {  status = cudaMemcpy(dstData, srcData, OcDType_size(dtype) * n, cudaMemcpyDeviceToHost);
   }

   /* Check for errors and synchronize */
   if (status != cudaSuccess) { OcCuda_setStatus(status); return -1; }
   if (OcStorage_update(dstStorage) != 0) return -1;

   /* Update the read event */
   if (OcStorage_finishRead(srcStorage, srcStorage) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcBufferGPU_copyHostToStorage(OcStorage *storage, OcIndex offset,
                                  OcSize nbytes, void *ptr)
/* -------------------------------------------------------------------- */
{  OcDevice    *device;
   cudaError_t  status;

   /* Get the storage device */
   device = OcStorage_device(storage);

   /* Synchronization */
   if (OcStorage_synchronize(storage) != 0) return -1;

   /* Synchronous data copy */
   status = cudaSetDevice(device -> index);
   if (status == cudaSuccess)
   {  status = cudaMemcpy((void *)(((char *)OcStorage_data(storage)) + offset),
                          ptr, nbytes, cudaMemcpyHostToDevice);
   }

   /* Check for errors and synchronize */
   if (status != cudaSuccess) { OcCuda_setStatus(status); return -1; }
   if (OcStorage_update(storage) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcBufferGPU_copyStorageToHost(OcStorage *storage, OcIndex offset,
                                  OcSize nbytes, void *ptr)
/* -------------------------------------------------------------------- */
{  OcDevice    *device;
   cudaError_t  status;

   /* Get the storage device */
   device = storage -> stream -> device;

   /* Synchronization */
   if (OcStorage_synchronize(storage) != 0) return -1;

   /* Synchronous data copy */
   status = cudaSetDevice(device -> index);
   if (status == cudaSuccess)
   {  status = cudaMemcpy(ptr, (void *)(((char *)OcStorage_data(storage)) + offset),
                          nbytes, cudaMemcpyDeviceToHost);
   }

   /* Check for errors and synchronize */
   if (status != cudaSuccess) { OcCuda_setStatus(status); return -1; }

   /* Update the read event */
   if (OcStorage_finishRead(storage, storage) != 0) return -1;

   return 0;
}


/* -------------------------------------------------------------------- */
int OcBufferGPU_zero(OcStorage *storage, void *data, OcDType dtype, OcSize n)
/* -------------------------------------------------------------------- */
{  OcDevice    *device;
   cudaError_t  status;

   /* Get the device */
   device = storage -> stream -> device;

   /* Synchronous data copy */
   status = cudaSetDevice(device -> index);
   if (status == cudaSuccess)
   {  status = cudaMemsetAsync(data, 0x00, OcDType_size(dtype) * n, OcStorageGPU_cudaStream(storage));
   }

   /* Check for errors and synchronize */
   if (status != cudaSuccess) { OcCuda_setStatus(status); return -1; }
   if (OcStorage_update(storage) != 0) return -1;

   return 0;
}

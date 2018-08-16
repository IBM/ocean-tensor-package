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
#include "ocean/core/gpu/cuda.h"
#include "ocean.h"

#include "solid_cpu.h"        /* Parallel function evaluation */
#include "solid_core_gpu.h"   /* Dummy kernel                 */

#include <string.h>


/* ===================================================================== */
/* Local and global variables                                            */
/* ===================================================================== */

OcDevice **OcGPU = NULL;
OcDeviceType *oc_device_type_gpu = NULL;

int oc_device_gpu_initialized = 0;
int oc_device_gpu_count       = 0;     /* Number of GPU devices */


/* ===================================================================== */
/* Internal function declarations                                        */
/* ===================================================================== */

static int       OcDeviceGPU_getProperties(OcDeviceGPU *device);
static OcStream *OcDeviceGPU_createStream (OcDevice *device);
static int       OcDeviceGPU_createEvent  (OcDevice *device, OcEvent **event);
static void      OcDeviceGPU_deleteStream (OcStream *stream);
static void      OcDeviceGPU_deleteEvent  (OcDevice *device, OcEvent *event);
static int       OcStreamGPU_synchronize  (OcStream *stream);
static int       OcEventGPU_synchronize   (OcEvent *event);
static int       OcEventGPU_record        (OcEvent *event, OcStream *stream);
static int       OcStreamGPU_waitEvent    (OcStream *stream, OcEvent *event);


/* ===================================================================== */
/* Functions for device creation and deletion                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
void OcFreeDeviceGPU(OcDevice *self)
/* -------------------------------------------------------------------- */
{  OcDeviceGPU *device = (OcDeviceGPU *)self;
   OcStream    *stream;
   OcEvent     *event;

   /* Free all buffered streams */
   while ((stream = device -> streamBuffer) != NULL)
   {  device -> streamBuffer = stream -> next;
      cudaStreamDestroy(((OcStreamGPU *)stream) -> cudaStream);
      OcFree(stream);
   }

   /* Free all buffered events */
   while ((event = device -> eventBuffer) != NULL)
   {  device -> eventBuffer = event -> next;
      cudaEventDestroy(((OcEventGPU *)event) -> cudaEvent);
      OcFree(event);
   }
}


/* -------------------------------------------------------------------- */
OcDevice *OcCreateDeviceGPU(OcDeviceType *type, int index, const char *name)
/* -------------------------------------------------------------------- */
{  OcDeviceGPU *deviceGPU;
   OcDevice    *device;

   /* Create the device structure */
   device    = OcCreateDevice(type, index, name, sizeof(OcDeviceGPU));
   deviceGPU = (OcDeviceGPU *)device;

   /* Initialize the device */
   if (device != NULL)
   {
      /* GPU device-specific initialization */
      device -> finalize = OcFreeDeviceGPU;
      device -> endianness = 0; /* Little endian */
      device -> requiresAlignedData = 1;

      /* Stream and event related functions */
      device -> create_stream = OcDeviceGPU_createStream;
      device -> create_event  = OcDeviceGPU_createEvent;
      device -> delete_stream = OcDeviceGPU_deleteStream;
      device -> delete_event  = OcDeviceGPU_deleteEvent;
      device -> sync_stream   = OcStreamGPU_synchronize;
      device -> sync_event    = OcEventGPU_synchronize;
      device -> record_event  = OcEventGPU_record;
      device -> wait_event    = OcStreamGPU_waitEvent;

      /* Buffering of streams and events */
      deviceGPU -> streamBuffer = NULL;
      deviceGPU -> streamCount  = 0;
      deviceGPU -> eventBuffer  = NULL;
      deviceGPU -> eventCount   = 0;

      /* Get the device properties */
      if (OcDeviceGPU_getProperties(deviceGPU) != 0)
      {  OcDecrefDevice(device);
         return NULL;
      }
   }

   return device;
}


/* -------------------------------------------------------------------- */
void OcFinalizeDevicesGPU(void)
/* -------------------------------------------------------------------- */
{  int i;

   /* Finalize can be called even if the module was not initialized. */
   if (oc_device_gpu_initialized == 0) return ;
   oc_device_gpu_initialized = 0;

   /* Clean up the device type reference */
   if (oc_device_type_gpu) OcDecrefDeviceType(oc_device_type_gpu);

   /* Finalize the GPU devices */
   if (OcGPU != NULL)
   {  for (i = 0; i < oc_device_gpu_count; i++)
      {  if (OcGPU[i] != NULL)
         {  OcDecrefDevice(OcGPU[i]);
            OcGPU[i] = NULL;
         }
      }

      /* Free the device pointer buffer */
      OcFree(OcGPU);
      OcGPU = NULL;
   }
   oc_device_gpu_count = 0;
}


/* -------------------------------------------------------------------- */
void OcInitDevicesGPU_intrnlInit(int index, void *data)
/* -------------------------------------------------------------------- */
{  cudaError_t  status;
   cudaEvent_t  event;

   /* Activate the device */
   (void)cudaGetLastError();
   if ((status = cudaSetDevice(index)) == cudaSuccess)
   {  /* Create the event */
      if (cudaEventCreate(&event) == cudaSuccess)
      {  cudaEventDestroy(event);
      }

      /* Call a dummy kernel */
      solid_gpu_dummy_kernel();
   }
   (void)cudaGetLastError();
}


/* -------------------------------------------------------------------- */
int OcInitDevicesGPU(void)
/* -------------------------------------------------------------------- */
{  OcDeviceType *type;
   OcDevice     *device;
   char          deviceName[12];
   int           deviceCount;
   int           index = 0;

   /* Make sure we are initialized only once */
   if (oc_device_gpu_initialized)
      OcError(-1, "Device GPU was already initialized");

   /* Get the number of devices */
   if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
   {  deviceCount = 0;
   }

   /* The first set of calls to Cuda can be slow. Here we */
   /* use multiple threads to create and delete events to */
   /* try and reduce the start-up time.                   */
   solid_omp_run_parallel(OcInitDevicesGPU_intrnlInit, deviceCount, -1, NULL);

   /* Create the device type */
   type = OcCreateDeviceType("GPU");
   if (type == NULL) return -1;

   /* Allocate memory for the GPU device buffer */
   if (deviceCount > 0)
   {  OcGPU = (OcDevice **)OcMalloc(sizeof(OcDevice *) * deviceCount);
      if (OcGPU == NULL) OcError(-1, "Error allocating GPU device buffer");

      /* Reset all device pointers */
      for (index = 0; index < deviceCount; index++)
      {  OcGPU[index] = NULL;
      }
   }


   /* Set the initialization flag. This indicates          */
   /* successful initialization, and allows us to clean up */
   /* any intermediate state using the finalize function,  */
   /* in case any error occurs during initialization.      */
   oc_device_gpu_initialized = 1;

   /* Instantiate the GPU devices */
   for (index = 0; index < deviceCount; index++)
   {  snprintf(deviceName, 12, "gpu%d", index);
      device = OcCreateDeviceGPU(type, index, deviceName);
      if ((device != NULL) && (OcRegisterDevice(device) == 0))
      {  OcGPU[index] = OcIncrefDevice(device);
      }
      else
      {  OcFinalizeDevicesGPU();
         return -1;
      }
   }

   /* Set the device count */
   oc_device_gpu_count = deviceCount;

   /* Get the device type */
   if (deviceCount > 0)
   {  oc_device_type_gpu = OcIncrefDeviceType(OcGPU[0] -> type);
   }

   return OcFinalizeAddHandler(OcFinalizeDevicesGPU, "Finalize devices GPU");
}



/* ===================================================================== */
/* Stream and event functions                                            */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static OcStream *OcDeviceGPU_createStream(OcDevice *device)
/* -------------------------------------------------------------------- */
{  OcDeviceGPU *gpu = (OcDeviceGPU *)device;
   OcStream    *stream = NULL;

   if (gpu -> streamCount > 0)
   {  /* Recycle a stream */
      stream = gpu -> streamBuffer;
      gpu -> streamBuffer = stream -> next;
      gpu -> streamCount --;
   }
   else
   {  /* Create a new stream */
      if (cudaSetDevice(device -> index) == cudaSuccess)
         stream = (OcStream *)OcMalloc(sizeof(OcStreamGPU));
      if ((stream == NULL) ||
          (cudaStreamCreate(&(((OcStreamGPU *)stream) -> cudaStream)) != cudaSuccess))
      {  if (stream) OcFree(stream);
         OcError(NULL, "Error allocating GPU stream");
      }
   }

   /* Initialize the stream */
   stream -> device   = OcIncrefDevice(device);
   stream -> refcount = 1;

   return stream;
}


/* -------------------------------------------------------------------- */
static int OcDeviceGPU_createEvent(OcDevice *device, OcEvent **event)
/* -------------------------------------------------------------------- */
{  OcDeviceGPU *gpu = (OcDeviceGPU *)device;
   OcEventGPU  *eventGPU = NULL;

   if (gpu -> eventCount > 0)
   {  /* Recycle an event */
      *event = gpu -> eventBuffer;
      gpu -> eventBuffer = (*event) -> next;
      gpu -> eventCount --;
   }
   else
   {  int flags = cudaEventDisableTiming | cudaEventBlockingSync;

      /* Create a new event */
      if (cudaSetDevice(device -> index) == cudaSuccess)
         eventGPU = (OcEventGPU *)OcMalloc(sizeof(OcEventGPU));
      if ((eventGPU == NULL) ||
          (cudaEventCreateWithFlags(&(eventGPU -> cudaEvent), flags) != cudaSuccess))
      {  *event = NULL;
         if (eventGPU) OcFree(eventGPU);
         OcError(-1, "Error allocating GPU event");
      }

      /* Set the event */
      *event = (OcEvent *)eventGPU;
   }

   /* Success */
   return 0;
}


/* -------------------------------------------------------------------- */
static void OcDeviceGPU_deleteStream(OcStream *stream)
/* -------------------------------------------------------------------- */
{  OcDeviceGPU *device;

   if (stream)
   {
      /* Get the stream device */
      device = (OcDeviceGPU *)(stream -> device);

      /* Check if we want to recycle the stream (maximum of 64) */
      if (device -> streamCount < 64)
      {  stream -> next = device -> streamBuffer;
         device -> streamBuffer = stream;
         device -> streamCount ++;

         /* Decrement the device reference count to make sure that  */
         /* devices can be freed while containing buffered streams. */
         OcDecrefDevice(stream -> device);
      }
      else
      {  /* Decrement the device reference count */
         OcDecrefDevice(stream -> device);

         /* Delete the cuda stream */
         cudaStreamDestroy(((OcStreamGPU *)stream) -> cudaStream);

         /* Delete the stream object */
         OcFree(stream);
      }
   }
}


/* -------------------------------------------------------------------- */
static void OcDeviceGPU_deleteEvent(OcDevice *device, OcEvent *event)
/* -------------------------------------------------------------------- */
{  OcDeviceGPU *gpu = (OcDeviceGPU *)device;

   if (event)
   {
      /* Check if we want to recycle the event (maximum of 64) */
      if (gpu -> eventCount < 64)
      {  event -> next = gpu -> eventBuffer;
         gpu -> eventBuffer = event;
         gpu -> eventCount ++;
      }
      else
      {  /* Delete the cuda event */
         cudaEventDestroy(((OcEventGPU *)event) -> cudaEvent);

         /* Delete the event object */
         OcFree(event);
      }
   }
}


/* -------------------------------------------------------------------- */
static int OcStreamGPU_synchronize(OcStream *stream)
/* -------------------------------------------------------------------- */
{  cudaError_t status;

   status = cudaStreamSynchronize(((OcStreamGPU *)stream) -> cudaStream);
   if (status != cudaSuccess) OcError(-1, "Error synchronizing GPU stream");

   return 0;
}


/* -------------------------------------------------------------------- */
static int OcEventGPU_synchronize (OcEvent *event)
/* -------------------------------------------------------------------- */
{  cudaError_t status;

   status = cudaEventSynchronize(((OcEventGPU *)event) -> cudaEvent);
   if (status != cudaSuccess) OcError(-1, "Error synchronizing GPU event");

   return 0;
}


/* -------------------------------------------------------------------- */
static int OcEventGPU_record(OcEvent *event, OcStream *stream)
/* -------------------------------------------------------------------- */
{  cudaError_t status;

   status = cudaEventRecord(((OcEventGPU *)event) -> cudaEvent,
                            ((OcStreamGPU *)stream) -> cudaStream);
   if (status != cudaSuccess) OcError(-1, "Error recording GPU event");

   return 0;
}


/* -------------------------------------------------------------------- */
static int OcStreamGPU_waitEvent(OcStream *stream, OcEvent *event)
/* -------------------------------------------------------------------- */
{  cudaError_t status;

   status = cudaStreamWaitEvent(((OcStreamGPU *)stream) -> cudaStream,
                                ((OcEventGPU *)event) -> cudaEvent, 0);
   if (status != cudaSuccess) OcError(-1, "Error waiting for GPU event");

   return 0;
}



/* ===================================================================== */
/*                    Device instance query functions                    */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int OcDeviceGPUCount(void)
/* -------------------------------------------------------------------- */
{
   return oc_device_gpu_count;
}


/* -------------------------------------------------------------------- */
OcDevice *OcDeviceGPUByIndex(int index)
/* -------------------------------------------------------------------- */
{
   if ((index < 0) || (index >= oc_device_gpu_count)) return NULL;

   return OcGPU[index];
}


/* -------------------------------------------------------------------- */
int OcDeviceGPU_peerAccess(int device1, int device2)
/* -------------------------------------------------------------------- */
{  int result = 0;

   cudaDeviceCanAccessPeer(&result, device1, device2);

   return result;
}


/* -------------------------------------------------------------------- */
int OcDeviceGPU_isInstance(OcDevice *device)
/* -------------------------------------------------------------------- */
{
   return (device -> type == oc_device_type_gpu) ? 1 : 0;
}



/* ===================================================================== */
/*                       Device property functions                       */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
static int OcDeviceGPU_getProperties(OcDeviceGPU *device)
/* -------------------------------------------------------------------- */
{  struct cudaDeviceProp prop;

   /* Get the device properties */
   if (cudaGetDeviceProperties(&prop, ((OcDevice *)device) -> index) != cudaSuccess)
      OcError(-1, "Error getting device properties for GPU[%d]", ((OcDevice *)device) -> index);

   /* Copy the properties */
#if CUDA_VERSION < 5000
   OcError(-1, "Device properties are currently not supported for Cuda versions lower than 5.0");
#else

   /* Properties for Cuda 5.0 and higher */
   memcpy((void *)(device -> properties.name), (void *)(prop.name), 256 * sizeof(char));
   device -> properties.totalGlobalMem              = prop.totalGlobalMem;
   device -> properties.freeGlobalMem               = 0; /* Dynamic */
   device -> properties.sharedMemPerBlock           = prop.sharedMemPerBlock;
   device -> properties.regsPerBlock                = prop.regsPerBlock;
   device -> properties.warpSize                    = prop.warpSize;
   device -> properties.memPitch                    = prop.memPitch;
   device -> properties.maxThreadsPerBlock          = prop.maxThreadsPerBlock;
   device -> properties.maxThreadsDim[0]            = prop.maxThreadsDim[0];
   device -> properties.maxThreadsDim[1]            = prop.maxThreadsDim[1];
   device -> properties.maxThreadsDim[2]            = prop.maxThreadsDim[2];
   device -> properties.maxGridSize[0]              = prop.maxGridSize[0];
   device -> properties.maxGridSize[1]              = prop.maxGridSize[1];
   device -> properties.maxGridSize[2]              = prop.maxGridSize[2];
   device -> properties.clockRate                   = prop.clockRate;
   device -> properties.totalConstMem               = prop.totalConstMem;
   device -> properties.major                       = prop.major;
   device -> properties.minor                       = prop.minor;
   device -> properties.deviceOverlap               = prop.deviceOverlap ? 1 : 0;
   device -> properties.multiProcessorCount         = prop.multiProcessorCount;
   device -> properties.kernelExecTimeoutEnabled    = prop.kernelExecTimeoutEnabled ? 1 : 0;
   device -> properties.integrated                  = prop.integrated ? 1 : 0;
   device -> properties.canMapHostMemory            = prop.canMapHostMemory ? 1 : 0;
   device -> properties.computeMode                 = prop.computeMode;
   device -> properties.concurrentKernels           = prop.concurrentKernels ? 1 : 0;
   device -> properties.ECCEnabled                  = prop.ECCEnabled ? 1 : 0;
   device -> properties.pciBusID                    = prop.pciBusID;
   device -> properties.pciDeviceID                 = prop.pciDeviceID;
   device -> properties.pciDomainID                 = prop.pciDomainID;
   device -> properties.tccDriver                   = prop.tccDriver;
   device -> properties.asyncEngineCount            = prop.asyncEngineCount;
   device -> properties.unifiedAddressing           = prop.unifiedAddressing ? 1 : 0;
   device -> properties.memoryClockRate             = prop.memoryClockRate;
   device -> properties.memoryBusWidth              = prop.memoryBusWidth;
   device -> properties.l2CacheSize                 = prop.l2CacheSize;
   device -> properties.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
   device -> properties.streamPrioritiesSupported   = prop.streamPrioritiesSupported ? 1 : 0;

   /* Properties for Cuda 6.5 and higher */
#if CUDA_VERSION >= 6050
   device -> properties.globalL1CacheSupported      = prop.globalL1CacheSupported ? 1 : 0;
   device -> properties.localL1CacheSupported       = prop.localL1CacheSupported ? 1: 0;
   device -> properties.sharedMemPerMultiprocessor  = prop.sharedMemPerMultiprocessor;
   device -> properties.regsPerMultiprocessor       = prop.regsPerMultiprocessor;
   device -> properties.managedMemSupported         = -1; /* managedMemSupported not found by gcc */
   device -> properties.isMultiGpuBoard             = prop.isMultiGpuBoard ? 1 : 0;
   device -> properties.multiGpuBoardGroupID        = prop.multiGpuBoardGroupID;
#else
   device -> properties.globalL1CacheSupported      = -1;
   device -> properties.localL1CacheSupported       = -1;
   device -> properties.sharedMemPerMultiprocessor  =  0;
   device -> properties.regsPerMultiprocessor       =  0;
   device -> properties.managedMemSupported         = -1;
   device -> properties.isMultiGpuBoard             = -1;
   device -> properties.multiGpuBoardGroupID        =  0;

   /* Determine shared memory based on compute capability */
   if (prop.major == 1)
      device -> properties.sharedMemPerMultiprocessor    = 16 * 1024;  /* Compute capability 1.x       */
   else if (prop.major == 2)
      device -> properties.sharedMemPerMultiprocessor    = 48 * 1024;  /* Compute capability 2.x       */
   else if (prop.major == 3)
   {  if (prop.minor <= 5)
         device -> properties.sharedMemPerMultiprocessor = 48 * 1024;  /* Compute capability 3.0 - 3.5 */
      else
         device -> properties.sharedMemPerMultiprocessor = 112 * 1024; /* Compute capability 3.6 - 3.x */
   }
   else if (prop.major == 5)
   {  if (prop.minor < 2)
         device -> properties.sharedMemPerMultiprocessor = 64 * 1024;  /* Compute capability 5.0 - 5.1 */
      else if (prop.minor < 3)
         device -> properties.sharedMemPerMultiprocessor = 96 * 1024;  /* Compute capability 5.2       */
      else
         device -> properties.sharedMemPerMultiprocessor = 96 * 1024;  /* Compute capability 5.3 - 5.x */
   }
   else if (prop.major == 6)
   {  if (prop.minor == 0)
         device -> properties.sharedMemPerMultiprocessor = 96 * 1024;  /* Compute capability 6.0       */
      else if (prop.minor == 1)
         device -> properties.sharedMemPerMultiprocessor = 96 * 1024;  /* Compute capability 6.1       */
      else if (prop.minor == 2)
         device -> properties.sharedMemPerMultiprocessor = 96 * 1024;  /* Compute capability 6.2       */
   }

#endif

   /* Properties for Cuda 8.0 and higher */
#if CUDA_VERSION >= 8000
   device -> properties.singleToDoublePrecisionPerfRatio = prop.singleToDoublePrecisionPerfRatio;
   device -> properties.pageableMemoryAccess             = prop.pageableMemoryAccess ? 1 : 0;
   device -> properties.concurrentManagedAccess          = prop.concurrentManagedAccess ? 1 : 0;
   device -> properties.hostNativeAtomicSupported        = -1;
#else
   device -> properties.singleToDoublePrecisionPerfRatio =  0;
   device -> properties.pageableMemoryAccess             = -1;
   device -> properties.concurrentManagedAccess          = -1;
   device -> properties.hostNativeAtomicSupported        = -1;
#endif

   /* Properties derived from compute capability */
   if (prop.major == 1)
   {  if (prop.minor <= 1)
      {  device -> properties.maxBlocksPerMultiProcessor = 8;    /* Compute capability 1.0 - 1.1 */
         device -> properties.maxWarpsPerMultiProcessor  = 24;
      }
      else
      {  device -> properties.maxBlocksPerMultiProcessor = 8;    /* Compute capability 1.2 - 1.x */
         device -> properties.maxWarpsPerMultiProcessor  = 32;
      }
   }
   else if (prop.major == 2)
   {  if (prop.minor >= 0)
      {  device -> properties.maxBlocksPerMultiProcessor = 8;    /* Compute capability 2.x       */
         device -> properties.maxWarpsPerMultiProcessor  = 48;
      }
   }
   else if (prop.major == 3)
   {  if (prop.minor >= 0)
      {  device -> properties.maxBlocksPerMultiProcessor = 16;   /* Compute capability 3.x       */
         device -> properties.maxWarpsPerMultiProcessor  = 64;
      }
   }
   else if (prop.major >= 5)
   {  if (prop.minor >= 0)
      {  device -> properties.maxBlocksPerMultiProcessor = 32;   /* Compute capability 5.x - y.x */
         device -> properties.maxWarpsPerMultiProcessor  = 64;
      }
   }

   return OcDeviceGPU_updateProperties(device);
#endif /* Version 5.0 or higher */
}


/* -------------------------------------------------------------------- */
int OcDeviceGPU_updateProperties(OcDeviceGPU *device)
/* -------------------------------------------------------------------- */
{  cudaError_t status;
   size_t      free, total;

   /* Set the device */
   if ((status = cudaSetDevice(((OcDevice *)device) -> index)) != cudaSuccess)
      OcError(-1, "Error setting cuda device in update properties function");

   /* Query the memory information */
   if ((status = cudaMemGetInfo(&free, &total)) != cudaSuccess)
      OcError(-1, "Error getting memory info in update properties function");
   
   /* Update the free memory field */
   device -> properties.freeGlobalMem = free;

   return 0;
}

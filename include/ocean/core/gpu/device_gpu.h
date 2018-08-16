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

#ifndef __OC_DEVICE_GPU_H__
#define __OC_DEVICE_GPU_H__

#include "ocean/base/api.h"
#include "ocean/base/device.h"
#include "ocean/core/gpu/cuda.h"

#include <stddef.h>


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  char      name[256];
   size_t    totalGlobalMem;
   size_t    freeGlobalMem;       /* Dynamic */
   size_t    sharedMemPerBlock;
   int       regsPerBlock;
   int       warpSize;
   size_t    memPitch;
   int       maxThreadsPerBlock;
   int       maxThreadsDim[3];
   int       maxGridSize[3];
   int       clockRate;
   size_t    totalConstMem;
   int       major;
   int       minor;
/* size_t    textureAlignment; */
/* size_t    texturePitchAlignment; */
   int       deviceOverlap;
   int       multiProcessorCount;
   int       kernelExecTimeoutEnabled;
   int       integrated;
   int       canMapHostMemory;
   int       computeMode;
/* int       maxTexture1D; */
/* int       maxTexture1DMipmap; */
/* int       maxTexture1DLinear; */
/* int       maxTexture2D[2]; */
/* int       maxTexture2DMipmap[2]; */
/* int       maxTexture2DLinear[3]; */
/* int       maxTexture2DGather[2]; */
/* int       maxTexture3D[3]; */
/* int       maxTexture3DAlt[3]; */
/* int       maxTextureCubemap; */
/* int       maxTexture1DLayered[2]; */
/* int       maxTexture2DLayered[3]; */
/* int       maxTextureCubemapLayered[2]; */
/* int       maxSurface1D; */
/* int       maxSurface2D[2]; */
/* int       maxSurface3D[3]; */
/* int       maxSurface1DLayered[2]; */
/* int       maxSurface2DLayered[3]; */
/* int       maxSurfaceCubemap; */
/* int       maxSurfaceCubemapLayered[2]; */
/* size_t    surfaceAlignment; */
   int       concurrentKernels;
   int       ECCEnabled;
   int       pciBusID;
   int       pciDeviceID;
   int       pciDomainID;
   int       tccDriver;
   int       asyncEngineCount;
   int       unifiedAddressing;
   int       memoryClockRate;
   int       memoryBusWidth;
   int       l2CacheSize;
   int       maxThreadsPerMultiProcessor;
   int       streamPrioritiesSupported;
   int       globalL1CacheSupported;           /* Version 6.5 */
   int       localL1CacheSupported;            /* Version 6.5 */
   size_t    sharedMemPerMultiprocessor;       /* Version 6.5 */
   int       regsPerMultiprocessor;            /* Version 6.5 */
   int       managedMemSupported;              /* Version 6.5 */
   int       isMultiGpuBoard;                  /* Version 6.5 */
   int       multiGpuBoardGroupID;             /* Version 6.5 */
   int       singleToDoublePrecisionPerfRatio; /* Version 8.0 */
   int       pageableMemoryAccess;             /* Version 8.0 */
   int       concurrentManagedAccess;          /* Version 8.0 */
   int       hostNativeAtomicSupported;        /* Documented in version 8.0.61 but not available */

   /* Implicit properties based on compute capability */
   int       maxBlocksPerMultiProcessor;
   int       maxWarpsPerMultiProcessor;
   int       maxSharedMemPerSM;
} OcDevicePropGPU;


typedef struct
{  OcDevice HEAD;

   /* Fields specific to the GPU device */
   OcDevicePropGPU properties;

   /* Stream and event buffers */
   OcStream *streamBuffer;  /* Recycled streams         */
   OcEvent  *eventBuffer;   /* Recycled events          */
   int       streamCount;   /* Streams in stream buffer */
   int       eventCount;    /* Events in event buffer   */
} OcDeviceGPU;
 

typedef struct
{  OcStream HEAD;

   /* Fields specific to GPU streams */
   cudaStream_t cudaStream;
} OcStreamGPU;


typedef struct
{  OcEvent HEAD;

   /* Fields specific to GPU events */
   cudaEvent_t cudaEvent;
} OcEventGPU;



/* ===================================================================== */
/* Global variables                                                      */
/* ===================================================================== */

/* Declaration of the global GPU devices */
extern OcDevice **OcGPU;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Function to query devices instances */
OC_API int       OcDeviceGPUCount(void);
OC_API OcDevice *OcDeviceGPUByIndex(int index);
OC_API int       OcDeviceGPU_peerAccess(int device1, int device2);
OC_API int       OcDeviceGPU_isInstance(OcDevice *device);

/* Functions to query device properties */
OC_API int       OcDeviceGPU_updateProperties(OcDeviceGPU *device);

/* Internal functions for initializing device information */
OC_API int       OcInitDevicesGPU(void);

#endif

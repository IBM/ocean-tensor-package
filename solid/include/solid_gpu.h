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

#ifndef __SOLID_GPU_H__
#define __SOLID_GPU_H__

#include "solid.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  int    major;
   int    minor;
   int    multiprocessor_count;
   int    max_threads_per_block;
   int    max_threads_per_multiprocessor;
   int    max_blocks_per_multiprocessor;
   int    max_warps_per_multiprocessor;
   int    max_resident_blocks_per_device; /* multiprocessor_count * max_blocks_per_multiprocessor */
   int    max_shared_mem_per_multiprocessor;
   int    max_shared_mem_per_threadblock;
   int    max_gridsize[3];
   int    max_blocksize[3];
} solid_gpu_properties;


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Initialization and finalization */
SOLID_API int  solid_gpu_initialize(void);
SOLID_API void solid_gpu_finalize(void);

/* Query functions */
SOLID_API int                   solid_gpu_check_status                  (void);
SOLID_API int                   solid_gpu_get_current_device            (int *device);
SOLID_API int                   solid_gpu_multiprocessor_count          (int device);
SOLID_API int                   solid_gpu_max_threads_per_multiprocessor(int device);
SOLID_API solid_gpu_properties *solid_gpu_get_device_properties         (int device);
SOLID_API solid_gpu_properties *solid_gpu_get_current_device_properties (void);
#endif

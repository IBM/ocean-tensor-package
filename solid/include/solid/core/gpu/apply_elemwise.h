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

#ifndef __SOLID_GPU_APPLY_ELEMWISE_H__
#define __SOLID_GPU_APPLY_ELEMWISE_H__

#include "solid.h"
#include "solid/base/gpu/types_gpu.h"

SOLID_API int solid_gpu_config_elemwise(solid_gpu_config *config, size_t nelem, int threads_per_block);
SOLID_API int solid_gpu_large_indexing(int ndims, const size_t *size, const ptrdiff_t *strides, int elemsize);


/* --------------------------------------------------------------------- */
/* Data types for size and index                                         */
/* --------------------------------------------------------------------- */
#define SOLID_ELEMWISE_INDEX_TYPE_SMALL   int
#define SOLID_ELEMWISE_INDEX_TYPE_LARGE   long int
#define SOLID_ELEMWISE_INDEX_TYPE_B(SIZE) SOLID_ELEMWISE_INDEX_TYPE_##SIZE
#define SOLID_ELEMWISE_INDEX_TYPE(SIZE)   SOLID_ELEMWISE_INDEX_TYPE_B(SIZE)

#define SOLID_ELEMWISE_SIZE_TYPE_SMALL    unsigned int
#define SOLID_ELEMWISE_SIZE_TYPE_LARGE    unsigned long int
#define SOLID_ELEMWISE_SIZE_TYPE_B(SIZE)  SOLID_ELEMWISE_SIZE_TYPE_##SIZE
#define SOLID_ELEMWISE_SIZE_TYPE(SIZE)    SOLID_ELEMWISE_SIZE_TYPE_B(SIZE)

#endif

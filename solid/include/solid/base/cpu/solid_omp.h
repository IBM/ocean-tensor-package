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

#ifndef __SOLID_OMP_H__
#define __SOLID_OMP_H__

#include "solid/base/generic/api.h"
#include "solid/base/cpu/solid_omp_config.h"

#if SOLID_ENABLE_OMP
   #include <omp.h>
   #define SOLID_OMP_MAX_THREADS 2048
#endif


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

SOLID_API int solid_omp_get_max_threads(void);
SOLID_API int solid_omp_get_thread_num(void);
SOLID_API int solid_omp_run_parallel(void (*funptr)(int,void *), int n, int maxThreads, void *data);

#endif

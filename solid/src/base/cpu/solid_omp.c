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

#include "solid_cpu.h"
#include "solid/base/cpu/solid_omp.h"


/* ===================================================================== */
#if SOLID_ENABLE_OMP /* Function implementation with OMP                 */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int solid_omp_get_max_threads(void)
/* -------------------------------------------------------------------- */
{  int result = omp_get_max_threads();

   if (result <= SOLID_OMP_MAX_THREADS)
        return result;
   else return SOLID_OMP_MAX_THREADS;
}


/* -------------------------------------------------------------------- */
int solid_omp_run_parallel(void (*funptr)(int,void *), int n, int maxThreads, void *data)
/* -------------------------------------------------------------------- */
{
   if (maxThreads < 0)
   {  _Pragma("omp parallel")
      {  int rank = omp_get_thread_num();
         int nthreads = omp_get_num_threads();
         int i;

         for (i = rank; i < n; i+=nthreads) funptr(i, data);
      }
   }
   else if (maxThreads > 1)
   {  _Pragma("omp parallel num_threads(maxThreads)")
      {  int rank = omp_get_thread_num();
         int nthreads = omp_get_num_threads();
         int i;

         for (i = rank; i < n; i+=nthreads) funptr(i, data);
      }
   }
   else
   {  funptr(0, data);
   }

   return 0;
}


/* ===================================================================== */
#else /* Function implementation without OMP                             */
/* ===================================================================== */

/* -------------------------------------------------------------------- */
int solid_omp_get_max_threads(void)
/* -------------------------------------------------------------------- */
{  return 1;
}


/* -------------------------------------------------------------------- */
int solid_omp_run_parallel(void (*funptr)(int,void *), int n, int maxThreads, void *data)
/* -------------------------------------------------------------------- */
{  int i;

   for (i = 0; i < n; i++) funptr(i, data);

   return 0;
}


/* ===================================================================== */
#endif /* #if SOLID_ENABLE_OMP */
/* ===================================================================== */


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

#include "solid.h"


/* ------------------------------------------------------------------------ */
__global__ void solid_gpu_dummy_kernel_intrnl(int *data)
/* ------------------------------------------------------------------------ */
{
   if (data != NULL) *data = 1;
}


/* ------------------------------------------------------------------------ */
SOLID_API int solid_gpu_dummy_kernel(void)
/* ------------------------------------------------------------------------ */
{  dim3 blockSize(1,1,1);
   dim3 gridSize(1,1,1);

   solid_gpu_dummy_kernel_intrnl<<<gridSize, blockSize>>>(NULL);

   return 0;
}

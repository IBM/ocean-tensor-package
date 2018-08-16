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

#include "solid/core/gpu/apply_elemwise.h"
#include "solid/core/gpu/apply_elemwise2.h"
#include "solid_gpu.h"


/* ------------------------------------------ */
/* Kernel index table                         */
/*                                            */
/* Index   Ndims   Unroll count   Index range */
/* -----   -----   ------------   ----------- */
/* 0       1D      1              SMALL       */
/* 1       2D      1              SMALL       */
/* 2       3D      1              SMALL       */
/* 3       ND      1              SMALL       */
/* 4       1D      1              LARGE       */
/* 5       2D      1              LARGE       */
/* 6       3D      1              LARGE       */
/* 7       ND      1              LARGE       */
/* 8       2D      2              LARGE       */
/* 9       3D      2              LARGE       */
/* 10      ND      2              LARGE       */
/* 11      ND      4              LARGE       */
/* ------------------------------------------ */


/* ------------------------------------------------------------------------ */
int solid_gpu_elemwise2_analyze(int ndims, const size_t *size,
                                const ptrdiff_t *strides1, void *ptr1,
                                const ptrdiff_t *strides2, void *ptr2,
                                int elemsize1, int elemsize2, int flag_unroll, 
                                solid_elemwise2_data_small *data_small,
                                solid_elemwise2_data_large *data_large,
                                solid_gpu_config *config, int *kernel_index)
/* ------------------------------------------------------------------------ */
{  size_t       nelem;
   size_t       size_zero;
   ptrdiff_t    strides_zero;
   int          unroll_factor = 1;
   int          flag_large = 0;
   int          result = 0;
   int          i;

   /* Initialize the size for scalar */
   if (ndims == 0)
   {  nelem        = 1;
      ndims        = 1;
      size_zero    = 1;
      strides_zero = 0;
      size         = &size_zero;
      strides1     = &strides_zero;
      strides2     = &strides_zero;
   }
   else
   {  /* Determine the number of elements */
      for (nelem = 1, i = 0; i < ndims; i++)
      {  nelem *= size[i];
      }

      /* Determine the indexing mode */
      flag_large = (solid_gpu_large_indexing(ndims, size, strides1, elemsize1) ||
                    solid_gpu_large_indexing(ndims, size, strides2, elemsize2));
   }
   
   /* ------------------------------------------------------- */
   /* Small-range indexing                                    */
   /* ------------------------------------------------------- */
   if (flag_large == 0)
   {
      /* Copy the tensor layout */
      for (i = 0; i < ndims; i++)
      {  data_small -> size[i]     = size[i];
         data_small -> strides1[i] = strides1[i];
         data_small -> strides2[i] = strides2[i];
      }
      data_small -> ndims = ndims;
      data_small -> ptr1  = (char *)ptr1;
      data_small -> ptr2  = (char *)ptr2;
      data_small -> nelem = nelem;
      data_large -> nelem = nelem;

      /* Select the kernel index */
      *kernel_index = (ndims <= 3) ? ndims-1 : 3;
   }
   /* ------------------------------------------------------- */
   /* Large-range indexing                                    */
   /* ------------------------------------------------------- */
   else
   {
      /* Copy the tensor layout */
      for (i = 0; i < ndims; i++)
      {  data_large -> size[i]     = size[i];
         data_large -> strides1[i] = strides1[i];
         data_large -> strides2[i] = strides2[i];
      }
      data_large -> ndims = ndims;
      data_large -> ptr1  = (char *)ptr1;
      data_large -> ptr2  = (char *)ptr2;
      data_large -> nelem = nelem;
      data_small -> nelem = nelem;

      /* Select the kernel index */
      *kernel_index = (ndims <= 3) ? ndims+3 : 7;

      if (flag_unroll)
      {  if (((size[0] % 2) == 0) && (ndims > 1))
         {  unroll_factor = 2;
            switch(ndims)
            {  case 2 : *kernel_index =  8; break;
               case 3 : *kernel_index =  9; break;
               default: *kernel_index = 10; break;
            }
         }

         if (((size[0] % 4) == 0) && (ndims > 3))
         {  unroll_factor = 4;
            (*kernel_index) ++;
         }
      }
   }

   /* ------------------------------------------------------- */
   /* Determine the number of threads and blocks              */
   /* ------------------------------------------------------- */
   if (config)
   {  int threads_per_block = 512;

      /* Determine the configuration */
      result = solid_gpu_config_elemwise(config, nelem / unroll_factor, threads_per_block);
   }

   return result;
}

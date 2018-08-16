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
#include "solid/core/gpu/apply_elemwise2b.h"
#include "solid_gpu.h"


/* ---------------------------------------------------- */
/* Kernel index table                                   */
/*                                                      */
/* Index   Ndims1   Ndims2   Unroll count   Index range */
/* -----   ------   ------   ------------   ----------- */
/* 0       1D       1D       1              SMALL       */
/* 1       1D       2D       1              SMALL       */
/* 2       1D       ND       1              SMALL       */
/* 3       2D       1D       1              SMALL       */
/* 4       2D       2D       1              SMALL       */
/* 5       2D       ND       1              SMALL       */
/* 6       ND       1D       1              SMALL       */
/* 7       ND       2D       1              SMALL       */
/* 8       ND       ND       1              SMALL       */
/*                                                      */
/* 9       1D       1D       1              LARGE       */
/* 10      1D       2D       1              LARGE       */
/* 11      1D       ND       1              LARGE       */
/* 12      2D       1D       1              LARGE       */
/* 13      2D       2D       1              LARGE       */
/* 14      2D       ND       1              LARGE       */
/* 15      ND       1D       1              LARGE       */
/* 16      ND       2D       1              LARGE       */
/* 17      ND       ND       1              LARGE       */
/*                                                      */
/* 18      ND       ND       2 (8)          SMALL       */
/*                                                      */
/* 19      1D       2D       2 (10)         LARGE       */
/* 20      1D       ND       2 (11)         LARGE       */
/* 21      1D       ND       4 (11)         LARGE       */
/* 22      2D       1D       2 (12)         LARGE       */
/* 23      2D       2D       2 (13)         LARGE       */
/* 24      2D       ND       2 (14)         LARGE       */
/* 25      2D       ND       4 (14)         LARGE       */
/* 26      ND       1D       2 (15)         LARGE       */
/* 27      ND       1D       4 (15)         LARGE       */
/* 28      ND       2D       2 (16)         LARGE       */
/* 29      ND       2D       4 (16)         LARGE       */
/* 30      ND       ND       2 (17)         LARGE       */
/* 31      ND       ND       4 (17)         LARGE       */
/* ---------------------------------------------------- */


/* ------------------------------------------------------------------------ */
int solid_gpu_elemwise2b_analyze(int ndims1, const size_t *size1, const ptrdiff_t *strides1, void *ptr1,
                                 int ndims2, const size_t *size2, const ptrdiff_t *strides2, void *ptr2,
                                 int elemsize1, int elemsize2, int flag_unroll,
                                 solid_elemwise2b_data_small *data_small,
                                 solid_elemwise2b_data_large *data_large,
                                 solid_gpu_config *config, int *kernel_index)
/* ------------------------------------------------------------------------ */
{  size_t       nelem;
   size_t       size_zero;
   ptrdiff_t    strides_zero;
   int          unroll_factor = 1;
   int          flag_large = 0;
   int          result = 0;
   int          i, j, k;

   /* Initialize the size for scalar */
   if ((ndims1 == 0) || (ndims2 == 0))
   {  nelem        = 1;
      ndims1       = 1;
      ndims2       = 1;
      size_zero    = 1;
      strides_zero = 0;
      size1        = &size_zero;
      size2        = &size_zero;
      strides1     = &strides_zero;
      strides2     = &strides_zero;
   }
   else
   {  /* Determine the number of elements */
      for (nelem = 1, i = 0; i < ndims1; i++)
      {  nelem *= size1[i];
      }

      /* Determine the indexing mode */
      flag_large = (solid_gpu_large_indexing(ndims1, size1, strides1, elemsize1) ||
                    solid_gpu_large_indexing(ndims2, size2, strides2, elemsize2));
   }
   
   /* ------------------------------------------------------- */
   /* Small-range indexing                                    */
   /* ------------------------------------------------------- */
   if (flag_large == 0)
   {
      /* Copy tensor #1 layout */
      data_small -> layout1.ndims = ndims1;
      data_small -> layout1.ptr   = (char *)ptr1;
      for (i = 0; i < ndims1; i++)
      {  data_small -> layout1.size[i]    = size1[i];
         data_small -> layout1.strides[i] = strides1[i];
      }

      /* Copy tensor #2 layout */
      data_small -> layout2.ndims = ndims2;
      data_small -> layout2.ptr   = (char *)ptr2;
      for (i = 0; i < ndims2; i++)
      {  data_small -> layout2.size[i]    = size2[i];
         data_small -> layout2.strides[i] = strides2[i];
      }
      data_small -> nelem = nelem;
      data_large -> nelem = nelem;

      /* Select the kernel index */
      i = (ndims1 <= 2) ? ndims1-1 : 2;
      j = (ndims2 <= 2) ? ndims2-1 : 2;
      *kernel_index = (i * 3) + j;

      if (flag_unroll)
      {  if ((i >= 2) && (j >= 2) && ((size1[0] % 2) == 0) && ((size2[0] % 2) == 0))
         {  unroll_factor = 2;
            *kernel_index = 18;
         }
      }
   }
   /* ------------------------------------------------------- */
   /* Large-range indexing                                    */
   /* ------------------------------------------------------- */
   else
   {
      /* Copy tensor #1 layout */
      data_large -> layout1.ndims = ndims1;
      data_large -> layout1.ptr   = (char *)ptr1;
      for (i = 0; i < ndims1; i++)
      {  data_large -> layout1.size[i]    = size1[i];
         data_large -> layout1.strides[i] = strides1[i];
      }

      /* Copy tensor #2 layout */
      data_large -> layout2.ndims = ndims2;
      data_large -> layout2.ptr   = (char *)ptr2;
      for (i = 0; i < ndims2; i++)
      {  data_large -> layout2.size[i]    = size2[i];
         data_large -> layout2.strides[i] = strides2[i];
      }
      data_large -> nelem = nelem;
      data_small -> nelem = nelem;

      /* Select the kernel index */
      i = (ndims1 <= 2) ? ndims1 : 3;  /* 1-based index */
      j = (ndims2 <= 2) ? ndims2 : 3;  /* 1-based index */
      *kernel_index = 5 + (i * 3) + j; /* 5 = 9 - 3*1 - 1 */

      if (flag_unroll)
      {  if (((size1[0] % 2) == 0) && ((size2[0] % 2) == 0))
         {  k = 0;
            if (i == 1)
            {  if (j >= 2) k = 17 + j; /* 19 or 20 */
            }
            else if (i == 2)
            {  k = 21 + j; /* 22, 23, or 24 */
            }
            else
            {  k = 24 + 2 * j; /* 26, 28, or 30 */
            }
            if (k != 0)
            {  unroll_factor = 2;
               *kernel_index = k;
            }
         }

         if (((size1[0] % 4) == 0) && ((size2[0] % 4) == 0))
         {  k = 0;
            if (i <= 2)
            {  if (j == 3) k = 1; /* 21 or 25 */
            }
            else
            {  k = 1; /* 27, 29, or 31 */
            }
            if (k != 0)
            {  unroll_factor = 4;
               (*kernel_index) ++;
            }
         }
      }
   }

   /* ------------------------------------------------------- */
   /* Determine the number of threads and blocks              */
   /* ------------------------------------------------------- */
   if (config)
   {  int threads_per_block = 512;

      result = solid_gpu_config_elemwise(config, nelem / unroll_factor, threads_per_block);
   }

   return result;
}

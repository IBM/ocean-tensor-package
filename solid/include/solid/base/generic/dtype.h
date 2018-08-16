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

#ifndef __SOLID_DTYPE_H__
#define __SOLID_DTYPE_H__

#include "solid/base/generic/types.h"

/* Type index values */
typedef enum { SD_BOOL    = 0,
               SD_UINT8   = 1,
               SD_UINT16  = 2,
               SD_UINT32  = 3,
               SD_UINT64  = 4,
               SD_INT8    = 5,
               SD_INT16   = 6,
               SD_INT32   = 7,
               SD_INT64   = 8,
               SD_HALF    = 9,
               SD_FLOAT   = 10,
               SD_DOUBLE  = 11,
               SD_CHALF   = 12,
               SD_CFLOAT  = 13,
               SD_CDOUBLE = 14
             } solid_type_index;

#endif

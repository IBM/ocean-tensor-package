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

#ifndef __SOLID_SCALAR_H__
#define __SOLID_SCALAR_H__

#include "solid/base/generic/dtype.h"


/* ===================================================================== */
/* Scalars                                                               */
/* ===================================================================== */
             
typedef union
{  solid_bool    _bool;
   solid_uint8   _uint8;
   solid_uint16  _uint16;
   solid_uint32  _uint32;
   solid_uint64  _uint64;
   solid_int8    _int8;
   solid_int16   _int16;
   solid_int32   _int32;
   solid_int64   _int64;
   solid_half    _half;
   solid_float   _float;
   solid_double  _double;
   solid_chalf   _chalf;
   solid_cfloat  _cfloat;
   solid_cdouble _cdouble;
} solid_scalar;

/* Macro to get value from scalar pointer */
#define SOLID_SCALAR_TYPE_B(TYPE, SCALAR) ((SCALAR)._##TYPE)
#define SOLID_SCALAR_TYPE(TYPE, SCALAR) SOLID_SCALAR_TYPE_B(TYPE, SCALAR)
#define SOLID_SCALAR(SCALAR) SOLID_SCALAR_TYPE(SDXTYPE,SCALAR)

#define SOLID_SCALAR_VALUE_TYPE_B(TYPE, SCALAR) ((SCALAR) -> _##TYPE)
#define SOLID_SCALAR_VALUE_TYPE(TYPE, SCALAR) SOLID_SCALAR_VALUE_TYPE_B(TYPE, SCALAR)
#define SOLID_SCALAR_VALUE(SCALAR) SOLID_SCALAR_VALUE_TYPE(SDXTYPE,SCALAR)

#define SOLID_SCALAR_C_VALUE_TYPE_B(CTYPE, SCALAR) (*((CTYPE *)&(SCALAR)))
#define SOLID_SCALAR_C_VALUE_TYPE(TYPE, SCALAR) SOLID_SCALAR_C_VALUE_TYPE_B(SOLID_C_TYPE(TYPE), SCALAR)
#define SOLID_SCALAR_C_VALUE(SCALAR) SOLID_SCALAR_C_VALUE_TYPE(SDXTYPE,SCALAR)

#endif

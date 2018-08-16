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

#ifndef __OC_SCALAR_H__
#define __OC_SCALAR_H__

#include "ocean/base/dtype.h"
#include "ocean/base/generate_macros.h"
#include "ocean/base/api.h"
#include "ocean/base/types.h"


/* ===================================================================== */
/* Structure definitions                                                 */
/* ===================================================================== */

typedef struct
{  /* The dtype field has type OcFlags instead of OcDType to  */
   /* allow encoding of weak-type information. It can be seen */
   /* as a combination of the OcDType value (enumerator) with */
   /* the weak flags. In most cases the field is initialized  */
   /* directly with a OcDType value, which is permitted in C  */
   /* and results in an automatic type-cast to int. This way  */
   /* the weak-type flags are directly set to zero, and thus  */
   /* avoids having to intitialize a separate flag field. The */
   /* only disadvantage is that the dtype field can no longer */
   /* be read directly, and instead the OcScalar_dtype macro  */
   /* should be used, which returns a OcDType value (casting  */
   /* from an integer to enum is also permitted in C). The    */
   /* flags can be extracted using OcScalar_flags macros.     */

   OcDType  dtype;
   union
   {  OcBool    sBool;
      OcUInt8   sUInt8;
      OcUInt16  sUInt16;
      OcUInt32  sUInt32;
      OcUInt64  sUInt64;
      OcInt8    sInt8;
      OcInt16   sInt16;
      OcInt32   sInt32;
      OcInt64   sInt64;
      OcHalf    sHalf;
      OcFloat   sFloat;
      OcDouble  sDouble;
      OcCHalf   sCHalf;
      OcCFloat  sCFloat;
      OcCDouble sCDouble;
   } value;
} OcScalar;


/* ===================================================================== */
/* Macros                                                                */
/* ===================================================================== */

#define OcScalar_isSigned(scalar)    (OcDType_isSigned((scalar) -> dtype))
#define OcScalar_isComplex(scalar)   (OcDType_isComplex((scalar) -> dtype))
#define OcScalar_data(scalar)        ((void *)(&((scalar) -> value)))

#endif

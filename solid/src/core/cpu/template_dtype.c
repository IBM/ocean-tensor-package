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
#include "solid/base/generic/dtype_macros.h"
#include "solid/base/cpu/dtype_cpu.h"

const int solid_cpu_type_size[] =
   {  sizeof(SOLID_C_TYPE(bool)),
      sizeof(SOLID_C_TYPE(uint8)),
      sizeof(SOLID_C_TYPE(uint16)),
      sizeof(SOLID_C_TYPE(uint32)),
      sizeof(SOLID_C_TYPE(uint64)),
      sizeof(SOLID_C_TYPE(int8)),
      sizeof(SOLID_C_TYPE(int16)),
      sizeof(SOLID_C_TYPE(int32)),
      sizeof(SOLID_C_TYPE(int64)),
      sizeof(SOLID_C_TYPE(half)),
      sizeof(SOLID_C_TYPE(float)),
      sizeof(SOLID_C_TYPE(double)),
      sizeof(SOLID_C_TYPE(chalf)),
      sizeof(SOLID_C_TYPE(cfloat)),
      sizeof(SOLID_C_TYPE(cdouble)),
   };

const int solid_cpu_worktype_size[] =
   {  sizeof(SOLID_C_WORKTYPE_TYPE(bool)),
      sizeof(SOLID_C_WORKTYPE_TYPE(uint8)),
      sizeof(SOLID_C_WORKTYPE_TYPE(uint16)),
      sizeof(SOLID_C_WORKTYPE_TYPE(uint32)),
      sizeof(SOLID_C_WORKTYPE_TYPE(uint64)),
      sizeof(SOLID_C_WORKTYPE_TYPE(int8)),
      sizeof(SOLID_C_WORKTYPE_TYPE(int16)),
      sizeof(SOLID_C_WORKTYPE_TYPE(int32)),
      sizeof(SOLID_C_WORKTYPE_TYPE(int64)),
      sizeof(SOLID_C_WORKTYPE_TYPE(half)),
      sizeof(SOLID_C_WORKTYPE_TYPE(float)),
      sizeof(SOLID_C_WORKTYPE_TYPE(double)),
      sizeof(SOLID_C_WORKTYPE_TYPE(chalf)),
      sizeof(SOLID_C_WORKTYPE_TYPE(cfloat)),
      sizeof(SOLID_C_WORKTYPE_TYPE(cdouble)),
   };

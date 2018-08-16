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

#ifndef __SOLID_GENERATE_MACROS_H__
#define __SOLID_GENERATE_MACROS_H__

#include "solid/base/generic/dtype_macros.h"

/* Name prefix (solid_device_name) */
#define SOLID_NAME_C(NAME, DEVICE)               solid ## _ ## DEVICE ## _ ## NAME
#define SOLID_NAME_B(NAME, DEVICE)               SOLID_NAME_C(NAME, DEVICE)
#define SOLID_NAME(NAME)                         SOLID_NAME_B(NAME, SOLID_DEVICE)

/* Function name - one type (solid_device_name_type) */
#define SOLID_FUNCTION_TYPE_C(NAME,TYPE)         NAME ## _ ## TYPE
#define SOLID_FUNCTION_TYPE_B(NAME,TYPE)         SOLID_FUNCTION_TYPE_C(NAME,TYPE)
#define SOLID_FUNCTION_TYPE(NAME,TYPE)           SOLID_FUNCTION_TYPE_B(SOLID_NAME(NAME),TYPE)
#define SOLID_FUNCTION(NAME)                     SOLID_FUNCTION_TYPE(NAME,SDXTYPE)

/* Function name - two types (solid_device_name_type1)       */
/*                           (solid_device_name_type1_type2) */
#define SOLID_FUNCTION2_TYPES_C1(NAME,TYPE1,TYPE2)          NAME ## _ ## TYPE1
#define SOLID_FUNCTION2_TYPES_C2(NAME,TYPE1,TYPE2)          NAME ## _ ## TYPE1 ## _ ## TYPE2
#define SOLID_FUNCTION2_TYPES_B(NAME,NAME_TYPE,TYPE1,TYPE2) SOLID_FUNCTION2_TYPES_C##NAME_TYPE(NAME,TYPE1,TYPE2)
#define SOLID_FUNCTION2_TYPES(NAME,NAME_TYPE,TYPE1,TYPE2)   SOLID_FUNCTION2_TYPES_B(SOLID_NAME(NAME),NAME_TYPE,TYPE1,TYPE2)
#define SOLID_FUNCTION2(NAME)                               SOLID_FUNCTION2_TYPES(NAME,2,SDXTYPE,SDXTYPE2)

/* Function name - three types (solid_device_name_type1)             */
/*                             (solid_device_name_type1_type2)       */
/*                             (solid_device_name_type1_type2_type3) */
#define SOLID_FUNCTION3_TYPES_C1(NAME,TYPE1,TYPE2,TYPE3)          NAME ## _ ## TYPE1
#define SOLID_FUNCTION3_TYPES_C2(NAME,TYPE1,TYPE2,TYPE3)          NAME ## _ ## TYPE1 ## _ ## TYPE2
#define SOLID_FUNCTION3_TYPES_C3(NAME,TYPE1,TYPE2,TYPE3)          NAME ## _ ## TYPE1 ## _ ## TYPE2 ## _ ## TYPE3
#define SOLID_FUNCTION3_TYPES_B(NAME,NAME_TYPE,TYPE1,TYPE2,TYPE3) SOLID_FUNCTION3_TYPES_C##NAME_TYPE(NAME,TYPE1,TYPE2,TYPE3)
#define SOLID_FUNCTION3_TYPES(NAME,NAME_TYPE,TYPE1,TYPE2,TYPE3)   SOLID_FUNCTION3_TYPES_B(SOLID_NAME(NAME),NAME_TYPE,TYPE1,TYPE2,TYPE3)


/* ------------------------------------------------------------------------ */
/* Kernel parameters - user defined                                         */
/* ------------------------------------------------------------------------ */
#define SOLID_PARAM_PREFIX_B(PREFIX) PREFIX ## _param
#define SOLID_PARAM_PREFIX(PREFIX)   SOLID_PARAM_PREFIX_B(PREFIX)
#define SOLID_PARAM_TYPE(NAME,TYPE)  SOLID_PARAM_PREFIX(SOLID_FUNCTION_TYPE(NAME,TYPE))
#define SOLID_PARAM(NAME)            SOLID_PARAM_TYPE(NAME,SDXTYPE)

#define SOLID_PARAM_STRUCT_0(PREFIX,PARAM) \
  /* Empty */

#define SOLID_PARAM_STRUCT_1(PREFIX,PARAM) \
  typedef struct \
  PARAM \
  SOLID_PARAM_PREFIX(PREFIX);

#define SOLID_PARAM_STRUCT(PREFIX,BODY) \
   SOLID_PARAM_STRUCT_1(PREFIX,BODY)
#define SOLID_PARAM_STRUCT_FLAG_B(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_PARAM_STRUCT_##FLAG_PARAM(PREFIX, PARAM)

#define SOLID_PARAM_STRUCT_FLAG(PREFIX, FLAG_PARAM, PARAM) \
   SOLID_PARAM_STRUCT_FLAG_B(PREFIX, FLAG_PARAM, PARAM)

#endif

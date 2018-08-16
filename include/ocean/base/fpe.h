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

#ifndef __OC_FPE_H__
#define __OC_FPE_H__

#include "ocean/base/api.h"

#include <fenv.h>

#define OC_FPE_DIVIDE_BY_ZERO   0x01
#define OC_FPE_OVERFLOW         0x02
#define OC_FPE_UNDERFLOW        0x04
#define OC_FPE_INEXACT          0x08
#define OC_FPE_INVALID          0x10

OC_API int  OcFPE_getStatus(void);
OC_API int  OcFPE_testStatus(int exception);
OC_API void OcFPE_clear(void);
OC_API void OcFPE_raise(int exception);

#endif

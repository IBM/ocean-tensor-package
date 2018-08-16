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

#ifndef __SOLID_FPE_H__
#define __SOLID_FPE_H__

#include "solid/base/generic/api.h"

#include <fenv.h>

#define SD_FPE_DIVIDE_BY_ZERO   0x01
#define SD_FPE_OVERFLOW         0x02
#define SD_FPE_UNDERFLOW        0x04
#define SD_FPE_INEXACT          0x08
#define SD_FPE_INVALID          0x10

SOLID_API int  solid_fpe_get_status(void);
SOLID_API int  solid_fpe_test_status(int exception);
SOLID_API void solid_fpe_clear(void);
SOLID_API void solid_fpe_raise(int exception);

#endif

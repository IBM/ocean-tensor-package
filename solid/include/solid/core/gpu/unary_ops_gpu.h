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

#ifndef __SOLID_GPU_CORE_UNARY_OPS_H__
#define __SOLID_GPU_CORE_UNARY_OPS_H__

#include "solid/base/gpu/solid_cuda.h"
#include "math_constants.h"


/* Single-precision functions and constants */
#define SOLID_FLOAT_CONST_LN2   CUDART_LN2_F
#define SOLID_FLOAT_CONST_LN10  CUDART_LNT_F
#define SOLID_FLOAT_CONST_INF   CUDART_INF_F

#define SOLID_FLOAT_SIN         sinf
#define SOLID_FLOAT_COS         cosf
#define SOLID_FLOAT_TAN         tanf
#define SOLID_FLOAT_SINH        sinhf
#define SOLID_FLOAT_COSH        coshf
#define SOLID_FLOAT_TANH        tanhf
#define SOLID_FLOAT_ASIN        asinf
#define SOLID_FLOAT_ACOS        acosf
#define SOLID_FLOAT_ATAN        atanf
#define SOLID_FLOAT_ASINH       asinhf
#define SOLID_FLOAT_ACOSH       acoshf
#define SOLID_FLOAT_ATANH       atanhf
#define SOLID_FLOAT_ATAN2       atan2f

#define SOLID_FLOAT_POW         powf
#define SOLID_FLOAT_SQRT        sqrtf
#define SOLID_FLOAT_CBRT        cbrtf
#define SOLID_FLOAT_HYPOT       hypotf
#define SOLID_FLOAT_EXP         expf
#define SOLID_FLOAT_EXP2        exp2f
#define SOLID_FLOAT_EXP10       exp10f
#define SOLID_FLOAT_EXPM1       expm1f
#define SOLID_FLOAT_LOG         logf
#define SOLID_FLOAT_LOG2        log2f
#define SOLID_FLOAT_LOG10       log10f
#define SOLID_FLOAT_LOG1P       log1pf

#define SOLID_FLOAT_FABS        fabsf
#define SOLID_FLOAT_CEIL        ceilf
#define SOLID_FLOAT_FLOOR       floorf
#define SOLID_FLOAT_TRUNC       truncf
#define SOLID_FLOAT_ROUND       roundf
#define SOLID_FLOAT_FMOD        fmodf

#define SOLID_FLOAT_ISFINITE    isfinite
#define SOLID_FLOAT_ISINF       isinf
#define SOLID_FLOAT_ISNAN       isnan


/* Double-precision functions */
#define SOLID_DOUBLE_CONST_LN2  CUDART_LN2
#define SOLID_DOUBLE_CONST_LN10 CUDART_LNT
#define SOLID_DOUBLE_CONST_INF  CUDART_INF

#define SOLID_DOUBLE_SIN        sin
#define SOLID_DOUBLE_COS        cos
#define SOLID_DOUBLE_TAN        tan
#define SOLID_DOUBLE_SINH       sinh
#define SOLID_DOUBLE_COSH       cosh
#define SOLID_DOUBLE_TANH       tanh
#define SOLID_DOUBLE_ASIN       asin
#define SOLID_DOUBLE_ACOS       acos
#define SOLID_DOUBLE_ATAN       atan
#define SOLID_DOUBLE_ASINH      asinh
#define SOLID_DOUBLE_ACOSH      acosh
#define SOLID_DOUBLE_ATANH      atanh
#define SOLID_DOUBLE_ATAN2      atan2

#define SOLID_DOUBLE_POW        pow
#define SOLID_DOUBLE_SQRT       sqrt
#define SOLID_DOUBLE_CBRT       cbrt
#define SOLID_DOUBLE_HYPOT      hypot
#define SOLID_DOUBLE_EXP        exp
#define SOLID_DOUBLE_EXP2       exp2
#define SOLID_DOUBLE_EXP10      exp10
#define SOLID_DOUBLE_EXPM1      expm1
#define SOLID_DOUBLE_LOG        log
#define SOLID_DOUBLE_LOG2       log2
#define SOLID_DOUBLE_LOG10      log10
#define SOLID_DOUBLE_LOG1P      log1p

#define SOLID_DOUBLE_FABS       fabs
#define SOLID_DOUBLE_CEIL       ceil
#define SOLID_DOUBLE_FLOOR      floor
#define SOLID_DOUBLE_TRUNC      trunc
#define SOLID_DOUBLE_ROUND      round
#define SOLID_DOUBLE_FMOD       fmod

#define SOLID_DOUBLE_ISFINITE   isfinite
#define SOLID_DOUBLE_ISINF      isinf
#define SOLID_DOUBLE_ISNAN      isnan


/* Half-precision operations are processed through single-precision floats */
#define SOLID_HALF_CONST_LN2    SOLID_FLOAT_CONST_LN2
#define SOLID_HALF_CONST_LN10   SOLID_FLOAT_CONST_LN10
#define SOLID_HALF_CONST_INF    SOLID_FLOAT_CONST_INF

#define SOLID_HALF_SIN          SOLID_FLOAT_SIN
#define SOLID_HALF_COS          SOLID_FLOAT_COS
#define SOLID_HALF_TAN          SOLID_FLOAT_TAN
#define SOLID_HALF_SINH         SOLID_FLOAT_SINH
#define SOLID_HALF_COSH         SOLID_FLOAT_COSH
#define SOLID_HALF_TANH         SOLID_FLOAT_TANH
#define SOLID_HALF_ASIN         SOLID_FLOAT_ASIN
#define SOLID_HALF_ACOS         SOLID_FLOAT_ACOS
#define SOLID_HALF_ATAN         SOLID_FLOAT_ATAN
#define SOLID_HALF_ASINH        SOLID_FLOAT_ASINH
#define SOLID_HALF_ACOSH        SOLID_FLOAT_ACOSH
#define SOLID_HALF_ATANH        SOLID_FLOAT_ATANH
#define SOLID_HALF_ATAN2        SOLID_FLOAT_ATAN2

#define SOLID_HALF_POW          SOLID_FLOAT_POW
#define SOLID_HALF_SQRT         SOLID_FLOAT_SQRT
#define SOLID_HALF_CBRT         SOLID_FLOAT_CBRT
#define SOLID_HALF_HYPOT        SOLID_FLOAT_HYPOT
#define SOLID_HALF_EXP          SOLID_FLOAT_EXP
#define SOLID_HALF_EXP2         SOLID_FLOAT_EXP2
#define SOLID_HALF_EXP10        SOLID_FLOAT_EXP10
#define SOLID_HALF_EXPM1        SOLID_FLOAT_EXPM1
#define SOLID_HALF_LOG          SOLID_FLOAT_LOG
#define SOLID_HALF_LOG2         SOLID_FLOAT_LOG2
#define SOLID_HALF_LOG10        SOLID_FLOAT_LOG10
#define SOLID_HALF_LOG1P        SOLID_FLOAT_LOG1P

#define SOLID_HALF_FABS         SOLID_FLOAT_FABS
#define SOLID_HALF_FLOOR        SOLID_FLOAT_FLOOR
#define SOLID_HALF_CEIL         SOLID_FLOAT_CEIL
#define SOLID_HALF_TRUNC        SOLID_FLOAT_TRUNC
#define SOLID_HALF_ROUND        SOLID_FLOAT_ROUND
#define SOLID_HALF_FMOD         SOLID_FLOAT_FMOD

#define SOLID_HALF_ISFINITE     SOLID_FLOAT_ISFINITE
#define SOLID_HALF_ISINF        SOLID_FLOAT_ISINF
#define SOLID_HALF_ISNAN        SOLID_FLOAT_ISNAN

#endif

#include "solid/core/generic/unary_ops.h"

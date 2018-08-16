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

#ifndef SDXTYPE
#error "Tensor unary operations must be included with SDXTYPE set"
#else

#ifdef SOLID_OP_BITNOT
#undef SOLID_OP_BITNOT
#endif
#ifdef SOLID_OP_NEGATIVE
#undef SOLID_OP_NEGATIVE
#endif
#ifdef SOLID_OP_CONJ
#undef SOLID_OP_CONJ
#endif
#ifdef SOLID_OP_RECIPROCAL
#undef SOLID_OP_RECIPROCAL
#endif

#ifdef SOLID_OP_SIN
#undef SOLID_OP_SIN
#endif
#ifdef SOLID_OP_COS
#undef SOLID_OP_COS
#endif
#ifdef SOLID_OP_TAN
#undef SOLID_OP_TAN
#endif
#ifdef SOLID_OP_SINH
#undef SOLID_OP_SINH
#endif
#ifdef SOLID_OP_COSH
#undef SOLID_OP_COSH
#endif
#ifdef SOLID_OP_TANH
#undef SOLID_OP_TANH
#endif
#ifdef SOLID_OP_ARCSIN
#undef SOLID_OP_ARCSIN
#endif
#ifdef SOLID_OP_ARCCOS
#undef SOLID_OP_ARCCOS
#endif
#ifdef SOLID_OP_ARCTAN
#undef SOLID_OP_ARCTAN
#endif
#ifdef SOLID_OP_ARCSINH
#undef SOLID_OP_ARCSINH
#endif
#ifdef SOLID_OP_ARCCOSH
#undef SOLID_OP_ARCCOSH
#endif
#ifdef SOLID_OP_ARCTANH
#undef SOLID_OP_ARCTANH
#endif

#ifdef SOLID_OP_SQRT
#undef SOLID_OP_SQRT
#endif
#ifdef SOLID_OP_SQRT_312
#undef SOLID_OP_SQRT_312
#endif
#ifdef SOLID_OP_CBRT
#undef SOLID_OP_CBRT
#endif
#ifdef SOLID_OP_SQUARE
#undef SOLID_OP_SQUARE
#endif
#ifdef SOLID_OP_EXP
#undef SOLID_OP_EXP
#endif
#ifdef SOLID_OP_EXP2
#undef SOLID_OP_EXP2
#endif
#ifdef SOLID_OP_EXP10
#undef SOLID_OP_EXP10
#endif
#ifdef SOLID_OP_EXPM1
#undef SOLID_OP_EXPM1
#endif
#ifdef SOLID_OP_LOG
#undef SOLID_OP_LOG
#endif
#ifdef SOLID_OP_LOG2
#undef SOLID_OP_LOG2
#endif
#ifdef SOLID_OP_LOG10
#undef SOLID_OP_LOG10
#endif
#ifdef SOLID_OP_LOG1P
#undef SOLID_OP_LOG1P
#endif


#ifdef SOLID_OP_ABSOLUTE
#undef SOLID_OP_ABSOLUTE
#endif
#ifdef SOLID_OP_FABS
#undef SOLID_OP_FABS
#endif
#ifdef SOLID_OP_SIGN
#undef SOLID_OP_SIGN
#endif
#ifdef SOLID_OP_CEIL
#undef SOLID_OP_CEIL
#endif
#ifdef SOLID_OP_FLOOR
#undef SOLID_OP_FLOOR
#endif
#ifdef SOLID_OP_TRUNC
#undef SOLID_OP_TRUNC
#endif
#ifdef SOLID_OP_ROUND
#undef SOLID_OP_ROUND
#endif
#ifdef SOLID_OP_ISFINITE
#undef SOLID_OP_ISFINITE
#endif
#ifdef SOLID_OP_ISINF
#undef SOLID_OP_ISINF
#endif
#ifdef SOLID_OP_ISNAN
#undef SOLID_OP_ISNAN
#endif
#ifdef SOLID_OP_ISPOSINF
#undef SOLID_OP_ISPOSINF
#endif
#ifdef SOLID_OP_ISNEGINF
#undef SOLID_OP_ISNEGINF
#endif

#ifdef SOLID_OP_FILLNAN
#undef SOLID_OP_FILLNAN
#endif
#ifdef SOLID_OP_COPYSIGN
#undef SOLID_OP_COPYSIGN
#endif



/* ============================================================================== */
/* UNDEFINE HELPER OPERATIONS - CPU                                               */
/* ============================================================================== */

/* ============================================================================== */
/* HELPER OPERATIONS                                                              */
/* ============================================================================== */

#include "solid/base/generic/macros.h"

#ifdef SD_FUN
#undef SD_FUN
#endif
#ifdef SD_CONST
#undef SD_CONST
#endif

#if (SDTYPE_MATCHES(SDXTYPE, half) || SDTYPE_MATCHES(SDXTYPE, chalf))
   #define SD_FUN(OP)     SOLID_CONCAT_2(SOLID_HALF_, OP)
   #define SD_CONST(NAME) SOLID_CONCAT_2(SOLID_HALF_CONST_, NAME)
#elif (SDTYPE_MATCHES(SDXTYPE, float) || SDTYPE_MATCHES(SDXTYPE, cfloat))
   #define SD_FUN(OP)     SOLID_CONCAT_2(SOLID_FLOAT_, OP)
   #define SD_CONST(NAME) SOLID_CONCAT_2(SOLID_FLOAT_CONST_, NAME)
#else
   #define SD_FUN(OP)     SOLID_CONCAT_2(SOLID_DOUBLE_, OP)
   #define SD_CONST(NAME) SOLID_CONCAT_2(SOLID_DOUBLE_CONST_, NAME)
#endif

#ifndef SD_FUN_TYPE
#define SD_FUN_TYPE_C_half(OP)     SOLID_CONCAT_2(SOLID_HALF_, OP)
#define SD_FUN_TYPE_C_float(OP)    SOLID_CONCAT_2(SOLID_FLOAT_, OP)
#define SD_FUN_TYPE_C_double(OP)   SOLID_CONCAT_2(SOLID_DOUBLE_, OP)
#define SD_FUN_TYPE_B(OP, TYPE) SD_FUN_TYPE_C_##TYPE(OP)
#define SD_FUN_TYPE(OP, TYPE) SD_FUN_TYPE_B(OP, TYPE)
#endif

#ifdef SDTYPE_C_TEMPTYPE
#undef SDTYPE_C_TEMPTYPE
#endif


/* ============================================================================== */
/* UNARY OPERATIONS - CPU                                                         */
/* NOTE: These macros cannot contain commas                                       */
/* ============================================================================== */


/* -------------------------------------------------------------------- */
#if (SDTYPE_IS_BOOL(SDXTYPE) || SDTYPE_IS_INT(SDXTYPE))
/* -------------------------------------------------------------------- */
   #define SOLID_OP_ISINF(X,Y)      { Y = 0; (void)(X); }
   #define SOLID_OP_ISNAN(X,Y)      { Y = 0; (void)(X); }
   #define SOLID_OP_ISFINITE(X,Y)   { Y = 1; (void)(X); }
   #define SOLID_OP_ISPOSINF(X,Y)   { Y = 0; (void)(X); }
   #define SOLID_OP_ISNEGINF(X,Y)   { Y = 0; (void)(X); }
#endif


/* -------------------------------------------------------------------- */
#if SDTYPE_IS_BOOL(SDXTYPE)
/* -------------------------------------------------------------------- */

   #define SOLID_OP_BITNOT(X,Y)     { Y = (X) ^ 0x01; }
   #define SOLID_OP_FABS(X,Y)       { Y = (X); }
   #define SOLID_OP_SIGN(X,Y)       { Y = (X); }


/* -------------------------------------------------------------------- */
#elif SDTYPE_IS_INT(SDXTYPE)
/* -------------------------------------------------------------------- */

   #define SOLID_OP_BITNOT(X,Y)     { Y = ~(X); }
   #define SOLID_OP_NEGATIVE(X,Y)   { Y = -1 * (X); }
   #define SOLID_OP_RECIPROCAL(X,Y) { const SOLID_C_TYPE(SDXTYPE) x_ = X; Y = (x_ == 0) ? 0 : 1 / x_; }

   #define SOLID_OP_SIN(X,Y)        { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(SIN  )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_COS(X,Y)        { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(COS  )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_TAN(X,Y)        { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(TAN  )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_SINH(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(SINH )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_COSH(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(COSH )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_TANH(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(TANH )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_ARCSIN(X,Y)     { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(ASIN )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_ARCCOS(X,Y)     { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(ACOS )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_ARCTAN(X,Y)     { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(ATAN )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_ARCSINH(X,Y)    { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(ASINH)((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_ARCCOSH(X,Y)    { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(ACOSH)((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_ARCTANH(X,Y)    { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(ATANH)((SOLID_C_TYPE(double))(X)); }

   #define SOLID_OP_SQRT_312(X,Y)   SOLID_OP_SQRT(X,Y)
   #define SOLID_OP_CBRT(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(CBRT)((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_SQUARE(X,Y)     { const SOLID_C_TYPE(SDXTYPE) x_ = X; Y = x_ * x_; }
   #define SOLID_OP_EXP(X,Y)        { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(EXP  )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_EXP2(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(EXP2 )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_EXP10(X,Y)      { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(EXP10)((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_EXPM1(X,Y)      { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(EXPM1)((SOLID_C_TYPE(double))(X)); }

#if SDTYPE_IS_SIGNED_INT(SDXTYPE)
   #define SOLID_OP_SQRT(X,Y)       { const SOLID_C_TYPE(SDXTYPE) x_ = X; \
                                      Y = (x_ <= 0) ? 0 : (SOLID_C_TYPE(SDXTYPE))SD_FUN(SQRT)((SOLID_C_TYPE(double))(x_)); }
   #define SOLID_OP_LOG(X,Y)        { const SOLID_C_TYPE(SDXTYPE) x_ = X; \
                                      Y = (x_ <= 0) ? 0 : (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG  )((SOLID_C_TYPE(double))(x_)); }
   #define SOLID_OP_LOG2(X,Y)       { const SOLID_C_TYPE(SDXTYPE) x_ = X; \
                                      Y = (x_ <= 0) ? 0 : (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG2 )((SOLID_C_TYPE(double))(x_)); }
   #define SOLID_OP_LOG10(X,Y)      { const SOLID_C_TYPE(SDXTYPE) x_ = X; \
                                      Y = (x_ <= 0) ? 0 : (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG10)((SOLID_C_TYPE(double))(x_)); }
   #define SOLID_OP_LOG1P(X,Y)      { const SOLID_C_TYPE(double) x_ = X; \
                                      Y = (x_ <= -1) ? 0 : (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG1P)((SOLID_C_TYPE(double))(x_)); }
   #define SOLID_OP_FABS(X,Y)       { const SOLID_C_TYPE(SDXTYPE) x_ = X; \
                                      Y = ((x_ >= 0) - (x_ < 0)) * x_; \
                                    }
   #define SOLID_OP_SIGN(X,Y)       { const SOLID_C_TYPE(SDXTYPE) x_ = X; \
                                      Y = ((x_ > 0) - (x_ < 0)); \
                                    }

#else
   #define SOLID_OP_SQRT(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(SQRT )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_LOG(X,Y)        { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG  )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_LOG2(X,Y)       { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG2 )((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_LOG10(X,Y)      { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG10)((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_LOG1P(X,Y)      { Y = (SOLID_C_TYPE(SDXTYPE))SD_FUN(LOG1P)((SOLID_C_TYPE(double))(X)); }
   #define SOLID_OP_FABS(X,Y)       { Y = X; }
   #define SOLID_OP_SIGN(X,Y)       { Y = ((X) > 0); }
#endif


/* -------------------------------------------------------------------- */
#elif SDTYPE_IS_REAL(SDXTYPE)
/* -------------------------------------------------------------------- */

   #define SOLID_OP_NEGATIVE(X,Y)   { Y = SOLID_FROM_WORKTYPE(-1 * SOLID_TO_WORKTYPE(X)); }
   #define SOLID_OP_RECIPROCAL(X,Y) { Y = SOLID_FROM_WORKTYPE(1. / SOLID_TO_WORKTYPE(X)); }
   
   #define SOLID_OP_SIN(X,Y)        { Y = SOLID_FROM_WORKTYPE(SD_FUN(SIN)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_COS(X,Y)        { Y = SOLID_FROM_WORKTYPE(SD_FUN(COS)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_TAN(X,Y)        { Y = SOLID_FROM_WORKTYPE(SD_FUN(TAN)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_SINH(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(SINH)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_COSH(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(COSH)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_TANH(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(TANH)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ARCSIN(X,Y)     { Y = SOLID_FROM_WORKTYPE(SD_FUN(ASIN)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ARCCOS(X,Y)     { Y = SOLID_FROM_WORKTYPE(SD_FUN(ACOS)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ARCTAN(X,Y)     { Y = SOLID_FROM_WORKTYPE(SD_FUN(ATAN)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ARCSINH(X,Y)    { Y = SOLID_FROM_WORKTYPE(SD_FUN(ASINH)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ARCCOSH(X,Y)    { Y = SOLID_FROM_WORKTYPE(SD_FUN(ACOSH)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ARCTANH(X,Y)    { Y = SOLID_FROM_WORKTYPE(SD_FUN(ATANH)(SOLID_TO_WORKTYPE(X))); }
   
   #define SOLID_OP_SQRT(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(SQRT)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_SQRT_312(X,Y)   SOLID_OP_SQRT(X,Y)
   #define SOLID_OP_CBRT(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(CBRT)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_SQUARE(X,Y)     { SOLID_C_WORKTYPE x_ = SOLID_TO_WORKTYPE(X); \
                                      Y = SOLID_FROM_WORKTYPE(x_ * x_); }
   #define SOLID_OP_EXP(X,Y)        { Y = SOLID_FROM_WORKTYPE(SD_FUN(EXP)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_EXP2(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(EXP2)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_EXP10(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(EXP10)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_EXPM1(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(EXPM1)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_LOG(X,Y)        { Y = SOLID_FROM_WORKTYPE(SD_FUN(LOG)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_LOG2(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(LOG2)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_LOG10(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(LOG10)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_LOG1P(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(LOG1P)(SOLID_TO_WORKTYPE(X))); }
   
   #define SOLID_OP_FABS(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(FABS)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_SIGN(X,Y)       { SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                      Y = SD_FUN(ISNAN)(_x) ? X : SOLID_FROM_WORKTYPE((_x > 0) - (_x < 0)); \
                                    }

   #define SOLID_OP_CEIL(X,Y)       { Y = SOLID_FROM_WORKTYPE(SD_FUN(CEIL)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_FLOOR(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(FLOOR)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_TRUNC(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(TRUNC)(SOLID_TO_WORKTYPE(X))); }
   #define SOLID_OP_ROUND(X,Y)      { Y = SOLID_FROM_WORKTYPE(SD_FUN(ROUND)(SOLID_TO_WORKTYPE(X))); }

   #define SOLID_OP_ISINF(X,Y)      { Y = (SD_FUN(ISINF)(SOLID_TO_WORKTYPE(X))) ? 1 : 0; }
   #define SOLID_OP_ISNAN(X,Y)      { Y = (SD_FUN(ISNAN)(SOLID_TO_WORKTYPE(X))) ? 1 : 0; }
   #define SOLID_OP_ISFINITE(X,Y)   { Y = (SD_FUN(ISFINITE)(SOLID_TO_WORKTYPE(X))) ? 1 : 0; }
   #define SOLID_OP_ISPOSINF(X,Y)   { SOLID_C_WORKTYPE x_ = SOLID_TO_WORKTYPE(X); \
                                      Y = ((x_ > 0) && (SD_FUN(ISINF)(x_))) ? 1 : 0; }
   #define SOLID_OP_ISNEGINF(X,Y)   { SOLID_C_WORKTYPE x_ = SOLID_TO_WORKTYPE(X); \
                                      Y = ((x_ < 0) && (SD_FUN(ISINF)(x_))) ? 1 : 0; }

   #define SOLID_OP_FILLNAN(X,Y)    { SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                      if (SD_FUN(ISNAN)(_x)) X = Y; \
                                    }

/* -------------------------------------------------------------------- */
#elif SDTYPE_IS_COMPLEX(SDXTYPE)
/* -------------------------------------------------------------------- */

   #if SDTYPE_MATCHES(SDXTYPE,half)
   #define SOLID_C_TEMPTYPE  SOLID_C_TYPE(float)
   #else
   #define SOLID_C_TEMPTYPE  SOLID_C_ELEMWORKTYPE
   #endif

   #define SOLID_OP_NEGATIVE(X,Y)   { (Y).real = SOLID_FROM_ELEMWORKTYPE(-1 * SOLID_TO_ELEMWORKTYPE((X).real)); \
                                      (Y).imag = SOLID_FROM_ELEMWORKTYPE(-1 * SOLID_TO_ELEMWORKTYPE((X).imag)); }

   #define SOLID_OP_CONJ(X,Y)       { (Y).real = (X).real; \
                                      (Y).imag = SOLID_FROM_ELEMWORKTYPE(-1 * SOLID_TO_ELEMWORKTYPE((X).imag)); }

   #define SOLID_OP_RECIPROCAL(X,Y) {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = 1. / ((xreal*xreal) + (ximag*ximag)); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(xreal * scale); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(ximag * scale * -1); \
                                    }

   #define SOLID_OP_SIN(X,Y)        {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SIN)(xreal) * SD_FUN(COSH)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(COS)(xreal) * SD_FUN(SINH)(ximag)); \
                                    }

   #define SOLID_OP_COS(X,Y)        {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(COS)(xreal) * SD_FUN(COSH)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SIN)(xreal) * SD_FUN(SINH)(ximag) * -1); \
                                    }

   #define SOLID_OP_TAN(X,Y)        {  /*             sin(2*a) + i sinh(2*b) */ \
                                       /* tan(a+ib) = ---------------------- */ \
                                       /*              cos(2*a) + cosh(2*b)  */ \
                                       SOLID_C_ELEMWORKTYPE xreal  = 2 * SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag  = 2 * SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE r; \
                                       r = (SD_FUN(COS)(xreal) + SD_FUN(COSH)(ximag)); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SIN)(xreal) / r); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SINH)(ximag) / r); \
                                     }                                    

   #define SOLID_OP_SINH(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SINH)(xreal) * SD_FUN(COS)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(COSH)(xreal) * SD_FUN(SIN)(ximag)); \
                                    }

   #define SOLID_OP_COSH(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(COSH)(xreal) * SD_FUN(COS)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SINH)(xreal) * SD_FUN(SIN)(ximag)); \
                                    }

   #define SOLID_OP_TANH(X,Y)       {  /*              sinh(2*a) + i sin(2*b) */ \
                                       /* tanh(a+ib) = ---------------------- */ \
                                       /*               cosh(2*a) + cos(2*b)  */ \
                                       SOLID_C_ELEMWORKTYPE xreal  = 2 * SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag  = 2 * SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE r; \
                                       r = 1. / (SD_FUN(COSH)(xreal) + SD_FUN(COS)(ximag)); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SINH)(xreal) * r); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(SIN)(ximag) * r); \
                                    }

   #define SOLID_OP_ARCSIN(X,Y)     {  /* arcsin(z) = 1/i * log(iz + sqrt(|1 - z^2|) * exp(i/2 * arg(1-z^2))) */ \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v1    = 1 - (xreal * xreal) + (ximag * ximag); \
                                       SOLID_C_TEMPTYPE v2    = -2 * xreal * ximag; \
                                       SOLID_C_TEMPTYPE gamma = SD_FUN(SQRT)(SD_FUN(HYPOT)(v1,v2)); \
                                       SOLID_C_TEMPTYPE theta = SD_FUN(ATAN2)(v2, v1) / 2; \
                                       v1 = -ximag + gamma * SD_FUN(COS)(theta); \
                                       v2 =  xreal + gamma * SD_FUN(SIN)(theta); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(v2,v1)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(v1*v1 + v2*v2) / -2); \
                                    }

   #define SOLID_OP_ARCCOS(X,Y)     {  /* arccos(z) = 1/i * log(z + i*sqrt(|1 - z^2|) * exp(i/2 * arg(1-z^2))) */ \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v1    = 1 - (xreal * xreal) + (ximag * ximag); \
                                       SOLID_C_TEMPTYPE v2    = -2 * xreal * ximag; \
                                       SOLID_C_TEMPTYPE gamma = SD_FUN(SQRT)(SD_FUN(HYPOT)(v1,v2)); \
                                       SOLID_C_TEMPTYPE theta = SD_FUN(ATAN2)(v2, v1) / 2; \
                                       v1 = xreal - gamma * SD_FUN(SIN)(theta); \
                                       v2 = ximag + gamma * SD_FUN(COS)(theta); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(v2,v1)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(v1*v1 + v2*v2) / -2); \
                                    }

   #define SOLID_OP_ARCTAN(X,Y)     {  /* arctan(z) = 1/(2i) * log((i - z) / (i + z)) */ \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v1    = (xreal * xreal); \
                                       SOLID_C_TEMPTYPE v2    = (ximag * ximag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(2 * xreal, 1 - v1 - v2) / 2); \
                                       v2 = 1 + ximag; \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(1 - 4*ximag / (v2*v2 + v1)) / -4); \
                                    }

   #define SOLID_OP_ARCSINH(X,Y)    {  /* arcsinh(z) = log(z + sqrt(|1 + z^2|) * exp(i/2 * arg(1+z^2))) */ \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v1    = 1 + (xreal * xreal) - (ximag * ximag); \
                                       SOLID_C_TEMPTYPE v2    = 2 * xreal * ximag; \
                                       SOLID_C_TEMPTYPE gamma = SD_FUN(SQRT)(SD_FUN(HYPOT)(v1,v2)); \
                                       SOLID_C_TEMPTYPE theta = SD_FUN(ATAN2)(v2, v1) / 2; \
                                       v1 = xreal + gamma * SD_FUN(COS)(theta); \
                                       v2 = ximag + gamma * SD_FUN(SIN)(theta); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(v1*v1 + v2*v2) / 2); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(v2,v1)); \
                                    }

   #define SOLID_OP_ARCCOSH(X,Y)    {  /* arccosh(z) = log(z + sqrt(|z^2 - 1|) * exp(i/2 * arg(z^2-1))) */ \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v1    = (xreal * xreal) - (ximag * ximag) - 1; \
                                       SOLID_C_TEMPTYPE v2    = 2 * xreal * ximag; \
                                       SOLID_C_TEMPTYPE gamma = SD_FUN(SQRT)(SD_FUN(HYPOT)(v1,v2)); \
                                       SOLID_C_TEMPTYPE theta = SD_FUN(ATAN2)(v2, v1) / 2; \
                                       v1 = xreal + gamma * SD_FUN(COS)(theta); \
                                       v2 = ximag + gamma * SD_FUN(SIN)(theta); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(v1*v1 + v2*v2) / 2); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(v2,v1)); \
                                    }

   #define SOLID_OP_ARCTANH(X,Y)    {  /* arctanh(z) = 1/2 * log((1 + z) / (1 - z)) */ \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v1    = (xreal * xreal); \
                                       SOLID_C_TEMPTYPE v2    = (ximag * ximag); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(2 * ximag, 1 - v1 - v2) / 2); \
                                       v1 = 1 - xreal; \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(1 + 4*xreal / (v1*v1 + v2)) / 4); \
                                    }

   #define SOLID_OP_SQRT(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = ((xreal * xreal) + (ximag * ximag)); \
                                       SOLID_C_TEMPTYPE angle; \
                                       if (scale > 0) \
                                       {  scale = SD_FUN(POW)(scale,0.25); \
                                          angle = SD_FUN(ATAN2)(ximag, xreal) / 2; \
                                          (Y).real = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(COS)(angle)); \
                                          (Y).imag = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(SIN)(angle)); \
                                       } \
                                       else \
                                       {  (Y).real = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                          (Y).imag = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                       } \
                                    }

   #define SOLID_OP_SQRT_312(X,Y)   {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE v = SD_FUN(HYPOT(xreal,ximag)); \
                                       \
                                       /* Implementation of: Paul Friedland, "Algorithm 312: Absolute */ \
                                       /* value and square root of a complex number", Communications  */ \
                                       /* of the ACM, 10(10), p. 665, 1967.                           */ \
                                       \
                                       if (!(v > 0)) \
                                       {  /* Both zero or at least one NaN */ \
                                          (Y).real = SOLID_FROM_ELEMWORKTYPE(v); \
                                          (Y).imag = (Y).real; \
                                       } \
                                       else if (SD_FUN(ISINF)(ximag)) \
                                       {  (Y).real = SD_CONST(INF); \
                                          (Y).imag = (X).imag; \
                                       } \
                                       else if (SD_FUN(ISINF)(xreal)) \
                                       {  (Y).real = SOLID_FROM_ELEMWORKTYPE((xreal > 0) ? xreal : 0); \
                                          (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(COPYSIGN)((xreal < 0) ? xreal : 0, ximag)); \
                                       } \
                                       else if (xreal >= 0) \
                                       {  v = SD_FUN(SQRT)((xreal + v) * 0.5); \
                                          (Y).real = SOLID_FROM_ELEMWORKTYPE(v); \
                                          (Y).imag = SOLID_FROM_ELEMWORKTYPE(ximag / (2 * v)); \
                                       } \
                                       else \
                                       {  v = SD_FUN(SQRT)((-xreal + v) * 0.5); \
                                          (Y).real = SOLID_FROM_ELEMWORKTYPE(((ximag >= 0) ? ximag : -ximag) / (2 * v)); \
                                          (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(COPYSIGN)(v, ximag)); \
                                       } \
                                    }

   #define SOLID_OP_CBRT(X,Y)       {  SOLID_C_TEMPTYPE scale; \
                                       SOLID_C_TEMPTYPE angle; \
                                       SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       scale = ((xreal * xreal) + (ximag * ximag)); \
                                       if (scale > 0) \
                                       {  scale = SD_FUN(POW)(scale, 1./6); \
                                          angle = SD_FUN(ATAN2)(ximag, xreal) / 3; \
                                          (Y).real = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(COS)(angle)); \
                                          (Y).imag = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(SIN)(angle)); \
                                       } \
                                       else \
                                       {  (Y).real = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                          (Y).imag = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                       } \
                                    }

   #define SOLID_OP_SQUARE(X,Y)     {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(xreal*xreal - ximag*ximag); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(2 * xreal * ximag); \
                                    }

   #define SOLID_OP_EXP(X,Y)        {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = SD_FUN(EXP)(xreal); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(COS)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(SIN)(ximag)); \
                                    }

   #define SOLID_OP_EXP2(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag) * SD_CONST(LN2); \
                                       SOLID_C_TEMPTYPE scale = SD_FUN(EXP2)(xreal); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(COS)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(SIN)(ximag)); \
                                    }

   #define SOLID_OP_EXP10(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag) * SD_CONST(LN10); \
                                       SOLID_C_TEMPTYPE scale = SD_FUN(EXP10)(xreal); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(COS)(ximag)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(SIN)(ximag)); \
                                    }

   #define SOLID_OP_EXPM1(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = SD_FUN(EXP)(xreal); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(COS)(ximag) - 1); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(scale * SD_FUN(SIN)(ximag)); \
                                    }

   #define SOLID_OP_LOG(X,Y)        {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = (xreal*xreal + ximag*ximag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(scale) * 0.5); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(ximag,xreal)); \
                                    }

   #define SOLID_OP_LOG2(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = (xreal*xreal + ximag*ximag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG2)(scale) * 0.5); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(ximag,xreal) / SD_CONST(LN2)); \
                                    }

   #define SOLID_OP_LOG10(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = (xreal*xreal + ximag*ximag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG10)(scale) * 0.5); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(ximag,xreal) / SD_CONST(LN10)); \
                                    }

   #define SOLID_OP_LOG1P(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real) + 1; \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       SOLID_C_TEMPTYPE scale = (xreal*xreal + ximag*ximag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(LOG)(scale) * 0.5); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ATAN2)(ximag,xreal)); \
                                    }

   #define SOLID_OP_ABSOLUTE(X,Y)   {  SOLID_C_ELEMWORKTYPE xreal  = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag  = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y) = SOLID_FROM_ELEMWORKTYPE(SD_FUN(HYPOT)(xreal,ximag)); \
                                    }

   #define SOLID_OP_FABS(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal  = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag  = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(HYPOT)(xreal,ximag)); \
                                       (Y).imag = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                    }

   #define SOLID_OP_SIGN(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal  = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag  = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       if (SD_FUN(ISNAN)(xreal) || SD_FUN(ISNAN)(ximag)) \
                                       {  (Y).real = SOLID_FROM_ELEMWORKTYPE(xreal + ximag); \
                                       } \
                                       else \
                                       {  xreal = (((xreal > 0) - (xreal < 0)) + (xreal == 0) * ((ximag > 0) - (ximag < 0))); \
                                          (Y).real = SOLID_FROM_ELEMWORKTYPE(xreal); \
                                       } \
                                       (Y).imag = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                    }

   #define SOLID_OP_CEIL(X,Y)       {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(CEIL)(xreal)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(CEIL)(ximag)); \
                                     }

   #define SOLID_OP_FLOOR(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(FLOOR)(xreal)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(FLOOR)(ximag)); \
                                    }

   #define SOLID_OP_TRUNC(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(TRUNC)(xreal)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(TRUNC)(ximag)); \
                                    }

   #define SOLID_OP_ROUND(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       (Y).real = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ROUND)(xreal)); \
                                       (Y).imag = SOLID_FROM_ELEMWORKTYPE(SD_FUN(ROUND)(ximag)); \
                                    }


   #define SOLID_OP_ISINF(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       Y = ((SD_FUN(ISINF)(xreal)) || (SD_FUN(ISINF)(ximag))) ? 1 : 0; \
                                    }

   #define SOLID_OP_ISNAN(X,Y)      {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       Y = ((SD_FUN(ISNAN)(xreal)) || (SD_FUN(ISINF)(ximag))) ? 1 : 0; \
                                    }

   #define SOLID_OP_ISFINITE(X,Y)   {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       Y = ((SD_FUN(ISFINITE)(xreal)) && (SD_FUN(ISFINITE)(ximag))) ? 1 : 0; \
                                    }


   #define SOLID_OP_FILLNAN(X,Y)    {  SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                       SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                       if ((SD_FUN(ISNAN)(xreal)) || (SD_FUN(ISINF)(ximag))) X = Y; \
                                    }

                                 
/* -------------------------------------------------------------------- */
#else /* Unsupported type */
/* -------------------------------------------------------------------- */

#error "Unsupported SDXTYPE encountered"

#endif

#endif

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
#error "Tensor binary operations must be included with SDXTYPE set"
#else

#include "solid/core/generic/unary_ops.h"

#ifdef SOLID_OP_ADD
#undef SOLID_OP_ADD
#endif
#ifdef SOLID_OP_SUBTRACT
#undef SOLID_OP_SUBTRACT
#endif
#ifdef SOLID_OP_MULTIPLY
#undef SOLID_OP_MULTIPLY
#endif
#ifdef SOLID_OP_DIVIDE
#undef SOLID_OP_DIVIDE
#endif
#ifdef SOLID_OP_TRUE_DIVIDE
#undef SOLID_OP_TRUE_DIVIDE
#endif
#ifdef SOLID_OP_FLOOR_DIVIDE
#undef SOLID_OP_FLOOR_DIVIDE
#endif
#ifdef SOLID_OP_MOD
#undef SOLID_OP_MOD
#endif
#ifdef SOLID_OP_FMOD
#undef SOLID_OP_FMOD
#endif
#ifdef SOLID_OP_POWER
#undef SOLID_OP_POWER
#endif

#ifdef SOLID_OP_BITXOR
#undef SOLID_OP_BITXOR
#endif
#ifdef SOLID_OP_BITAND
#undef SOLID_OP_BITAND
#endif
#ifdef SOLID_OP_BITOR
#undef SOLID_OP_BITOR
#endif
#ifdef SOLID_OP_BITSHIFT_LEFT
#undef SOLID_OP_BITSHIFT_LEFT
#endif
#ifdef SOLID_OP_BITSHIFT_RIGHT
#undef SOLID_OP_BITSHIFT_RIGHT
#endif

#ifdef SOLID_OP_LT
#undef SOLID_OP_LT
#endif
#ifdef SOLID_OP_LE
#undef SOLID_OP_LE
#endif
#ifdef SOLID_OP_EQ
#undef SOLID_OP_EQ
#endif
#ifdef SOLID_OP_NE
#undef SOLID_OP_NE
#endif
#ifdef SOLID_OP_GE
#undef SOLID_OP_GE
#endif
#ifdef SOLID_OP_GT
#undef SOLID_OP_GT
#endif

#ifdef SOLID_OP_MIN
#undef SOLID_OP_MIN
#endif
#ifdef SOLID_OP_MAX
#undef SOLID_OP_MAX
#endif
#ifdef SOLID_OP_FMIN
#undef SOLID_OP_FMIN
#endif
#ifdef SOLID_OP_FMAX
#undef SOLID_OP_FMAX
#endif


/* ============================================================================== */
/* UNDEFINE HELPER OPERATIONS - CPU                                               */
/* ============================================================================== */


/* ============================================================================== */
/* BINARY OPERATIONS - CPU                                                        */
/* NOTE: These macros cannot contain commas                                       */
/* ============================================================================== */


/* -------------------------------------------------------------------- */
#if SDTYPE_IS_BOOL(SDXTYPE)
/* -------------------------------------------------------------------- */

   #define SOLID_OP_ADD(X,Y,Z)            { Z = (X) | (Y); }
   #define SOLID_OP_SUBTRACT(X,Y,Z)       { Z = (X) & ~(Y); }
   #define SOLID_OP_MULTIPLY(X,Y,Z)       { Z = (X) & (Y); }
   #define SOLID_OP_TRUE_DIVIDE(X,Y,Z)    { Z = (X) & (Y); } /* x / False = False */
   #define SOLID_OP_DIVIDE(X,Y,Z)         SOLID_OP_TRUE_DIVIDE(X,Y,Z)
   #define SOLID_OP_FLOOR_DIVIDE(X,Y,Z)   SOLID_OP_TRUE_DIVIDE(X,Y,Z)
   #define SOLID_OP_MOD(X,Y,Z)            { Z = 0; (void)(X); (void)(Y); }
   #define SOLID_OP_FMOD(X,Y,Z)           { Z = 0; (void)(X); (void)(Y); }
   #define SOLID_OP_POWER(X,Y,Z)          { Z = (((X) != 0) | ((Y) == 0)); }

   #define SOLID_OP_BITXOR(X,Y,Z)         { Z = (X) ^ (Y); }
   #define SOLID_OP_BITAND(X,Y,Z)         { Z = (X) & (Y); }
   #define SOLID_OP_BITOR(X,Y,Z)          { Z = (X) | (Y); }
   #define SOLID_OP_BITSHIFT_LEFT(X,Y,Z)  { Z = ((Y) == 0) ? ((X) & 0x01) : 0; }
   #define SOLID_OP_BITSHIFT_RIGHT(X,Y,Z) { Z = ((Y) == 0) ? ((X) & 0x01) : 0; }

   #define SOLID_OP_LT(X,Y,Z)             { Z = ((X) <  (Y)) ? 1 : 0; }
   #define SOLID_OP_LE(X,Y,Z)             { Z = ((X) <= (Y)) ? 1 : 0; }
   #define SOLID_OP_EQ(X,Y,Z)             { Z = ((X) == (Y)) ? 1 : 0; }
   #define SOLID_OP_NE(X,Y,Z)             { Z = ((X) != (Y)) ? 1 : 0; }
   #define SOLID_OP_GE(X,Y,Z)             { Z = ((X) >= (Y)) ? 1 : 0; }
   #define SOLID_OP_GT(X,Y,Z)             { Z = ((X) >  (Y)) ? 1 : 0; }


/* -------------------------------------------------------------------- */
#elif SDTYPE_IS_INT(SDXTYPE)
/* -------------------------------------------------------------------- */

   #define SOLID_OP_ADD(X,Y,Z)           { Z = (X) + (Y); }
   #define SOLID_OP_SUBTRACT(X,Y,Z)      { Z = (X) - (Y); }
   #define SOLID_OP_MULTIPLY(X,Y,Z)      { Z = (X) * (Y); }

   #define SOLID_OP_TRUE_DIVIDE(X,Y,Z)   { SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                           Z = (_y == 0) ? 0 : ((X) / _y); \
                                         }

   #if SDTYPE_IS_UNSIGNED_INT(SDXTYPE)
   #define SOLID_OP_DIVIDE(X,Y,Z)        SOLID_OP_TRUE_DIVIDE(X,Y,Z)
   #else
   #define SOLID_OP_DIVIDE(X,Y,Z)        { SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                           SOLID_C_TYPE(SDXTYPE) _x; \
                                           if (_y == 0) \
                                           {  Z = 0; \
                                           } \
                                           else \
                                           {  _x = (X); \
                                              /* We want to implement the following division operation: */ \
                                              /*                                                        */ \
                                              /*    x /  y    desired    C                              */ \
                                              /*   --------  ---------  ---                             */ \
                                              /*    4 /  3       1       1                              */ \
                                              /*    4 / -3      -2      -1                              */ \
                                              /*   -4 /  3      -2      -1                              */ \
                                              /*   -4 / -3       1       1                              */ \
                                              /*                                                        */ \
                                              /* We can update x such that standard division by C works.*/ \
                                              /* Note that we do not need any correction whenever the   */ \
                                              /* signs of x and y match. We therefore use (_x*_y < 0)   */ \
                                              /* as a multiplicative factor, which is 1 when the signs  */ \
                                              /* of x and y differ and 0 otherwise. In case of a sign   */ \
                                              /* difference we need to subtract y and then compensate   */ \
                                              /* by an addition -1 when y < 0 and +1 when y > 0. Once x */ \
                                              /* is corrected we can divide by y.                       */ \
                                              Z = (_x - (_x*_y < 0) * (_y + ((_y < 0) - (_y > 0)))) / _y; \
                                           } \
                                         }
   #endif

   #if SDTYPE_IS_UNSIGNED_INT(SDXTYPE)
   #define SOLID_OP_FLOOR_DIVIDE(X,Y,Z)  SOLID_OP_TRUE_DIVIDE(X,Y,Z)
   #else
   #define SOLID_OP_FLOOR_DIVIDE(X,Y,Z)  { SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                           SOLID_C_TYPE(SDXTYPE) _x; \
                                           if (_y == 0) \
                                           {  Z = 0; \
                                           } \
                                           else \
                                           {  _x = (X); \
                                              /* In the following code we try to evaluate floor((float)x / y)  */ \
                                              /* without conversion to float, and without the need to branch.  */ \
                                              /* When _x and _y have the same sign it suffices to take regular */ \
                                              /* division. Otherwise we need to subtract (_y + 1) when _y < 0  */ \
                                              /* and (_y - 1) when _y > 0. We determine +1 or -1 factor by     */ \
                                              /* computing (_y < 0) - (_y > 0). The conditional on _x * _y < 0 */ \
                                              /* is included by multiplication by ((_x * _y) < 0).             */ \
                                              _x -= ((_x*_y) < 0) * (_y + ((_y < 0) - (_y > 0))); \
                                              Z = _x / _y; \
                                           } \
                                         }
   #endif

   #define SOLID_OP_FMOD(X,Y,Z)     { SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                      Z = (_y == 0) ? 0 : (X) % _y; \
                                    }

   #if SDTYPE_IS_UNSIGNED_INT(SDXTYPE)
   #define SOLID_OP_MOD(X,Y,Z)      SOLID_OP_FMOD(X,Y,Z)
   #else
   #define SOLID_OP_MOD(X,Y,Z)      { SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                      SOLID_C_TYPE(SDXTYPE) _x; \
                                      SOLID_C_TYPE(SDXTYPE) _z; \
                                      if (_y == 0) \
                                      {  Z = 0; \
                                      } \
                                      else \
                                      {  _x = (X); \
                                         /* We want to have a modulo operation that gives the following: */ \
                                         /*                                                              */ \
                                         /*   x mod y      Desired result     C modulo                   */ \
                                         /*  ----------   ----------------   ----------                  */ \
                                         /*   3 mod  4           3                3                      */ \
                                         /*   3 mod -4          -1                3                      */ \
                                         /*  -3 mod  4           1               -3                      */ \
                                         /*  -3 mod -4          -3               -3                      */ \
                                         /*                                                              */ \
                                         /* When x and y have the same sign or when the result is zero   */ \
                                         /* the C modulo, otherwise we add y multiplied by the sign of x.*/ \
                                         _z = _x % _y; \
                                         Z = _z + ((_x*_y < 0) && (_z != 0)) * (_y); \
                                      } \
                                    }
   #endif

   #define SOLID_OP_POWER(X,Y,Z)          { SOLID_C_TYPE(SDXTYPE) _x = (X); \
                                            SOLID_C_TYPE(SDXTYPE) _z; \
                                            int64_t _exp = (Y); \
                                            _z = (_exp >= 0); \
                                            while (_exp > 0) \
                                            {  if (_exp & 0x01) _z *= _x; \
                                               _exp >>= 1; \
                                               _x *= _x; \
                                            } \
                                            (Z) = _z; \
                                          }
                                    
   #define SOLID_OP_BITXOR(X,Y,Z)         { Z = (X) ^ (Y); }
   #define SOLID_OP_BITAND(X,Y,Z)         { Z = (X) & (Y); }
   #define SOLID_OP_BITOR(X,Y,Z)          { Z = (X) | (Y); }
   #define SOLID_OP_BITSHIFT_LEFT(X,Y,Z)  { int8_t _shift = (Y); Z = (_shift < 0) ? ((X) >> -_shift) : ((X) << _shift); }
   #define SOLID_OP_BITSHIFT_RIGHT(X,Y,Z) { int8_t _shift = (Y); Z = (_shift < 0) ? ((X) << -_shift) : ((X) >> _shift); }

   #define SOLID_OP_LT(X,Y,Z)       { Z = ((X) <  (Y)) ? 1 : 0; }
   #define SOLID_OP_LE(X,Y,Z)       { Z = ((X) <= (Y)) ? 1 : 0; }
   #define SOLID_OP_EQ(X,Y,Z)       { Z = ((X) == (Y)) ? 1 : 0; }
   #define SOLID_OP_NE(X,Y,Z)       { Z = ((X) != (Y)) ? 1 : 0; }
   #define SOLID_OP_GE(X,Y,Z)       { Z = ((X) >= (Y)) ? 1 : 0; }
   #define SOLID_OP_GT(X,Y,Z)       { Z = ((X) >  (Y)) ? 1 : 0; }

   #define SOLID_OP_MIN(X,Y,Z)      { SOLID_C_TYPE(SDXTYPE) _x = (X); \
                                      SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                      Z = (_x <= _y) ? _x : _y; \
                                    }

   #define SOLID_OP_MAX(X,Y,Z)      { SOLID_C_TYPE(SDXTYPE) _x = (X); \
                                      SOLID_C_TYPE(SDXTYPE) _y = (Y); \
                                      Z = (_x >= _y) ? _x : _y; \
                                    }

   #define SOLID_OP_FMIN(X,Y,Z)    SOLID_OP_MIN(X,Y,Z)
   #define SOLID_OP_FMAX(X,Y,Z)    SOLID_OP_MAX(X,Y,Z)


/* -------------------------------------------------------------------- */
#elif SDTYPE_IS_REAL(SDXTYPE)
/* -------------------------------------------------------------------- */

   #define SOLID_OP_ADD(X,Y,Z)           { Z = SOLID_FROM_WORKTYPE(SOLID_TO_WORKTYPE(X) + SOLID_TO_WORKTYPE(Y)); }
   #define SOLID_OP_SUBTRACT(X,Y,Z)      { Z = SOLID_FROM_WORKTYPE(SOLID_TO_WORKTYPE(X) - SOLID_TO_WORKTYPE(Y)); }
   #define SOLID_OP_MULTIPLY(X,Y,Z)      { Z = SOLID_FROM_WORKTYPE(SOLID_TO_WORKTYPE(X) * SOLID_TO_WORKTYPE(Y)); }

   #define SOLID_OP_TRUE_DIVIDE(X,Y,Z)   { Z = SOLID_FROM_WORKTYPE(SOLID_TO_WORKTYPE(X) / SOLID_TO_WORKTYPE(Y)); }
   #define SOLID_OP_DIVIDE(X,Y,Z)        SOLID_OP_TRUE_DIVIDE(X,Y,Z)
   #define SOLID_OP_FLOOR_DIVIDE(X,Y,Z)  { SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                           SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                           Z = SOLID_FROM_WORKTYPE(SD_FUN(FLOOR)(_x / _y)); \
                                         }

   #define SOLID_OP_MOD(X,Y,Z)           { SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                           SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                           SOLID_C_WORKTYPE _z = SOLID_TO_WORKTYPE(Y); \
                                           _z = SD_FUN(FLOOR)(_x / _y); \
                                           Z = SOLID_FROM_WORKTYPE((_x - _z * _y)); \
                                         }

   #define SOLID_OP_FMOD(X,Y,Z)          { SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                           SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                           Z = SOLID_FROM_WORKTYPE(SD_FUN(FMOD)(_x,_y)); \
                                         }

   #define SOLID_OP_POWER(X,Y,Z)         { SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                           SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                           Z = SOLID_FROM_WORKTYPE(SD_FUN(POW)(_x,_y)); \
                                         }

   #define SOLID_OP_LT(X,Y,Z)       { Z = (SOLID_TO_WORKTYPE(X) <  SOLID_TO_WORKTYPE(Y)) ? 1 : 0; }
   #define SOLID_OP_LE(X,Y,Z)       { Z = (SOLID_TO_WORKTYPE(X) <= SOLID_TO_WORKTYPE(Y)) ? 1 : 0; }
   #define SOLID_OP_EQ(X,Y,Z)       { Z = (SOLID_TO_WORKTYPE(X) == SOLID_TO_WORKTYPE(Y)) ? 1 : 0; }
   #define SOLID_OP_NE(X,Y,Z)       { Z = (SOLID_TO_WORKTYPE(X) != SOLID_TO_WORKTYPE(Y)) ? 1 : 0; }
   #define SOLID_OP_GE(X,Y,Z)       { Z = (SOLID_TO_WORKTYPE(X) >= SOLID_TO_WORKTYPE(Y)) ? 1 : 0; }
   #define SOLID_OP_GT(X,Y,Z)       { Z = (SOLID_TO_WORKTYPE(X) >  SOLID_TO_WORKTYPE(Y)) ? 1 : 0; }

   #define SOLID_OP_MIN(X,Y,Z)      { /* Propagate NaN */ \
                                      SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                      SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                      if ((SD_FUN(ISNAN)(_y)) || (_y < _x)) _x = _y; \
                                      Z = SOLID_FROM_WORKTYPE(_x); \
                                    }

   #define SOLID_OP_MAX(X,Y,Z)      { /* Propagate NaN */ \
                                      SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                      SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                      if ((SD_FUN(ISNAN)(_y)) || (_y > _x)) _x = _y; \
                                      Z = SOLID_FROM_WORKTYPE(_x); \
                                    }

   #define SOLID_OP_FMIN(X,Y,Z)     { /* Do not propagate NaN */ \
                                      SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                      SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                      if ((SD_FUN(ISNAN)(_x)) || (_y <= _x)) _x = _y; \
                                      Z = SOLID_FROM_WORKTYPE(_x); \
                                    }

   #define SOLID_OP_FMAX(X,Y,Z)     { /* Do not propagate NaN */ \
                                      SOLID_C_WORKTYPE _x = SOLID_TO_WORKTYPE(X); \
                                      SOLID_C_WORKTYPE _y = SOLID_TO_WORKTYPE(Y); \
                                      if ((SD_FUN(ISNAN)(_x)) || (_y >= _x)) _x = _y; \
                                      Z = SOLID_FROM_WORKTYPE(_x); \
                                    }


/* -------------------------------------------------------------------- */
#elif SDTYPE_IS_COMPLEX(SDXTYPE)
/* -------------------------------------------------------------------- */

   #if SDTYPE_MATCHES(SDXTYPE,half)
   #define SOLID_C_TEMPTYPE  SOLID_C_TYPE(float)
   #else
   #define SOLID_C_TEMPTYPE  SOLID_C_ELEMWORKTYPE
   #endif

   #define SOLID_OP_ADD(X,Y,Z)      { (Z).real = SOLID_FROM_ELEMWORKTYPE(SOLID_TO_ELEMWORKTYPE((X).real) + SOLID_TO_ELEMWORKTYPE((Y).real)); \
                                      (Z).imag = SOLID_FROM_ELEMWORKTYPE(SOLID_TO_ELEMWORKTYPE((X).imag) + SOLID_TO_ELEMWORKTYPE((Y).imag)); \
                                    }

   #define SOLID_OP_SUBTRACT(X,Y,Z) { (Z).real = SOLID_FROM_ELEMWORKTYPE(SOLID_TO_ELEMWORKTYPE((X).real) - SOLID_TO_ELEMWORKTYPE((Y).real)); \
                                      (Z).imag = SOLID_FROM_ELEMWORKTYPE(SOLID_TO_ELEMWORKTYPE((X).imag) - SOLID_TO_ELEMWORKTYPE((Y).imag)); \
                                    }

   #define SOLID_OP_MULTIPLY(X,Y,Z) { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      (Z).real = SOLID_FROM_ELEMWORKTYPE(xreal * yreal - ximag * yimag); \
                                      (Z).imag = SOLID_FROM_ELEMWORKTYPE(xreal * yimag + yreal * ximag); \
                                    }
                                 
   #define SOLID_OP_TRUE_DIVIDE(X,Y,Z) \
                                    { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      SOLID_C_ELEMWORKTYPE r = yreal * yreal + yimag * yimag; \
                                      (Z).real = SOLID_FROM_ELEMWORKTYPE((xreal * yreal + ximag * yimag) / r); \
                                      (Z).imag = SOLID_FROM_ELEMWORKTYPE((ximag * yreal - xreal * yimag) / r); \
                                    }

   #define SOLID_OP_DIVIDE(X,Y,Z)   SOLID_OP_TRUE_DIVIDE(X,Y,Z)

   #define SOLID_OP_POWER(X,Y,Z)    { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      SOLID_C_ELEMWORKTYPE scale = xreal * xreal + ximag * ximag; \
                                      SOLID_C_ELEMWORKTYPE zreal; \
                                      SOLID_C_ELEMWORKTYPE zimag; \
                                      if (scale == 0) \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE((yreal == 0) && (yimag == 0)); \
                                         (Z).imag = SDTYPE_ZERO(SDTYPE_ELEMTYPE(SDXTYPE)); \
                                      } \
                                      else \
                                      { /* Logarithm: z = log(x) */ \
                                        zimag = SD_FUN(ATAN2)(ximag,xreal); \
                                        zreal = SD_FUN(LOG)(scale) * 0.5; \
                                        \
                                        /* Multiply: x = y * log(x) */ \
                                        xreal = yreal * zreal - yimag * zimag; \
                                        ximag = yreal * zimag + yimag * zreal; \
                                        \
                                        /* Exponent: exp(y * log(x)) */ \
                                        scale = SD_FUN(EXP)(xreal); \
                                        xreal = scale * SD_FUN(COS)(ximag); \
                                        ximag = scale * SD_FUN(SIN)(ximag); \
                                        (Z).real = SOLID_FROM_ELEMWORKTYPE(xreal); \
                                        (Z).imag = SOLID_FROM_ELEMWORKTYPE(ximag); \
                                      } \
                                    }

   #define SOLID_OP_LT(X,Y,Z)       { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      if (xreal != yreal) \
                                      {  (Z) = (xreal < yreal) ? 1 : 0; \
                                      } \
                                      else \
                                      {  (Z) = (SOLID_TO_ELEMWORKTYPE((X).imag) < SOLID_TO_ELEMWORKTYPE((Y).imag)) ? 1 : 0; \
                                      } \
                                    }

   #define SOLID_OP_LE(X,Y,Z)       { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      if (xreal != yreal) \
                                      {  (Z) = (xreal < yreal) ? 1 : 0; \
                                      } \
                                      else \
                                      {  (Z) = (SOLID_TO_ELEMWORKTYPE((X).imag) <= SOLID_TO_ELEMWORKTYPE((Y).imag)) ? 1 : 0; \
                                      } \
                                    }

   #define SOLID_OP_GE(X,Y,Z)       { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      if (xreal != yreal) \
                                      {  (Z) = (xreal > yreal) ? 1 : 0; \
                                      } \
                                      else \
                                      {  (Z) = (SOLID_TO_ELEMWORKTYPE((X).imag) >= SOLID_TO_ELEMWORKTYPE((Y).imag)) ? 1 : 0; \
                                      } \
                                    }

   #define SOLID_OP_GT(X,Y,Z)       { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      if (xreal != yreal) \
                                      {  (Z) = (xreal > yreal) ? 1 : 0; \
                                      } \
                                      else \
                                      {  (Z) = (SOLID_TO_ELEMWORKTYPE((X).imag) > SOLID_TO_ELEMWORKTYPE((Y).imag)) ? 1 : 0; \
                                      } \
                                    }


   #define SOLID_OP_EQ(X,Y,Z)       { (Z) = ((SOLID_TO_ELEMWORKTYPE((X).real) == SOLID_TO_ELEMWORKTYPE((Y).real)) && \
                                             (SOLID_TO_ELEMWORKTYPE((X).imag) == SOLID_TO_ELEMWORKTYPE((Y).imag))) ? 1 : 0; \
                                    }

   #define SOLID_OP_NE(X,Y,Z)       { (Z) = ((SOLID_TO_ELEMWORKTYPE((X).real) != SOLID_TO_ELEMWORKTYPE((Y).real)) || \
                                             (SOLID_TO_ELEMWORKTYPE((X).imag) != SOLID_TO_ELEMWORKTYPE((Y).imag))) ? 1 : 0; \
                                    }


   #define SOLID_OP_MIN(X,Y,Z)      { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      if ((SD_FUN(ISNAN)(yreal) || SD_FUN(ISNAN(yimag))) || \
                                           (yreal < xreal) || ((yreal == xreal) && (yimag < ximag))) \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(yreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(yimag); \
                                      } \
                                      else \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(xreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(ximag); \
                                      } \
                                    }
                                     
   #define SOLID_OP_MAX(X,Y,Z)      { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      if ((SD_FUN(ISNAN)(yreal) || SD_FUN(ISNAN(yimag))) || \
                                          (yreal > xreal) || ((yreal == xreal) && (yimag > ximag))) \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(yreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(yimag); \
                                      } \
                                      else \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(xreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(ximag); \
                                      } \
                                    }

   #define SOLID_OP_FMIN(X,Y,Z)     { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      if (((yreal < xreal) || (SD_FUN(ISNAN)(xreal))) || \
                                          ((xreal == yreal) && ((yimag < ximag) || (SD_FUN(ISNAN)(ximag))))) \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(yreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(yimag); \
                                      } \
                                      else \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(xreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(ximag); \
                                      } \
                                    } 

   #define SOLID_OP_FMAX(X,Y,Z)     { SOLID_C_ELEMWORKTYPE xreal = SOLID_TO_ELEMWORKTYPE((X).real); \
                                      SOLID_C_ELEMWORKTYPE ximag = SOLID_TO_ELEMWORKTYPE((X).imag); \
                                      SOLID_C_ELEMWORKTYPE yreal = SOLID_TO_ELEMWORKTYPE((Y).real); \
                                      SOLID_C_ELEMWORKTYPE yimag = SOLID_TO_ELEMWORKTYPE((Y).imag); \
                                      if (((yreal > xreal) || (SD_FUN(ISNAN)(xreal))) || \
                                          ((xreal == yreal) && ((yimag > ximag) || (SD_FUN(ISNAN)(ximag))))) \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(yreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(yimag); \
                                      } \
                                      else \
                                      {  (Z).real = SOLID_FROM_ELEMWORKTYPE(xreal); \
                                         (Z).imag = SOLID_FROM_ELEMWORKTYPE(ximag); \
                                      } \
                                    } 

/* -------------------------------------------------------------------- */
#else /* Unsupported type */
/* -------------------------------------------------------------------- */

#error "Unsupported SDXTYPE encountered"

#endif

#endif

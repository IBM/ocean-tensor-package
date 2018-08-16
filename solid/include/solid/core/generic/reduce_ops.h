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


/* ------------------------------------------------------ */
/* SD_TEMPLATE_REDUCE(Operation name,                     */
/*                    Input data type,                    */
/*                    Output data type,                   */
/*                    Flag initialize empty tensor (cpu)  */
/*                    Flag finalization,                  */
/*                    Flag parameters,                    */
/*                    Code - Initialization,              */
/*                    Code - Accumulation,                */
/*                    Code - Reduction,                   */
/*                    Code - Finalization,                */
/*                    Parameter structure)                */
/* ------------------------------------------------------ */

/* Define intermediate template to ensure parameter expansion */
#ifndef SD_TEMPLATE_REDUCE_
#define SD_TEMPLATE_REDUCE_(NAME, DTYPE1, DTYPE2, FLAG_INIT, FLAG_FINALIZE, FLAG_PARAM, \
                            CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, CODE_FINALIZE, PARAM) \
         SD_TEMPLATE_REDUCE(NAME, DTYPE1, DTYPE2, FLAG_PARAM, PARAM, FLAG_INIT, FLAG_FINALIZE, \
                            CODE_INIT, CODE_ACCUMULATE, CODE_REDUCE, CODE_FINALIZE)

#endif

/* Undefine reduction operators */
#ifdef SOLID_OP_REDUCE_ANY
#undef SOLID_OP_REDUCE_ANY
#endif
#ifdef SOLID_OP_REDUCE_ALL
#undef SOLID_OP_REDUCE_ALL
#endif
#ifdef SOLID_OP_REDUCE_ALL_FINITE
#undef SOLID_OP_REDUCE_ALL_FINITE
#endif
#ifdef SOLID_OP_REDUCE_ANY_INF
#undef SOLID_OP_REDUCE_ANY_INF
#endif
#ifdef SOLID_OP_REDUCE_ANY_NAN
#undef SOLID_OP_REDUCE_ANY_NAN
#endif
#ifdef SOLID_OP_REDUCE_NNZ
#undef SOLID_OP_REDUCE_NNZ
#endif
#ifdef SOLID_OP_REDUCE_NNZ_NAN
#undef SOLID_OP_REDUCE_NNZ_NAN
#endif
#ifdef SOLID_OP_REDUCE_ALL_LT
#undef SOLID_OP_REDUCE_ALL_LT
#endif
#ifdef SOLID_OP_REDUCE_ALL_LE
#undef SOLID_OP_REDUCE_ALL_LE
#endif
#ifdef SOLID_OP_REDUCE_ALL_GT
#undef SOLID_OP_REDUCE_ALL_GT
#endif
#ifdef SOLID_OP_REDUCE_ALL_GE
#undef SOLID_OP_REDUCE_ALL_GE
#endif
#ifdef SOLID_OP_REDUCE_ALL_GTLT
#undef SOLID_OP_REDUCE_ALL_GTLT
#endif
#ifdef SOLID_OP_REDUCE_ALL_GTLE
#undef SOLID_OP_REDUCE_ALL_GTLE
#endif
#ifdef SOLID_OP_REDUCE_ALL_GELT
#undef SOLID_OP_REDUCE_ALL_GELT
#endif
#ifdef SOLID_OP_REDUCE_ALL_GELE
#undef SOLID_OP_REDUCE_ALL_GELE
#endif
#ifdef SOLID_OP_REDUCE_SUM
#undef SOLID_OP_REDUCE_SUM
#endif
#ifdef SOLID_OP_REDUCE_PROD
#undef SOLID_OP_REDUCE_PROD
#endif
#ifdef SOLID_OP_REDUCE_SUM_NAN
#undef SOLID_OP_REDUCE_SUM_NAN
#endif
#ifdef SOLID_OP_REDUCE_PROD_NAN
#undef SOLID_OP_REDUCE_PROD_NAN
#endif
#ifdef SOLID_OP_REDUCE_SUM_ABS
#undef SOLID_OP_REDUCE_SUM_ABS
#endif
#ifdef SOLID_OP_REDUCE_SUM_ABS_NAN
#undef SOLID_OP_REDUCE_SUM_ABS_NAN
#endif
#ifdef SOLID_OP_REDUCE_MAXIMUM
#undef SOLID_OP_REDUCE_MAXIMUM
#endif
#ifdef SOLID_OP_REDUCE_MINIMUM
#undef SOLID_OP_REDUCE_MINIMUM
#endif
#ifdef SOLID_OP_REDUCE_MAXIMUM_ABS
#undef SOLID_OP_REDUCE_MAXIMUM_ABS
#endif
#ifdef SOLID_OP_REDUCE_MINIMUM_ABS
#undef SOLID_OP_REDUCE_MINIMUM_ABS
#endif
#ifdef SOLID_OP_REDUCE_NORM
#undef SOLID_OP_REDUCE_NORM
#endif
#ifdef SOLID_OP_REDUCE_NORM_NAN
#undef SOLID_OP_REDUCE_NORM_NAN
#endif
#ifdef SOLID_OP_REDUCE_NORM2
#undef SOLID_OP_REDUCE_NORM2
#endif
#ifdef SOLID_OP_REDUCE_NORM2_NAN
#undef SOLID_OP_REDUCE_NORM2_NAN
#endif



/* ------------------------------------------------------------------------ */
/* Any                                                                      */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ANY \
      SD_TEMPLATE_REDUCE_(any, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if (!(SOLID_TO_WORKTYPE(*_ptr) == 0)) /* Works for NaN */ \
                            { _accumulate = 1; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result |= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ANY \
      SD_TEMPLATE_REDUCE_(any, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if ((!((SOLID_TO_ELEMWORKTYPE(_ptr -> real)) == 0)) || \
                                (!((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) == 0))) \
                          { _accumulate = 1; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result |= *_partial; }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* All                                                                      */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL \
      SD_TEMPLATE_REDUCE_(all, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          { if (SOLID_TO_WORKTYPE(*_ptr) == 0) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL \
      SD_TEMPLATE_REDUCE_(all, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          { if (((SOLID_TO_ELEMWORKTYPE(_ptr -> real)) == 0) && \
                                ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) == 0)) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* All finite (defined only for floating-point types)                       */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_ALL_FINITE \
      SD_TEMPLATE_REDUCE_(all_finite, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          { if (!SD_FUN(ISFINITE)(SOLID_TO_WORKTYPE(*_ptr))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_FINITE \
      SD_TEMPLATE_REDUCE_(all_finite, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          { if ((!SD_FUN(ISFINITE)(SOLID_TO_ELEMWORKTYPE(_ptr -> real))) || \
                                (!SD_FUN(ISFINITE)(SOLID_TO_ELEMWORKTYPE(_ptr -> imag)))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, { })
#endif
#endif


/* ------------------------------------------------------------------------ */
/* Any inf (defined only for floating-point types)                          */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_ANY_INF \
      SD_TEMPLATE_REDUCE_(any_inf, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if (SD_FUN(ISINF)(SOLID_TO_WORKTYPE(*_ptr))) \
                            { _accumulate = 1; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result |= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ANY_INF \
      SD_TEMPLATE_REDUCE_(any_inf, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if ((SD_FUN(ISINF)(SOLID_TO_ELEMWORKTYPE(_ptr -> real))) || \
                                (SD_FUN(ISINF)(SOLID_TO_ELEMWORKTYPE(_ptr -> imag)))) \
                            { _accumulate = 1; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result |= *_partial; }, \
                          { }, { })
#endif
#endif


/* ------------------------------------------------------------------------ */
/* Any nan (defined only for floating-point types)                          */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_ANY_NAN \
      SD_TEMPLATE_REDUCE_(any_nan, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if (SD_FUN(ISNAN)(SOLID_TO_WORKTYPE(*_ptr))) \
                            { _accumulate = 1; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result |= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ANY_NAN \
      SD_TEMPLATE_REDUCE_(any_nan, SDXTYPE, bool, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if ((SD_FUN(ISNAN)(SOLID_TO_ELEMWORKTYPE(_ptr -> real))) || \
                                (SD_FUN(ISNAN)(SOLID_TO_ELEMWORKTYPE(_ptr -> imag)))) \
                            { _accumulate = 1; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result |= *_partial; }, \
                          { }, { })
#endif
#endif


/* ------------------------------------------------------------------------ */
/* NNZ (number of non-zeros)                                                */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_NNZ \
      SD_TEMPLATE_REDUCE_(nnz, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if (!(SOLID_TO_WORKTYPE(*_ptr) == 0)) /* Works for NaN */ \
                            { _accumulate ++; } \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_NNZ \
      SD_TEMPLATE_REDUCE_(nnz, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { if ((!((SOLID_TO_ELEMWORKTYPE(_ptr -> real)) == 0)) || \
                                (!((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) == 0))) \
                            { _accumulate ++; } \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* NNZ NaN (number of non-zeros, excluding NaN)                             */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_NNZ_NAN \
      SD_TEMPLATE_REDUCE_(nnz_nan, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (_temp != 0)) \
                            { _accumulate ++; } \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_NNZ_NAN \
      SD_TEMPLATE_REDUCE_(nnz_nan, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag)) && \
                                ((((SOLID_TO_ELEMWORKTYPE(_real)) != 0)) || \
                                 (((SOLID_TO_ELEMWORKTYPE(_imag)) != 0)))) \
                            { _accumulate ++; } \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#endif
#endif


/* ------------------------------------------------------------------------ */
/* All less than                                                            */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_LT \
      SD_TEMPLATE_REDUCE_(all_lt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if (SOLID_TO_WORKTYPE(*_ptr) >= param.bound) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_LT \
      SD_TEMPLATE_REDUCE_(all_lt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real > param.bound.real)) || \
                                ((_real == param.bound.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) >= param.bound.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#endif


/* ------------------------------------------------------------------------ */
/* All less than or equal                                                   */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_LE \
      SD_TEMPLATE_REDUCE_(all_le, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if (SOLID_TO_WORKTYPE(*_ptr) > param.bound) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_LE \
      SD_TEMPLATE_REDUCE_(all_le, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real > param.bound.real)) || \
                                ((_real == param.bound.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) > param.bound.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#endif


/* ------------------------------------------------------------------------ */
/* All greater than                                                         */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_GT \
      SD_TEMPLATE_REDUCE_(all_gt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if (SOLID_TO_WORKTYPE(*_ptr) <= param.bound) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_GT \
      SD_TEMPLATE_REDUCE_(all_gt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real < param.bound.real)) || \
                                ((_real == param.bound.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) <= param.bound.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#endif


/* ------------------------------------------------------------------------ */
/* All greater than or equal                                                */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_GE \
      SD_TEMPLATE_REDUCE_(all_ge, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if (SOLID_TO_WORKTYPE(*_ptr) < param.bound) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_GE \
      SD_TEMPLATE_REDUCE_(all_ge, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real < param.bound.real)) || \
                                ((_real == param.bound.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) < param.bound.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE bound; })
#endif


/* ------------------------------------------------------------------------ */
/* All in range (lower, upper)                                              */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_GTLT \
      SD_TEMPLATE_REDUCE_(all_gtlt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if ((SOLID_TO_WORKTYPE(*_ptr) <= param.lower) || \
                                (SOLID_TO_WORKTYPE(*_ptr) >= param.upper)) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_GTLT \
      SD_TEMPLATE_REDUCE_(all_gtlt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real < param.lower.real)) || (_real > param.upper.real) || \
                                ((_real == param.lower.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) <= param.lower.imag)) || \
                                ((_real == param.upper.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) >= param.upper.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#endif


/* ------------------------------------------------------------------------ */
/* All in range (lower, upper]                                              */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_GTLE \
      SD_TEMPLATE_REDUCE_(all_gtle, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if ((SOLID_TO_WORKTYPE(*_ptr) <= param.lower) || \
                                (SOLID_TO_WORKTYPE(*_ptr) >  param.upper)) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_GTLE \
      SD_TEMPLATE_REDUCE_(all_gtle, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real < param.lower.real)) || (_real > param.upper.real) || \
                                ((_real == param.lower.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) <= param.lower.imag)) || \
                                ((_real == param.upper.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) >  param.upper.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#endif


/* ------------------------------------------------------------------------ */
/* All in range [lower, upper)                                              */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_GELT \
      SD_TEMPLATE_REDUCE_(all_gelt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if ((SOLID_TO_WORKTYPE(*_ptr) <  param.lower) || \
                                (SOLID_TO_WORKTYPE(*_ptr) >= param.upper)) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_GELT \
      SD_TEMPLATE_REDUCE_(all_gelt, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real < param.lower.real)) || (_real > param.upper.real) || \
                                ((_real == param.lower.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) <  param.lower.imag)) || \
                                ((_real == param.upper.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) >= param.upper.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#endif


/* ------------------------------------------------------------------------ */
/* All in range [lower, upper]                                              */
/* ------------------------------------------------------------------------ */
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_ALL_GELE \
      SD_TEMPLATE_REDUCE_(all_gele, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { if ((SOLID_TO_WORKTYPE(*_ptr) < param.lower) || \
                                (SOLID_TO_WORKTYPE(*_ptr) > param.upper)) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_ALL_GELE \
      SD_TEMPLATE_REDUCE_(all_gele, SDXTYPE, bool, 1, 0, 1, \
                          { _accumulate = 1; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            if (((_real < param.lower.real)) || (_real > param.upper.real) || \
                                ((_real == param.lower.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) < param.lower.imag)) || \
                                ((_real == param.upper.real) && ((SOLID_TO_ELEMWORKTYPE(_ptr -> imag)) > param.upper.imag))) \
                            { _accumulate = 0; SOLID_OP_REDUCE_BREAK; } \
                          }, \
                          { *_result &= *_partial; }, \
                          { }, \
                          { SOLID_C_WORKTYPE lower; \
                            SOLID_C_WORKTYPE upper; \
                          })
#endif


/* ------------------------------------------------------------------------ */
/* Sum                                                                      */
/* ------------------------------------------------------------------------ */

/* Data types */
#ifdef SDXTYPE_OUTPUT_SUM
#undef SDXTYPE_OUTPUT_SUM
#endif
#if SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE)
#define SDXTYPE_OUTPUT_SUM uint64
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
#define SDXTYPE_OUTPUT_SUM int64
#else
#define SDXTYPE_OUTPUT_SUM SDXTYPE
#endif

#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_SUM \
      SD_TEMPLATE_REDUCE_(sum, SDXTYPE, SDXTYPE_OUTPUT_SUM, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { _accumulate += SOLID_TO_WORKTYPE(*_ptr); }, \
                          { *_result += *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_SUM \
      SD_TEMPLATE_REDUCE_(sum, SDXTYPE, SDXTYPE_OUTPUT_SUM, 1, 0, 0, \
                          { _accumulate.real = 0; _accumulate.imag = 0; }, \
                          { _accumulate.real += SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            _accumulate.imag += SOLID_TO_ELEMWORKTYPE(_ptr -> imag); }, \
                          { _result -> real += _partial -> real; \
                            _result -> imag += _partial -> imag; }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Prod                                                                     */
/* ------------------------------------------------------------------------ */

/* Data types */
#ifdef SDXTYPE_OUTPUT_PROD
#undef SDXTYPE_OUTPUT_PROD
#endif
#if SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE)
#define SDXTYPE_OUTPUT_PROD uint64
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
#define SDXTYPE_OUTPUT_PROD int64
#else
#define SDXTYPE_OUTPUT_PROD SDXTYPE
#endif

#if SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_PROD \
      SD_TEMPLATE_REDUCE_(prod, SDXTYPE, SDXTYPE_OUTPUT_PROD, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          { _accumulate *= SOLID_TO_WORKTYPE(*_ptr); }, \
                          { *_result *= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_PROD \
      SD_TEMPLATE_REDUCE_(prod, SDXTYPE, SDXTYPE_OUTPUT_PROD, 1, 0, 0, \
                          { _accumulate.real = 1; \
                            _accumulate.imag = 0; \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            SOLID_C_ELEMWORKTYPE _temp = _accumulate.real; \
                            _accumulate.real = _temp * _real - _accumulate.imag * _imag; \
                            _accumulate.imag = _temp * _imag + _accumulate.imag * _real; \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _temp = _result -> real; \
                            _result -> real = _temp * _partial -> real - _result -> imag * _partial -> imag; \
                            _result -> imag = _temp * _partial -> imag + _result -> imag * _partial -> real; \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Sum NaN                                                                  */
/* ------------------------------------------------------------------------ */

/* Data types */
#ifdef SDXTYPE_OUTPUT_SUM_NAN
#undef SDXTYPE_OUTPUT_SUM_NAN
#endif
#if SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE)
#define SDXTYPE_OUTPUT_SUM_NAN uint64
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
#define SDXTYPE_OUTPUT_SUM_NAN int64
#endif

#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* For integer types this version is the same as reduce sum */
   #define SOLID_OP_REDUCE_SUM_NAN \
      SD_TEMPLATE_REDUCE_(sum_nan, SDXTYPE, SDXTYPE_OUTPUT_SUM_NAN, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { _accumulate += SOLID_TO_WORKTYPE(*_ptr); }, \
                          { *_result += *_partial; }, \
                          { }, { })

#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_SUM_NAN \
      SD_TEMPLATE_REDUCE_(sum_nan, SDXTYPE, SDXTYPE, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          {  SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                             if (!SD_FUN(ISNAN)(_temp)) _accumulate += _temp; \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_SUM_NAN \
      SD_TEMPLATE_REDUCE_(sum_nan, SDXTYPE, SDXTYPE, 1, 0, 0, \
                          { _accumulate.real = 0; \
                            _accumulate.imag = 0; \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag))) \
                            { _accumulate.real += _real; \
                              _accumulate.imag += _imag; \
                            } \
                          }, \
                          { _result -> real += _partial -> real; \
                            _result -> imag += _partial -> imag; \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Prod NaN                                                                 */
/* ------------------------------------------------------------------------ */

/* Data types */
#ifdef SDXTYPE_OUTPUT_PROD_NAN
#undef SDXTYPE_OUTPUT_PROD_NAN
#endif
#if SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE)
#define SDXTYPE_OUTPUT_PROD_NAN uint64
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
#define SDXTYPE_OUTPUT_PROD_NAN int64
#endif

#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* For integer types this version is the same as reduce prod */
   #define SOLID_OP_REDUCE_PROD_NAN \
      SD_TEMPLATE_REDUCE_(prod_nan, SDXTYPE, SDXTYPE_OUTPUT_PROD_NAN, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          { _accumulate *= SOLID_TO_WORKTYPE(*_ptr); }, \
                          { *_result *= *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_PROD_NAN \
      SD_TEMPLATE_REDUCE_(prod_nan, SDXTYPE, SDXTYPE, 1, 0, 0, \
                          { _accumulate = 1; }, \
                          {  SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                             if (!SD_FUN(ISNAN)(_temp)) _accumulate *= _temp; \
                          }, \
                          { *_result *= *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_PROD_NAN \
      SD_TEMPLATE_REDUCE_(prod_nan, SDXTYPE, SDXTYPE, 1, 0, 0, \
                          { _accumulate.real = 1; \
                             _accumulate.imag = 0; \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            SOLID_C_ELEMWORKTYPE _temp; \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag))) \
                            { _temp = _accumulate.real; \
                              _accumulate.real = _temp * _real - _accumulate.imag * _imag; \
                              _accumulate.imag = _temp * _imag - _accumulate.imag * _real; \
                            } \
                          }, \
                          { SOLID_C_ELEMWORKTYPE_TYPE(SDXTYPE) _temp = _result -> real; \
                            _result -> real = _temp * _partial -> real - _result -> imag * _partial -> imag; \
                            _result -> imag = _temp * _partial -> imag + _result -> imag * _partial -> real; \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Sum absolute                                                             */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* For unsigned integer types this version is the same as reduce sum */
   #define SOLID_OP_REDUCE_SUM_ABS \
      SD_TEMPLATE_REDUCE_(sum_abs, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { _accumulate += SOLID_TO_WORKTYPE(*_ptr); }, \
                          { *_result += *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   /* Signed integer */
   #define SOLID_OP_REDUCE_SUM_ABS \
      SD_TEMPLATE_REDUCE_(sum_abs, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          {  SOLID_C_TYPE(SDXTYPE) _temp = *_ptr;\
                             if (_temp < 0) \
                                  _accumulate -= _temp; \
                             else _accumulate += _temp; \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_SUM_ABS \
      SD_TEMPLATE_REDUCE_(sum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { _accumulate += SD_FUN(FABS)(SOLID_TO_WORKTYPE(*_ptr)); }, \
                          { *_result += *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_SUM_ABS \
      SD_TEMPLATE_REDUCE_(sum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                           _accumulate += SD_FUN(HYPOT)(_real, _imag); \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Sum absolute NaN                                                         */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* For unsigned integer types this version is the same as sum */
   #define SOLID_OP_REDUCE_SUM_ABS_NAN \
      SD_TEMPLATE_REDUCE_(sum_abs_nan, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { _accumulate += SOLID_TO_WORKTYPE(*_ptr); }, \
                          { *_result += *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   /* For signed integer types this version is the same as sum abs */
   #define SOLID_OP_REDUCE_SUM_ABS_NAN \
      SD_TEMPLATE_REDUCE_(sum_abs_nan, SDXTYPE, uint64, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          {  SOLID_C_TYPE(SDXTYPE) _temp = *_ptr;\
                            if (_temp < 0) \
                                   _accumulate -= _temp; \
                             else _accumulate += _temp; \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_SUM_ABS_NAN \
      SD_TEMPLATE_REDUCE_(sum_abs_nan, SDXTYPE, SDXTYPE, 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (!SD_FUN(ISNAN)(_temp)) _accumulate += SD_FUN(FABS)(_temp); }, \
                          { *_result += *_partial; }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_SUM_ABS_NAN \
      SD_TEMPLATE_REDUCE_(sum_abs_nan, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 0, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag))) \
                            { _accumulate += SD_FUN(HYPOT)(_real, _imag); } \
                          }, \
                          { *_result += *_partial; }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Maximum                                                                  */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* Integer types */
   #define SOLID_OP_REDUCE_MAXIMUM \
      SD_TEMPLATE_REDUCE_(maximum, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SOLID_TO_WORKTYPE(*_ptr); }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (_accumulate <_temp) _accumulate = _temp; \
                          }, \
                          { if (*_result < *_partial) *_result = *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex floating point */
   #define SOLID_OP_REDUCE_MAXIMUM \
      SD_TEMPLATE_REDUCE_(maximum, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SOLID_TO_WORKTYPE(*_ptr); }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(_accumulate >= _temp))) _accumulate = _temp; \
                          }, \
                          { SOLID_C_WORKTYPE _temp = *_partial; \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(*_result >= _temp))) *_result = _temp; \
                          }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_MAXIMUM \
      SD_TEMPLATE_REDUCE_(maximum, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate.real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            _accumulate.imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag)) && \
                                ((!(_accumulate.real >= _real)) || \
                                 ((!(_accumulate.real < _real)) && (!(_accumulate.real > _real)) && (!(_accumulate.imag >= _imag))) || \
                                 (SD_FUN(ISNAN)(_accumulate.imag)))) \
                             { _accumulate.real = _real; \
                               _accumulate.imag = _imag; \
                             } \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = _partial -> real; \
                            SOLID_C_ELEMWORKTYPE _imag = _partial -> imag; \
                            SOLID_C_ELEMWORKTYPE _resreal = _result -> real; \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag)) && \
                                ((!(_resreal >= _real)) || \
                                 ((!(_resreal < _real)) && (!(_resreal > _real)) && (!(_result -> imag >= _imag))) || \
                                 (SD_FUN(ISNAN)(_result -> imag)))) \
                            { _result -> real = _real; \
                              _result -> imag = _imag; \
                            } \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Minimum                                                                  */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* Integer types */
   #define SOLID_OP_REDUCE_MINIMUM \
      SD_TEMPLATE_REDUCE_(minimum, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SOLID_TO_WORKTYPE(*_ptr); }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (_accumulate >_temp) _accumulate = _temp; \
                          }, \
                          { if (*_result > *_partial) *_result = *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_MINIMUM \
      SD_TEMPLATE_REDUCE_(minimum, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SOLID_TO_WORKTYPE(*_ptr); }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(_accumulate <= _temp))) _accumulate = _temp; \
                          }, \
                          { SOLID_C_WORKTYPE _temp = *_partial; \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(*_result <= _temp))) *_result = _temp; \
                          }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_MINIMUM \
      SD_TEMPLATE_REDUCE_(minimum, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate.real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            _accumulate.imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag)) && \
                                ((!(_accumulate.real <= _real)) || \
                                 ((!(_accumulate.real < _real)) && (!(_accumulate.real > _real)) && (!(_accumulate.imag <= _imag))) || \
                                 (SD_FUN(ISNAN)(_accumulate.imag)))) \
                             { _accumulate.real = _real; \
                               _accumulate.imag = _imag; \
                             } \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = _partial -> real; \
                            SOLID_C_ELEMWORKTYPE _imag = _partial -> imag; \
                            SOLID_C_ELEMWORKTYPE _resreal = _result -> real; \
                            if ((!SD_FUN(ISNAN)(_real)) && (!SD_FUN(ISNAN)(_imag)) && \
                                ((!(_resreal <= _real)) || \
                                 ((!(_resreal < _real)) && (!(_resreal > _real)) && (!(_result -> imag <= _imag))) || \
                                 (SD_FUN(ISNAN)(_result -> imag)))) \
                            { _result -> real = _real; \
                              _result -> imag = _imag; \
                            } \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Maximum absolute                                                         */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* For unsigned integer types this version is the same as maximum */
   #define SOLID_OP_REDUCE_MAXIMUM_ABS \
      SD_TEMPLATE_REDUCE_(maximum_abs, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SOLID_TO_WORKTYPE(*_ptr); }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (_accumulate <_temp) _accumulate = _temp; \
                          }, \
                          { if (*_result < *_partial) *_result = *_partial; }, \
                          { }, { })

#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   /* Signed integers */
   #define SOLID_OP_REDUCE_MAXIMUM_ABS \
      SD_TEMPLATE_REDUCE_(maximum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 0, 0, 0, \
                          { SOLID_C_TYPE(SDXTYPE) _temp = *_ptr; \
                            if (_temp < 0) \
                                 _accumulate = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))(-(_temp)); \
                            else _accumulate = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))( (_temp)); \
                          }, \
                          { SOLID_C_TYPE(SDXTYPE) _temp = *_ptr; \
                            SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)) _abs; \
                            if (_temp < 0) \
                                 _abs = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))(-(_temp)); \
                            else _abs = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))( (_temp)); \
                            if (_accumulate < _abs) _accumulate = _abs; \
                          }, \
                          { if ((*_result < *_partial)) *_result = *_partial; }, \
                          { }, { })

#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_MAXIMUM_ABS \
      SD_TEMPLATE_REDUCE_(maximum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 0, 0, 0, \
                          { _accumulate = SD_FUN(FABS)(SOLID_TO_WORKTYPE(*_ptr)); }, \
                          { SOLID_C_WORKTYPE _temp = SD_FUN(FABS)(SOLID_TO_WORKTYPE(*_ptr)); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(_accumulate >= _temp))) _accumulate = _temp; \
                          }, \
                          { SOLID_C_WORKTYPE _temp = *_partial; \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(*_result >= _temp))) *_result = _temp; \
                          }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_MAXIMUM_ABS \
      SD_TEMPLATE_REDUCE_(maximum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 0, 0, 0, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            _accumulate = SD_FUN(HYPOT)(_real, _imag); \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            SOLID_C_ELEMWORKTYPE _temp = SD_FUN(HYPOT)(_real, _imag); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(_accumulate >= _temp))) _accumulate = _temp; \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _temp = *_partial; \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(*_result >= _temp))) *_result = _temp; \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Minimum absolute                                                         */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_UNSIGNED_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* For unsigned integer types this version is the same as minimum */
   #define SOLID_OP_REDUCE_MINIMUM_ABS \
      SD_TEMPLATE_REDUCE_(minimum_abs, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SOLID_TO_WORKTYPE(*_ptr); }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (_accumulate >_temp) _accumulate = _temp; \
                          }, \
                          { if (*_result > *_partial) *_result = *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_SIGNED_INT(SDXTYPE)
   /* Signed integers */
   #define SOLID_OP_REDUCE_MINIMUM_ABS \
      SD_TEMPLATE_REDUCE_(minimum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 0, 0, 0, \
                          { SOLID_C_TYPE(SDXTYPE) _temp = *_ptr; \
                            if (_temp < 0) \
                                 _accumulate = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))(-(_temp)); \
                            else _accumulate = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))( (_temp)); \
                          }, \
                          { SOLID_C_TYPE(SDXTYPE) _temp = *_ptr; \
                            SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)) _abs; \
                            if (_temp < 0) \
                                 _abs = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))(-(_temp)); \
                            else _abs = (SOLID_C_TYPE(SDTYPE_ABSTYPE(SDXTYPE)))( (_temp)); \
                            if (_accumulate > _abs) _accumulate = _abs; \
                          }, \
                          { if ((*_result > *_partial)) *_result = *_partial; }, \
                          { }, { })
#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_MINIMUM_ABS \
      SD_TEMPLATE_REDUCE_(minimum_abs, SDXTYPE, SDXTYPE, 0, 0, 0, \
                          { _accumulate = SD_FUN(FABS)(SOLID_TO_WORKTYPE(*_ptr)); }, \
                          { SOLID_C_WORKTYPE _temp = SD_FUN(FABS)(SOLID_TO_WORKTYPE(*_ptr)); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(_accumulate <= _temp))) _accumulate = _temp; \
                          }, \
                          { SOLID_C_WORKTYPE _temp = *_partial; \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(*_result <= _temp))) *_result = _temp; \
                          }, \
                          { }, { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_MINIMUM_ABS \
      SD_TEMPLATE_REDUCE_(minimum_abs, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 0, 0, 0, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            _accumulate = SD_FUN(HYPOT)(_real, _imag); \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _real = SOLID_TO_ELEMWORKTYPE(_ptr -> real); \
                            SOLID_C_ELEMWORKTYPE _imag = SOLID_TO_ELEMWORKTYPE(_ptr -> imag); \
                            SOLID_C_ELEMWORKTYPE _temp = SD_FUN(HYPOT)(_real, _imag); \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(_accumulate <= _temp))) _accumulate = _temp; \
                          }, \
                          { SOLID_C_ELEMWORKTYPE _temp = *_partial; \
                            if ((!SD_FUN(ISNAN)(_temp)) && (!(*_result <= _temp))) *_result = _temp; \
                          }, \
                          { }, { })
#endif


/* ------------------------------------------------------------------------ */
/* Norm                                                                     */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* Integer and Boolean */
   #define SOLID_OP_REDUCE_NORM \
      SD_TEMPLATE_REDUCE_(norm, SDXTYPE, double, 1, 1, 1, \
                          { _accumulate = 0; }, \
                          { double _temp = (double)(SOLID_TO_WORKTYPE(*_ptr)); \
                            _temp *= _temp; \
                            _accumulate += SD_FUN_TYPE(POW,double)(_temp, param.phalf); \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(POW,double)(*_partial, param.pinv); }, \
                          { double phalf; \
                            double pinv; \
                          })

#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_NORM \
      SD_TEMPLATE_REDUCE_(norm, SDXTYPE, SDXTYPE, 1, 1, 1, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            _temp *= _temp; \
                            _accumulate += SD_FUN_TYPE(POW,SDXTYPE)(_temp, param.phalf); \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(POW,SDXTYPE)(*_partial, param.pinv); }, \
                          { SOLID_C_WORKTYPE_TYPE(SDXTYPE) phalf; \
                            SOLID_C_WORKTYPE_TYPE(SDXTYPE) pinv; \
                          })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_NORM \
      SD_TEMPLATE_REDUCE_(norm, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 1, 1, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE     _temp1 = SOLID_TO_WORKTYPE(*_ptr); \
                            SOLID_C_ELEMWORKTYPE _temp2 = (_temp1.real * _temp1.real) + (_temp1.imag * _temp1.imag); \
                            _accumulate += SD_FUN_TYPE(POW,SOLID_ELEMWORKTYPE(SDXTYPE))(_temp2, param.phalf); \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(POW,SDTYPE_ABSTYPE(SDXTYPE))(*_partial, param.pinv); }, \
                          { SOLID_C_ELEMWORKTYPE_TYPE(SDXTYPE) phalf; \
                            SOLID_C_ELEMWORKTYPE_TYPE(SDXTYPE) pinv; \
                          })
#endif


/* ------------------------------------------------------------------------ */
/* Norm NaN                                                                 */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_NORM_NAN \
      SD_TEMPLATE_REDUCE_(norm_nan, SDXTYPE, SDXTYPE, 1, 1, 1, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (!SD_FUN(ISNAN)(_temp)) \
                            { _temp *= _temp; \
                              _accumulate += SD_FUN_TYPE(POW,SDXTYPE)(_temp, param.phalf); \
                            } \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(POW,SDXTYPE)(*_partial, param.pinv); }, \
                          { SOLID_C_WORKTYPE_TYPE(SDXTYPE) phalf; \
                            SOLID_C_WORKTYPE_TYPE(SDXTYPE) pinv; \
                          })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_NORM_NAN \
      SD_TEMPLATE_REDUCE_(norm_nan, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 1, 1, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE     _temp1 = SOLID_TO_WORKTYPE(*_ptr); \
                            SOLID_C_ELEMWORKTYPE _temp2 = (_temp1.real * _temp1.real) + (_temp1.imag * _temp1.imag); \
                            if (!SD_FUN(ISNAN)(_temp2)) \
                            { _accumulate += SD_FUN_TYPE(POW,SOLID_ELEMWORKTYPE(SDXTYPE))(_temp2, param.phalf); \
                            } \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(POW,SDTYPE_ABSTYPE(SDXTYPE))(*_partial, param.pinv); }, \
                          { SOLID_C_ELEMWORKTYPE_TYPE(SDXTYPE) phalf; \
                            SOLID_C_ELEMWORKTYPE_TYPE(SDXTYPE) pinv; \
                          })
#endif
#endif


/* ------------------------------------------------------------------------ */
/* Norm2                                                                    */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_INT(SDXTYPE) || SDTYPE_IS_BOOL(SDXTYPE))
   /* Integer and Boolean */
   #define SOLID_OP_REDUCE_NORM2 \
      SD_TEMPLATE_REDUCE_(norm2, SDXTYPE, double, 1, 1, 0, \
                          { _accumulate = 0; }, \
                          { double _temp = (double)(SOLID_TO_WORKTYPE(*_ptr)); \
                            _accumulate += _temp * _temp; \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(SQRT,double)(*_partial); }, \
                          { })

#elif SDTYPE_IS_REAL(SDXTYPE)
   /* Non-complex */
   #define SOLID_OP_REDUCE_NORM2 \
      SD_TEMPLATE_REDUCE_(norm2, SDXTYPE, SDXTYPE, 1, 1, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            _accumulate += _temp * _temp; \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(SQRT,SDXTYPE)(*_partial); }, \
                          { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_NORM2 \
      SD_TEMPLATE_REDUCE_(norm2, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 1, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            _accumulate += (_temp.real * _temp.real) + (_temp.imag * _temp.imag); \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(SQRT,SDTYPE_ABSTYPE(SDXTYPE))(*_partial); }, \
                          { })
#endif


/* ------------------------------------------------------------------------ */
/* Norm2 NaN                                                                */
/* ------------------------------------------------------------------------ */
#if (SDTYPE_IS_FLOAT(SDXTYPE) || SDTYPE_IS_COMPLEX(SDXTYPE))
#if SDTYPE_IS_REAL(SDXTYPE)
   /* Real floating point */
   #define SOLID_OP_REDUCE_NORM2_NAN \
      SD_TEMPLATE_REDUCE_(norm2_nan, SDXTYPE, SDXTYPE, 1, 1, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp = SOLID_TO_WORKTYPE(*_ptr); \
                            if (!SD_FUN(ISNAN)(_temp)) _accumulate += _temp * _temp; \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(SQRT,SDXTYPE)(*_partial); }, \
                          { })
#else
   /* Complex floating point */
   #define SOLID_OP_REDUCE_NORM2_NAN \
      SD_TEMPLATE_REDUCE_(norm2_nan, SDXTYPE, SDTYPE_ABSTYPE(SDXTYPE), 1, 1, 0, \
                          { _accumulate = 0; }, \
                          { SOLID_C_WORKTYPE _temp1 = SOLID_TO_WORKTYPE(*_ptr); \
                            SOLID_C_ELEMWORKTYPE _temp2;\
                            _temp2 = (_temp1.real * _temp1.real) + (_temp1.imag * _temp1.imag); \
                            if (!SD_FUN(ISNAN)(_temp2)) _accumulate += _temp2; \
                          }, \
                          { *_result += *_partial; }, \
                          { *_result = SD_FUN_TYPE(SQRT,SDTYPE_ABSTYPE(SDXTYPE))(*_partial); }, \
                          { })
#endif
#endif

#endif

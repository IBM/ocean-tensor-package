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

#ifndef __OC_ARCH_H__
#define __OC_ARCH_H__

#define OC_ARCH_NAME ""

/* -------------------------------------------------------------------- */
/* AMD64                                                                */
/* -------------------------------------------------------------------- */
#if defined(__amd64__) || defined(_amd64) || defined(__x64_64__) || \
    defined(__x64_64)  || defined(_M_X64) || defined(_M_AMD_64)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "AMD64"
#define OC_ARCH_AMD64
#endif


/* -------------------------------------------------------------------- */
/* Intel x86                                                            */
/* -------------------------------------------------------------------- */
#if defined(i386)     || defined(__i386)    || defined(__i386__)      || \
    defined(__i486__) || defined(__i586__)  || defined(__i686__)      || \
    defined(__IA32__) || defined(_M_I86)    || defined(M_IX86)        || \
    defined(_M_IX86)  || defined(__X86__)   || defined(__THW_INTEL__) || \
    defined(__I86__)  || defined(__INTEL__) || defined(__386)         || \
    defined(i486)     || defined(i586)      || defined(i686)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "Intel x64"
#define OC_ARCH_X64
#endif


/* -------------------------------------------------------------------- */
/* Intel Itanium (IA-64)                                                */
/* -------------------------------------------------------------------- */
#if defined(__ia64__) || defined(_IA64)   || defined(__IA64__) || \
    defined(__ia64)   || defined(_M_IA64) || defined(__itanium__)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "Intel Itanium (IA-64)"
#define OC_ARCH_IA64
#endif


/* -------------------------------------------------------------------- */
/* Motorola 68k                                                         */
/* -------------------------------------------------------------------- */
#if defined(__m68k__)    || defined(M68000)    || defined(__MC68K) || \
    defined(__mc68000__) || defined(__mc68000) || defined(mc68000) || defined(__MC68000__) || \
    defined(__mc68100__) || defined(__mc68100) || defined(mc68100) || defined(__MC68100__) || \
    defined(__mc68200__) || defined(__mc68200) || defined(mc68200) || defined(__MC68200__) || \
    defined(__mc68300__) || defined(__mc68300) || defined(mc68300) || defined(__MC68300__) || \
    defined(__mc68400__) || defined(__mc68400) || defined(mc68400) || defined(__MC68400__) || \
    defined(__mc68600__) || defined(__mc68600) || defined(mc68600) || defined(__MC68600__)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "Motorola 68k"
#define OC_ARCH_M68K
#endif


/* -------------------------------------------------------------------- */
/* PowerPC                                                              */
/* -------------------------------------------------------------------- */
#if defined(__powerpc)   || defined(__powerpc__) || defined(__powerpc64__) || \
    defined(__POWERPC__) || defined(__ppc__)     || defined(__ppc64__)     || \
    defined(__PPC__)     || defined(__PPC64__)   || defined(_ARCH_PPC)     || \
    defined(__ppc601__)  || defined(__ppc603__)  || defined(__ppc604__)    || \
    defined(_M_PPC)      || defined(_ARCH_440)   || defined(_ARCH_450)     || \
    defined(_ARCH_601)   || defined(_ARCH_603)   || defined(_ARCH_604)     || \
    defined(__ppc)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "PowerPC"
#define OC_ARCH_PPC
#endif


/* -------------------------------------------------------------------- */
/* Sparc                                                                */
/* -------------------------------------------------------------------- */
#if defined(__sparc__)    || defined(__sparc)   || defined(__sparc_v8__) || \
    defined(__sparc_v9__) || defined(__sparcv8) || defined(__sparcv9)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "Sparc"
#define OC_ARCH_SPARC
#endif


/* -------------------------------------------------------------------- */
/* SystemZ                                                              */
/* -------------------------------------------------------------------- */
#if defined(__370__)   || defined(__THW_370__) || defined(__s390__) || \
    defined(__s390x__) || defined(__zarch__)   || defined(__SYSC_ZARCH__)
#undef OC_ARCH_NAME
#define OC_ARCH_NAME "SystemZ"
#define OC_ARCH_SYSTEM_Z
#endif

#endif

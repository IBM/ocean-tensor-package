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

#ifndef __PYOCEAN_CORE_H__
#define __PYOCEAN_CORE_H__

#include <Python.h>

#include "ocean.h"


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */
PyObject *pyOceanCore_intrnl_ensure        (PyObject *obj, OcDType dtype, OcDevice *device, int flagInplace);
PyObject *pyOceanCore_intrnl_cast          (PyObject *obj, OcDType dtype, OcDevice *device);

PyObject *pyOceanCore_intrnl_add           (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_subtract      (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_scale         (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_divide        (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_trueDivide    (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_floorDivide   (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_mod           (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_fmod          (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_min           (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_max           (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_fmin          (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_fmax          (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_bitwiseAnd    (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_bitwiseOr     (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_bitwiseXor    (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_logicalAnd    (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_logicalOr     (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_logicalXor    (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_bitshiftLeft  (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_bitshiftRight (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_power         (PyObject *src1, PyObject *src2, OcTensor *dst, char mode);

int       pyOceanCore_intrnl_iadd          (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_isubtract     (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_iscale        (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_idivide       (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ifloorDivide  (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_imod          (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ibitwiseAnd   (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ibitwiseOr    (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ibitwiseXor   (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ibitshiftLeft (OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ibitshiftRight(OcTensor *dst, PyObject *src);
int       pyOceanCore_intrnl_ipower        (OcTensor *dst, PyObject *src, char mode);

PyObject *pyOceanCore_intrnl_mtimes      (PyObject *src1, PyObject *src2, OcTensor *dst);
PyObject *pyOceanCore_intrnl_gemm        (PyObject *alpha, OcTensor *A, char transA,
                                                           OcTensor *B, char transB,
                                          PyObject *beta,  OcTensor *C);

#endif

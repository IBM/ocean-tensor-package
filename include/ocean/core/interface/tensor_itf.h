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

#ifndef __OC_MODULE_CORE_ITF_TENSOR_H__
#define __OC_MODULE_CORE_ITF_TENSOR_H__

#include "ocean/base/dtype.h"
#include "ocean/base/device.h"
#include "ocean/base/scalar.h"
#include "ocean/base/storage.h"
#include "ocean/base/tensor.h"
#include "ocean/base/index.h"
#include "ocean/base/api.h"


#define OC_TENSOR_CAST_NONE    0
#define OC_TENSOR_CAST_DOUBLE  1
#define OC_TENSOR_CAST_COMPLEX 2


/* ===================================================================== */
/* Function declarations                                                 */
/* ===================================================================== */

/* Configuration */
OC_API int       OcTensor_getAutoBroadcastMode(void);
OC_API void      OcTensor_setAutoBroadcastMode(int flag);
OC_API int       OcTensor_getAutoTypecastMode (void);
OC_API void      OcTensor_setAutoTypecastMode (int flag);
OC_API int       OcTensor_checkAutoTypecast   (OcTensor *tensor, OcDType dtype, OcDevice *device);
OC_API char      OcTensor_getDefaultMathMode  (void);
OC_API int       OcTensor_setDefaultMathMode  (char mode);
OC_API int       OcTensor_validateMathMode    (char *mode);

/* Tensor creation */
OC_API OcTensor *OcTensor_create              (int ndims, OcSize *size, OcIndex *strides, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_createTemporary     (int ndims, OcSize *size, OcIndex *strides, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_createWithStream    (int ndims, OcSize *size, OcIndex *strides, OcDType dtype, OcStream *stream);
OC_API OcTensor *OcTensor_createFromStorage   (OcStorage *storage, int ndims, OcSize *size, OcIndex *strides,
                                               OcIndex offset, OcDType dtype);
OC_API OcTensor *OcTensor_createFromScalar    (OcScalar *scalar, OcDType dtype, OcDevice *device, int flagTemporary);
OC_API OcTensor *OcTensor_createFromData      (void *data, OcSize size, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_createContiguousLike(OcTensor *tensor, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_createEmpty         (OcDType dtype, OcDevice *device);

/* Additional tensor creation */
OC_API OcTensor *OcTensor_zeros             (int ndims, OcSize *size, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_ones              (int ndims, OcSize *size, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_full              (int ndims, OcSize *size, OcScalar *value, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_emptyLike         (OcTensor *tensor);
OC_API OcTensor *OcTensor_zerosLike         (OcTensor *tensor);
OC_API OcTensor *OcTensor_onesLike          (OcTensor *tensor);
OC_API OcTensor *OcTensor_fullLike          (OcTensor *tensor, OcScalar *value);
OC_API OcTensor *OcTensor_eye               (OcSize rows, OcSize columns, OcIndex index, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_diagonal          (OcTensor *tensor, OcIndex index, OcDType dtype, OcDevice *device);

/* Generic tensor operations */
OC_API int       OcTensor_detach            (OcTensor **tensor);
OC_API int       OcTensor_detachTensor      (OcTensor **tensor);
OC_API int       OcTensor_detachStorage     (OcTensor *tensor);
OC_API int       OcTensor_copy              (OcTensor *src, OcTensor *dst);
OC_API OcTensor *OcTensor_clone             (OcTensor *tensor);
OC_API OcTensor *OcTensor_cloneTo           (OcTensor *tensor, OcDevice *device);
OC_API OcTensor *OcTensor_cloneFlags        (OcTensor *tensor, OcDType dtype, OcDevice *device, int flagByteswapped, int flagTemporary);
OC_API OcTensor *OcTensor_replicate         (OcTensor *tensor);
OC_API OcTensor *OcTensor_replicateTo       (OcTensor *tensor, OcDevice *device);
OC_API OcTensor *OcTensor_contiguous        (OcTensor *tensor);
OC_API OcTensor *OcTensor_contiguousType    (OcTensor *tensor, char type);
OC_API int       OcTensor_toScalar          (OcTensor *tensor, OcScalar *scalar);

/* Ensure the data type or device - new or incref      */
/* The input tensor reference count is decrements when */
/* the operation failed and the result pointer is NULL */
OC_API int       OcTensor_ensureDType       (OcTensor **tensorPtr, OcDType dtype, OcTensor **result);
OC_API int       OcTensor_ensureDevice      (OcTensor **tensorPtr, OcDevice *device, OcTensor **result);
OC_API int       OcTensor_ensure            (OcTensor **tensorPtr, OcDType dtype, OcDevice *device, OcTensor **result);
OC_API int       OcTensor_ensureFlags       (OcTensor **tensorPtr, OcDType dtype, OcDevice *device, OcTensor **result, int flagTemporary);
OC_API int       OcTensor_ensureByteOrder   (OcTensor **tensorPtr, OcTensor **result);

/* Cast - new or incref */
OC_API OcTensor *OcTensor_castDType         (OcTensor *tensor, OcDType dtype);
OC_API OcTensor *OcTensor_castDevice        (OcTensor *tensor, OcDevice *device);
OC_API OcTensor *OcTensor_cast              (OcTensor *tensor, OcDType dtype, OcDevice *device);

/* Byte order */
OC_API int       OcTensor_byteswap          (OcTensor *tensor);
OC_API int       OcTensor_byteswapNoFlag    (OcTensor *tensor);
OC_API int       OcTensor_hasHostByteOrder  (OcTensor *tensor);
OC_API void      OcTensor_setByteswapped    (OcTensor *tensor, int flag);
OC_API int       OcTensor_isByteswapped     (OcTensor *tensor);

/* Shape and layout - when the result pointer is NULL the original tensor pointer */
/* is replaced by the result of the operation. In case of error, this means that  */
/* the reference cound of the original tensor is decremented and the *tensorPtr   */
/* is assigned the NULL value. When the result pointer is set, the result will be */
/* assigned to the given pointer. In case of failure *result will be set to NULL. */
OC_API int       OcTensor_reshape           (OcTensor **tensorPtr, int ndims, OcSize *size, OcTensor **result);
OC_API int       OcTensor_reshapeLike       (OcTensor **tensorPtr, OcTensor *reference, OcTensor **result);
OC_API int       OcTensor_broadcastTo       (OcTensor **tensorPtr, int ndims, OcSize *size, int mode, OcTensor **result);
OC_API int       OcTensor_broadcastLike     (OcTensor **tensorPtr, OcTensor *reference, int mode, OcTensor **result);
OC_API int       OcTensor_flipAxis          (OcTensor **tensorPtr, int axis, OcTensor **result);
OC_API int       OcTensor_fliplr            (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_flipud            (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_transpose         (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_ctranspose        (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_swapAxes          (OcTensor **tensorPtr, int axis1, int axis2, OcTensor **result);
OC_API int       OcTensor_reverseAxes       (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_reverseAxes2      (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_permuteAxes       (OcTensor **tensorPtr, int ndims, OcIndex *dims, OcTensor **result);
OC_API int       OcTensor_squeeze           (OcTensor **tensorPtr, OcTensor **result);
OC_API int       OcTensor_squeezeDim        (OcTensor **tensorPtr, int dim, OcTensor **result);
OC_API int       OcTensor_unsqueezeDim      (OcTensor **tensorPtr, int dim, OcTensor **result);
OC_API int       OcTensor_flatten           (OcTensor **tensorPtr, char type, OcTensor **result);

/* Tensor indexing */
OC_API OcTensor *OcTensor_getIndex          (OcTensor *tensor, OcTensorIndex *index);
OC_API OcTensor *OcTensor_getIndexFlags     (OcTensor *tensor, OcTensorIndex *index, int *flagScalar, int *flagView);
OC_API int       OcTensor_setIndex          (OcTensor *tensor, OcTensorIndex *index, OcTensor *value);
OC_API int       OcTensor_fillIndex         (OcTensor *tensor, OcTensorIndex *index, OcScalar *value);
OC_API int       OcTensor_getIndexValue     (OcTensor *tensor, OcIndex *indices, OcScalar *value);
OC_API int       OcTensor_setIndexValue     (OcTensor *tensor, OcIndex *indices, OcScalar *value);

/* Tensor indexing - primitives */
OC_API OcTensor *OcTensor_find              (OcTensor *tensor);
OC_API OcTensor *OcTensor_maskToOffset      (OcTensor *tensor, OcIndex *strides);
OC_API int       OcTensor_indexToOffset     (OcTensor *tensor, OcIndex *strides, OcTensor **result);
OC_API int       OcTensor_addIfNegative     (OcTensor *tensor, OcScalar *value);

/* Tensor extraction */
OC_API OcTensor *OcTensor_diag              (OcTensor *tensor, OcIndex offset, int axis1, int axis2);
OC_API OcTensor *OcTensor_real              (OcTensor *tensor);
OC_API OcTensor *OcTensor_imag              (OcTensor *tensor);
OC_API OcTensor *OcTensor_slice             (OcTensor *tensor, int axis, OcSize offset, OcSize size);

/* Basic tensor operations */
OC_API int       OcTensor_zero              (OcTensor *tensor);
OC_API int       OcTensor_fillOnes          (OcTensor *tensor);
OC_API int       OcTensor_fill              (OcTensor *tensor, OcScalar *value);
OC_API int       OcTensor_fillNaN           (OcTensor *tensor, OcScalar *value);
OC_API int       OcTensor_fillDouble        (OcTensor *tensor, OcDouble value);
OC_API int       OcTensor_fillInt64         (OcTensor *tensor, OcInt64 value);
OC_API int       OcTensor_fillUInt64        (OcTensor *tensor, OcUInt64 value);
OC_API int       OcTensor_maskedFill        (OcTensor *tensor, OcTensor *mask, OcScalar *value);
OC_API OcTensor *OcTensor_range             (OcScalar *start, OcScalar *stop, OcScalar *step, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_rangeDouble       (OcDouble start, OcDouble stop, OcDouble step, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_rangeInt64        (OcInt64 start, OcInt64 stop, OcInt64 step, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_linspace          (OcScalar *start, OcScalar *stop, OcSize nSamples, OcSize nIntervals,
                                             OcScalar *spacing, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_linspaceDouble    (OcDouble start, OcDouble stop, OcSize nSamples, OcSize nIntervals,
                                             OcScalar *spacing, OcDType dtype, OcDevice *device);
OC_API OcTensor *OcTensor_linspaceCDouble   (OcCDouble start, OcCDouble stop, OcSize nSamples, OcSize nIntervals,
                                             OcScalar *spacing, OcDType dtype, OcDevice *device);

/* Tensor element properties */
OC_API int       OcTensor_isinf             (OcTensor *src, OcTensor **dst);
OC_API int       OcTensor_isnan             (OcTensor *src, OcTensor **dst);
OC_API int       OcTensor_isneginf          (OcTensor *src, OcTensor **dst);
OC_API int       OcTensor_isposinf          (OcTensor *src, OcTensor **dst);
OC_API int       OcTensor_isfinite          (OcTensor *src, OcTensor **dst);

/* Unary tensor operations */
#define OC_TEMPLATE(NAME, X, CHECK, Y) OC_TEMPLATE_B##CHECK(NAME)
#define OC_TEMPLATE_B0(NAME) \
OC_API int       OcTensor_##NAME            (OcTensor *src, OcTensor **dst);
#define OC_TEMPLATE_B1(NAME) \
OC_API int       OcTensor_##NAME            (OcTensor *src, OcTensor **dst, char mode);
#include "ocean/core/generic/generate_tensor_unary.h"
#undef OC_TEMPLATE_B0
#undef OC_TEMPLATE_B1
#undef OC_TEMPLATE

/* Binary tensor operations */
#define OC_TEMPLATE(NAME, X, CHECK, Y) OC_TEMPLATE_B##CHECK(NAME)
#define OC_TEMPLATE_B0(NAME) \
OC_API int       OcTensor_##NAME            (OcTensor *src1, OcTensor *src2, OcTensor **dst);
#define OC_TEMPLATE_B1(NAME) \
OC_API int       OcTensor_##NAME            (OcTensor *src1, OcTensor *src2, OcTensor **dst, char mode);
#include "ocean/core/generic/generate_tensor_binary.h"
#undef OC_TEMPLATE_B0
#undef OC_TEMPLATE_B1
#undef OC_TEMPLATE

/* Binary tensor operations with scalar */
OC_API int       OcTensor_addScalar          (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_addScalarIfNegative(OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_subtractScalar     (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_multiplyScalar     (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_divideScalar       (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_trueDivideScalar   (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_floorDivideScalar  (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_modScalar          (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_fmodScalar         (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_bitwiseAndScalar   (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_bitwiseOrScalar    (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_bitwiseXorScalar   (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_bitshiftLeftScalar (OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_bitshiftRightScalar(OcTensor *src, OcScalar *s, OcTensor **dst);
OC_API int       OcTensor_powerScalar        (OcTensor *src, OcScalar *s, OcTensor **dst, char mode);

/* Tensor domain reductions */
OC_API int       OcTensor_anyEQZero           (OcTensor *tensor);
OC_API int       OcTensor_anyLTZero           (OcTensor *tensor);
OC_API int       OcTensor_anyLTOne            (OcTensor *tensor);
OC_API int       OcTensor_anyLTNegOne         (OcTensor *tensor);
OC_API int       OcTensor_anyGTOneAbs         (OcTensor *tensor);
OC_API int       OcTensor_allLessThan         (OcTensor *tensor, OcScalar *value);
OC_API int       OcTensor_allLessEqual        (OcTensor *tensor, OcScalar *value);
OC_API int       OcTensor_allGreaterThan      (OcTensor *tensor, OcScalar *value);
OC_API int       OcTensor_allGreaterEqual     (OcTensor *tensor, OcScalar *value);
OC_API int       OcTensor_allInRange          (OcTensor *tensor, OcScalar *lower, int lowerInclusive,
                                                                 OcScalar *upper, int upperInclusive);

/* Tensor global reduction */
OC_API int       OcTensor_any                (OcTensor *tensor, int *result);
OC_API int       OcTensor_all                (OcTensor *tensor, int *result);
OC_API int       OcTensor_allFinite          (OcTensor *tensor, int *result);
OC_API int       OcTensor_anyInf             (OcTensor *tensor, int *result);
OC_API int       OcTensor_anyNaN             (OcTensor *tensor, int *result);
OC_API int       OcTensor_nnz                (OcTensor *tensor, OcUInt64 *result);
OC_API int       OcTensor_nnzNaN             (OcTensor *tensor, OcUInt64 *result);
OC_API int       OcTensor_sum                (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_prod               (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_sumNaN             (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_prodNaN            (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_sumAbs             (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_sumAbsNaN          (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_maximum            (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_minimum            (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_maximumAbs         (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_minimumAbs         (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_norm               (OcTensor *tensor, double p, OcScalar *result);
OC_API int       OcTensor_normNaN            (OcTensor *tensor, double p, OcScalar *result);
OC_API int       OcTensor_norm1              (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_norm2              (OcTensor *tensor, OcScalar *result);
OC_API int       OcTensor_normInf            (OcTensor *tensor, OcScalar *result);

/* Tensor single axis reduction */
OC_API int       OcTensor_axisAny            (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisAll            (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisAllFinite      (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisAnyInf         (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisAnyNaN         (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNnz            (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNnzNaN         (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisSum            (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisProd           (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisSumNaN         (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisProdNaN        (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisSumAbs         (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisSumAbsNaN      (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisMaximum        (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisMinimum        (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisMaximumAbs     (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisMinimumAbs     (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNorm           (OcTensor *src, double p, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNormNaN        (OcTensor *src, double p, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNorm1          (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNorm2          (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);
OC_API int       OcTensor_axisNormInf        (OcTensor *src, int n, int *axes, int keepdims, OcTensor **dst);

/* Tensor multiplication */
OC_API int       OcTensor_mtimes             (OcTensor *src1, OcTensor *src2, OcTensor **dst);
OC_API int       OcTensor_gemm               (OcScalar *alpha, OcTensor *A, char modeA, OcTensor *B, char modeB,
                                              OcScalar *beta,  OcTensor **ptrC);
OC_API int       OcTensor_bcastgemm          (OcTensor *alpha, OcTensor *A, char modeA, OcTensor *B, char modeB,
                                              OcTensor *beta, OcTensor **ptrC);
OC_API int       OcTensor_gemmSupportedOn    (OcDevice *device, OcDType dtype);

/* Tensor dimension broadcasting */
OC_API OcTensor *OcTensor_autoBroadcastLike  (OcTensor *tensor, OcTensor *reference); /* New or incref */
OC_API int       OcTensor_canAutoBroadcast   (OcTensor *tensor, OcTensor *reference);
OC_API int       OcTensor_canBroadcast       (OcTensor *tensor, OcTensor *reference);

/* Internal operations */
OC_API int        OcTensor_intrnlCopy        (OcTensor *src, OcTensor *dst);
OC_API int        OcTensor_intrnlCopyDevices (OcTensor *src, OcTensor *dst);
OC_API int        OcTensor_setResult         (OcTensor **tensorPtr, OcTensor **result, OcTensor *tensor, int status);

/* Tensor formatting */
OC_API int       OcTensor_format (OcTensor *tensor, char **str, const char *header, const char *footer);
OC_API int       OcTensor_formatFooter(OcTensor *tensor, char **str, const char *pre, const char *post);
OC_API int       OcTensor_display(OcTensor *tensor);
OC_API void      OcTensor_displayShape(OcTensor *tensor);
OC_API int       OcTensor_displayFooter(OcTensor *tensor);

/* Module initialization */
OC_API int       OcModuleCore_initializeTensorItf(void);

/* Tensor warnings - internal usage only */
OC_API int oc_tensor_warning_reciprocal;
OC_API int oc_tensor_warning_sqrt;
OC_API int oc_tensor_warning_arcsin;
OC_API int oc_tensor_warning_arccos;
OC_API int oc_tensor_warning_arccosh;
OC_API int oc_tensor_warning_arctanh;
OC_API int oc_tensor_warning_log;
OC_API int oc_tensor_warning_log2;
OC_API int oc_tensor_warning_log10;
OC_API int oc_tensor_warning_log1p;
OC_API int oc_tensor_warning_modulo_zero;

#endif

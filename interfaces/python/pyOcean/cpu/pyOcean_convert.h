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

#ifndef __PYOC_CONVERT_H__
#define __PYOC_CONVERT_H__

#include "ocean.h"

#include <Python.h>


/* ===================================================================== */
/* Types definitions for tensor import and export                        */
/* ===================================================================== */

typedef int (*pyOcean_fptrIsScalar       )(PyObject *);
typedef int (*pyOcean_fptrIsTensor       )(PyObject *);
typedef int (*pyOcean_fptrIsScalarTensor )(PyObject *);

typedef int (*pyOcean_fptrGetScalar      )(PyObject *, OcScalar *);
typedef int (*pyOcean_fptrGetScalarTensor)(PyObject *, OcTensor **);
typedef int (*pyOcean_fptrGetTensor      )(PyObject *, OcTensor **);

typedef int (*pyOcean_fptrGetScalarType  )(PyObject *, OcDType *, int);
typedef int (*pyOcean_fptrGetTensorLayout)(PyObject *, int *, OcSize *, OcIndex *, OcDType *, OcDevice **, int);
typedef int (*pyOcean_fptrExportTensor   )(OcTensor *, PyObject **, int);


typedef struct __pyOceanConvert
{  pyOcean_fptrIsScalar         isScalar;
   pyOcean_fptrIsTensor         isTensor;
   pyOcean_fptrIsScalarTensor   isScalarTensor;
   pyOcean_fptrGetScalar        getScalar;
   pyOcean_fptrGetScalarTensor  getScalarTensor;
   pyOcean_fptrGetTensor        getTensor;
   pyOcean_fptrGetScalarType    getScalarType;
   pyOcean_fptrGetTensorLayout  getTensorLayout;
   pyOcean_fptrExportTensor     exportTensor;
   
   const char                  *name;
   struct __pyOceanConvert     *next;
} pyOceanConvert;


/* Type-check functions */
int pyOcean_isScalar           (PyObject *obj);   /* Scalar type only                            */
int pyOcean_isScalarLike       (PyObject *obj);   /* Scalar type or tensor-like with one element */
int pyOcean_isScalarTensor     (PyObject *obj);   /* Tensor type with one element only           */
int pyOcean_isScalarTensorLike (PyObject *obj);   /* Tensor-like type with one element           */
int pyOcean_isWeakScalar       (PyObject *obj);   /* Weakly-typed scalar only                    */
int pyOcean_isTensor           (PyObject *obj);   /* Tensor type only                            */
int pyOcean_isTensorLike       (PyObject *obj);   /* Tensor-like type, including scalars         */
int pyOcean_isTensorLikeOnly   (PyObject *obj);   /* Tensor-like type, excluding scalar          */

/* Conversion functions */
int pyOcean_getScalar             (PyObject *obj, OcScalar  *scalar);
int pyOcean_getScalarLike         (PyObject *obj, OcScalar  *scalar);
int pyOcean_getScalarTensor       (PyObject *obj, OcTensor **tensor);
int pyOcean_getScalarTensorLike   (PyObject *obj, OcScalar  *scalar);
int pyOcean_getTensor             (PyObject *obj, OcTensor **tensor);
int pyOcean_getTensorLike         (PyObject *obj, OcTensor **tensor, OcDType dtype, OcDevice *device);
int pyOcean_getTensorLikeOnly     (PyObject *obj, OcTensor **tensor, OcScalar *fill, OcDType dtype, OcDevice *device, char order);
int pyOcean_intrnlGetTensorLike   (PyObject *obj, OcTensor **tensor, OcScalar *fill, OcDType dtype, OcDevice *device, char order);

/* Tensor indexing */
int  pyOcean_convertIndex         (PyObject *obj, OcTensorIndex *index);
int  pyOcean_convertIndices       (PyObject *args, OcTensorIndex **index);
int  pyOcean_convertScalarIndices (PyObject *args, int ndims, OcIndex *indices);

/* Data type and layout functions */
int pyOcean_getScalarType         (PyObject *obj, OcDType *dtype, int verbose);
int pyOcean_getTensorLayout       (PyObject *obj, int *ndims, OcSize *size, OcIndex *strides,
                                   OcDType *dtype, OcDevice **device, int verbose);
int pyOcean_getTensorLikeLayout   (PyObject *obj, int *ndims, OcSize *size,
                                   OcDType *dtype, OcDevice **device, int padded, int verbose);

/* Export functions */
int pyOcean_exportTensor          (OcTensor *tensor, const char *name, PyObject **obj, int deepcopy);

/* Management functions */
int       pyOcean_registerConverter(pyOceanConvert *type);
PyObject *pyOcean_getImportTypes(void);
PyObject *pyOcean_getExportTypes(void);

#endif

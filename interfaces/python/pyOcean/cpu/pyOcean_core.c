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

#include "pyOcean_core.h"
#include "pyOcean_convert.h"
#include "pyOcean_scalar.h"
#include "pyOcean_tensor.h"
#include "pyOcean_storage.h"
#include "pyOcean_index.h"


/* --------------------------------------------------------------------------------------- */
PyObject *pyOceanCore_intrnl_ensure(PyObject *obj, OcDType dtype, OcDevice *device, int flagInplace)
/* --------------------------------------------------------------------------------------- */
{  PyObject      *result = NULL;
   OcStorage     *storage = NULL;
   OcTensor      *tensor = NULL;
   OcTensorIndex *index = NULL;
   OcScalar      *scalar;
   int            status;

   /* Process the different object types */
   if (PyOceanStorage_Check(obj))
   {  storage = ((pyOcStorage *)obj) -> storage;
      if (dtype == OcDTypeNone) dtype = (OcStorage_isRaw(storage)) ? OcDTypeNone : storage -> dtype;
      status = OcStorage_ensure(&(((pyOcStorage *)obj) -> storage), dtype, device, (flagInplace ? NULL : &storage));
      if (status == 0)
      {  if (flagInplace)
         {  Py_INCREF(Py_None); result = Py_None;
         }
         else
         {  result = PyOceanStorage_Wrap(storage);
         }
      }
   }
   else if (PyOceanTensorIndex_Check(obj))
   {  /* Make sure dtype is none */
      if (dtype != OcDTypeNone)
         OcError(NULL, "Type casting does not apply to tensor index objects");

      /* Create a new index object */
      status = OcTensorIndex_setDevice(&(PYOC_GET_TENSOR_INDEX(obj)), device, &index);
      if (status == 0)
      {  if (flagInplace)
         {  OcDecrefTensorIndex(PYOC_GET_TENSOR_INDEX(obj));
            PYOC_GET_TENSOR_INDEX(obj) = index;
            Py_INCREF(Py_None); result = Py_None;
         }
         else
         {  result = PyOceanTensorIndex_Wrap(index);
         }
      }
   }
   else if (flagInplace)
   {  /* Object must be a tensor or scalar */
      if (PyOceanScalar_Check(obj))
      {  if (device != NULL)
         {  OcErrorMessage("In-place ensure device does not apply to scalar objects");
         }
         else
         {  /* In-place type conversion of scalar */
            scalar = PYOC_GET_SCALAR(obj);
            OcScalar_castTo(scalar, dtype, NULL);
            Py_INCREF(Py_None); result = Py_None;
         }
      }
      else if (PyOceanTensor_Check(obj))
      {  /* Ensure the correct device */
         status = OcTensor_ensure(&(((pyOcTensor *)obj) -> tensor), dtype, device, NULL);
         if (status == 0)
         {  Py_INCREF(Py_None); result = Py_None;
         }
      }
      else
      {  OcErrorMessage("In-place ensure applies only to tensor, scalar, and storage objects");
      }
   }
   else
   {  /* Object must be tensor-like (which includes scalars) */
      if ((device == NULL) && (pyOcean_isScalar(obj)))
      {  /* Return a scalar object */
         OcScalar s;
         if (pyOcean_getScalar(obj, &s) == 1)
         {  OcScalar_castTo(&s, dtype, NULL);
            result = PyOceanScalar_New(&s);
         }
      }
      else if (pyOcean_getTensorLike(obj, &tensor, dtype, device) == 1)
      {  /* Detach the tensor object */
         status = OcTensor_detachTensor(&tensor);
         if (status == 0)
              result = PyOceanTensor_Wrap(tensor);
         else OcDecrefTensor(tensor);
      }
   }

   return result;
}


/* --------------------------------------------------------------------------------------- */
PyObject *pyOceanCore_intrnl_cast(PyObject *obj, OcDType dtype, OcDevice *device)
/* --------------------------------------------------------------------------------------- */
{  PyObject    *result = NULL;
   OcStorage   *storage = NULL;
   OcTensor    *tensor = NULL;

   /* Process the different object types */
   if (PyOceanStorage_Check(obj))
   {  storage = ((pyOcStorage *)obj) -> storage;
      if (dtype == OcDTypeNone) dtype = (OcStorage_isRaw(storage)) ? OcDTypeNone : storage -> dtype;
      storage = OcStorage_cast(((pyOcStorage *)obj) -> storage, dtype, device);
      result = PyOceanStorage_Wrap(storage);
   }
   else
   {  /* Object must be tensor-like (which includes scalars) */
      if ((device == NULL) && (pyOcean_isScalar(obj)))
      {  /* Return a scalar object */
         OcScalar s;
         if (pyOcean_getScalar(obj, &s) == 1)
         {  OcScalar_castTo(&s, dtype, NULL);
            result = PyOceanScalar_New(&s);
         }
      }
      else if (pyOcean_getTensorLike(obj, &tensor, dtype, device) == 1)
      {  /* Detach the tensor object */
         if (OcTensor_detach(&tensor) == 0)
              result = PyOceanTensor_Wrap(tensor);
         else OcDecrefTensor(tensor);
      }
   }

   return result;
}



/* --------------------------------------------------------------------------------------- */
/* PyObject *pyOceanCore_intrnl_add        (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_subtract   (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_scale      (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_divide     (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_floorDivide(PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_mod        (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_fmod       (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_min        (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_max        (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_fmin       (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_fmax       (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_logicalAnd (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_logicalOr  (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* PyObject *pyOceanCore_intrnl_logicalXor (PyObject *src1, PyObject *src2, OcTensor *dst) */
/* --------------------------------------------------------------------------------------- */
#define OC_TEMPLATE(OP, OPS, TENSOR_OP) \
PyObject *pyOceanCore_intrnl_##OP(PyObject *src1, PyObject *src2, OcTensor *dst) \
{  OcTensor *tensor1 = NULL, *tensor2 = NULL; \
   OcScalar  scalar1, scalar2, scalar3; \
   OcDType   dtype; \
   int       status, flagDest = (dst != NULL); \
   int       result = -1; \
   \
   /* Scalar operations */ \
   if ((dst == NULL) && (pyOcean_isScalar(src1)) && (pyOcean_isScalar(src2))) \
   {  /* Scalar inputs */ \
      if ((pyOcean_getScalar(src1, &scalar1) == 1) && \
          (pyOcean_getScalar(src2, &scalar2) == 1))   \
      {  status = OcScalar_##OPS(&scalar1, &scalar2, &scalar3); \
         return (status == 0) ? PyOceanScalar_New(&scalar3) : NULL; \
      } \
      else \
      {  OcError(NULL, "Error parsing scalar parameters"); \
      } \
   } \
   \
   /* Tensor operations */ \
   else \
   {  /* Tensor or scalar #1 */ \
      if (!pyOcean_isWeakScalar(src1)) \
           status = pyOcean_getTensorLike(src1, &tensor1, OcDTypeNone, NULL); \
      else status = pyOcean_getScalar(src1, &scalar1); \
      if (status != 1) { OcErrorMessage("Error parsing first input argument"); goto final; } \
      \
      /* Tensor or scalar #2 */ \
      if (!pyOcean_isWeakScalar(src2)) \
           status = pyOcean_getTensorLike(src2, &tensor2, OcDTypeNone, NULL); \
      else status = pyOcean_getScalar(src2, &scalar2); \
      if (status != 1) { OcErrorMessage("Error parsing second input argument"); goto final; } \
      \
      /* Convert scalars to tensors */ \
      if (tensor1 == NULL) \
      {  dtype = OcScalar_getCommonType(&scalar1, (tensor2) ? tensor2 -> dtype : scalar2.dtype); \
         tensor1 = OcTensor_createFromScalar(&scalar1, dtype, (tensor2) ? tensor2 -> device : OcCPU, 1); \
         if (tensor1 == NULL) goto final; \
      } \
      if (tensor2 == NULL) \
      {  status = pyOcean_getScalar(src2, &scalar2); \
         dtype = OcScalar_getCommonType(&scalar2, (tensor1) ? tensor1 -> dtype : scalar1.dtype); \
         tensor2 = OcTensor_createFromScalar(&scalar2, dtype, (tensor1) ? tensor1 -> device : OcCPU, 1); \
         if (tensor2 == NULL) goto final; \
      } \
      \
      /* Apply the operation */ \
      result = OcTensor_##TENSOR_OP(tensor1, tensor2, &dst); \
      \
   } \
\
final : ; \
   /* Free the tensors */ \
   OcXDecrefTensor(tensor1); \
   OcXDecrefTensor(tensor2); \
   \
   if (result != 0) return NULL; \
   if (!flagDest) return PyOceanTensor_Wrap(dst); else Py_RETURN_NONE; \
}

OC_TEMPLATE(add,           add,           add          )
OC_TEMPLATE(subtract,      subtract,      subtract     )
OC_TEMPLATE(scale,         multiply,      scale        )
OC_TEMPLATE(divide,        divide,        divide       )
OC_TEMPLATE(trueDivide,    trueDivide,    trueDivide   )
OC_TEMPLATE(floorDivide,   floorDivide,   floorDivide  )
OC_TEMPLATE(mod,           mod,           mod          )
OC_TEMPLATE(fmod,          fmod,          fmod         )
OC_TEMPLATE(min,           min,           elemwiseMin  )
OC_TEMPLATE(max,           max,           elemwiseMax  )
OC_TEMPLATE(fmin,          fmin,          elemwiseFMin )
OC_TEMPLATE(fmax,          fmax,          elemwiseFMax )
OC_TEMPLATE(bitwiseAnd,    bitwiseAnd,    bitwiseAnd   )
OC_TEMPLATE(bitwiseOr,     bitwiseOr,     bitwiseOr    )
OC_TEMPLATE(bitwiseXor,    bitwiseXor,    bitwiseXor   )
OC_TEMPLATE(logicalAnd,    logicalAnd,    logicalAnd   )
OC_TEMPLATE(logicalOr,     logicalOr,     logicalOr    )
OC_TEMPLATE(logicalXor,    logicalXor,    logicalXor   )
OC_TEMPLATE(bitshiftLeft,  bitshiftLeft,  bitshiftLeft )
OC_TEMPLATE(bitshiftRight, bitshiftRight, bitshiftRight)
#undef OC_TEMPLATE


/* ----------------------------------------------------------------------------------------- */
PyObject *pyOceanCore_intrnl_power(PyObject *src1, PyObject *src2, OcTensor *dst, char mode)
/* ----------------------------------------------------------------------------------------- */
{  OcTensor *tensor1 = NULL, *tensor2 = NULL;
   OcScalar  scalar1, scalar2, scalar3;
   OcDType   dtype;
   int       status, flagDest = (dst != NULL);
   int       result = -1;

   /* Scalar operations */
   if ((dst == NULL) && (pyOcean_isScalar(src1)) && (pyOcean_isScalar(src2)))
   {  /* Scalar inputs */
      if ((pyOcean_getScalar(src1, &scalar1) == 1) &&
          (pyOcean_getScalar(src2, &scalar2) == 1))
      {  status = OcScalar_power(&scalar1, &scalar2, &scalar3);
         return (status == 0) ? PyOceanScalar_New(&scalar3) : NULL;
      }
      else
      {  OcError(NULL, "Error parsing scalar parameters");
      }
   }

   /* Tensor operations */
   else
   {  /* Tensor or scalar #1 */
      if (!pyOcean_isWeakScalar(src1))
           status = pyOcean_getTensorLike(src1, &tensor1, OcDTypeNone, NULL);
      else status = pyOcean_getScalar(src1, &scalar1);
      if (status != 1) { OcErrorMessage("Error parsing first input argument"); goto final; }

      /* Tensor or scalar #2 */ \
      if (!pyOcean_isWeakScalar(src2))
           status = pyOcean_getTensorLike(src2, &tensor2, OcDTypeNone, NULL);
      else status = pyOcean_getScalar(src2, &scalar2);
      if (status != 1) { OcErrorMessage("Error parsing second input argument"); goto final; }

      /* Convert scalars to tensors */
      if (tensor1 == NULL)
      {  /* Scalar ** tensor */
         dtype = scalar1.dtype;
         tensor1 = OcTensor_createFromScalar(&scalar1, dtype, tensor2 -> device, 1);
         if (tensor1 == NULL) goto final;
      }
      if (tensor2 == NULL)
      {  /* Tensor ** scalar */
         if (OcDType_isInteger(tensor1 -> dtype) && OcDType_isInteger(scalar2.dtype))
              dtype = OcDTypeInt16;
         else dtype = OcScalar_getCommonType(&scalar2, tensor1 -> dtype);
         tensor2 = OcTensor_createFromScalar(&scalar2, dtype, tensor1 -> device, 1);
         if (tensor2 == NULL) goto final;
      }

      /* Apply the operation */
      result = OcTensor_power(tensor1, tensor2, &dst, mode);
   }

final : ;
   /* Free the tensors */
   OcXDecrefTensor(tensor1);
   OcXDecrefTensor(tensor2);

   if (result != 0) return NULL;
   if (!flagDest) return PyOceanTensor_Wrap(dst); else Py_RETURN_NONE;
}



/* ----------------------------------------------------------------- */
/* int pyOceanCore_intrnl_iadd        (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_isubtract   (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_iscale      (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_idivide     (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_ifloorDivide(OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_imod        (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_ibitwiseAnd (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_ibitwiseOr  (OcTensor *dst, PyObject *src) */
/* int pyOceanCore_intrnl_ibitwiseXor (OcTensor *dst, PyObject *src) */
/* ----------------------------------------------------------------- */
#define OC_TEMPLATE(OP, TENSOR_OP, SCALAR_OP) \
int pyOceanCore_intrnl_##OP(OcTensor *dst, PyObject *src) \
{  OcTensor *tensor = NULL; \
   OcScalar  scalar; \
   int       status, result = -1; \
   \
   /* Tensor or scalar */ \
   if (!pyOcean_isWeakScalar(src)) \
        status = pyOcean_getTensorLike(src, &tensor, OcDTypeNone, NULL); \
   else status = pyOcean_getScalar(src, &scalar); \
   if (status != 1) { OcErrorMessage("Error parsing the right-hand side"); goto final; } \
   \
   /* Apply the operation */ \
   if (tensor) \
        result = OcTensor_##TENSOR_OP(dst, tensor, &dst); \
   else result = OcTensor_##SCALAR_OP(dst,&scalar, &dst); \
\
final : ; \
   /* Free the tensor */ \
   OcXDecrefTensor(tensor); \
   \
   return result; \
}

OC_TEMPLATE(iadd,           add,           addScalar          )
OC_TEMPLATE(isubtract,      subtract,      subtractScalar     )
OC_TEMPLATE(iscale,         scale,         multiplyScalar     )
OC_TEMPLATE(idivide,        divide,        divideScalar       )
OC_TEMPLATE(ifloorDivide,   floorDivide,   floorDivideScalar  )
OC_TEMPLATE(imod,           mod,           modScalar          )
OC_TEMPLATE(ibitwiseAnd,    bitwiseAnd,    bitwiseAndScalar   )
OC_TEMPLATE(ibitwiseOr,     bitwiseOr,     bitwiseOrScalar    )
OC_TEMPLATE(ibitwiseXor,    bitwiseXor,    bitwiseXorScalar   )
OC_TEMPLATE(ibitshiftLeft,  bitshiftLeft,  bitshiftLeftScalar )
OC_TEMPLATE(ibitshiftRight, bitshiftRight, bitshiftRightScalar)
#undef OC_TEMPLATE


/* -------------------------------------------------------------------- */
int pyOceanCore_intrnl_ipower(OcTensor *dst, PyObject *src, char mode)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor = NULL;
   OcScalar  scalar;
   int       status, result = -1;

   /* Tensor or scalar */
   if (!pyOcean_isWeakScalar(src))
        status = pyOcean_getTensorLike(src, &tensor, OcDTypeNone, NULL);
   else status = pyOcean_getScalar(src, &scalar);
   if (status != 1) { OcErrorMessage("Error parsing the right-hand side"); goto final; }

   /* Apply the operation */
   if (tensor)
        result = OcTensor_power(dst, tensor, &dst, mode);
   else result = OcTensor_powerScalar(dst,&scalar, &dst, mode);

final : ;
   /* Free the tensor */
   OcXDecrefTensor(tensor);

   return result;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_intrnl_mtimes(PyObject *src1, PyObject *src2, OcTensor *dst)
/* -------------------------------------------------------------------- */
{  OcTensor *tensor1 = NULL, *tensor2 = NULL;
   int       flagDest = (dst != NULL);
   int       result = -1;

   /* Scalar multiplication */
   if (pyOcean_isScalar(src1) || pyOcean_isScalar(src2))
      return pyOceanCore_intrnl_scale(src1, src2, dst);

   /* Tensor operations */
   if ((pyOcean_getTensorLike(src1, &tensor1, OcDTypeNone, NULL) != 1) ||
       (pyOcean_getTensorLike(src2, &tensor2, OcDTypeNone, NULL) != 1))
   {  OcErrorMessage("Error parsing input parameters");
      goto final;
   }

   /* Call the multiplication function */
   result = OcTensor_gemm(NULL, tensor1, 'N', tensor2, 'N', NULL, &dst);

final : ;
   /* Free the tensors */
   OcXDecrefTensor(tensor1);
   OcXDecrefTensor(tensor2);

   if (result != 0) return NULL;
   if (!flagDest) return PyOceanTensor_Wrap(dst); else Py_RETURN_NONE;
}


/* -------------------------------------------------------------------- */
PyObject *pyOceanCore_intrnl_gemm(PyObject *alpha, OcTensor *A, char transA,
                                                   OcTensor *B, char transB,
                                  PyObject *beta, OcTensor *C)
/* -------------------------------------------------------------------- */
{  OcTensor *tensorAlpha = NULL, *tensorBeta = NULL;
   OcScalar *ptrAlpha, *ptrBeta;
   OcScalar  scalarAlpha, scalarBeta;
   OcDType   dtype, basetype;
   int       flagDest = (C != NULL);
   int       weakAlpha = 1, weakBeta = 1;
   int       result = -1;

   /* Determine the common data type */
   dtype  = OcDType_getCommonType(A -> dtype, B -> dtype);
   if (C) dtype = OcDType_getCommonType(dtype, C -> dtype);

   /* Scalar parameters */
   if (((alpha == NULL) || (pyOcean_isScalar(alpha))) &&
       ((beta  == NULL) || (pyOcean_isScalar(beta))))
   {  /* ---------------------------------------- */
      /* Gemm call with scalar alpha and beta     */
      /* ---------------------------------------- */

      /* Get the scalars */
      if (alpha)
      {  if (pyOcean_getScalar(alpha, &scalarAlpha) != 1)
        {  OcErrorMessage("Error parsing parameter alpha"); goto final;  }
         if ((weakAlpha = pyOcean_isWeakScalar(alpha)) != 0)
              dtype = OcScalar_getCommonType(&scalarAlpha, dtype);
         else dtype = OcDType_getCommonType(dtype, scalarAlpha.dtype);
      }
      if (beta)
      {  if (pyOcean_getScalar(beta,  &scalarBeta ) != 1)
         {  OcErrorMessage("Error parsing parameter beta"); goto final;  }
         if ((weakBeta = pyOcean_isWeakScalar(beta)) != 0)
              dtype = OcScalar_getCommonType(&scalarBeta, dtype);
         else dtype = OcDType_getCommonType(dtype, scalarBeta.dtype);
      }

      /* Get the base type */
      basetype = OcDType_getBaseType(dtype); /* Non-complex part */

      /* Update the scalars */
      if (alpha)
      {  if (weakAlpha)
              dtype = OcScalar_getCommonType(&scalarAlpha, basetype);
         else dtype = OcDType_getCommonType(basetype, scalarAlpha.dtype);
         OcScalar_castTo(&scalarAlpha, dtype, NULL);
      }
      if (beta)
      {  if (weakBeta)
              dtype = OcScalar_getCommonType(&scalarBeta, basetype);
         else dtype = OcDType_getCommonType(basetype, scalarBeta.dtype);
         OcScalar_castTo(&scalarBeta, dtype, NULL);
      }

      /* Set the pointers */
      ptrAlpha = (alpha) ? &scalarAlpha : NULL;
      ptrBeta  = (beta ) ? &scalarBeta  : NULL;

      /* Call the gemm function */
      result = OcTensor_gemm(ptrAlpha, A, transA, B, transB, ptrBeta, &C);
   }
   else
   {  /* ---------------------------------------- */
      /* Gemm call with tensor alpha and beta     */
      /* ---------------------------------------- */

      /* Parse tensor objects */
      if (alpha && (!pyOcean_isWeakScalar(alpha)))
      {  if (pyOcean_getTensorLike(alpha, &tensorAlpha, OcDTypeNone, NULL) != 1)
         {  OcErrorMessage("Error parsing parameter alpha"); goto final;  }
         dtype = OcDType_getCommonType(dtype, tensorAlpha -> dtype);
      }
      if (beta && (!pyOcean_isWeakScalar(beta)))
      {  if (pyOcean_getTensorLike(beta, &tensorBeta, OcDTypeNone, NULL) != 1)
         {  OcErrorMessage("Error parsing parameter beta"); goto final;  }
         dtype = OcDType_getCommonType(dtype, tensorBeta -> dtype);
      }

      /* Get the base type */
      basetype = OcDType_getBaseType(dtype); /* Non-complex part */

      /* Create the alpha tensor if needed */
      if (tensorAlpha == NULL)
      {  if (alpha)
         {  if (pyOcean_getScalar(alpha, &scalarAlpha) != 1)
            {  OcErrorMessage("Error parsing parameter alpha"); goto final;  }
            dtype = OcScalar_getCommonType(&scalarAlpha, basetype);
            OcScalar_castTo(&scalarAlpha, dtype, NULL);
         }
         else
         {  scalarAlpha.dtype = basetype;
            OcScalar_fromInt64(&scalarAlpha, 1);
         }

         /* Create the tensor */
         tensorAlpha = OcTensor_createFromScalar(&scalarAlpha, scalarAlpha.dtype, OcCPU, 1);
         if (tensorAlpha == NULL) goto final;
      }

      /* Create the beta tensor if needed */
      if ((tensorBeta == NULL) && (C != NULL))
      {  if (beta)
         {  if (pyOcean_getScalar(beta, &scalarBeta) != 1)
            {  OcErrorMessage("Error parsing parameter beta"); goto final;  }
            dtype = OcScalar_getCommonType(&scalarBeta, basetype);
            OcScalar_castTo(&scalarBeta, dtype, NULL);
         }
         else
         {  scalarBeta.dtype = basetype;
            OcScalar_fromInt64(&scalarBeta, 0);
         }

         /* Create the tensor */
         tensorBeta = OcTensor_createFromScalar(&scalarBeta, scalarBeta.dtype, OcCPU, 1);
         if (tensorBeta == NULL) goto final;
      }

      /* Call the broadcast gemm function */
      result = OcTensor_bcastgemm(tensorAlpha, A, transA, B, transB, tensorBeta, &C);
   }

final : ;

   /* Finalize tensors */
   OcXDecrefTensor(tensorAlpha);
   OcXDecrefTensor(tensorBeta);

   if (result != 0) return NULL;
   if (!flagDest) return PyOceanTensor_Wrap(C); else Py_RETURN_NONE;
}

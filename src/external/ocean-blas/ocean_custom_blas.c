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

#include "ocean/external/ocean-blas/ocean_blas.h"
#include "ocean_blas_config.h"


/* ===================================================================== */
/* Local type definitions                                                */
/* ===================================================================== */

typedef struct
{  float real;
   float imag;
} OcBlasCFloat;

typedef struct
{  double real;
   double imag;
} OcBlasCDouble;


/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
void ocblas_sgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  float alpha, const float *ptrA, int lda, const float *ptrB, int ldb,
                  float beta, float *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  const float *ptr;
   float       *ptrC_, v;
   int          i, j, t;

   /* Assumption: m, n, and k are nonnegative */

   /* Check if any operation is needed */
   if ((n <= 0) || (m <= 0) ||                      /* Empty output matrix */
       (((alpha == 0) || (k <= 0)) && (beta == 1))) /* Add nothing         */
   {  return ;
   }

   /* Normalize the ordering */
   if (order == OcBlasRowMajor)
   {
      /* Compute C = B(transB) * A(transA) - because of the row-major order */
      /* we can simply swap the order and leave the transformation flags as */
      /* they were and only swap the pointers and dimensions.               */
      ptr = ptrA; ptrA = ptrB; ptrB = ptr;
      t = m; m = n; n = t;
      t = lda; lda = ldb; ldb = t;
   }

   /* Scale C */
   if (alpha == 0)
   {  ptrC_ = ptrC;
      if (beta == 0)
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  ptrC_[i] = 0;
            }
            ptrC_ += ldc;
         }
      }
      else
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  ptrC_[i] *= beta;
            }
            ptrC_ += ldc;
         }
      }
      return ;
   }

   /* Matrix multiplications */
   if (transA == OcBlasNoTrans)
   {  if (transB == OcBlasNoTrans)
      {  /* C <- alpha * A * B + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (beta == 0)
            {  for (i = 0; i < m; i++) ptrC[i] = 0;
            }
            else if (beta != 1)
            {  for (i = 0; i < m; i++) ptrC[j] *= beta;
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA;
            for (t = 0; t < k; t++)
            {  v = alpha * ptrB[t];
               for (i = 0; i < m; i++)
               {  ptrC[i] += ptr[i] * v;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB += ldb; ptrC += ldc;
         }
      }
      else
      {  /* C <- alpha * A * B^T + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (beta == 0)
            {  for (i = 0; i < m; i++) ptrC[i] = 0;
            }
            else if (beta != 1)
            {  for (i = 0; i < m; i++) ptrC[j] *= beta;
            }


            /* Add the contributions from the j-th column of B */
            ptr = ptrA;
            for (t = 0; t < k; t++)
            {  v = alpha * ptrB[t*ldb];
               for (i = 0; i < m; i++)
               {  ptrC[i] += ptr[i] * v;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB ++; ptrC += ldc;
         }
      }
   }
   else
   {  if (transB == OcBlasNoTrans)
      {  /* C = A^T * B */
         for (j = 0; j < n; j++)
         {  ptr = ptrA;
            for (i = 0; i < m; i++)
            {  v = 0;
               for (t = 0; t < k; t++)
               {  v += ptr[t] * ptrB[t];
               }
               if (beta == 0)
               {  ptrC[i] = alpha * v;
               }
               else
               {  ptrC[i] = beta * ptrC[i] + alpha * v;
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB += ldb; ptrC += ldc;
         }
      }
      else
      {  /* C = A^T * B^T */
         for (j = 0; j < n; j++)
         {  ptr = ptrA;
            for (i = 0; i < m; i++)
            {  v = 0;
               for (t = 0; t < k; t++)
               {  v += ptr[t] * ptrB[t*ldb];
               }
               if (beta == 0)
               {  ptrC[i] = alpha * v;
               }
               else
               {  ptrC[i] = beta * ptrC[i] + alpha * v;
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB ++; ptrC += ldc;
         }
      }
   }
}



/* --------------------------------------------------------------------- */
void ocblas_dgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  double alpha, const double *ptrA, int lda, const double *ptrB, int ldb,
                  double beta, double *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  const double *ptr;
   double       *ptrC_, v;
   int           i, j, t;

   /* Assumption: m, n, and k are nonnegative */

   /* Check if any operation is needed */
   if ((n <= 0) || (m <= 0) ||                      /* Empty output matrix */
       (((alpha == 0) || (k <= 0)) && (beta == 1))) /* Add nothing         */
   {  return ;
   }

   /* Normalize the ordering */
   if (order == OcBlasRowMajor)
   {
      /* Compute C = B(transB) * A(transA) - because of the row-major order */
      /* we can simply swap the order and leave the transformation flags as */
      /* they were and only swap the pointers and dimensions.               */
      ptr = ptrA; ptrA = ptrB; ptrB = ptr;
      t = m; m = n; n = t;
      t = lda; lda = ldb; ldb = t;
   }

   /* Scale C */
   if (alpha == 0)
   {  ptrC_ = ptrC;
      if (beta == 0)
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  ptrC_[i] = 0;
            }
            ptrC_ += ldc;
         }
      }
      else
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  ptrC_[i] *= beta;
            }
            ptrC_ += ldc;
         }
      }
      return ;
   }

   /* Matrix multiplications */
   if (transA == OcBlasNoTrans)
   {  if (transB == OcBlasNoTrans)
      {  /* C <- alpha * A * B + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (beta == 0)
            {  for (i = 0; i < m; i++) ptrC[i] = 0;
            }
            else if (beta != 1)
            {  for (i = 0; i < m; i++) ptrC[j] *= beta;
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA;
            for (t = 0; t < k; t++)
            {  v = alpha * ptrB[t];
               for (i = 0; i < m; i++)
               {  ptrC[i] += ptr[i] * v;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB += ldb; ptrC += ldc;
         }
      }
      else
      {  /* C <- alpha * A * B^T + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (beta == 0)
            {  for (i = 0; i < m; i++) ptrC[i] = 0;
            }
            else if (beta != 1)
            {  for (i = 0; i < m; i++) ptrC[j] *= beta;
            }


            /* Add the contributions from the j-th column of B */
            ptr = ptrA;
            for (t = 0; t < k; t++)
            {  v = alpha * ptrB[t*ldb];
               for (i = 0; i < m; i++)
               {  ptrC[i] += ptr[i] * v;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB ++; ptrC += ldc;
         }
      }
   }
   else
   {  if (transB == OcBlasNoTrans)
      {  /* C = A^T * B */
         for (j = 0; j < n; j++)
         {  ptr = ptrA;
            for (i = 0; i < m; i++)
            {  v = 0;
               for (t = 0; t < k; t++)
               {  v += ptr[t] * ptrB[t];
               }
               if (beta == 0)
               {  ptrC[i] = alpha * v;
               }
               else
               {  ptrC[i] = beta * ptrC[i] + alpha * v;
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB += ldb; ptrC += ldc;
         }
      }
      else
      {  /* C = A^T * B^T */
         for (j = 0; j < n; j++)
         {  ptr = ptrA;
            for (i = 0; i < m; i++)
            {  v = 0;
               for (t = 0; t < k; t++)
               {  v += ptr[t] * ptrB[t*ldb];
               }
               if (beta == 0)
               {  ptrC[i] = alpha * v;
               }
               else
               {  ptrC[i] = beta * ptrC[i] + alpha * v;
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB ++; ptrC += ldc;
         }
      }
   }
}


/* --------------------------------------------------------------------- */
void ocblas_cgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  const void *alpha, const void *ptrA, int lda, const void *ptrB, int ldb,
                  const void *beta, void *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  const OcBlasCFloat *ptr, *ptrA_, *ptrB_;
   OcBlasCFloat       *ptrC_, v, va, vb;
   float               fr, fi;
   float               ar, ai, br, bi;
   int                 flagAlphaZero, flagBetaZero, flagBetaOne;
   int                 i, j, t;

   /* Assumption: m, n, and k are nonnegative */

   /* Get the alpha and beta values and flags */
   ptr = (const OcBlasCFloat *)alpha; ar = ptr -> real; ai = ptr -> imag;
   ptr = (const OcBlasCFloat *)beta;  br = ptr -> real; bi = ptr -> imag;

   flagAlphaZero = ((ar == 0) && (ai == 0)) ? 1 : 0;
   flagBetaZero  = ((br == 0) && (bi == 0)) ? 1 : 0;
   flagBetaOne   = ((br == 1) && (bi == 0)) ? 1 : 0;

   /* Check if any operation is needed */
   if ((n <= 0) || (m <= 0) ||                           /* Empty output matrix */
       (((flagAlphaZero) || (k <= 0)) && (flagBetaOne))) /* Add nothing         */
   {  return ;
   }

   /* Normalize the ordering */
   if (order == OcBlasRowMajor)
   {  /* Compute C = B(transB) * A(transA) - because of the row-major order */
      /* we can simply swap the order and leave the transformation flags as */
      /* they were and only swap the pointers and dimensions.               */
      ptrA_ = (const OcBlasCFloat *)ptrB;
      ptrB_ = (const OcBlasCFloat *)ptrA;
      t = m; m = n; n = t;
      t = lda; lda = ldb; ldb = t;
   }
   else
   {  ptrA_ = (const OcBlasCFloat *)ptrA;
      ptrB_ = (const OcBlasCFloat *)ptrB;
   }

   /* Scale C */
   ptrC_ = (OcBlasCFloat *)ptrC;
   if (flagAlphaZero)
   {  if (flagBetaZero)
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  ptrC_[i].real = 0;
               ptrC_[i].imag = 0;
            }
            ptrC_ += ldc;
         }
      }
      else
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  fr = ptrC_[i].real;
               fi = ptrC_[i].imag;
               ptrC_[i].real = br * fr - bi * fi;
               ptrC_[i].imag = bi * fr + br * fi;
            }
            ptrC_ += ldc;
         }
      }
      return ;
   }

   /* Matrix multiplications */
   if (transA == OcBlasNoTrans)
   {  if (transB == OcBlasNoTrans)
      {  /* C <- alpha * A * B + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (flagBetaZero)
            {  for (i = 0; i < m; i++)
               {  ptrC_[i].real = 0;
                  ptrC_[i].imag = 0;
               }
            }
            else if (!flagBetaOne)
            {  for (i = 0; i < m; i++)
               {  fr = ptrC_[j].real;
                  fi = ptrC_[j].imag;
                  ptrC_[i].real += br * fr - bi * fi;
                  ptrC_[i].imag += bi * fr + br * fi;
               }
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA_;
            for (t = 0; t < k; t++)
            {  v = ptrB_[t];
               fr = ar * v.real - ai * v.imag;
               fi = ai * v.real + ar * v.imag;
               for (i = 0; i < m; i++)
               {  v = ptr[i];
                  ptrC_[i].real += fr * v.real - fi * v.imag;
                  ptrC_[i].imag += fi * v.real + fr * v.imag;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ += ldb; ptrC_ += ldc;
         }
      }
      else if (transB == OcBlasTrans)
      {  /* C <- alpha * A * B^T + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (flagBetaZero)
            {  for (i = 0; i < m; i++)
               {  ptrC_[i].real = 0;
                  ptrC_[i].imag = 0;
               }
            }
            else if (!flagBetaOne)
            {  for (i = 0; i < m; i++)
               {  fr = ptrC_[j].real;
                  fi = ptrC_[j].imag;
                  ptrC_[i].real += br * fr - bi * fi;
                  ptrC_[i].imag += bi * fr + br * fi;
               }
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA_;
            for (t = 0; t < k; t++)
            {  v = ptrB_[t*ldb];
               fr = ar * v.real - ai * v.imag;
               fi = ai * v.real + ar * v.imag;
               for (i = 0; i < m; i++)
               {  v = ptr[i];
                  ptrC_[i].real += fr * v.real - fi * v.imag;
                  ptrC_[i].imag += fi * v.real + fr * v.imag;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
      else /* (transB == OcBlasConjTrans) */
      {  /* C <- alpha * A * B^H + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (flagBetaZero)
            {  for (i = 0; i < m; i++)
               {  ptrC_[i].real = 0;
                  ptrC_[i].imag = 0;
               }
            }
            else if (!flagBetaOne)
            {  for (i = 0; i < m; i++)
               {  fr = ptrC_[j].real;
                  fi = ptrC_[j].imag;
                  ptrC_[i].real += br * fr - bi * fi;
                  ptrC_[i].imag += bi * fr + br * fi;
               }
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA_;
            for (t = 0; t < k; t++)
            {  v = ptrB_[t*ldb];
               fr = ar * v.real + ai * v.imag;
               fi = ai * v.real - ar * v.imag;
               for (i = 0; i < m; i++)
               {  v = ptr[i];
                  ptrC_[i].real += fr * v.real - fi * v.imag;
                  ptrC_[i].imag += fi * v.real + fr * v.imag;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
   }
   else if (transA == OcBlasTrans)
   {  if (transB == OcBlasNoTrans)
      {  /* C = A^T * B */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t];
                  v.real += va.real * vb.real - va.imag * vb.imag;
                  v.imag += va.imag * vb.real + va.real * vb.imag;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ += ldb; ptrC_ += ldc;
         }
      }
      else if (transB == OcBlasTrans)
      {  /* C = A^T * B^T */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real - va.imag * vb.imag;
                  v.imag += va.imag * vb.real + va.real * vb.imag;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
      else /* (transB == OcBlasConjTrans) */
      {  /* C = A^T * B^H */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real + va.imag * vb.imag;
                  v.imag += va.imag * vb.real - va.real * vb.imag;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
   }
   else /* (transA == OcBlasConjTrans) */
   {  if (transB == OcBlasNoTrans)
      {  /* C = A^H * B */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t];
                  v.real += va.real * vb.real + va.imag * vb.imag;
                  v.imag += va.real * vb.imag - va.imag * vb.real;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ += ldb; ptrC_ += ldc;
         }
      }
      else if (transB == OcBlasTrans)
      {  /* C = A^H * B^T */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real + va.imag * vb.imag;
                  v.imag += va.real * vb.imag - va.imag * vb.real;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
      else /* (transB == OcBlasConjTrans) */
      {  /* C = A^H * B^H */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real - va.imag * vb.imag;
                  v.imag -= va.real * vb.imag + va.imag * vb.real;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
   }
}


/* --------------------------------------------------------------------- */
void ocblas_zgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  const void *alpha, const void *ptrA, int lda, const void *ptrB, int ldb,
                  const void *beta, void *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  const OcBlasCDouble *ptr, *ptrA_, *ptrB_;
   OcBlasCDouble       *ptrC_, v, va, vb;
   double               fr, fi;
   double               ar, ai, br, bi;
   int                  flagAlphaZero, flagBetaZero, flagBetaOne;
   int                  i, j, t;

   /* Assumption: m, n, and k are nonnegative */

   /* Get the alpha and beta values and flags */
   ptr = (const OcBlasCDouble *)alpha; ar = ptr -> real; ai = ptr -> imag;
   ptr = (const OcBlasCDouble *)beta;  br = ptr -> real; bi = ptr -> imag;

   flagAlphaZero = ((ar == 0) && (ai == 0)) ? 1 : 0;
   flagBetaZero  = ((br == 0) && (bi == 0)) ? 1 : 0;
   flagBetaOne   = ((br == 1) && (bi == 0)) ? 1 : 0;

   /* Check if any operation is needed */
   if ((n <= 0) || (m <= 0) ||                           /* Empty output matrix */
       (((flagAlphaZero) || (k <= 0)) && (flagBetaOne))) /* Add nothing         */
   {  return ;
   }

   /* Normalize the ordering */
   if (order == OcBlasRowMajor)
   {  /* Compute C = B(transB) * A(transA) - because of the row-major order */
      /* we can simply swap the order and leave the transformation flags as */
      /* they were and only swap the pointers and dimensions.               */
      ptrA_ = (const OcBlasCDouble *)ptrB;
      ptrB_ = (const OcBlasCDouble *)ptrA;
      t = m; m = n; n = t;
      t = lda; lda = ldb; ldb = t;
   }
   else
   {  ptrA_ = (const OcBlasCDouble *)ptrA;
      ptrB_ = (const OcBlasCDouble *)ptrB;
   }

   /* Scale C */
   ptrC_ = (OcBlasCDouble *)ptrC;
   if (flagAlphaZero)
   {  if (flagBetaZero)
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  ptrC_[i].real = 0;
               ptrC_[i].imag = 0;
            }
            ptrC_ += ldc;
         }
      }
      else
      {  for (j = 0; j < n; j++)
         {  for (i = 0; i < m; i++)
            {  fr = ptrC_[i].real;
               fi = ptrC_[i].imag;
               ptrC_[i].real = br * fr - bi * fi;
               ptrC_[i].imag = bi * fr + br * fi;
            }
            ptrC_ += ldc;
         }
      }
      return ;
   }

   /* Matrix multiplications */
   if (transA == OcBlasNoTrans)
   {  if (transB == OcBlasNoTrans)
      {  /* C <- alpha * A * B + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (flagBetaZero)
            {  for (i = 0; i < m; i++)
               {  ptrC_[i].real = 0;
                  ptrC_[i].imag = 0;
               }
            }
            else if (!flagBetaOne)
            {  for (i = 0; i < m; i++)
               {  fr = ptrC_[j].real;
                  fi = ptrC_[j].imag;
                  ptrC_[i].real += br * fr - bi * fi;
                  ptrC_[i].imag += bi * fr + br * fi;
               }
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA_;
            for (t = 0; t < k; t++)
            {  v = ptrB_[t];
               fr = ar * v.real - ai * v.imag;
               fi = ai * v.real + ar * v.imag;
               for (i = 0; i < m; i++)
               {  v = ptr[i];
                  ptrC_[i].real += fr * v.real - fi * v.imag;
                  ptrC_[i].imag += fi * v.real + fr * v.imag;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ += ldb; ptrC_ += ldc;
         }
      }
      else if (transB == OcBlasTrans)
      {  /* C <- alpha * A * B^T + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (flagBetaZero)
            {  for (i = 0; i < m; i++)
               {  ptrC_[i].real = 0;
                  ptrC_[i].imag = 0;
               }
            }
            else if (!flagBetaOne)
            {  for (i = 0; i < m; i++)
               {  fr = ptrC_[j].real;
                  fi = ptrC_[j].imag;
                  ptrC_[i].real += br * fr - bi * fi;
                  ptrC_[i].imag += bi * fr + br * fi;
               }
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA_;
            for (t = 0; t < k; t++)
            {  v = ptrB_[t*ldb];
               fr = ar * v.real - ai * v.imag;
               fi = ai * v.real + ar * v.imag;
               for (i = 0; i < m; i++)
               {  v = ptr[i];
                  ptrC_[i].real += fr * v.real - fi * v.imag;
                  ptrC_[i].imag += fi * v.real + fr * v.imag;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
      else /* (transB == OcBlasConjTrans) */
      {  /* C <- alpha * A * B^H + beta * C */
         for (j = 0; j < n; j++)
         {  /* Scale the j-th column of C */
            if (flagBetaZero)
            {  for (i = 0; i < m; i++)
               {  ptrC_[i].real = 0;
                  ptrC_[i].imag = 0;
               }
            }
            else if (!flagBetaOne)
            {  for (i = 0; i < m; i++)
               {  fr = ptrC_[j].real;
                  fi = ptrC_[j].imag;
                  ptrC_[i].real += br * fr - bi * fi;
                  ptrC_[i].imag += bi * fr + br * fi;
               }
            }

            /* Add the contributions from the j-th column of B */
            ptr = ptrA_;
            for (t = 0; t < k; t++)
            {  v = ptrB_[t*ldb];
               fr = ar * v.real + ai * v.imag;
               fi = ai * v.real - ar * v.imag;
               for (i = 0; i < m; i++)
               {  v = ptr[i];
                  ptrC_[i].real += fr * v.real - fi * v.imag;
                  ptrC_[i].imag += fi * v.real + fr * v.imag;
               }
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
   }
   else if (transA == OcBlasTrans)
   {  if (transB == OcBlasNoTrans)
      {  /* C = A^T * B */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t];
                  v.real += va.real * vb.real - va.imag * vb.imag;
                  v.imag += va.imag * vb.real + va.real * vb.imag;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ += ldb; ptrC_ += ldc;
         }
      }
      else if (transB == OcBlasTrans)
      {  /* C = A^T * B^T */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real - va.imag * vb.imag;
                  v.imag += va.imag * vb.real + va.real * vb.imag;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
      else /* (transB == OcBlasConjTrans) */
      {  /* C = A^T * B^H */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real + va.imag * vb.imag;
                  v.imag += va.imag * vb.real - va.real * vb.imag;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
   }
   else /* (transA == OcBlasConjTrans) */
   {  if (transB == OcBlasNoTrans)
      {  /* C = A^H * B */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t];
                  v.real += va.real * vb.real + va.imag * vb.imag;
                  v.imag += va.real * vb.imag - va.imag * vb.real;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ += ldb; ptrC_ += ldc;
         }
      }
      else if (transB == OcBlasTrans)
      {  /* C = A^H * B^T */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real + va.imag * vb.imag;
                  v.imag += va.real * vb.imag - va.imag * vb.real;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
      else /* (transB == OcBlasConjTrans) */
      {  /* C = A^H * B^H */
         for (j = 0; j < n; j++)
         {  ptr = ptrA_;
            for (i = 0; i < m; i++)
            {  v.real = 0;
               v.imag = 0;
               for (t = 0; t < k; t++)
               {  va = ptr[t];
                  vb = ptrB_[t*ldb];
                  v.real += va.real * vb.real - va.imag * vb.imag;
                  v.imag -= va.real * vb.imag + va.imag * vb.real;
               }
               if (flagBetaZero)
               {  ptrC_[i].real = ar * v.real - ai * v.imag;
                  ptrC_[i].imag = ai * v.real + ar * v.imag;
               }
               else
               {  va = ptrC_[i];
                  ptrC_[i].real = (br * va.real - bi * va.imag) + (ar * v.real - ai * v.imag);
                  ptrC_[i].imag = (bi * va.real + br * va.imag) + (ai * v.real + ar * v.imag);
               }

               /* Update the pointer for A */
               ptr += lda;
            }

            /* Update the pointers */
            ptrB_ ++; ptrC_ += ldc;
         }
      }
   }
}

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

#ifdef OCBLAS_UNDERSCORE
#define OCBLAS(X) X##_
#else
#define OCBLAS(X) X
#endif

/* ===================================================================== */
/* Function prototypes                                                   */
/* ===================================================================== */

extern void OCBLAS(sgemm)(const char *transa, const char *transb,
                          const int *m, const int *n, const int *k,
                          const float *alpha, const float *a,
                          const int *lda, const float *b, const int *ldb,
                          const float *beta, float *c, const int *ldc);

extern void OCBLAS(dgemm)(const char *transa, const char *transb,
                          const int *m, const int *n, const int *k,
                          const double *alpha, const double *a,
                          const int *lda, const double *b, const int *ldb,
                          const double *beta, double *c, const int *ldc);

extern void OCBLAS(cgemm)(const char *transa, const char *transb,
                          const int *m, const int *n, const int *k,
                          const void *alpha, const void *a,
                          const int *lda, const void *b, const int *ldb,
                          const void *beta, void *c, const int *ldc);

extern void OCBLAS(zgemm)(const char *transa, const char *transb,
                          const int *m, const int *n, const int *k,
                          const void *alpha, const void *a,
                          const int *lda, const void *b, const int *ldb,
                          const void *beta, void *c, const int *ldc);


/* ===================================================================== */
/* Function implementation                                               */
/* ===================================================================== */

/* --------------------------------------------------------------------- */
void ocblas_sgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  float alpha, const float *ptrA, int lda, const float *ptrB, int ldb,
                  float beta, float *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  char modeA, modeB;

   if (order == OcBlasColMajor)
   {  modeA = (transA == OcBlasNoTrans ? 'N' : 'T');
      modeB = (transB == OcBlasNoTrans ? 'N' : 'T');
      OCBLAS(sgemm)(&modeA, &modeB, &m, &n, &k, &alpha, ptrA, &lda, ptrB, &ldb, &beta, ptrC, &ldc);
   }
   else
   {  modeA = (transA == OcBlasNoTrans ? 'T' : 'N');
      modeB = (transB == OcBlasNoTrans ? 'T' : 'N');
      OCBLAS(sgemm)(&modeA, &modeB, &m, &n, &k, &alpha, ptrA, &lda, ptrB, &ldb, &beta, ptrC, &ldc);
   }
}


/* --------------------------------------------------------------------- */
void ocblas_dgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  double alpha, const double *ptrA, int lda, const double *ptrB, int ldb,
                  double beta, double *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  char modeA, modeB;

   if (order == OcBlasColMajor)
   {  modeA = (transA == OcBlasNoTrans ? 'N' : 'T');
      modeB = (transB == OcBlasNoTrans ? 'N' : 'T');
      OCBLAS(dgemm)(&modeA, &modeB, &m, &n, &k, &alpha, ptrA, &lda, ptrB, &ldb, &beta, ptrC, &ldc);
   }
   else
   {  modeA = (transA == OcBlasNoTrans ? 'T' : 'N');
      modeB = (transB == OcBlasNoTrans ? 'T' : 'N');
      OCBLAS(dgemm)(&modeA, &modeB, &m, &n, &k, &alpha, ptrA, &lda, ptrB, &ldb, &beta, ptrC, &ldc);
   }
}


/* --------------------------------------------------------------------- */
void ocblas_cgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  const void *alpha, const void *ptrA, int lda, const void *ptrB, int ldb,
                  const void *beta, void *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  char  modeA, modeB;
   const void *ptr;
   int   t;

   if (order == OcBlasRowMajor)
   {  /* Compute C = B(transB) * A(transA) - because of the row-major order */
      /* we can simply swap the order and leave the transformation flags as */
      /* they were and only swap the pointers and dimensions.               */
      ptr = ptrA; ptrA = ptrB; ptrB = ptr;
      t = m; m = n; n = t;
      t = lda; lda = ldb; ldb = t;

   }

   /* Set the transformation flags */
   if (transA == OcBlasNoTrans)
        modeA = 'N';
   else modeA = (transA == OcBlasTrans) ? 'T' : 'C';

   if (transB == OcBlasNoTrans)
        modeB = 'N';
   else modeB = (transB == OcBlasTrans) ? 'T' : 'C';

   /* Call BLAS */
   OCBLAS(cgemm)(&modeA, &modeB, &m, &n, &k, alpha, ptrA, &lda, ptrB, &ldb, beta, ptrC, &ldc);
}


/* --------------------------------------------------------------------- */
void ocblas_zgemm(OcBlas_Order order, OcBlas_Transpose transA,
                  OcBlas_Transpose transB, int m, int n, int k,
                  const void *alpha, const void *ptrA, int lda, const void *ptrB, int ldb,
                  const void *beta, void *ptrC, int ldc)
/* --------------------------------------------------------------------- */
{  char  modeA, modeB;
   const void *ptr;
   int   t;

   if (order == OcBlasRowMajor)
   {  /* Compute C = B(transB) * A(transA) - because of the row-major order */
      /* we can simply swap the order and leave the transformation flags as */
      /* they were and only swap the pointers and dimensions.               */
      ptr = ptrA; ptrA = ptrB; ptrB = ptr;
      t = m; m = n; n = t;
      t = lda; lda = ldb; ldb = t;
   }

   /* Set the transformation flags */
   if (transA == OcBlasNoTrans)
        modeA = 'N';
   else modeA = (transA == OcBlasTrans) ? 'T' : 'C';

   if (transB == OcBlasNoTrans)
        modeB = 'N';
   else modeB = (transB == OcBlasTrans) ? 'T' : 'C';

   /* Call BLAS */
   OCBLAS(zgemm)(&modeA, &modeB, &m, &n, &k, alpha, ptrA, &lda, ptrB, &ldb, beta, ptrC, &ldc);
}

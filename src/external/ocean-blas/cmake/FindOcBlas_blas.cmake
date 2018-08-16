# ---------------------------------------------------------------------------
# The following variables are defined:
#    OCBLAS_FOUND          Set to 1 if a compatible BLAS was found, 0 otherwise
#    OCBLAS_LIBRARIES      Library files needed for BLAS
#    OCBLAS_INCLUDE_PATHS  Location of the include paths for compilation
#    OCBLAS_INCLUDE        Filename to include
#    OCBLAS_TYPE           BLAS / CBLAS
#    OCBLAS_UNDERSCORE     Set if function names contain a postfix underscore
# ---------------------------------------------------------------------------

set(OCBLAS_FOUND 0)
find_package(BLAS)

if (BLAS_FOUND)

   # Check if BLAS uses underscores in the function names
   if (NOT OCBLAS_FOUND)
      file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/CMakeFiles/OcBlas_Test_Blas1.c" OCBLAS_TEST_BLAS1_C)
      file(WRITE ${OCBLAS_TEST_BLAS1_C}
          "/* Test Blas compilation with undercore */
           extern void dgemm_(const char *transA, const char *transB,
                              const int *m, const int *n, const int *k,
                              const double *alpha, const double *A,
                              const int *lda, const double *b, const int *ldb,
                              const double *beta, double *c, const int *ldc);

           int main(void)
           {  double  A = 1, B = 2, C = 3;
              double  alpha = 3, beta = 1;
              char    transA= 'N', transB = 'N';
              int     lda = 1, ldb = 1, ldc = 1;
              int     m = 1, n = 1, k = 1;
              dgemm_(&transA, &transB, &m, &n, &k, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc);
              return (((int)C) == 9) ? 0 : 1;
            }
           ")
    
      try_run(OCBLAS_TEST_BLAS1_ERROR OCBLAS_TEST_BLAS1_COMPILE
              ${CMAKE_BINARY_DIR} ${OCBLAS_TEST_BLAS1_C}
              LINK_LIBRARIES ${BLAS_LIBRARIES})

      if ((OCBLAS_TEST_BLAS1_COMPILE) AND (NOT OCBLAS_TEST_BLAS1_ERROR))
         set(OCBLAS_FOUND       1)
         set(OCBLAS_UNDERSCORE  1)
         message(STATUS "Compatible BLAS library with underscore found (${BLAS_LIBRARIES})")
      endif()
   endif()

   # Check if BLAS omits underscores in the function names
   if (NOT OCBLAS_FOUND)
     file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/CMakeFiles/OcBlas_Test_Blas2.c" OCBLAS_TEST_BLAS2_C)
     file(WRITE ${OCBLAS_TEST_BLAS2_C}
          "/* Test Blas compilation without undercore */
           extern void dgemm(const char *transA, const char *transB,
                             const int *m, const int *n, const int *k,
                             const double *alpha, const double *A,
                             const int *lda, const double *b, const int *ldb,
                             const double *beta, double *c, const int *ldc);

           int main(void)
           {  double  A = 1, B = 2, C = 3;
              double  alpha = 3, beta = 1;
              char    transA= 'N', transB = 'N';
              int     lda = 1, ldb = 1, ldc = 1;
              int     m = 1, n = 1, k = 1;
              dgemm(&transA, &transB, &m, &n, &k, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc);
              return (((int)C) == 9) ? 0 : 1;
            }
           ")

      try_run(OCBLAS_TEST_BLAS2_ERROR OCBLAS_TEST_BLAS2_COMPILE
              ${CMAKE_BINARY_DIR} ${OCBLAS_TEST_BLAS1_C}
              LINK_LIBRARIES ${BLAS_LIBRARIES})

      if ((OCBLAS_TEST_BLAS2_COMPILE) AND (NOT OCBLAS_TEST_BLAS2_ERROR))
         set(OCBLAS_FOUND       1)
         set(OCBLAS_UNDERSCORE  0)
         message(STATUS "Compatible BLAS library without underscore found (${BLAS_LIBRARIES})")
      endif()
   endif()

   # Set BLAS parameters if needed
   if (OCBLAS_FOUND)
      set(OCBLAS_LIBRARIES     ${BLAS_LIBRARIES})
      set(OCBLAS_INCLUDE_PATHS                  )
      set(OCBLAS_INCLUDE                        )
      set(OCBLAS_TYPE          "BLAS"           )
      set(OCBLAS_C_FILE        "ocean_blas.c"   )
   endif()
endif()


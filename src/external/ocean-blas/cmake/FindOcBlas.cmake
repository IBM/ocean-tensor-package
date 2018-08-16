
set(OCBLAS_FOUND 0)

if (NOT OCBLAS_FOUND)
   find_package(OcBlas_cblas)
endif()

if (NOT OCBLAS_FOUND)
   find_package(OcBlas_blas)
endif()

if (NOT OCBLAS_FOUND)
   find_package(OcBlas_custom)
endif()


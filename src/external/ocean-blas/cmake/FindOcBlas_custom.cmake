set(OCBLAS_FOUND 1)
set(OCBLAS_TYPE CUSTOM)
set(OCBLAS_C_FILE "ocean_custom_blas.c")

set(OCBLAS_LIBRARIES)
set(OCBLAS_INCLUDE_PATHS)
set(OCBLAS_INCLUDE)

message(STATUS "")
message(STATUS "----------------------------------------------------------\n   WARNING: Unable to find a compatible BLAS library, using\n   internal library. For better performance please recompile\n   with optimized BLAS library.")
message(STATUS "----------------------------------------------------------")


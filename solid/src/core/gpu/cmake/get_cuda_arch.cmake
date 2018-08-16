enable_language(CUDA)

function(CUDA_CHECK_ARCH CUDA_ARCH_ID)
  execute_process(COMMAND nvcc "-gencode" "arch=compute_${CUDA_ARCH_ID},code=compute_${CUDA_ARCH_ID}"
                                "-c" "${CMAKE_CUDA_TEMP_DIR}/test.cu"  "-o" "${CMAKE_CUDA_TEMP_DIR}/test.o"
                  RESULT_VARIABLE CUDA_ARCH_FAILED OUTPUT_QUIET ERROR_QUIET)

  if (NOT CUDA_ARCH_FAILED)
     set(CUDA_ARCHITECTURE_FLAGS "${CUDA_ARCHITECTURE_FLAGS};-gencode;arch=compute_${CUDA_ARCH_ID},code=compute_${CUDA_ARCH_ID}" PARENT_SCOPE)
     set(CUDA_ARCHITECTURE ${CUDA_ARCH_ID} PARENT_SCOPE)
     message(STATUS "Supporting CUDA compute capacity ${CUDA_ARCH_ID}")
  endif()
  set(CUDA_ARCH_FAILED ${CUDA_ARCH_FAILED} PARENT_SCOPE)
endfunction(CUDA_CHECK_ARCH)


function(CUDA_CHECK_ARCHITECTURES output_variable)
   # Initialize
   set(CUDA_ARCHITECTURE_FLAGS)
   set(CUDA_ARCHITECTURE)
   set(CMAKE_CUDA_TEMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles")

   # Create the test source file for nvcc compute capability tests
   file(WRITE ${CMAKE_CUDA_TEMP_DIR}/test.cu "__global__ void test(void) { }")

   # Check if we can get GPU device information
   message(STATUS "Querying device information . . .")
   file(WRITE ${CMAKE_CUDA_TEMP_DIR}/test_device.cu
              "#include <stdio.h>\n"
              "int main(void)\n"
              "{  struct cudaDeviceProp prop;\n"
              "   int deviceCount;\n"
              "   if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) return -1;\n"
              "   if (deviceCount == 0) return -1;\n"
              "   if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;\n"
              "   printf(\"%d;%d\", deviceCount, 10 * prop.major + prop.minor);\n"
              "   return 0;\n"
              "}")

   try_run(CUDA_RUN_RESULT CUDA_COMPILE_RESULT ${CMAKE_CUDA_TEMP_DIR} ${CMAKE_CUDA_TEMP_DIR}/test_device.cu
           CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
           LINK_LIBRARIES ${CUDA_LIBRARIES}
           RUN_OUTPUT_VARIABLE CUDA_OUTPUT)

   # Check validity of the output
   if ((CUDA_COMPILE_RESULT) AND (CUDA_RUN_RESULT EQUAL "0"))
      list(GET CUDA_OUTPUT 0 CUDA_DEVICE_COUNT)
      list(GET CUDA_OUTPUT 1 CUDA_COMPUTE_CAPACITY)

      message(STATUS "Found ${CUDA_DEVICE_COUNT} device(s) with compute capacity ${CUDA_COMPUTE_CAPACITY}")
      CUDA_CHECK_ARCH(${CUDA_COMPUTE_CAPACITY})
      if (CUDA_ARCH_FAILED)
         message(STATUS "Compute capacity ${CUDA_COMPUTE_CAPACITY} not supported by nvcc")
       endif ()
   else()
      set(CUDA_ARCH_FAILED 1)
   endif()

   # Check different architectures
   if (CUDA_ARCH_FAILED)
      message(STATUS "Detecting nvcc supported GPU architectures . . .")
      CUDA_CHECK_ARCH(30)
      CUDA_CHECK_ARCH(35)
      CUDA_CHECK_ARCH(50)
      CUDA_CHECK_ARCH(60)
      CUDA_CHECK_ARCH(70)
   endif()

   # Add the default architecture
   if (CUDA_ARCHITECTURE)
      set(CUDA_ARCHITECTURE_FLAGS "-arch=sm_${CUDA_ARCHITECTURE};${CUDA_ARCHITECTURE_FLAGS}")
   endif()
   set(${output_variable} ${CUDA_ARCHITECTURE_FLAGS} PARENT_SCOPE)
endfunction(CUDA_CHECK_ARCHITECTURES)

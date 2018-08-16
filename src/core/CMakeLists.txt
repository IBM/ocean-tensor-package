cmake_minimum_required(VERSION 3.8)
project(ocean-core)
enable_language(CXX)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


# ------------------------------------------------------------------
# Generic configuration
# ------------------------------------------------------------------

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Add the solid include directory
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../solid/include" SOLID_INCLUDE)
get_filename_component(SOLID_INCLUDE "${SOLID_INCLUDE}" ABSOLUTE)
include_directories(BEFORE "${SOLID_INCLUDE}")

# Add the ocean include directory
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../include" OCEAN_INCLUDE)
get_filename_component(OCEAN_INCLUDE "${OCEAN_INCLUDE}" ABSOLUTE)
include_directories(BEFORE "${OCEAN_INCLUDE}")

# Get the processor name
if (CMAKE_UNAME)
   execute_process( COMMAND uname -m OUTPUT_VARIABLE OCEAN_PROCESSOR)
   string(STRIP "${OCEAN_PROCESSOR}" OCEAN_PROCESSOR)
endif (CMAKE_UNAME)

# Get the system name
set(OCEAN_SYSTEM ${CMAKE_SYSTEM_NAME})

# Determine the library paths
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../solid/lib/${OCEAN_SYSTEM}_${OCEAN_PROCESSOR}" SOLID_LIB_PATH)
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/${OCEAN_SYSTEM}_${OCEAN_PROCESSOR}" OCEAN_LIB_PATH)
get_filename_component(SOLID_LIB_PATH "${SOLID_LIB_PATH}" ABSOLUTE)
get_filename_component(OCEAN_LIB_PATH "${OCEAN_LIB_PATH}" ABSOLUTE)
file(MAKE_DIRECTORY "${OCEAN_LIB_PATH}")

# Add the link directories
link_directories(${SOLID_LIB_PATH})
link_directories(${OCEAN_LIB_PATH})

# Set the output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OCEAN_LIB_PATH}")
set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${SOLID_LIB_PATH}")
set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${OCEAN_LIB_PATH}")
set(CMAKE_INSTALL_NAME_DIR "${OCEAN_LIB_PATH}")


# ------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------

# Optimization flags for CXX:
# [-Wno-implicit-function-declarations]
#    Disable warnings on implicit function declarations: the cublasHgemm
#    function is declared in cublas_api.h (included from cublas_v2.h)
#    but only when __cplusplus is defined.

set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing -Wno-implicit-function-declaration")
set(CMAKE_C_FLAGS   "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing -Wno-implicit-function-declaration")


# ------------------------------------------------------------------
# Check for Cuda
# ------------------------------------------------------------------
find_package(CUDA)

# Add global library dependencies
if (CUDA_FOUND)
   foreach (CUDA_LIBRARY ${CUDA_CBLAS_LIBRARIES})
      if (EXISTS ${CUDA_LIBRARY})
         get_filename_component(CUDA_LIBRARY_PATH ${CUDA_LIBRARY} DIRECTORY)
         set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${CUDA_LIBRARY_PATH}")
      endif()
   endforeach (CUDA_LIBRARY)
endif()


# ------------------------------------------------------------------
# Ocean library (CPU)
# ------------------------------------------------------------------

FILE(GLOB OCEAN_ITF     "${CMAKE_CURRENT_SOURCE_DIR}/interface/*.c")
FILE(GLOB OCEAN_GENERIC "${CMAKE_CURRENT_SOURCE_DIR}/generic/*.c"  )
FILE(GLOB OCEAN_CPU     "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.c"      )
FILE(GLOB OCEAN_CPU_OP  "${CMAKE_CURRENT_SOURCE_DIR}/cpu/op/*.c"   )

add_library(ocean SHARED ${OCEAN_ITF} ${OCEAN_GENERIC} ${OCEAN_CPU} ${OCEAN_CPU_OP})
target_link_libraries(ocean solid_base solid_cpu solid_core_cpu ocean_base ocean_solid ocean_blas)
set_target_properties(ocean PROPERTIES C_STANDARD 99
                                       MACOSX_RPATH 0
                                       BUILD_WITH_INSTALL_NAME_DIR 1
                                       INSTALL_NAME_DIR ${OCEAN_LIB_PATH})


# ------------------------------------------------------------------
# Ocean library (GPU)
# ------------------------------------------------------------------

if(CUDA_FOUND)
   enable_language(CUDA)

   FILE(GLOB OCEAN_GPU     "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.c"      )
   FILE(GLOB OCEAN_GPU_OP  "${CMAKE_CURRENT_SOURCE_DIR}/gpu/op/*.c"   )

   add_library(ocean_gpu SHARED ${OCEAN_GPU} ${OCEAN_GPU_OP})
   target_include_directories(ocean_gpu PUBLIC ${CUDA_INCLUDE_DIRS})
   target_include_directories(ocean_gpu PUBLIC ${SOLID_INCLUDE} ${OCEAN_INCLUDE})
   target_link_libraries(ocean_gpu solid_base solid_cpu solid_gpu solid_core_gpu ocean_base ocean_solid ocean)
   target_link_libraries(ocean_gpu ${CUDA_CUBLAS_LIBRARIES})
   target_link_libraries(ocean_gpu ${CUDA_LIBRARIES})

   # Use the C11 standard when possible to avoid compile warnings
   # when including cuda.h
   set_target_properties(ocean_gpu PROPERTIES C_STANDARD 11
                                              MACOSX_RPATH 0
                                              BUILD_WITH_INSTALL_NAME_DIR 1
                                              INSTALL_NAME_DIR ${OCEAN_LIB_PATH})
endif()


# ------------------------------------------------------------------
# Display list of paths to add
# ------------------------------------------------------------------
if (${OCEAN_CMAKE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
   if ((OCEAN_LIBRARY_PATHS) AND (NOT APPLE))
      string(SUBSTRING ${OCEAN_LIBRARY_PATHS} 1 -1 OCEAN_LIBRARY_PATHS)
      separate_arguments(OCEAN_LIBRARY_PATHS)
      list(REMOVE_DUPLICATES OCEAN_LIBRARY_PATHS)
      string(REPLACE ";" ":" OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS}")
      message(STATUS "")
      message(STATUS "----------------------------------------------------------")
      message(STATUS "Add the following paths to the library search path:\n   ${OCEAN_LIBRARY_PATHS}")
      message(STATUS "----------------------------------------------------------")
   endif()
else()
   set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS}" PARENT_SCOPE)
endif()


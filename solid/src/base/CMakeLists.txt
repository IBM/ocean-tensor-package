cmake_minimum_required(VERSION 3.8)
project(solid-base)
enable_language(CXX)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


# ------------------------------------------------------------------
# Generic path settings
# ------------------------------------------------------------------

# Add the solid include directory
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../include" SOLID_INCLUDE)
get_filename_component(SOLID_INCLUDE "${SOLID_INCLUDE}" ABSOLUTE)
include_directories(BEFORE "${SOLID_INCLUDE}")

# Get the processor name
if (CMAKE_UNAME)
   execute_process( COMMAND uname -m OUTPUT_VARIABLE OCEAN_PROCESSOR)
   string(STRIP "${OCEAN_PROCESSOR}" SOLID_PROCESSOR)
endif (CMAKE_UNAME)

# Get the system name
set(SOLID_SYSTEM ${CMAKE_SYSTEM_NAME})

# Determine the library path
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/${SOLID_SYSTEM}_${SOLID_PROCESSOR}" SOLID_LIB_PATH)
get_filename_component(SOLID_LIB_PATH "${SOLID_LIB_PATH}" ABSOLUTE)
file(MAKE_DIRECTORY "${SOLID_LIB_PATH}")

# Set the output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOLID_LIB_PATH}")
set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${SOLID_LIB_PATH}")


# ------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing -std=c99")


# ------------------------------------------------------------------
# Check for OMP
# ------------------------------------------------------------------

find_package(OpenMP)
if(OPENMP_FOUND)
   message(STATUS "Enabling OpenMP support")
   set(SOLID_ENABLE_OMP 1)
   set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
   message(STATUS "Disabling OpenMP support")
   set(SOLID_ENABLE_OMP 0)
endif()

# Write the configuration header file
set(SOLID_OMP_CONFIG_H "${SOLID_INCLUDE}/solid/base/cpu/solid_omp_config.h")
file(WRITE  ${SOLID_OMP_CONFIG_H} "/* This file was automatically generated from solid/src/base/CMakeList.txt */\n")
file(APPEND ${SOLID_OMP_CONFIG_H} "#ifndef __SOLID_OMP_CONFIG_H__\n")
file(APPEND ${SOLID_OMP_CONFIG_H} "#define __SOLID_OMP_CONFIG_H__\n")
file(APPEND ${SOLID_OMP_CONFIG_H} "#define SOLID_ENABLE_OMP ${SOLID_ENABLE_OMP}\n")
file(APPEND ${SOLID_OMP_CONFIG_H} "#endif\n")


# ------------------------------------------------------------------
# Check for Cuda
# ------------------------------------------------------------------

find_package(CUDA)
if (CUDA_FOUND)
   message(STATUS "Enabling CUDA support")
else()
   message(STATUS "Disabling CUDA support")
endif()

# Add global library dependencies
if (CUDA_FOUND)
   foreach (CUDA_LIBRARY ${CUDA_LIBRARIES})
      if (EXISTS ${CUDA_LIBRARY})
         get_filename_component(CUDA_LIBRARY_PATH ${CUDA_LIBRARY} DIRECTORY)
         set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${CUDA_LIBRARY_PATH}")
      endif()
   endforeach (CUDA_LIBRARY)
endif()


# ------------------------------------------------------------------
# Shared libaries
# ------------------------------------------------------------------

# Shared library for solid
FILE(GLOB SOLID_FILES "${CMAKE_CURRENT_SOURCE_DIR}/generic/*.c")
add_library(solid_base SHARED ${SOLID_FILES} ${SOLID_OMP_CONFIG_H})
set_target_properties(solid_base PROPERTIES MACOSX_RPATH 0
                                            BUILD_WITH_INSTALL_NAME_DIR 1
                                            INSTALL_NAME_DIR ${SOLID_LIB_PATH})

# Shared library for solid_cpu
FILE(GLOB SOLID_CPU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.c")
add_library(solid_cpu SHARED ${SOLID_CPU_FILES} ${SOLID_OMP_CONFIG_H})
target_link_libraries(solid_cpu solid_base)
set_target_properties(solid_cpu PROPERTIES MACOSX_RPATH 0
                                           BUILD_WITH_INSTALL_NAME_DIR 1
                                           INSTALL_NAME_DIR ${SOLID_LIB_PATH})

# Shared library for solid_gpu
if (CUDA_FOUND)
   FILE(GLOB SOLID_GPU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.c")
   include_directories(AFTER ${CUDA_INCLUDE_DIRS})
   add_library(solid_gpu SHARED ${SOLID_GPU_FILES} ${SOLID_OMP_CONFIG_H})
   target_link_libraries(solid_gpu solid_base ${CUDA_LIBRARIES})
   set_target_properties(solid_gpu PROPERTIES MACOSX_RPATH 0
                                              BUILD_WITH_INSTALL_NAME_DIR 1
                                              INSTALL_NAME_DIR ${SOLID_LIB_PATH})
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


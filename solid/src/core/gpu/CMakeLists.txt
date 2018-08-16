cmake_minimum_required(VERSION 3.8)
project(solid-core-gpu)
enable_language(CXX)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


# ------------------------------------------------------------------
# Generic configuration
# ------------------------------------------------------------------

# Add the solid include directory
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../include" SOLID_INCLUDE)
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
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/${SOLID_SYSTEM}_${SOLID_PROCESSOR}" SOLID_LIB_PATH)
get_filename_component(SOLID_LIB_PATH "${SOLID_LIB_PATH}" ABSOLUTE)
file(MAKE_DIRECTORY "${SOLID_LIB_PATH}")

# Add the link directory
link_directories(${SOLID_LIB_PATH})

# Set the output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOLID_LIB_PATH}")
set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${SOLID_LIB_PATH}")


# ------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------

# Optimization flags for CXX (-pedantic distabled to avoid NVCC warnings)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wno-variadic-macros -fno-strict-aliasing -Wno-unused-function")
set(CMAKE_C_FLAGS   "-O3 -Wall -Wno-variadic-macros -fno-strict-aliasing -Wno-unused-function -std=c99")


# ------------------------------------------------------------------
# Check for Cuda
# ------------------------------------------------------------------

find_package(CUDA)

if (CUDA_FOUND)
   message(STATUS "Configuring solid/core for GPU")
   enable_language(CUDA)
else()
   message(STATUS "Omitting solid/core module for GPU")
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

# Determine the CUDA architecture
if (CUDA_FOUND)
   include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cuda_arch.cmake")
   CUDA_CHECK_ARCHITECTURES(CUDA_ARCHITECTURE_FLAGS)
   list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCHITECTURE_FLAGS})
endif()


# ------------------------------------------------------------------
# Find the python interpreter to generate the header files
# ------------------------------------------------------------------

find_package(PythonInterp REQUIRED)

file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/generate.py" SOLID_GENERATE_PY)
get_filename_component(SOLID_GENERATE_PY "${SOLID_GENERATE_PY}" ABSOLUTE)
if(NOT EXISTS "${SOLID_GENERATE_PY}")
   message(FATAL_ERROR "Could not find the required file '${SOLID_GENERATE_PY}'")
endif()


# ------------------------------------------------------------------
# Automatically generated GPU header file and look-up table
# ------------------------------------------------------------------

if (CUDA_FOUND)
   file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/solid_core_gpu.src SOLID_CORE_GPU_SRC)
   file(TO_NATIVE_PATH ${SOLID_INCLUDE}/solid_core_gpu.h SOLID_CORE_GPU_H)
   file(TO_NATIVE_PATH ${PROJECT_BINARY_DIR}/CMakeFiles/ SOLID_CORE_GPU_LUT_DIR)
   file(TO_NATIVE_PATH ${SOLID_CORE_GPU_LUT_DIR}/solid_core_gpu.lut.c SOLID_CORE_GPU_LUT_C)
   file(MAKE_DIRECTORY ${SOLID_CORE_GPU_LUT_DIR})

   add_custom_command(
      OUTPUT   ${SOLID_CORE_GPU_H}
      COMMAND  ${PYTHON_EXECUTABLE} ${SOLID_GENERATE_PY} -h ${SOLID_CORE_GPU_SRC} ${SOLID_INCLUDE}
      DEPENDS  ${SOLID_CORE_GPU_SRC}
   )

   add_custom_command(
      OUTPUT   ${SOLID_CORE_GPU_LUT_C}
      COMMAND  ${PYTHON_EXECUTABLE} ${SOLID_GENERATE_PY} -c ${SOLID_CORE_GPU_SRC} ${SOLID_CORE_GPU_LUT_DIR}
      DEPENDS  ${SOLID_CORE_GPU_SRC}
   )
endif()


# ------------------------------------------------------------------
# GPU code
# ------------------------------------------------------------------

if (CUDA_FOUND)
   FILE(GLOB SOLID_CORE_GPU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
   FILE(GLOB SOLID_CORE_GPU_OP "${CMAKE_CURRENT_SOURCE_DIR}/op/*.c")
   include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../")
   include_directories(AFTER ${CUDA_INCLUDE_DIRS})
   CUDA_ADD_LIBRARY(solid_core_gpu SHARED ${SOLID_CORE_GPU} ${SOLID_CORE_GPU_OP} ${SOLID_CORE_GPU_LUT_C} ${SOLID_CORE_CPU_H} ${SOLID_CORE_GPU_H})
   target_link_libraries(solid_core_gpu solid_base solid_gpu solid_core_cpu ${CUDA_LIBRARIES})
   set_target_properties(solid_core_gpu PROPERTIES MACOSX_RPATH 0
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


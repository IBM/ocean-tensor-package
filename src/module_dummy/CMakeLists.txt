cmake_minimum_required(VERSION 3.8)
project(ocean-module-dummy)
enable_language(CXX)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


# ------------------------------------------------------------------
# Generic configuration
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------

# Optimization flags for CXX
set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing")
set(CMAKE_C_FLAGS   "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing")


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
# Module Dummy - Interface
# ------------------------------------------------------------------
FILE(GLOB MODULE_DUMMY_ITF  "${CMAKE_CURRENT_SOURCE_DIR}/interface/*.c")
add_library(ocean_dummy_itf SHARED ${MODULE_DUMMY_ITF})
target_link_libraries(ocean_dummy_itf ocean ocean_base)
set_target_properties(ocean_dummy_itf PROPERTIES MACOSX_RPATH 0
                                                 BUILD_WITH_INSTALL_NAME_DIR 1
                                                 INSTALL_NAME_DIR ${OCEAN_LIB_PATH})


# ------------------------------------------------------------------
# Module Dummy - CPU
# ------------------------------------------------------------------
FILE(GLOB MODULE_DUMMY_CPU  "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.c")
add_library(ocean_dummy_cpu SHARED ${MODULE_DUMMY_CPU})
add_dependencies(ocean_dummy_cpu ocean_dummy_itf)
target_link_libraries(ocean_dummy_cpu ocean ocean_dummy_itf)
set_target_properties(ocean_dummy_cpu PROPERTIES MACOSX_RPATH 0
                                                 BUILD_WITH_INSTALL_NAME_DIR 1
                                                 INSTALL_NAME_DIR ${OCEAN_LIB_PATH})


# ------------------------------------------------------------------
# Module Dummy - GPU
# ------------------------------------------------------------------
if (CUDA_FOUND)
   FILE(GLOB MODULE_DUMMY_GPU  "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.c")
   include_directories(AFTER ${CUDA_INCLUDE_DIRS})
   add_library(ocean_dummy_gpu SHARED ${MODULE_DUMMY_GPU})
   add_dependencies(ocean_dummy_gpu ocean_dummy_itf)
   target_link_libraries(ocean_dummy_gpu ocean ocean_gpu ocean_dummy_itf)
   set_target_properties(ocean_dummy_gpu PROPERTIES C_STANDARD 11
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


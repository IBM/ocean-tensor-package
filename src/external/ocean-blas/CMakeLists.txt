cmake_minimum_required(VERSION 3.8)
project(ocean-external-blas)
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

# Add the ocean include directory
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../include" OCEAN_INCLUDE)
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
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/${OCEAN_SYSTEM}_${OCEAN_PROCESSOR}" OCEAN_LIB_PATH)
get_filename_component(OCEAN_LIB_PATH "${OCEAN_LIB_PATH}" ABSOLUTE)
file(MAKE_DIRECTORY "${OCEAN_LIB_PATH}")

# Set the output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OCEAN_LIB_PATH}")
set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${OCEAN_LIB_PATH}")


# ------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------

# Optimization flags for CXX
set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing")
set(CMAKE_C_FLAGS   "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing -std=c99")


# ------------------------------------------------------------------
# Check if blas is available
# ------------------------------------------------------------------

# Find a compatible Blas library
find_package(OcBlas)

# Output the configuration header file
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/" OCEAN_BLAS_CONFIG_DIR)
file(TO_NATIVE_PATH "${OCEAN_BLAS_CONFIG_DIR}/ocean_blas_config.h" OCEAN_BLAS_CONFIG_H)
file(WRITE  ${OCEAN_BLAS_CONFIG_H} "#ifndef __OCEAN_EXTERNAL_OCBLAS_CONFIG_H__\n")
file(APPEND ${OCEAN_BLAS_CONFIG_H} "#define __OCEAN_EXTERNAL_OCBLAS_CONFIG_H__\n") 
file(APPEND ${OCEAN_BLAS_CONFIG_H} "\n")

if (OCBLAS_UNDERSCORE)
   file(APPEND ${OCEAN_BLAS_CONFIG_H} "#define OCBLAS_UNDERSCORE ${OCBLAS_UNDERSCORE}\n")
endif()

if (OCBLAS_INCLUDE)
   file(APPEND ${OCEAN_BLAS_CONFIG_H} "#include \"${OCBLAS_INCLUDE}\"\n")
endif()

file(APPEND ${OCEAN_BLAS_CONFIG_H} "\n")
file(APPEND ${OCEAN_BLAS_CONFIG_H} "#endif\n")


# ------------------------------------------------------------------
# Ocean-blas interface library
# ------------------------------------------------------------------

add_library(ocean_blas SHARED ${OCBLAS_C_FILE})
target_include_directories(ocean_blas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) # ${OCBLAS_C_FILE}
target_include_directories(ocean_blas PUBLIC ${OCEAN_BLAS_CONFIG_DIR})    # ocean_blas_config.h
set_target_properties(ocean_blas PROPERTIES MACOSX_RPATH 0
                                            BUILD_WITH_INSTALL_NAME_DIR 1
                                            INSTALL_NAME_DIR ${OCEAN_LIB_PATH})

# Add global library dependencies
foreach (OCBLAS_LIBRARY ${OCBLAS_LIBRARIES})
   get_filename_component(OCBLAS_LIBRARY_PATH ${OCBLAS_LIBRARY} DIRECTORY)
   set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${OCBLAS_LIBRARY_PATH}")
endforeach (OCBLAS_LIBRARY)

if (OCBLAS_LIBRARIES)
   target_link_libraries(ocean_blas ${OCBLAS_LIBRARIES})
endif()

if (OCBLAS_INCLUDE_PATHS)
   target_include_directories(ocean_blas PUBLIC ${OCBLAS_INCLUDE_PATHS})
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


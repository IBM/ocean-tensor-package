cmake_minimum_required(VERSION 3.8)
project(ocean-base)
enable_language(CXX)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


# ------------------------------------------------------------------
# Generic configuration
# ------------------------------------------------------------------

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

# Add the link directory
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
set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing -Wno-unneeded-internal-declaration")
set(CMAKE_C_FLAGS   "-O3 -pedantic -Wall -Wno-variadic-macros -fno-strict-aliasing -Wno-unneeded-internal-declaration -std=c99")


# ------------------------------------------------------------------
# Ocean base library
# ------------------------------------------------------------------

FILE(GLOB OCEAN_BASE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
add_library(ocean_base SHARED ${OCEAN_BASE_FILES})
set_target_properties(ocean_base PROPERTIES MACOSX_RPATH 0
                                            BUILD_WITH_INSTALL_NAME_DIR 1
                                            INSTALL_NAME_DIR ${OCEAN_LIB_PATH})


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

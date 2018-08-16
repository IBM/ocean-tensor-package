cmake_minimum_required(VERSION 3.8)
project(solid-core-cpu)
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
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wno-variadic-macros -fno-strict-aliasing")
set(CMAKE_C_FLAGS   "-O3 -Wall -Wno-variadic-macros -fno-strict-aliasing -std=c99")


# ------------------------------------------------------------------
# Check for OpenMP
# ------------------------------------------------------------------

find_package(OpenMP)
if(OPENMP_FOUND)
   set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
# Automatically generated CPU header file and look-up table
# ------------------------------------------------------------------

file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/solid_core_cpu.src SOLID_CORE_CPU_SRC)
file(TO_NATIVE_PATH ${SOLID_INCLUDE}/solid_core_cpu.h SOLID_CORE_CPU_H)
file(TO_NATIVE_PATH ${PROJECT_BINARY_DIR}/CMakeFiles/ SOLID_CORE_CPU_LUT_DIR)
file(TO_NATIVE_PATH ${SOLID_CORE_CPU_LUT_DIR}/solid_core_cpu.lut.c SOLID_CORE_CPU_LUT_C)
file(MAKE_DIRECTORY ${SOLID_CORE_CPU_LUT_DIR})

add_custom_command(
   OUTPUT   ${SOLID_CORE_CPU_H}
   COMMAND  ${PYTHON_EXECUTABLE} ${SOLID_GENERATE_PY} -h ${SOLID_CORE_CPU_SRC} ${SOLID_INCLUDE}
   DEPENDS  ${SOLID_CORE_CPU_SRC}
)

add_custom_command(
   OUTPUT   ${SOLID_CORE_CPU_LUT_C}
   COMMAND  ${PYTHON_EXECUTABLE} ${SOLID_GENERATE_PY} -c ${SOLID_CORE_CPU_SRC} ${SOLID_CORE_CPU_LUT_DIR}
   DEPENDS  ${SOLID_CORE_CPU_SRC}
)


# ------------------------------------------------------------------
# CPU code
# ------------------------------------------------------------------

FILE(GLOB SOLID_CORE_CPU "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../")
add_library(solid_core_cpu SHARED ${SOLID_CORE_CPU} ${SOLID_CORE_CPU_LUT_C} ${SOLID_CORE_CPU_H})
target_link_libraries(solid_core_cpu solid_base solid_cpu)
set_target_properties(solid_core_cpu PROPERTIES MACOSX_RPATH 0
                                                BUILD_WITH_INSTALL_NAME_DIR 1
                                                INSTALL_NAME_DIR ${SOLID_LIB_PATH})


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


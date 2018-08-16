cmake_minimum_required(VERSION 3.8)
project(pyocean-core-cpu)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


# ------------------------------------------------------------------
# Python
# ------------------------------------------------------------------
find_package(PythonInterp)
if (NOT PYTHONINTERP_FOUND)
   message(STATUS "Python not found: Skipping pyOcean_cpu")
   return()
endif()


# ------------------------------------------------------------------
# Generic configuration
# ------------------------------------------------------------------

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Add the ocean include directory
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include" OCEAN_INCLUDE)
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
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../lib/${OCEAN_SYSTEM}_${OCEAN_PROCESSOR}" OCEAN_LIB_PATH)
file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../lib" PYOCEAN_LIB_BASE)
file(TO_NATIVE_PATH "${PYOCEAN_LIB_BASE}/${OCEAN_SYSTEM}_${OCEAN_PROCESSOR}" PYOCEAN_LIB_PATH)
get_filename_component(OCEAN_LIB_PATH "${OCEAN_LIB_PATH}" ABSOLUTE)
get_filename_component(PYOCEAN_LIB_BASE "${PYOCEAN_LIB_BASE}" ABSOLUTE)
get_filename_component(PYOCEAN_LIB_PATH "${PYOCEAN_LIB_PATH}" ABSOLUTE)
file(MAKE_DIRECTORY "${PYOCEAN_LIB_PATH}")

# Add the library directories
set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS} ${OCEAN_LIB_PATH}")
set(OCEAN_PYTHON_PATHS  "${OCEAN_PYTHON_PATHS} ${PYOCEAN_LIB_BASE} ${PYOCEAN_LIB_PATH}")


# ------------------------------------------------------------------
# Get the library name and add target
# ------------------------------------------------------------------

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import setup; import sys; sys.stdout.write(setup.libso('pyOcean_cpu'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE PYOCEAN_CPU_ERROR
                OUTPUT_VARIABLE PYOCEAN_CPU_LIBNAME)
if (PYOCEAN_CPU_ERROR)
   message(FATAL_ERROR "Could not get the name of the pyOcean_cpu library")
else()
   message(STATUS "Adding library ${PYOCEAN_CPU_LIBNAME}")
endif()

# Set the target
file(TO_NATIVE_PATH "${PYOCEAN_LIB_PATH}/${PYOCEAN_CPU_LIBNAME}" PYOCEAN_LIB_FILE)
file(GLOB PYOCEAN_CPU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_custom_command(OUTPUT ${PYOCEAN_LIB_FILE}
                   COMMAND ${PYTHON_EXECUTABLE}
                   ARGS setup.py build_ext --build-lib "${PYOCEAN_LIB_PATH}"
                   DEPENDS ${PYOCEAN_CPU_FILES}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(pyOcean_cpu ALL DEPENDS ${PYOCEAN_LIB_FILE})


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

   if (OCEAN_PYTHON_PATHS)
      string(SUBSTRING ${OCEAN_PYTHON_PATHS} 1 -1 OCEAN_PYTHON_PATHS)
      separate_arguments(OCEAN_PYTHON_PATHS)
      list(REMOVE_DUPLICATES OCEAN_PYTHON_PATHS)
      string(REPLACE ";" ":" OCEAN_PYTHON_PATHS "${OCEAN_PYTHON_PATHS}")
      message(STATUS "")
      message(STATUS "----------------------------------------------------------")
      message(STATUS "Add the following paths to the Python search path:\n   ${OCEAN_PYTHON_PATHS}")
      message(STATUS "----------------------------------------------------------")
   endif()
else()
   set(OCEAN_LIBRARY_PATHS "${OCEAN_LIBRARY_PATHS}" PARENT_SCOPE)
   set(OCEAN_PYTHON_PATHS  "${OCEAN_PYTHON_PATHS}" PARENT_SCOPE)
endif()


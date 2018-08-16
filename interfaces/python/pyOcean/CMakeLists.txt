cmake_minimum_required(VERSION 3.8)
project(pyocean-core)

# Initialize
if (NOT OCEAN_CMAKE_DIR)
   set(OCEAN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Add subdirectories
add_subdirectory(cpu)
add_subdirectory(gpu)


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
endif()


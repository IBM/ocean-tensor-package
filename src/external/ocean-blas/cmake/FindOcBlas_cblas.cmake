
# -------------------------------------------------------------------------
# Expand path
function(FindSubDirectories path pattern result)
# -------------------------------------------------------------------------
   # Initialize the pathlist with the existing values
   # so we can append all new entries. Note that ${result}
   # only gives the variable name, so double ${} is needed.
   set(pathlist "${${result}}")

   # Find all directories that match the given pattern,
   # then sort and reverse to approximately obtain newer
   # versions of a package first (this only works for
   # single-digit versions but it better than nothing).
   file(GLOB entries RELATIVE "${path}" "${path}/${pattern}")
   list(SORT entries)
   list(REVERSE entries)
   foreach(entry ${entries})
      file(TO_NATIVE_PATH "${path}/${entry}" directory)
      if(IS_DIRECTORY "${directory}")
         get_filename_component(directory "${directory}" ABSOLUTE)
         list(APPEND pathlist "${path}/${entry}")
      endif()
   endforeach()

   # Assign the result to the parent scope
   set(${result} ${pathlist} PARENT_SCOPE)
endfunction()


# -------------------------------------------------------------------------
# Find directories with given pattern
function(FindPathPattern pathlist patterns result)
# -------------------------------------------------------------------------
   set(results "")
   foreach(path ${pathlist})
      foreach(pattern ${patterns})
         FindSubDirectories("${path}" "${pattern}" results)
      endforeach()
   endforeach()
   set(${result} ${results} PARENT_SCOPE)
endfunction()

# -------------------------------------------------------------------------
# Find all valid paths of the form <prefix>/<postfix>
function(ExpandPathList prefixList postfixList result)
# -------------------------------------------------------------------------
   set(results "")
   foreach(prefix ${prefixList})
      if (IS_DIRECTORY "${prefix}")
         foreach(postfix ${postfixList})
            file(TO_NATIVE_PATH "${prefix}/${postfix}" directory)
            if (IS_DIRECTORY "${directory}")
               get_filename_component(directory "${directory}" ABSOLUTE)
               list(APPEND results "${directory}")
            endif()
         endforeach()
      endif()
   endforeach()
   set(${result} ${results} PARENT_SCOPE)
endfunction()


# -------------------------------------------------------------------------
# Verify a given CBlas installation
function(VerifyCBlasInstallation installation user_specified result)
# -------------------------------------------------------------------------
   # Set the default result
   set(${result} "" PARENT_SCOPE)

   # Make sure the installation is a list with two components
   if (user_specified)
      # Make sure that the installation has two parts
      list(LENGTH installation installation_parts)
      if (NOT (installation_parts EQUAL 2))
         message(STATUS "User-specified CBlas installation must have the format <full library name>;<full include name>")
         return ()
      endif()
   endif()

   # Extract the installation informaiton
   list(GET installation 0 installation_library)
   list(GET installation 1 installation_include)

   # Check the existence of the library and header file
   if (user_specified)
      if (NOT (EXISTS "${installation_library}"))
         message(STATUS "WARNING: Could not find user-specified CBlas library: '${installation_library}'")
         return ()
      endif()
      if (NOT (EXISTS "${installation_include}"))
         message(STATUS "WARNING: Could not find user-specified CBlas include file: '${installation_include}'")
         return ()
      endif()
   endif()

   # Extract the library and include properties
   get_filename_component(library_name "${installation_library}" NAME_WE)
   get_filename_component(library_path "${installation_library}" DIRECTORY)
   get_filename_component(include_file "${installation_include}" NAME)
   get_filename_component(include_path "${installation_include}" DIRECTORY)

   string(LENGTH "${CMAKE_STATIC_LIBRARY_PREFIX}" lib_prefix_length)
   if (lib_prefix_length GREATER 0)
      # The prefix is presumed to be equal to the library prefix
      string(SUBSTRING "${library_name}" ${lib_prefix_length} -1 library_name)
   endif()

   # Check whether the include/library combination compiles and runs
   file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/CMakeFiles/OcBlas_Test_CBlas.c" testfile)
   file(WRITE ${testfile}
        "/* Test CBlas compilation */
         \#include\"${include_file}\"
         int main(void)
         {  double  A = 1, B = 2, C = 3;
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 3.0, &A, 1, &B, 1, 1.0, &C, 1);
            return (((int)C) == 9) ? 0 : 1;
         }
         "
        )
   try_run(test_error test_compile ${CMAKE_BINARY_DIR} ${testfile}
           CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${include_path} -DLINK_DIRECTORIES=${library_path} -DLINK_LIBRARIES=${library_name})

   # Check the result
   if ((test_compile) AND (test_error EQUAL 0))
      set(${result} 1 PARENT_SCOPE) # Success
   else()
      if(user_specified)
         message(STATUS "WARNING: Could not run user-specified version of CBLAS")
      endif()
   endif()
endfunction()


# -------------------------------------------------------------------------
# Check for library in path
function(CheckCBlasInPath path libraryName includeFile includePaths result)
# -------------------------------------------------------------------------
   set(success)
   set(includePath)
   set(libraryPath "libraryPath-NOTFOUND")
   find_library(libraryPath NAMES ${libraryName} PATHS ${path} NO_DEFAULT_PATH)
   if (NOT (libraryPath STREQUAL "libraryPath-NOTFOUND"))
      # Check if we can find a corresponding include
      foreach(relativePath ${includePaths})
         file(TO_NATIVE_PATH "${path}/${relativePath}" directory)
         file(TO_NATIVE_PATH "${directory}/${includeFile}" filename)
         if (EXISTS "${filename}")
            set(installation "${libraryPath};${directory}/${includeFile}")
            VerifyCBlasInstallation("${installation}" 0 success)
            if (success)
               break()
            endif()
         endif()
      endforeach()
   endif()

   if (success)
      #set(${result} "${libraryFull};${includePath}" PARENT_SCOPE)
      set(${result} "${installation}" PARENT_SCOPE)
   else()
      set(${result} "" PARENT_SCOPE)
   endif()
endfunction()


# -------------------------------------------------------------------------
# Check for library in paths
function(CheckCBlasInPaths pattern libraryName includeFile cblasName result)
# -------------------------------------------------------------------------
   # Return whenever a result was already found
   if (${result})
      return()
   endif()

   # Expand the directory pattern if needed
   set(libraryPaths "/;/usr;/usr/lib64;/usr/lib;/usr/local;/opt/share")
   set(includePaths "/.;/../include;../../include")
   if (pattern)
      FindPathPattern("${libraryPaths}" "${pattern}" libraryPaths)
   endif()

   # Add possible library subdirectories
   ExpandPathList("${libraryPaths}"
                  ".;lib64;lib;lib64/${OCEAN_PROCESSOR};lib/${OCEAN_PROCESSOR};${OCEAN_PROCESSOR};${OCEAN_PROCESSOR}/lib64;${OCEAN_PROCESSOR}/lib"
                  libraryPaths)

   # Check paths for CBlas
   set(results)
   foreach(libraryPath ${libraryPaths})
      CheckCBlasInPath("${libraryPath}" "${libraryName}" "${includeFile}" "${includePaths}" results)
      if (results)
         # Found a compatible version of CBlas
         break()
      endif()
   endforeach()

   # Set the result
   if (results)
      set(${result} "${results};${cblasName}" PARENT_SCOPE)
   else()
      set(${result} "" PARENT_SCOPE)
   endif()

endfunction()


# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------

# Get the processor name
if (CMAKE_UNAME)
   execute_process( COMMAND uname -m OUTPUT_VARIABLE OCEAN_PROCESSOR)
   string(STRIP "${OCEAN_PROCESSOR}" OCEAN_PROCESSOR)
endif (CMAKE_UNAME)

# Check user-specified installation
option(CBlasInfo "CBlasInfo" "")
if (CBlasInfo)
   VerifyCBlasInstallation("${CBlasInfo}" 1 success)
   if (success)
      message(STATUS "Successfully tested user-specified version of CBLAS")
      set(CBlasInfo "${CBlasInfo};user-specified")
   else ()
      set(CBlasInfo "")
   endif()
endif()

# Check other possible installations
if (NOT CBlasInfo)
   CheckCBlasInPaths(""                    cblas    cblas.h "generic CBLAS" CBlasInfo)
   CheckCBlasInPaths("atlas*"              tatlas   cblas.h "Atlas BLAS"    CBlasInfo)
   CheckCBlasInPaths("OpenBlas*;OpenBLAS*" openblas cblas.h "OpenBLAS"      CBlasInfo)
endif()

# Check the result
if (CBlasInfo)
   # Basic properties
   set(OCBLAS_FOUND   1)
   set(OCBLAS_TYPE    "CBLAS")
   set(OCBLAS_C_FILE  "ocean_cblas.c")

   # Extract the informstion
   list(GET CBlasInfo 0 CBlasLibraryFile)
   list(GET CBlasInfo 1 CBlasIncludeFile)
   list(GET CBlasInfo 2 CBlasName)

   message(STATUS "Compatible CBLAS library found: ${CBlasName}")
   message(STATUS "- using CBLAS library file: ${CBlasLibraryFile}")
   message(STATUS "- using CBLAS include file: ${CBlasIncludeFile}")

   # Get the file components
   get_filename_component(OCBLAS_INCLUDE_PATHS "${CBlasIncludeFile}" DIRECTORY)
   get_filename_component(OCBLAS_INCLUDE       "${CBlasIncludeFile}" NAME)
   get_filename_component(OCBLAS_LIBRARIES     "${CBlasLibraryFile}" ABSOLUTE)

else()
   set(OCBLAS_FOUND 0)
endif()


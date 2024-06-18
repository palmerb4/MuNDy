# @HEADER
# ************************************************************************
#
#            TriBITS: Tribal Build, Integrate, and Test System
#                    Copyright 2013 Sandia Corporation
#
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ************************************************************************
# @HEADER


include(CMakeParseArguments)

# @FUNCTION: mundy_tribits_install_headers()
#
# Function used to (optionally) install header files and directories using the ``install()``
# command. This function facilitates flexible installation configurations, including the
# installation of individual header files, entire directories of headers while preserving
# their directory structure, and the ability to filter which files are installed based on
# pattern matching or regular expressions.
#
# Usage::
#
#   mundy_tribits_install_headers(
#     [HEADERS <h0> <h1> ...]
#     [DIRECTORIES <dir1> <dir2> ...]
#     [INSTALL_SUBDIR <subdir>]
#     [COMPONENT <component>]
#     [FILES_MATCHING [PATTERN <pattern> | REGEX <regex>] [EXCLUDE] [PERMISSIONS <permissions>...]]
#     )
#
# The formal arguments are:
#
#   ``HEADERS <h0> <h1> ...``
#
#     List of individual header files to install. These header files can be specified with
#     either relative path or absolute path. If relative paths are used, they are assumed to
#     be relative to the current source directory.
#
#   ``DIRECTORIES <dir1> <dir2> ...``
#
#     List of directories from which all headers will be installed. This option allows for
#     the installation of headers while preserving their directory structure within the specified
#     directories. Relative paths are considered relative to the current source directory, but 
#     absolute paths can also be used.
#
#   ``INSTALL_SUBDIR <subdir>``
#
#     Optional subdirectory under which the headers or directories will be installed within
#     the standard installation include directory. If specified, the headers or directories 
#     will be installed under ``${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/<subdir>``. If not 
#     specified or empty, they will be installed directly under 
#     ``${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/``.
#
#   ``COMPONENT <component>``
#
#     Optional. If specified, this argument will be passed to the ``install()`` command to 
#     associate the installed headers with a specific component, facilitating component-specific 
#     installations. If not specified, a default component, typically ``${PROJECT_NAME}``, will 
#     be used.
#
#   ``FILES_MATCHING``
#
#     Indicates that file matching filters will be applied to the directories specified in the
#     ``DIRECTORIES`` argument. This should be followed by one or more of the following options:
#
#     ``PATTERN <pattern>``
#       Install only the files matching the globbing pattern.
#
#     ``REGEX <regex>``
#       Install only the files that match the regular expression.
#
#     ``EXCLUDE``
#       Excludes files matching the pattern or regex. This must be used after specifying
#       a PATTERN or REGEX.
#
#     ``PERMISSIONS <permissions>...``
#       Sets the permissions for installed files. This can be a list of permissions such as
#       OWNER_READ, GROUP_READ, etc.
#
# If `${PROJECT_NAME}_INSTALL_LIBRARIES_AND_HEADERS` is ``FALSE``, then no headers will be installed, 
# irrespective of other arguments provided.
#
# Note: Combining ``DIRECTORIES`` with ``FILES_MATCHING`` and its sub-options provides a powerful 
# mechanism for precisely controlling which headers are installed and their permissions, enabling 
# robust installation configurations.
#
function(mundy_tribits_install_headers)

  if (${PROJECT_NAME}_VERBOSE_CONFIGURE)
    set(TRIBITS_INSTALL_HEADERS_DEBUG_DUMP TRUE)
  endif()

  if (TRIBITS_INSTALL_HEADERS_DEBUG_DUMP)
    message("\nTRIBITS_INSTALL_HEADERS: ${ARGN}")
  endif()

  # Parse the arguments, adding support for FILES_MATCHING and its associated sub-options
  cmake_parse_arguments(
    PARSE
    "FILES_MATCHING" # Option
    "INSTALL_SUBDIR;COMPONENT" # One value keywords
    "HEADERS;DIRECTORIES;PATTERN;REGEX;PERMISSIONS" # Multi value keywords
    ${ARGN}
  )

  tribits_check_for_unparsed_arguments()

  if (PARSE_INSTALL_SUBDIR)
    set(INSTALL_DIR "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/${PARSE_INSTALL_SUBDIR}")
  else()
    set(INSTALL_DIR "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}")
  endif()

  if (PARSE_COMPONENT)
    set(COMPONENT ${PARSE_COMPONENT})
  else()
    set(COMPONENT ${PROJECT_NAME})
  endif()

  if (${PROJECT_NAME}_INSTALL_LIBRARIES_AND_HEADERS)
    if (TRIBITS_INSTALL_HEADERS_DEBUG_DUMP)
      message("\nTRIBITS_INSTALL_HEADERS: Installing headers into '${INSTALL_DIR}'")
    endif()

    # Install individual header files
    if (PARSE_HEADERS)
      install(
        FILES ${PARSE_HEADERS}
        DESTINATION "${INSTALL_DIR}"
        COMPONENT ${COMPONENT}
      )
    endif()

    # Install directories with optional FILES_MATCHING logic
    if (PARSE_DIRECTORIES AND PARSE_FILES_MATCHING)
      foreach(dir IN LISTS PARSE_DIRECTORIES)
        # Initialize the install command with DIRECTORY, DESTINATION, and COMPONENT
        set(install_cmd "install(DIRECTORY ${dir} DESTINATION \"${INSTALL_DIR}\" COMPONENT \"${COMPONENT}\"")

        # Check for PATTERN or REGEX and add them to the command
        if (PARSE_PATTERN)
          foreach(pattern IN LISTS PARSE_PATTERN)
            list(APPEND install_cmd "PATTERN \"${pattern}\"")
          endforeach()
        endif()

        if (PARSE_REGEX)
          foreach(regex IN LISTS PARSE_REGEX)
            list(APPEND install_cmd "REGEX \"${regex}\"")
          endforeach()
        endif()

        # Check for EXCLUDE and add it to the command
        if (PARSE_EXCLUDE)
          list(APPEND install_cmd "EXCLUDE")
        endif()

        # Handle PERMISSIONS
        if (PARSE_PERMISSIONS)
          list(APPEND install_cmd "PERMISSIONS")
          foreach(permission IN LISTS PARSE_PERMISSIONS)
            list(APPEND install_cmd "${permission}")
          endforeach()
        endif()

        # Finalize and execute the install command
        string(APPEND install_cmd ")")
        eval(${install_cmd})
      endforeach()
    elseif(PARSE_DIRECTORIES)
      foreach(dir IN LISTS PARSE_DIRECTORIES)
        install(
          DIRECTORY ${dir}
          DESTINATION "${INSTALL_DIR}"
          COMPONENT ${COMPONENT}
        )
      endforeach()
    endif()
  endif()

endfunction()

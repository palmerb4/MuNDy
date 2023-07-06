// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_ASSERT_HPP_
#define MUNDY_ASSERT_HPP_

/// \file MundyAssert.cpp
/// \brief Declaration of our assertion macros

// C++ core libs
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Trilinos libs
#include "Teuchos_Assert.hpp"  // for TEUCHOS_TEST_FOR_EXCEPTION

/// \def MUNDY_THROW_ASSERT
/// \brief Throw an exception if the given assertion is false.
///
/// This macro is a a revised version of Teuchos' \c TEUCHOS_TEST_FOR_EXCEPTION macro with improved logic. Unlike
/// \c TEUCHOS_TEST_FOR_EXCEPTION, this macro will throw an exception if the assertion is false. If the assertion is
/// true, nothing happens. This macro is intended to be used in place of \c TEUCHOS_TEST_FOR_EXCEPTION in order to
/// improve code readability.
///
/// \param assertion_to_test The assertion to test
/// \param exception_to_throw The exception to throw if the assertion is false
/// \param message_to_print The message to print if the assertion is false
#define MUNDY_THROW_ASSERT(assertion_to_test, exception_to_throw, message_to_print)                  \
  do {                                                                                               \
    const bool assertion_failed = !(assertion_to_test);                                              \
    if (assertion_failed) {                                                                          \
      std::ostringstream omsg;                                                                       \
      omsg << "Assertion failed in " << __func__ << "\nFile: " << __FILE__ << "\nLine: " << __LINE__ \
           << "\nMessage: " << message_to_print << std::endl;                                        \
      const std::string &omsgstr = omsg.str();                                                       \
      throw exception_to_throw(omsgstr);                                                             \
    }                                                                                                \
  } while (0)

#endif  // MUNDY_ASSERT_HPP_
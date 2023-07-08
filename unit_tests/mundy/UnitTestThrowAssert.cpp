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

// External libs
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <stdexcept>  // for logic_error, invalid_argument, etc

// Mundy libs
#include <mundy/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace {

//! \name Throw assert tests
//@{

TEST(ThrowAssert, DoesNotThrowWhenTrue) {
  // Check that throw assert does not throw when the condition is true

  // Check that throw assert does not throw a logic error
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::logic_error, "Logic error"));

  // Check that throw assert does not throw an invalid argument
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::invalid_argument, "Invalid argument"));

  // Check that throw assert does not throw a runtime error
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::runtime_error, "Runtime error"));

  // Check that throw assert does not throw a domain error
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::domain_error, "Domain error"));

  // Check that throw assert does not throw a length error
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::length_error, "Length error"));

  // Check that throw assert does not throw an out of range error
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::out_of_range, "Out of range error"));

  // Check that throw assert does not throw a range error
  ASSERT_NO_THROW(MUNDY_THROW_ASSERT(true, std::range_error, "Range error"));
}

TEST(ThrowAssert, ThrowsCorrectErrorType) {
  // Check that throw assert throws the correct error type

  // Check that throw assert throws a logic error
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::logic_error, "Logic error"), std::logic_error);

  // Check that throw assert throws an invalid argument
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid argument"), std::invalid_argument);

  // Check that throw assert throws a runtime error
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::runtime_error, "Runtime error"), std::runtime_error);

  // Check that throw assert throws a domain error
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::domain_error, "Domain error"), std::domain_error);

  // Check that throw assert throws a length error
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::length_error, "Length error"), std::length_error);

  // Check that throw assert throws an out of range error
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::out_of_range, "Out of range error"), std::out_of_range);

  // Check that throw assert throws a range error
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, std::range_error, "Range error"), std::range_error);
}

TEST(ThrowAssert, ThrowsCorrectMessage) {
  // Check that throw assert throws the correct message

  auto some_error_message = "Some error message";
  using some_exception = std::logic_error;

  // Check that throw assert throws the correct message
  ASSERT_THROW(MUNDY_THROW_ASSERT(false, some_exception, some_error_message), some_exception);
  try {
    MUNDY_THROW_ASSERT(false, some_exception, some_error_message);
  } catch (const some_exception& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr(some_error_message));
  }
}
//@}

}  // namespace

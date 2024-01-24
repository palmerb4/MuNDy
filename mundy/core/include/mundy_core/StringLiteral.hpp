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

#ifndef MUNDY_CORE_STRINGLITERAL_HPP_
#define MUNDY_CORE_STRINGLITERAL_HPP_

/// \file StringLiteral.cpp
/// \brief Declaration of our StringLiteral class type.

#include <algorithm>
#include <iostream>

namespace mundy {

namespace core {

/// \brief Literal class type that wraps a constant expression string.
///
/// The StringLiteral struct allows templates to accept constant strings,
/// effectively turning them into compile-time constants. This can result in
/// performance benefits as computations using these strings can be performed at
/// compile time.
///
/// How It Works:
/// The `constexpr` keyword is used to ensure that all computations involving
/// StringLiteral are done at compile time. The StringLiteral struct takes in the
/// string and its size as template parameters and stores them. The string can
/// then be accessed with the `value` member, and the size with the `size` member.
///
/// Example Usage:
/// \code{.cpp}
/// template <StringLiteral StrLit>
/// void Print() {
///   // The size of the string is available as a constant expression.
///   constexpr auto size = StrLit.size;
///
///   // and so is the string's content.
///   constexpr auto contents = StrLit.value;
///
///   std::cout << "Size: " << size << ", Contents: " << contents << std::endl;
/// }
///
/// int main() {
///   Print<make_string_literal("literal string")>();  // Prints "Size: 15, Contents: literal string"
/// }
/// \endcode
///
/// Credit where credit is due: This design entirely originates from Kevin Hartman's Passing String Literals as Template
/// Parameters in C++20 blog post:
/// https://ctrpeach.io/posts/cpp20-string-literal-template-parameters/#:~:text=This%20would%20work%20by%20wrapping,e.g.%20char%5BN%5D%20).
///
/// \tparam StrSize The size of the string literal
template <size_t StrSize>
struct StringLiteral {
  /// @brief Constructor that copies the string literal into the struct.
  /// @param str The string literal to copy
  constexpr explicit StringLiteral(const char (&str)[StrSize]) {
    std::copy_n(str, StrSize, value);
  }

  /// @brief The string literal's content.
  char value[StrSize];

  /// @brief The string literal's size.
  static constexpr size_t size = StrSize;
};  // StringLiteral

/// \brief Helper function for creating a StringLiteral.
///
/// This function is used to create a StringLiteral without having to specify the template parameters.
///
/// \param str The string literal to copy
/// \return A StringLiteral with the same content as the input string literal
template <size_t N>
constexpr StringLiteral<N> make_string_literal(const char (&str)[N]) {
  return StringLiteral<N>(str);
}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_STRINGLITERAL_HPP_

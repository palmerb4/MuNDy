// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

#ifndef MUNDY_MATH_MATRIX3_HPP_
#define MUNDY_MATH_MATRIX3_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Matrix.hpp>        // for mundy::math::Matrix
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance

namespace mundy {

namespace math {

/// \brief Class for a 3x3 matrix with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
template <typename T, ValidAccessor<T> Accessor = Array<T, 9>, typename OwnershipType = Ownership::Owns>
using Matrix3 = Matrix<T, 3, 3, Accessor, OwnershipType>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 9>>
using Matrix3View = Matrix<T, 3, 3, Accessor, Ownership::Views>;

//! \name Matrix3<T, Accessor> views
//@{

/// \brief A helper function to create a Matrix3<T, Accessor> based on a given (valid) accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a Matrix3<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   Matrix3<T, Accessor> mat(data);
/// \endcode
/// you can write
/// \code
///   auto mat = get_matrix3_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION auto get_matrix3_view(Accessor& data) {
  return Matrix3View<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION auto get_matrix3_view(Accessor&& data) {
  return Matrix3View<T, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_MATRIX3_HPP_

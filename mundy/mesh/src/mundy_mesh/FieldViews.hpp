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

#ifndef MUNDY_MESH_FIELDVIEWS_HPP_
#define MUNDY_MESH_FIELDVIEWS_HPP_

/// \file FieldViews.hpp
/// \brief Declaration of the field view helper functions

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/FieldBase.hpp>  // for stk::mesh::FieldBase, stk::mesh::field_data
#include <stk_mesh/base/NgpField.hpp>   // for stk::mesh::NgpField

// Mundy
#include <mundy_math/Matrix3.hpp>     // for mundy::math::get_matrix3_view and mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy::math::get_quaternion_view and mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::math::get_vector3_view and mundy::math::Vector3

namespace mundy {

namespace mesh {

//! \name stk::mesh::Field data views
///@{

/// \brief A helper function for getting a view of a field's data as a scalar
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto& scalar_field_data(
    const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
    const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber)[0];
}

/// \brief A helper function for getting a view of a field's data as a Vector3
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto vector3_field_data(
    const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
    const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_vector3_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief A helper function for getting a view of a field's data as a Quaternion
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto quaternion_field_data(
    const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
    const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_quaternion_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief A helper function for getting a view of a field's data as a Matrix3
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto matrix3_field_data(
    const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
    const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_matrix3_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}
//@}

//! \name stk::mesh::NgpField data views
///@{

/// \brief A helper function for getting a view of a field's data as a Vector3
template <class FieldType>
inline auto vector3_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_vector3<typename FieldType::value_type>(f(i));
}

/// \brief A helper function for getting a view of a field's data as a Quaternion
template <class FieldType>
inline auto quaternion_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_quaternion<typename FieldType::value_type>(f(i));
}

/// \brief A helper function for getting a view of a field's data as a Matrix3
template <class FieldType>
inline auto matrix3_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_matrix3<typename FieldType::value_type>(f(i));
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FIELDVIEWS_HPP_

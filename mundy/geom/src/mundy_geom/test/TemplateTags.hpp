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

#ifndef MUNDY_GEOM_AGGREGATES_TAGS_HPP_
#define MUNDY_GEOM_AGGREGATES_TAGS_HPP_

// C++ core
#include <vector>  // for std::vector

// STK mesh
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/NgpField.hpp>  // for stk::mesh::NgpField

// Mundy mesh
#include <mundy_core/NgpVector.hpp>  // for mundy::core::NgpVector

namespace mundy {

namespace geom {

//! \name Helper stuff
//@{

/// \brief Tag indicating an invalid tag
struct INVALID {
  static constexpr unsigned value = 0;
};
//
constexpr invalid_tag_v = INVALID::value;

namespace tag_type {
struct AGGREGATE {};
struct FIELD {};
struct VECTOR_OF_FIELDS {};
struct SHARED {};
struct VECTOR_OF_SHARED {};
}  // namespace tag_type
//@}

//! \name Map tag to data type helpers
//@{

/// \brief Maps a tag (e.g., FIELD, VECTOR_OF_FIELDS) to its corresponding data type.
///
/// Template parameters:
/// \tparam Tag: The tag indicating the type category.
/// \tparam Scalar: The scalar type (e.g., float, double).
/// \tparam SharedType: (Optional) The type for shared data, used by SHARED or VECTOR_OF_SHARED tags.
template <typename Tag, typename Scalar, typename SharedType = void>
struct map_tag_to_data_type {
  using type = void;
};
//
template <typename Tag, typename Scalar, typename SharedType = void>
using map_tag_to_data_type_t = typename map_tag_to_data_type<Tag, Scalar, SharedType>::type;

/// @brief Specialization of map_tag_to_data_type for types that inherit from tag_type::FIELD
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<tag_type::FIELD, Tag>
struct map_tag_to_data_type<Tag, Scalar> {
  using type = stk::mesh::Field<Scalar>*;
};

/// @brief Specialization of map_tag_to_data_type for types that inherit from tag_type::VECTOR_OF_FIELDS
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, Tag>
struct map_tag_to_data_type<Tag, Scalar> {
  using type = std::vector<stk::mesh::Field<Scalar>*>;
};

/// @brief Specialization of map_tag_to_data_type for types that inherit from tag_type::SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<tag_type::SHARED, Tag>
struct map_tag_to_data_type<Tag, Scalar, SharedType> {
  using type = SharedType;
};

/// @brief Specialization of map_tag_to_data_type for types that inherit from tag_type::VECTOR_OF_SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<tag_type::VECTOR_OF_SHARED, Tag>
struct map_tag_to_data_type<Tag, Scalar, SharedType> {
  using type = std::vector<SharedType>;
};
//@}

//! \name Map tag to NGP data type helpers
//@{

/// \brief Maps a tag (e.g., FIELD, VECTOR_OF_FIELDS) to its corresponding NGP data type.
///
/// Template parameters:
/// \tparam Tag: The tag indicating the type category.
/// \tparam Scalar: The scalar type (e.g., float, double).
/// \tparam SharedType: (Optional) The type for shared data, used by SHARED or VECTOR_OF_SHARED tags.
template <typename Tag, typename Scalar, typename SharedType = void>
struct map_tag_to_ngp_data_type {
  using type = void;
};

template <typename Tag, typename Scalar, typename SharedType = void>
using map_tag_to_ngp_data_type_t = typename map_tag_to_ngp_data_type<Tag, Scalar, SharedType>::type;

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from tag_type::FIELD
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<tag_type::FIELD, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar> {
  using type = stk::mesh::NgpField<Scalar>;
};

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from tag_type::VECTOR_OF_FIELDS
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar> {
  using type = mundy::core::NgpVector<stk::mesh::NgpField<Scalar>>;
};

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from tag_type::SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<tag_type::SHARED, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar, SharedType> {
  using type = mundy::core::NgpVector<SharedType>;
};

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from tag_type::VECTOR_OF_SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<tag_type::VECTOR_OF_SHARED, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar, SharedType> {
  using type = mundy::core::NgpVector<SharedType>;
};
//@}

//! \name Map tag value to tag type helpers
//@{

/// @brief Maps a value to its corresponding tag type.
template <unsigned Value>
struct value_to_tag_type {
  using type = INVALID;
};
//
template <unsigned Value>
using value_to_tag_type_t = value_to_tag_type<Value>::type;
//@}

//! \name Map a tags to types
//@{

/// @brief Map a list of tags to the corresponding aggregate type.
/// This map is one-to-one with the ~set~ of tags, i.e., we must sort and unique
/// the tags before mapping them to the aggregate type.
template <typename... Tags>
struct map_tags_to_aggregate_type {
  using type = void;
};
//
template <typename... Tags>
using map_tags_to_aggregate_type_t = typename map_tags_to_aggregate_type<Tags...>::type;

/// @brief Map a list of tags to the corresponding entity view type.
template <typename... Tags>
struct map_tags_to_entity_view_type {
  using type = void;
};
//
template <typename... Tags>
using map_tags_to_entity_view_type_t = typename map_tags_to_entity_view_type<Tags...>::type;

//@}

//! \name All tags
//@{

all_tags_placeholder
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_TAGS_HPP_

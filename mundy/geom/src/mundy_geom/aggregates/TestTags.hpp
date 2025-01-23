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

//! \name Invalid tag
//@{

/// \brief Tag indicating an invalid tag
struct INVALID {
  static constexpr unsigned value = 0;
};
//
constexpr invalid_tag_v = INVALID::value;
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

/// @brief Specialization of map_tag_to_data_type for types that inherit from data_tag::FIELD
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<data_tag::FIELD, Tag>
struct map_tag_to_data_type<Tag, Scalar> {
  using type = stk::mesh::Field<Scalar>*;
};

/// @brief Specialization of map_tag_to_data_type for types that inherit from data_tag::VECTOR_OF_FIELDS
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<data_tag::VECTOR_OF_FIELDS, Tag>
struct map_tag_to_data_type<Tag, Scalar> {
  using type = std::vector<stk::mesh::Field<Scalar>*>;
};

/// @brief Specialization of map_tag_to_data_type for types that inherit from data_tag::SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<data_tag::SHARED, Tag>
struct map_tag_to_data_type<Tag, Scalar, SharedType> {
  using type = SharedType;
};

/// @brief Specialization of map_tag_to_data_type for types that inherit from data_tag::VECTOR_OF_SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<data_tag::VECTOR_OF_SHARED, Tag>
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

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from data_tag::FIELD
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<data_tag::FIELD, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar> {
  using type = stk::mesh::NgpField<Scalar>;
};

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from data_tag::VECTOR_OF_FIELDS
template <typename Tag, typename Scalar>
  requires std::is_base_of_v<data_tag::VECTOR_OF_FIELDS, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar> {
  using type = mundy::core::NgpVector<stk::mesh::NgpField<Scalar>>;
};

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from data_tag::SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<data_tag::SHARED, Tag>
struct map_tag_to_ngp_data_type<Tag, Scalar, SharedType> {
  using type = mundy::core::NgpVector<SharedType>;
};

/// @brief Specialization of map_tag_to_ngp_data_type for types that inherit from data_tag::VECTOR_OF_SHARED
template <typename Tag, typename Scalar, typename SharedType>
  requires std::is_base_of_v<data_tag::VECTOR_OF_SHARED, Tag>
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

/// @brief The Tag identifying our data type
struct TESTDATA1_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 521554769;
};
//
constexpr auto test_data1_is_field_v = TESTDATA1_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<521554769> {
  using type = TESTDATA1_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct TESTDATA1_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 209178398;
};
//
constexpr auto test_data1_is_shared_v = TESTDATA1_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<209178398> {
  using type = TESTDATA1_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct TESTDATA1_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 2596270206;
};
//
constexpr auto test_data1_is_vector_of_fields_v = TESTDATA1_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2596270206> {
  using type = TESTDATA1_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct TESTDATA1_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 3377808615;
};
//
constexpr auto test_data1_is_vector_of_shared_v = TESTDATA1_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3377808615> {
  using type = TESTDATA1_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct TESTDATA2_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 843188577;
};
//
constexpr auto test_data2_is_field_v = TESTDATA2_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<843188577> {
  using type = TESTDATA2_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct TESTDATA2_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1419683722;
};
//
constexpr auto test_data2_is_shared_v = TESTDATA2_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1419683722> {
  using type = TESTDATA2_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct TESTDATA2_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 1970425085;
};
//
constexpr auto test_data2_is_vector_of_fields_v = TESTDATA2_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1970425085> {
  using type = TESTDATA2_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct TESTDATA2_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2740490944;
};
//
constexpr auto test_data2_is_vector_of_shared_v = TESTDATA2_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2740490944> {
  using type = TESTDATA2_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct TEST_AGGREGATE : public tag_type::AGGREGATE {
  static constexpr unsigned value = 2939929087;
};
//
constexpr auto test_aggregate_v = TEST_AGGREGATE::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2939929087> {
  using type = TEST_AGGREGATE;
};

//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_TAGS_HPP_

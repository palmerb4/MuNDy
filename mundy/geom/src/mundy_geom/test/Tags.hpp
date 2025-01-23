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

/// @brief The Tag identifying our data type
struct ELLIPSOID : public tag_type::AGGREGATE {
  static constexpr unsigned value = 3295512787;
};
//
constexpr auto ellipsoid_v = ELLIPSOID::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3295512787> {
  using type = ELLIPSOID;
};


/// @brief The Tag identifying our data type
struct LINE : public tag_type::AGGREGATE {
  static constexpr unsigned value = 1451970172;
};
//
constexpr auto line_v = LINE::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1451970172> {
  using type = LINE;
};


/// @brief The Tag identifying our data type
struct POINT : public tag_type::AGGREGATE {
  static constexpr unsigned value = 960399992;
};
//
constexpr auto point_v = POINT::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<960399992> {
  using type = POINT;
};


/// @brief The Tag identifying our data type
struct SPHERE : public tag_type::AGGREGATE {
  static constexpr unsigned value = 727820882;
};
//
constexpr auto sphere_v = SPHERE::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<727820882> {
  using type = SPHERE;
};


/// @brief The Tag identifying our data type
struct SPHEROCYLINDER : public tag_type::AGGREGATE {
  static constexpr unsigned value = 2048848047;
};
//
constexpr auto spherocylinder_v = SPHEROCYLINDER::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2048848047> {
  using type = SPHEROCYLINDER;
};


/// @brief The Tag identifying our data type
struct SPHEROCYLINDER_SEGMENT : public tag_type::AGGREGATE {
  static constexpr unsigned value = 3959765592;
};
//
constexpr auto spherocylinder_segment_v = SPHEROCYLINDER_SEGMENT::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3959765592> {
  using type = SPHEROCYLINDER_SEGMENT;
};


/// @brief The Tag identifying our data type
struct V_SEGMENT : public tag_type::AGGREGATE {
  static constexpr unsigned value = 724720585;
};
//
constexpr auto v_segment_v = V_SEGMENT::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<724720585> {
  using type = V_SEGMENT;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 2700572390;
};
//
constexpr auto center_data_is_field_v = CENTER_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2700572390> {
  using type = CENTER_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 529410409;
};
//
constexpr auto center_data_is_shared_v = CENTER_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<529410409> {
  using type = CENTER_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 1431802533;
};
//
constexpr auto center_data_is_vector_of_fields_v = CENTER_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1431802533> {
  using type = CENTER_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 1113888891;
};
//
constexpr auto center_data_is_vector_of_shared_v = CENTER_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1113888891> {
  using type = CENTER_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 4205492020;
};
//
constexpr auto direction_data_is_field_v = DIRECTION_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4205492020> {
  using type = DIRECTION_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1000626469;
};
//
constexpr auto direction_data_is_shared_v = DIRECTION_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1000626469> {
  using type = DIRECTION_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 3724915481;
};
//
constexpr auto direction_data_is_vector_of_fields_v = DIRECTION_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3724915481> {
  using type = DIRECTION_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 125695553;
};
//
constexpr auto direction_data_is_vector_of_shared_v = DIRECTION_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<125695553> {
  using type = DIRECTION_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 71321604;
};
//
constexpr auto length_data_is_field_v = LENGTH_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<71321604> {
  using type = LENGTH_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1815487130;
};
//
constexpr auto length_data_is_shared_v = LENGTH_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1815487130> {
  using type = LENGTH_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 3415361540;
};
//
constexpr auto length_data_is_vector_of_fields_v = LENGTH_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3415361540> {
  using type = LENGTH_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 811099358;
};
//
constexpr auto length_data_is_vector_of_shared_v = LENGTH_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<811099358> {
  using type = LENGTH_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 3132960;
};
//
constexpr auto node_coords_data_is_field_v = NODE_COORDS_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3132960> {
  using type = NODE_COORDS_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 2819210058;
};
//
constexpr auto node_coords_data_is_shared_v = NODE_COORDS_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2819210058> {
  using type = NODE_COORDS_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 2805838448;
};
//
constexpr auto node_coords_data_is_vector_of_fields_v = NODE_COORDS_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2805838448> {
  using type = NODE_COORDS_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2704229092;
};
//
constexpr auto node_coords_data_is_vector_of_shared_v = NODE_COORDS_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2704229092> {
  using type = NODE_COORDS_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 3754328683;
};
//
constexpr auto orientation_data_is_field_v = ORIENTATION_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3754328683> {
  using type = ORIENTATION_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 2664864330;
};
//
constexpr auto orientation_data_is_shared_v = ORIENTATION_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2664864330> {
  using type = ORIENTATION_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 3467959128;
};
//
constexpr auto orientation_data_is_vector_of_fields_v = ORIENTATION_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3467959128> {
  using type = ORIENTATION_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2569831686;
};
//
constexpr auto orientation_data_is_vector_of_shared_v = ORIENTATION_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2569831686> {
  using type = ORIENTATION_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 1595799434;
};
//
constexpr auto radii_data_is_field_v = RADII_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1595799434> {
  using type = RADII_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1820459012;
};
//
constexpr auto radii_data_is_shared_v = RADII_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1820459012> {
  using type = RADII_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 479266086;
};
//
constexpr auto radii_data_is_vector_of_fields_v = RADII_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<479266086> {
  using type = RADII_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 3728880861;
};
//
constexpr auto radii_data_is_vector_of_shared_v = RADII_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3728880861> {
  using type = RADII_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 120913482;
};
//
constexpr auto radius_data_is_field_v = RADIUS_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<120913482> {
  using type = RADIUS_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1673798306;
};
//
constexpr auto radius_data_is_shared_v = RADIUS_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1673798306> {
  using type = RADIUS_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 2981683851;
};
//
constexpr auto radius_data_is_vector_of_fields_v = RADIUS_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2981683851> {
  using type = RADIUS_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 236077426;
};
//
constexpr auto radius_data_is_vector_of_shared_v = RADIUS_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<236077426> {
  using type = RADIUS_DATA_IS_VECTOR_OF_SHARED;
};

//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_TAGS_HPP_

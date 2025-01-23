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
struct ELLIPSOID : public tag_type::AGGREGATE {
  static constexpr unsigned value = 1877636769;
};
//
constexpr auto ellipsoid_v = ELLIPSOID::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1877636769> {
  using type = ELLIPSOID;
};


/// @brief The Tag identifying our data type
struct LINE : public tag_type::AGGREGATE {
  static constexpr unsigned value = 4234856900;
};
//
constexpr auto line_v = LINE::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4234856900> {
  using type = LINE;
};


/// @brief The Tag identifying our data type
struct POINT : public tag_type::AGGREGATE {
  static constexpr unsigned value = 1125053675;
};
//
constexpr auto point_v = POINT::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1125053675> {
  using type = POINT;
};


/// @brief The Tag identifying our data type
struct SPHERE : public tag_type::AGGREGATE {
  static constexpr unsigned value = 4253717452;
};
//
constexpr auto sphere_v = SPHERE::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4253717452> {
  using type = SPHERE;
};


/// @brief The Tag identifying our data type
struct SPHEROCYLINDER : public tag_type::AGGREGATE {
  static constexpr unsigned value = 736618394;
};
//
constexpr auto spherocylinder_v = SPHEROCYLINDER::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<736618394> {
  using type = SPHEROCYLINDER;
};


/// @brief The Tag identifying our data type
struct SPHEROCYLINDER_SEGMENT : public tag_type::AGGREGATE {
  static constexpr unsigned value = 3694052767;
};
//
constexpr auto spherocylinder_segment_v = SPHEROCYLINDER_SEGMENT::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3694052767> {
  using type = SPHEROCYLINDER_SEGMENT;
};


/// @brief The Tag identifying our data type
struct V_SEGMENT : public tag_type::AGGREGATE {
  static constexpr unsigned value = 4038278200;
};
//
constexpr auto v_segment_v = V_SEGMENT::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4038278200> {
  using type = V_SEGMENT;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 3360046322;
};
//
constexpr auto center_data_is_field_v = CENTER_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3360046322> {
  using type = CENTER_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 2474318228;
};
//
constexpr auto center_data_is_shared_v = CENTER_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2474318228> {
  using type = CENTER_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 2253179595;
};
//
constexpr auto center_data_is_vector_of_fields_v = CENTER_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2253179595> {
  using type = CENTER_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct CENTER_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 186288445;
};
//
constexpr auto center_data_is_vector_of_shared_v = CENTER_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<186288445> {
  using type = CENTER_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 210061914;
};
//
constexpr auto direction_data_is_field_v = DIRECTION_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<210061914> {
  using type = DIRECTION_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1095544457;
};
//
constexpr auto direction_data_is_shared_v = DIRECTION_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1095544457> {
  using type = DIRECTION_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 530126827;
};
//
constexpr auto direction_data_is_vector_of_fields_v = DIRECTION_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<530126827> {
  using type = DIRECTION_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct DIRECTION_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2517842629;
};
//
constexpr auto direction_data_is_vector_of_shared_v = DIRECTION_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2517842629> {
  using type = DIRECTION_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 3839538095;
};
//
constexpr auto length_data_is_field_v = LENGTH_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3839538095> {
  using type = LENGTH_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 3759456762;
};
//
constexpr auto length_data_is_shared_v = LENGTH_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3759456762> {
  using type = LENGTH_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 4158217842;
};
//
constexpr auto length_data_is_vector_of_fields_v = LENGTH_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4158217842> {
  using type = LENGTH_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct LENGTH_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2746286935;
};
//
constexpr auto length_data_is_vector_of_shared_v = LENGTH_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2746286935> {
  using type = LENGTH_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 4160273902;
};
//
constexpr auto node_coords_data_is_field_v = NODE_COORDS_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4160273902> {
  using type = NODE_COORDS_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 897728872;
};
//
constexpr auto node_coords_data_is_shared_v = NODE_COORDS_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<897728872> {
  using type = NODE_COORDS_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 3186010238;
};
//
constexpr auto node_coords_data_is_vector_of_fields_v = NODE_COORDS_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3186010238> {
  using type = NODE_COORDS_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct NODE_COORDS_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2482493857;
};
//
constexpr auto node_coords_data_is_vector_of_shared_v = NODE_COORDS_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2482493857> {
  using type = NODE_COORDS_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 2941241329;
};
//
constexpr auto orientation_data_is_field_v = ORIENTATION_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2941241329> {
  using type = ORIENTATION_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 2235066877;
};
//
constexpr auto orientation_data_is_shared_v = ORIENTATION_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2235066877> {
  using type = ORIENTATION_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 3317661792;
};
//
constexpr auto orientation_data_is_vector_of_fields_v = ORIENTATION_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3317661792> {
  using type = ORIENTATION_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct ORIENTATION_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 2789344524;
};
//
constexpr auto orientation_data_is_vector_of_shared_v = ORIENTATION_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<2789344524> {
  using type = ORIENTATION_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 3777548098;
};
//
constexpr auto radii_data_is_field_v = RADII_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<3777548098> {
  using type = RADII_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 4280415232;
};
//
constexpr auto radii_data_is_shared_v = RADII_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<4280415232> {
  using type = RADII_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 269965201;
};
//
constexpr auto radii_data_is_vector_of_fields_v = RADII_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<269965201> {
  using type = RADII_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct RADII_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 1714659070;
};
//
constexpr auto radii_data_is_vector_of_shared_v = RADII_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1714659070> {
  using type = RADII_DATA_IS_VECTOR_OF_SHARED;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_FIELD : public tag_type::FIELD {
  static constexpr unsigned value = 353356131;
};
//
constexpr auto radius_data_is_field_v = RADIUS_DATA_IS_FIELD::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<353356131> {
  using type = RADIUS_DATA_IS_FIELD;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_SHARED : public tag_type::SHARED {
  static constexpr unsigned value = 1275253671;
};
//
constexpr auto radius_data_is_shared_v = RADIUS_DATA_IS_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1275253671> {
  using type = RADIUS_DATA_IS_SHARED;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_VECTOR_OF_FIELDS : public tag_type::VECTOR_OF_FIELDS {
  static constexpr unsigned value = 1805038336;
};
//
constexpr auto radius_data_is_vector_of_fields_v = RADIUS_DATA_IS_VECTOR_OF_FIELDS::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<1805038336> {
  using type = RADIUS_DATA_IS_VECTOR_OF_FIELDS;
};


/// @brief The Tag identifying our data type
struct RADIUS_DATA_IS_VECTOR_OF_SHARED : public tag_type::VECTOR_OF_SHARED {
  static constexpr unsigned value = 563935248;
};
//
constexpr auto radius_data_is_vector_of_shared_v = RADIUS_DATA_IS_VECTOR_OF_SHARED::value;

/// @brief The inverse map from tag value to tag type
template <>
struct value_to_tag_type<563935248> {
  using type = RADIUS_DATA_IS_VECTOR_OF_SHARED;
};

//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_TAGS_HPP_

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

#ifndef MUNDY_GEOM_AGGREGATES_TYPES_HPP_
#define MUNDY_GEOM_AGGREGATES_TYPES_HPP_

#include <mundy_core/NgpVector.hpp>
#include <tuple>

namespace mundy {

namespace geom {

namespace tag_type {
struct AGGREGATE {};
struct FIELD {};
struct VECTOR_OF_FIELDS {};
struct SHARED {};
struct VECTOR_OF_SHARED {};
}  // namespace data_tag

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

template <typename Tag, typename Scalar, typename SharedType = void>
using map_tag_to_data_type_t = typename map_tag_to_data_type<Tag, Scalar, SharedType>::type;

template <typename Scalar>
struct map_tag_to_data_type<data_tag::FIELD, Scalar> {
  using type = stk::mesh::Field<Scalar>*;
};

template <typename Scalar>
struct map_tag_to_data_type<data_tag::VECTOR_OF_FIELDS, Scalar> {
  using type = std::vector<stk::mesh::Field<Scalar>*>;
};

template <typename Scalar, typename SharedType>
struct map_tag_to_data_type<data_tag::SHARED, Scalar, SharedType> {
  using type = SharedType;
};

template <typename Scalar, typename SharedType>
struct map_tag_to_data_type<data_tag::VECTOR_OF_SHARED, Scalar, SharedType> {
  using type = std::vector<SharedType>;
};

template <typename Tag, typename Scalar, typename SharedType = void>
  requires std::is_same_v<Tag, data_tag::FIELD> || std::is_same_v<Tag, data_tag::VECTOR_OF_FIELDS> ||
           std::is_same_v<Tag, data_tag::SHARED> || std::is_same_v<Tag, data_tag::VECTOR_OF_SHARED>
auto tagged_data_to_ngp(const map_tag_to_data_type_t<Tag, Scalar, SharedType>& tagged_data) {
  if constexpr (std::is_same_v<Tag, data_tag::FIELD>) {
    // stk::mesh::Field* -> stk::mesh::NgpField
    return stk::mesh::get_updated_ngp_field<Scalar>(*tagged_data);
  } else if constexpr (std::is_same_v<Tag, data_tag::VECTOR_OF_FIELDS>) {
    // std::vector<stk::mesh::Field*> -> mundy::core::NgpVector<stk::mesh::NgpField>
    mundy::core::NgpVector<stk::mesh::NgpField<Scalar>> ngp_fields;
    for (const auto center_field_ptr : tagged_data) {
      ngp_fields.push_back(stk::mesh::get_updated_ngp_field<Scalar>(*center_field_ptr));
    }
    return ngp_fields;
  } else if constexpr (std::is_same_v<Tag, data_tag::SHARED>) {
    // SharedType -> mundy::core::NgpVector<SharedType> (size 1)
    return mundy::core::NgpVector<SharedType>{tagged_data};  // N-value constructor
  } else {
    // std::vector<SharedType> -> mundy::core::NgpVector<SharedType>
    return mundy::core::NgpVector<SharedType>(tagged_data);  // Copy from std::vector
  }
}

/// \brief Maps a tag (e.g., FIELD, VECTOR_OF_FIELDS) to its corresponding ngp data type.
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

template <typename Scalar>
struct map_tag_to_ngp_data_type<data_tag::FIELD, Scalar> {
  using type = stk::mesh::NgpField<Scalar>;
};

template <typename Scalar>
struct map_tag_to_ngp_data_type<data_tag::VECTOR_OF_FIELDS, Scalar> {
  using type = mundy::core::NgpVector<stk::mesh::NgpField<Scalar>>;
};

template <typename Scalar, typename SharedType>
struct map_tag_to_ngp_data_type<data_tag::SHARED, Scalar, SharedType> {
  using type = mundy::core::NgpVector<SharedType>;
};

template <typename Scalar, typename SharedType>
struct map_tag_to_ngp_data_type<data_tag::VECTOR_OF_SHARED, Scalar, SharedType> {
  using type = mundy::core::NgpVector<SharedType>;
};

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

struct invalid_tag {
  static constexpr unsigned value = 0;
};
constexpr invalid_tag_v = invalid_tag::value;

template <unsigned Value>
struct value_to_tag_type {
  using type = invalid_tag;
};
//
template <unsigned Value>
using value_to_tag_type_t = value_to_tag_type<Value>::type;

template <typename... Tags>
struct aggregate_type {
  using tags = std::tuple<Tags...>;
  using type = void;
};
//
template <typename... Tags>
using aggregate_type_t = typename aggregate_type<Tags...>::type;

template <typename... Tags>
struct entity_view_type {
  using tags = std::tuple<Tags...>;
  using type = void;
};
//
template <typename... Tags>
using entity_view_type_t = typename entity_view_type<Tags...>::type;

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_TYPES_HPP_

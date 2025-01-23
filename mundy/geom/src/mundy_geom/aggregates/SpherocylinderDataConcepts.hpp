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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERDATACONCEPTS_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERDATACONCEPTS_HPP_

// C++ core
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy
#include <mundy_geom/primitives/Spherocylinder.hpp>  // for mundy::geom::ValidSpherocylinderType
#include <mundy_mesh/BulkData.hpp>                   // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_geom/aggregates/Types.hpp>  // For value_to_tag_type

/// @brief The Tag identifying our data type
struct spherocylinder_tag {
  static constexpr unsigned value = 3814515394;
};
//
constexpr spherocylinder_tag_v = spherocylinder_tag::value;

/// @brief The inverse map from tag value to tag type 
template <>
struct value_to_tag_type<3814515394> {
    using type = spherocylinder_tag;
};

namespace mundy {

namespace geom {

/// \brief Check if the type provides the same data as SpherocylinderData
template <typename Agg>
concept ValidSpherocylinderDataType = requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::get_topology() } -> std::convertible_to<stk::topology::topology_t>;
    }
  && std::convertible_to<decltype(std::declval<Agg>().bulk_data()), const stk::mesh::BulkData&>
  && std::convertible_to<decltype(std::declval<Agg>().center_data()), const stk::mesh::Field<typename Agg::scalar_t>&>
  && std::convertible_to<decltype(std::declval<Agg>().orientation_data()), const stk::mesh::Field<typename Agg::scalar_t>&>
  && (std::convertible_to<decltype(std::declval<Agg>().radius_data()), const stk::mesh::Field<typename Agg::scalar_t>&> ||
      std::convertible_to<decltype(std::declval<Agg>().radius_data()), const typename Agg::scalar_t&>)
  && (std::convertible_to<decltype(std::declval<Agg>().length_data()), const stk::mesh::Field<typename Agg::scalar_t>&> ||
      std::convertible_to<decltype(std::declval<Agg>().length_data()), const typename Agg::scalar_t&>);
    
/// \brief Check if the type provides the same data as NgpSpherocylinderData
template <typename Agg>
concept ValidNgpSpherocylinderDataType = requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::get_topology() } -> std::convertible_to<stk::topology::topology_t>;
    }
  && std::convertible_to<decltype(std::declval<Agg>().ngp_mesh()), stk::mesh::NgpMesh>
  && std::convertible_to<decltype(std::declval<Agg>().center_data()), stk::mesh::NgpField<typename Agg::scalar_t>>
  && std::convertible_to<decltype(std::declval<Agg>().orientation_data()), stk::mesh::NgpField<typename Agg::scalar_t>>
  && (std::convertible_to<decltype(std::declval<Agg>().radius_data()), stk::mesh::NgpField<typename Agg::scalar_t>> ||
      std::convertible_to<decltype(std::declval<Agg>().radius_data()), typename Agg::scalar_t>)
  && (std::convertible_to<decltype(std::declval<Agg>().length_data()), stk::mesh::NgpField<typename Agg::scalar_t>> ||
      std::convertible_to<decltype(std::declval<Agg>().length_data()), typename Agg::scalar_t>);

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERDATACONCEPTS_HPP_
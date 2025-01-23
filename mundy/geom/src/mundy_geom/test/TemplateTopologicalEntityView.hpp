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

#ifndef MUNDY_GEOM_AGGREGATES_EXAMPLENAMEENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_EXAMPLENAMEENTITYVIEW_HPP_

// C++ core
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an STK entity meant to represent a example_name
///
/// Use \ref create_example_name_entity_view to build an ExampleNameEntityView object with automatic template deduction.
template <typename Base, typename ExampleNameDataType>
class ExampleNameEntityView : public Base {
  static_assert_valid_topology_placeholder;

 public:
  using scalar_t = typename ExampleNameDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = ExampleNameDataType::topology_t;
  static constexpr stk::topology::rank_t rank_t = ExampleNameDataType::rank_t;

  ExampleNameEntityView(const Base& base, const ExampleNameDataType& data) : Base(base), data_(data) {
  }

  entity_getters_placeholder;

  data_getters_placeholder;

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_view(Args&&... args) const {
    return NextAugment<ExampleNameEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const ExampleNameDataType& data_;
};  // ExampleNameEntityView

/// @brief An ngp-compatible view of an example_name entity
template <typename Base, typename ExampleNameDataType>
class NgpExampleNameEntityView : public Base {
  static_assert_valid_topology_placeholder;

 public:
  using scalar_t = typename ExampleNameDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = ExampleNameDataType::topology_t;
  static constexpr stk::topology::rank_t rank_t = ExampleNameDataType::rank_t;

  KOKKOS_INLINE_FUNCTION
  NgpExampleNameEntityView(const Base& base, const ExampleNameDataType& data) : Base(base), data_(data) {
  }

  entity_index_getters_placeholder;

  data_getters_ngp_placeholder;

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_view(Args&&... args) const {
    return NextAugment<NgpExampleNameEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const ExampleNameDataType& data_;
};  // NgpExampleNameEntityView

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_EXAMPLENAMEENTITYVIEW_HPP_
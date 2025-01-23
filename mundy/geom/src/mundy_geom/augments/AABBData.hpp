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

#ifndef MUNDY_GEOM_AUGMENTS_AABBDATA_HPP_
#define MUNDY_GEOM_AUGMENTS_AABBDATA_HPP_

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
#include <mundy_geom/augments/AABBDataConcepts.hpp>  // for mundy::geom::ValidAABBDataType
#include <mundy_geom/augments/AABBEntityView.hpp>    // for mundy::geom::AABBEntityView
#include <mundy_geom/primitives/AABB.hpp>            // for mundy::geom::ValidAABBType
#include <mundy_mesh/BulkData.hpp>                   // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of objects with aabbs
/// This is designed as an augment on top of other data, such as adding aabb data to a collection of spheres.
///
/// The rank of an AABB does not change the access pattern for the underlying data.
///
/// \tparam Scalar The scalar type of the aabb data's aabb.
template <typename Base>
class AABBData : public Base {
 public:
  using scalar_t = typename Base::scalar_t;
  using aabb_data_t = stk::mesh::Field<scalar_t>;

  AABBData(const Base& base, const aabb_data_t& aabb_data) : Base(base), aabb_data_(aabb_data) {
    MUNDY_THROW_REQUIRE(aabb_data.entity_rank() == Base::get_rank(), std::invalid_argument,
                        "The rank of the aabb field must match the rank of the base");
  }

  const aabb_data_t& aabb_data() const {
    return aabb_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<AABBData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = AABBData<Base>;
    return Base::get_entity_view(entity).template augment_view<AABBEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = AABBData<Base>;
    return Base::get_entity_view(entity).template augment_view<AABBEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    return create_ngp_aabb_data(Base::get_updated_ngp_data(),  //
                                          stk::mesh::get_updated_ngp_field<scalar_t>(aabb_data_));
  }

 private:
  const aabb_data_t& aabb_data_;
};  // AABBData

/// \brief A struct to hold the data for a collection of NGP-compatible aabbs
/// See the discussion for AABBData for more information. Only difference is NgpFields over Fields.
template <typename Base>
class NgpAABBData : public Base {
 public:
  using scalar_t = typename Base::scalar_t;
  using aabb_data_t = stk::mesh::NgpField<scalar_t>;

  NgpAABBData(const Base& base, const aabb_data_t& aabb_data) : Base(base), aabb_data_(aabb_data) {
    MUNDY_THROW_REQUIRE(aabb_data.get_rank() == Base::get_rank(), std::invalid_argument,
                        "The rank of the aabb field must match the rank of the base");
  }

  const aabb_data_t& aabb_data() const {
    return aabb_data_;
  }

  aabb_data_t& aabb_data() {
    return aabb_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpAABBData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpAABBData<Base>;
    return Base::get_entity_view(entity_index).template augment_view<NgpAABBEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpAABBData<Base>;
    return Base::get_entity_view(entity_index).template augment_view<NgpAABBEntityView, our_t>(*this);
  }

 private:
  aabb_data_t aabb_data_;
};  // NgpAABBData

/// \brief A helper function to create an AABBData object
/// Typically, these objects are created via adding them as an augment to another object.
template <typename Base>  // deduced
auto create_aabb_data(const Base& base, const stk::mesh::Field<typename Base::scalar_t>& aabb_data) {
  return AABBData<Base>{base, aabb_data};
}

/// \brief A helper function to create an NgpEllipsoidData object
/// See the discussion for create_ellipsoid_data for more information. Only difference is NgpFields over Fields.
template <typename Base>  // deduced
auto create_ngp_aabb_data(const Base& base, const stk::mesh::NgpField<typename Base::scalar_t>& aabb_data) {
  return NgpAABBData<Base>{base, aabb_data};
}

/// \brief A helper function to get an updated NgpAABBData object from a AABBData object
/// \param data The AABBData object to convert
template <typename Base>  // deduced
auto get_updated_ngp_data(const AABBData<Base>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_AABB_HPP_
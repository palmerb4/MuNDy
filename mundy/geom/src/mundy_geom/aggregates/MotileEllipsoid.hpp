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

#ifndef MUNDY_GEOM_AGGREGATES_MOTILEELLIPSOID_HPP_
#define MUNDY_GEOM_AGGREGATES_MOTILEELLIPSOID_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/aggregates/GeomEllipsoid.hpp>  // for mundy::geom::EllipsoidData
#include <mundy_geom/primitives/Ellipsoid.hpp>      // for mundy::geom::ValidEllipsoidType
#include <mundy_mesh/BulkData.hpp>                  // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of motile_ellipsoids
template <ValidEllipsoidDataType EllipsoidDataType>
class MotileEllipsoidData : public EllipsoidDataType {
 public:
  using ellipsoid_data_t = EllipsoidDataType;
  using scalar_t = typename EllipsoidDataType::scalar_t;
  using center_data_t = typename EllipsoidDataType::center_data_t;
  using orientation_data_t = typename EllipsoidDataType::orientation_data_t;
  using axis_lengths_data_t = typename EllipsoidDataType::axis_lengths_data_t;
  using velocity_data_t = stk::mesh::Field<scalar_t>;
  using angular_velocity_data_t = stk::mesh::Field<scalar_t>;

  /// \brief Constructor
  MotileEllipsoidData(ellipsoid_data_t ellipsoid_data, velocity_data_t& velocity_data,
                      angular_velocity_data_t& angular_velocity_data)
      : EllipsoidDataType(ellipsoid_data),
        velocity_data_(velocity_data),
        angular_velocity_data_(angular_velocity_data) {
    stk::topology our_topology = ellipsoid_data.topology_t;
    MUNDY_THROW_ASSERT(velocity_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The velocity_data data must be a field of NODE_RANK");
    MUNDY_THROW_ASSERT(angular_velocity_data.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The angular_velocity_data data must be a field of the same rank as the motile_ellipsoid");
  }

  const velocity_data_t& velocity_data() const {
    return velocity_data_;
  }

  velocity_data_t& velocity_data() {
    return velocity_data_;
  }

  const angular_velocity_data_t& angular_velocity_data() const {
    return angular_velocity_data_;
  }

  angular_velocity_data_t& angular_velocity_data() {
    return angular_velocity_data_;
  }

 private:
  velocity_data_t& velocity_data_;
  angular_velocity_data_t& angular_velocity_data_;
};  // MotileEllipsoidData

/// \brief Aggregate to hold the data for a collection of NGP-compatible motile_ellipsoids
/// See the discussion for MotileEllipsoidData for more information. Only difference is NgpFields over Fields.
template <ValidNgpEllipsoidDataType NgpEllipsoidDataType>
class NgpMotileEllipsoidData : public NgpEllipsoidDataType {
 public:
  using ellipsoid_data_t = NgpEllipsoidDataType;
  using scalar_t = typename NgpEllipsoidDataType::scalar_t;
  using center_data_t = typename NgpEllipsoidDataType::center_data_t;
  using orientation_data_t = typename NgpEllipsoidDataType::orientation_data_t;
  using axis_lengths_data_t = typename NgpEllipsoidDataType::axis_lengths_data_t;
  using velocity_data_t = stk::mesh::NgpField<scalar_t>;
  using angular_velocity_data_t = stk::mesh::NgpField<scalar_t>;

  /// \brief Constructor
  NgpMotileEllipsoidData(ellipsoid_data_t ellipsoid_data, velocity_data_t& velocity_data,
                         angular_velocity_data_t& angular_velocity_data)
      : NgpEllipsoidDataType(ellipsoid_data),
        velocity_data_(velocity_data),
        angular_velocity_data_(angular_velocity_data) {
    stk::topology our_topology = NgpEllipsoidDataType::topology_t;
    MUNDY_THROW_ASSERT(velocity_data_.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The velocity_data data must be a field of NODE_RANK");
    MUNDY_THROW_ASSERT(angular_velocity_data_.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The angular_velocity_data data must be a field of the same rank as the motile_ellipsoid");
  }

  const velocity_data_t& velocity_data() const {
    return velocity_data_;
  }

  velocity_data_t& velocity_data() {
    return velocity_data_;
  }

  const angular_velocity_data_t& angular_velocity_data() const {
    return angular_velocity_data_;
  }

  angular_velocity_data_t& angular_velocity_data() {
    return angular_velocity_data_;
  }

 private:
  velocity_data_t& velocity_data_;
  angular_velocity_data_t& angular_velocity_data_;
};  // NgpMotileEllipsoidData

/// \brief A helper function to create a MotileEllipsoidData object
///
/// This function creates a MotileEllipsoidData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename EllipsoidDataType,        // deduced
          typename VelocityDataType,         // deduced
          typename AngularVelocityDataType>  // deduced
auto create_motile_ellipsoid_data(stk::mesh::BulkData& bulk_data, EllipsoidDataType ellipsoid_data,
                                  VelocityDataType& velocity_data, AngularVelocityDataType& angular_velocity_data) {
  return MotileEllipsoidData<EllipsoidDataType, VelocityDataType, AngularVelocityDataType>{
      ellipsoid_data, velocity_data, angular_velocity_data};
}

/// \brief A helper function to create a MotileEllipsoidData object
///
/// This function creates a MotileEllipsoidData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename CenterDataType,                // deduced
          typename OrientationDataType,           // deduced
          typename AxisLengthsDataType,           // deduced
          typename VelocityDataType,              // deduced
          typename AngularVelocityDataType>       // deduced
auto create_motile_ellipsoid_data(stk::mesh::BulkData& bulk_data, CenterDataType& center_data,
                                  OrientationDataType& orientation_data, AxisLengthsDataType& axis_lengths_data,
                                  VelocityDataType& velocity_data, AngularVelocityDataType& angular_velocity_data) {
  auto ellipsoid_data =
      create_ellipsoid_data<Scalar, OurTopology, CenterDataType, OrientationDataType, AxisLengthsDataType>(
          bulk_data, center_data, orientation_data, axis_lengths_data);
  return create_motile_ellipsoid_data(bulk_data, ellipsoid_data, velocity_data, angular_velocity_data);
}

/// \brief A helper function to create a NgpMotileEllipsoidData object
/// See the discussion for create_motile_ellipsoid_data for more information. Only difference is NgpFields over Fields.
template <typename NgpEllipsoidDataType,        // deduced
          typename VelocityDataType,         // deduced
          typename AngularVelocityDataType>  // deduced
auto create_ngp_motile_ellipsoid_data(stk::mesh::NgpMesh ngp_mesh, NgpEllipsoidDataType& ellipsoid_data,
                                      VelocityDataType& velocity_data, AngularVelocityDataType& angular_velocity_data) {
  return NgpMotileEllipsoidData<NgpEllipsoidDataType, VelocityDataType, AngularVelocityDataType>{
      ellipsoid_data, velocity_data, angular_velocity_data};
}

/// \brief A helper function to create a NgpMotileEllipsoidData object
/// See the discussion for create_motile_ellipsoid_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename CenterDataType,                // deduced
          typename OrientationDataType,           // deduced
          typename AxisLengthsDataType,           // deduced
          typename VelocityDataType,              // deduced
          typename AngularVelocityDataType>       // deduced
auto create_ngp_motile_ellipsoid_data(stk::mesh::NgpMesh ngp_mesh, CenterDataType& center_data,
                                      OrientationDataType& orientation_data, AxisLengthsDataType& axis_lengths_data,
                                      VelocityDataType& velocity_data, AngularVelocityDataType& angular_velocity_data) {
  auto ellipsoid_data =
      create_ngp_ellipsoid_data<Scalar, OurTopology, CenterDataType, OrientationDataType, AxisLengthsDataType>(
          ngp_mesh, center_data, orientation_data, axis_lengths_data);
  return create_ngp_motile_ellipsoid_data<Scalar, OurTopology>(ngp_mesh, ellipsoid_data, velocity_data,
                                                               angular_velocity_data);
}

/// \brief Check if the type provides the same data as MotileEllipsoidData
template <typename Agg>
concept ValidMotileEllipsoidDataType = requires(Agg agg) {
  ValidEllipsoidDataType<Agg>;
  typename Agg::velocity_data_t;
  typename Agg::angular_velocity_data_t;
  std::is_same_v<std::decay_t<typename Agg::velocity_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::angular_velocity_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.velocity_data() } -> std::convertible_to<typename Agg::velocity_data_t&>;
  { agg.angular_velocity_data() } -> std::convertible_to<typename Agg::angular_velocity_data_t&>;
};  // ValidMotileEllipsoidDataType

/// \brief Check if the type provides the same data as NgpMotileEllipsoidData
template <typename Agg>
concept ValidNgpMotileEllipsoidDataType = requires(Agg agg) {
  ValidNgpEllipsoidDataType<Agg>;
  typename Agg::velocity_data_t;
  typename Agg::angular_velocity_data_t;
  std::is_same_v<std::decay_t<typename Agg::velocity_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::angular_velocity_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.velocity_data() } -> std::convertible_to<typename Agg::velocity_data_t&>;
  { agg.angular_velocity_data() } -> std::convertible_to<typename Agg::angular_velocity_data_t&>;
};  // ValidNgpMotileEllipsoidDataType

/// \brief A helper function to get an updated NgpMotileEllipsoidData object from a MotileEllipsoidData object
/// \param data The MotileEllipsoidData object to convert
template <ValidMotileEllipsoidDataType MotileEllipsoidDataType>
auto get_updated_ngp_data(MotileEllipsoidDataType data) {
  using scalar_t = typename MotileEllipsoidDataType::scalar_t;
  using velocity_data_t = typename MotileEllipsoidDataType::velocity_data_t;
  using angular_velocity_data_t = typename MotileEllipsoidDataType::angular_velocity_data_t;
  return create_ngp_motile_ellipsoid_data<scalar_t, MotileEllipsoidDataType::topology_t>(
      create_ngp_ellipsoid_data(data.ellipsoid_data()),                  //
      stk::mesh::get_updated_ngp_field<scalar_t>(data.velocity_data()),  //
      stk::mesh::get_updated_ngp_field<scalar_t>(data.angular_velocity_data()));
}

/// \brief A traits class to provide abstracted access to a motile_ellipsoid's data via an aggregate
///
/// By default, this class is compatible with MotileEllipsoidData or any class the meets the
/// ValidMotileEllipsoidDataType concept. Users can specialize this class to support other aggregate types.
template <typename Agg>
struct MotileEllipsoidDataTraits : public EllipsoidDataTraits<Agg> {
  static_assert(ValidMotileEllipsoidDataType<Agg>,
                "Agg must satisfy the ValidMotileEllipsoidDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpMotileEllipsoidData but is free to "
                "extend it as needed without "
                "having to rely on inheritance.");

  using angular_velocity_data_t = typename Agg::angular_velocity_data_t;
  using velocity_data_t = typename Agg::velocity_data_t;

  static decltype(auto) velocity(Agg agg, stk::mesh::Entity motile_ellipsoid_node) {
    return mundy::mesh::vector3_field_data(agg.velocity_data(), motile_ellipsoid_node);
  }

  static decltype(auto) angular_velocity(Agg agg, stk::mesh::Entity motile_ellipsoid) {
    return mundy::mesh::vector3_field_data(agg.angular_velocity_data(), motile_ellipsoid);
  }
};  // MotileEllipsoidDataTraits

/// \brief A traits class to provide abstracted access to a motile_ellipsoid's data via an NGP-compatible aggregate
/// See the discussion for MotileEllipsoidDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpMotileEllipsoidDataTraits : public NgpEllipsoidDataTraits<Agg> {
  static_assert(ValidNgpMotileEllipsoidDataType<Agg>,
                "Agg must satisfy the ValidNgpMotileEllipsoidDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpMotileEllipsoidData but is free to "
                "extend it as needed without "
                "having to rely on inheritance.");

  using angular_velocity_data_t = typename Agg::angular_velocity_data_t;
  using velocity_data_t = typename Agg::velocity_data_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) velocity(Agg agg, stk::mesh::FastMeshIndex motile_ellipsoid_node_index) {
    return mundy::mesh::vector3_field_data(agg.velocity_data(), motile_ellipsoid_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) angular_velocity(Agg agg, stk::mesh::FastMeshIndex motile_ellipsoid_index) {
    return mundy::mesh::vector3_field_data(agg.angular_velocity_data(), motile_ellipsoid_index);
  }
};  // NgpMotileEllipsoidDataTraits

/// @brief A view of an STK entity meant to represent a motile_ellipsoid
///
/// We type specialize this class based on the valid set of topologies for an motile_ellipsoid entity.
///
/// Use \ref create_motile_ellipsoid_entity_view to build an MotileEllipsoidEntityView object with automatic template
/// deduction.
template <stk::topology::topology_t OurTopology, typename MotileEllipsoidDataType>
class MotileEllipsoidEntityView;

/// @brief A view of an STK entity meant to represent a motile_ellipsoid
template <typename MotileEllipsoidDataType>
class MotileEllipsoidEntityView<stk::topology::NODE, MotileEllipsoidDataType>
    : public EllipsoidEntityView<stk::topology::NODE, MotileEllipsoidDataType> {
 public:
  using data_access_t = MotileEllipsoidDataTraits<MotileEllipsoidDataType>;
  using ellipsoid_entity_view_t = EllipsoidEntityView<stk::topology::NODE, MotileEllipsoidDataType>;

  MotileEllipsoidEntityView(MotileEllipsoidDataType data, stk::mesh::Entity motile_ellipsoid)
      : ellipsoid_entity_view_t(data, motile_ellipsoid) {
  }

  decltype(auto) velocity() {
    return data_access_t::velocity(data(), get_ellipsoid());
  }

  decltype(auto) velocity() const {
    return data_access_t::velocity(data(), get_ellipsoid());
  }

  decltype(auto) angular_velocity() {
    return data_access_t::angular_velocity(data(), get_ellipsoid());
  }

  decltype(auto) angular_velocity() const {
    return data_access_t::angular_velocity(data(), get_ellipsoid());
  }
};  // MotileEllipsoidEntityView<NODE, MotileEllipsoidDataType>

template <typename MotileEllipsoidDataType>
class MotileEllipsoidEntityView<stk::topology::PARTICLE, MotileEllipsoidDataType>
    : public EllipsoidEntityView<stk::topology::PARTICLE, MotileEllipsoidDataType> {
 public:
  using data_access_t = MotileEllipsoidDataTraits<MotileEllipsoidDataType>;
  using ellipsoid_entity_view_t = EllipsoidEntityView<stk::topology::PARTICLE, MotileEllipsoidDataType>;

  MotileEllipsoidEntityView(MotileEllipsoidDataType data, stk::mesh::Entity motile_ellipsoid)
      : ellipsoid_entity_view_t(data, motile_ellipsoid) {
  }

  decltype(auto) velocity() {
    return data_access_t::velocity(data(), get_ellipsoid());
  }

  decltype(auto) velocity() const {
    return data_access_t::velocity(data(), get_ellipsoid());
  }

  decltype(auto) angular_velocity() {
    return data_access_t::angular_velocity(data(), get_ellipsoid());
  }

  decltype(auto) angular_velocity() const {
    return data_access_t::angular_velocity(data(), get_ellipsoid());
  }
};  // MotileEllipsoidEntityView<PARTICLE, MotileEllipsoidDataType>

/// @brief An ngp-compatible view of an motile_ellipsoid entity
template <stk::topology::topology_t OurTopology, typename NgpMotileEllipsoidDataType>
class NgpMotileEllipsoidEntityView;

template <typename NgpMotileEllipsoidDataType>
class NgpMotileEllipsoidEntityView<stk::topology::NODE, NgpMotileEllipsoidDataType>
    : public NgpEllipsoidEntityView<stk::topology::NODE, NgpMotileEllipsoidDataType> {
  static_assert(NgpMotileEllipsoidDataType::topology_t == stk::topology::NODE,
                "The topology of the motile_ellipsoid data must match the view");

 public:
  using data_access_t = NgpMotileEllipsoidDataTraits<NgpMotileEllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpMotileEllipsoidDataType>(),
                                                 std::declval<stk::mesh::FastMeshIndex>()));

  using ellipsoid_entity_view_t = NgpEllipsoidEntityView<stk::topology::NODE, NgpMotileEllipsoidDataType>;

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpMotileEllipsoidEntityView(NgpMotileEllipsoidDataType data, stk::mesh::FastMeshIndex motile_ellipsoid_index)
      : ellipsoid_entity_view_t(data, motile_ellipsoid_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) velocity() {
    return data_access_t::velocity(data(), get_ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) velocity() const {
    return data_access_t::velocity(data(), get_ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) angular_velocity() {
    return data_access_t::angular_velocity(data(), get_ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) angular_velocity() const {
    return data_access_t::angular_velocity(data(), get_ellipsoid_index());
  }
};  // NgpMotileEllipsoidEntityView<NODE, NgpMotileEllipsoidDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a motile_ellipsoid
template <typename NgpMotileEllipsoidDataType>
class NgpMotileEllipsoidEntityView<stk::topology::PARTICLE, NgpMotileEllipsoidDataType>
    : public NgpEllipsoidEntityView<stk::topology::PARTICLE, NgpMotileEllipsoidDataType> {
  static_assert(NgpMotileEllipsoidDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the motile_ellipsoid data must match the view");

 public:
  using data_access_t = NgpMotileEllipsoidDataTraits<NgpMotileEllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpMotileEllipsoidDataType>(),
                                                 std::declval<stk::mesh::FastMeshIndex>()));

  using ellipsoid_entity_view_t = NgpEllipsoidEntityView<stk::topology::PARTICLE, NgpMotileEllipsoidDataType>;

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpMotileEllipsoidEntityView(NgpMotileEllipsoidDataType data, stk::mesh::FastMeshIndex motile_ellipsoid_index)
      : ellipsoid_entity_view_t(data, motile_ellipsoid_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) velocity() {
    return data_access_t::velocity(data(), ellipsoid_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) velocity() const {
    return data_access_t::velocity(data(), ellipsoid_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) angular_velocity() {
    return data_access_t::angular_velocity(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) angular_velocity() const {
    return data_access_t::angular_velocity(data(), ellipsoid_index());
  }
};  // NgpMotileEllipsoidEntityView<PARTICLE, NgpMotileEllipsoidDataType>

/// \brief A helper function to create a MotileEllipsoidEntityView object with type deduction
template <typename MotileEllipsoidDataType>  // deduced
auto create_motile_ellipsoid_entity_view(MotileEllipsoidDataType& data, stk::mesh::Entity motile_ellipsoid) {
  return MotileEllipsoidEntityView<MotileEllipsoidDataType::topology_t, MotileEllipsoidDataType>(data,
                                                                                                 motile_ellipsoid);
}

/// \brief A helper function to create a NgpMotileEllipsoidEntityView object with type deduction
template <typename NgpMotileEllipsoidDataType>  // deduced
auto create_ngp_motile_ellipsoid_entity_view(NgpMotileEllipsoidDataType data,
                                             stk::mesh::FastMeshIndex motile_ellipsoid_index) {
  return NgpMotileEllipsoidEntityView<NgpMotileEllipsoidDataType::topology_t, NgpMotileEllipsoidDataType>(
      data, motile_ellipsoid_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_MOTILEELLIPSOID_HPP_
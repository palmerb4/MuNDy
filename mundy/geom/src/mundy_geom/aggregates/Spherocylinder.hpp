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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDER_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDER_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/Spherocylinder.hpp>  // for mundy::geom::ValidSpherocylinderType
#include <mundy_mesh/BulkData.hpp>                   // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of spherocylinders
///
/// The topology of a spherocylinder directly effects the access pattern for the underlying data:
///   - NODE: All data is stored on a single node
///   - PARTICLE: The center is stored on a node, whereas the orientation and radius are stored on the element-rank
///   particle
///
/// Use \ref create_spherocylinder_data to build a SpherocylinderData object with automatic template deduction.
///
/// \tparam Scalar The scalar type of the spherocylinder
/// \tparam OurTopology Can be NODE or PARTICLE.
/// \tparam CenterDataType Can either be a const or non-const stk::mesh::Field of scalars.
/// \tparam OrientationDataType Can either be a Quaternion or a const or non-const
/// \tparam RadiusDataType Can either be a scalar or an stk::mesh::Field of scalars.
/// \tparam LengthDataType Can either be a scalar or an stk::mesh::Field of scalars.
template <typename Scalar,                        //
          stk::topology::topology_t OurTopology,  //
          typename CenterDataType = stk::mesh::Field<Scalar>,
          typename OrientationDataType = stk::mesh::Field<Scalar>,  //
          typename RadiusDataType = stk::mesh::Field<Scalar>,       //
          typename LengthDataType = stk::mesh::Field<Scalar>>
class SpherocylinderData {
  static_assert(OurTopology == stk::topology::NODE || OurTopology == stk::topology::PARTICLE,
                "The topology of a spherocylinder must be either NODE or PARTICLE");
  static_assert((std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                 std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>) &&
                    (std::is_same_v<std::decay_t<LengthDataType>, Scalar> ||
                     std::is_same_v<std::decay_t<LengthDataType>, stk::mesh::Field<Scalar>>) &&
                    std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>> &&
                    (mundy::math::is_quaternion_v<std::decay_t<OrientationDataType>> ||
                     std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::Field<Scalar>>),
                "RadiusDataType must be either a scalar or a field of scalars\n"
                "LengthDataType must be either a scalar or a field of scalars\n"
                "CenterDataType must be either a const or non-const field of scalars\n"
                "OrientationDataType must be either a quaternion or a field of scalars");

 public:
  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using orientation_data_t = OrientationDataType;
  using radius_data_t = RadiusDataType;
  using length_data_t = LengthDataType;
  static constexpr stk::topology::topology_t topology_t = OurTopology;

  /// \brief Constructor
  SpherocylinderData(stk::mesh::BulkData& bulk_data, center_data_t& center_data, orientation_data_t& orientation_data,
                     radius_data_t& radius_data, length_data_t& length_data)
      : bulk_data_(bulk_data),
        center_data_(center_data),
        orientation_data_(orientation_data),
        radius_data_(radius_data),
        length_data_(length_data) {
    constexpr bool is_orientation_a_field = std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::Field<Scalar>>;
    constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>;
    constexpr bool is_length_a_field = std::is_same_v<std::decay_t<LengthDataType>, stk::mesh::Field<Scalar>>;
    stk::topology our_topology = OurTopology;
    if constexpr (is_orientation_a_field) {
      MUNDY_THROW_ASSERT(orientation_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The orientation data must be a field of the same rank as the spherocylinder");
    }
    if constexpr (is_radius_a_field) {
      MUNDY_THROW_ASSERT(radius_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the spherocylinder");
    }
    if constexpr (is_length_a_field) {
      MUNDY_THROW_ASSERT(length_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The length data must be a field of the same rank as the spherocylinder");
    }
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  stk::mesh::BulkData& bulk_data() {
    return bulk_data_;
  }

  const center_data_t& center_data() const {
    return center_data_;
  }

  center_data_t& center_data() {
    return center_data_;
  }

  const orientation_data_t& orientation_data() const {
    return orientation_data_;
  }

  orientation_data_t& orientation_data() {
    return orientation_data_;
  }

  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  radius_data_t& radius_data() {
    return radius_data_;
  }

  const length_data_t& length_data() const {
    return length_data_;
  }

  length_data_t& length_data() {
    return length_data_;
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  center_data_t& center_data_;
  orientation_data_t& orientation_data_;
  radius_data_t& radius_data_;
  length_data_t& length_data_;
};  // SpherocylinderData

/// \brief A struct to hold the data for a collection of NGP-compatible spherocylinders
/// See the discussion for SpherocylinderData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                                             //
          stk::topology::topology_t OurTopology,                       //
          typename CenterDataType = stk::mesh::NgpField<Scalar>,       //
          typename OrientationDataType = stk::mesh::NgpField<Scalar>,  //
          typename RadiusDataType = stk::mesh::NgpField<Scalar>, typename LengthDataType = stk::mesh::NgpField<Scalar>>
class NgpSpherocylinderData {
  static_assert(OurTopology == stk::topology::NODE || OurTopology == stk::topology::PARTICLE,
                "The topology of a spherocylinder must be either NODE or PARTICLE");
  static_assert((std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                 std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>) &&
                    (std::is_same_v<std::decay_t<LengthDataType>, Scalar> ||
                     std::is_same_v<std::decay_t<LengthDataType>, stk::mesh::NgpField<Scalar>>) &&
                    std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>> &&
                    (mundy::math::is_quaternion_v<std::decay_t<OrientationDataType>> ||
                     std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::NgpField<Scalar>>),
                "RadiusDataType must be either a scalar or a field of scalars\n"
                "LengthDataType must be either a scalar or a field of scalars\n"
                "CenterDataType must be either a const or non-const field of scalars\n"
                "OrientationDataType must be either a quaternion or a field of scalars");

 public:
  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using orientation_data_t = OrientationDataType;
  using radius_data_t = RadiusDataType;
  using length_data_t = LengthDataType;
  static constexpr stk::topology::topology_t topology_t = OurTopology;

  /// \brief Constructor
  NgpSpherocylinderData(stk::mesh::NgpMesh ngp_mesh, center_data_t& center_data, orientation_data_t& orientation_data,
                        radius_data_t& radius_data, length_data_t& length_data)
      : ngp_mesh_(ngp_mesh),
        center_data_(center_data),
        orientation_data_(orientation_data),
        radius_data_(radius_data),
        length_data_(length_data) {
    constexpr bool is_orientation_a_field =
        std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::NgpField<Scalar>>;
    constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>;
    constexpr bool is_length_a_field = std::is_same_v<std::decay_t<LengthDataType>, stk::mesh::NgpField<Scalar>>;
    stk::topology our_topology = OurTopology;
    if constexpr (is_orientation_a_field) {
      MUNDY_THROW_ASSERT(orientation_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The orientation data must be a field of the same rank as the spherocylinder");
    }
    if constexpr (is_radius_a_field) {
      MUNDY_THROW_ASSERT(radius_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the spherocylinder");
    }
    if constexpr (is_length_a_field) {
      MUNDY_THROW_ASSERT(length_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The length data must be a field of the same rank as the spherocylinder");
    }
  }

  stk::mesh::NgpMesh ngp_mesh() const {
    return ngp_mesh_;
  }

  const center_data_t& center_data() const {
    return center_data_;
  }

  center_data_t& center_data() {
    return center_data_;
  }

  const orientation_data_t& orientation_data() const {
    return orientation_data_;
  }

  orientation_data_t& orientation_data() {
    return orientation_data_;
  }

  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  radius_data_t& radius_data() {
    return radius_data_;
  }

  const length_data_t& length_data() const {
    return length_data_;
  }

  length_data_t& length_data() {
    return length_data_;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  center_data_t& center_data_;
  orientation_data_t& orientation_data_;
  radius_data_t& radius_data_;
  length_data_t& length_data_;
};  // NgpSpherocylinderData

/// \brief A helper function to create a SpherocylinderData object
///
/// This function creates a SpherocylinderData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename CenterDataType,                // deduced
          typename OrientationDataType,           // deduced
          typename RadiusDataType,                // deduced
          typename LengthDataType>                // deduced
auto create_spherocylinder_data(stk::mesh::BulkData& bulk_data, CenterDataType& center_data,
                                OrientationDataType& orientation_data, RadiusDataType& radius_data,
                                LengthDataType& length_data) {
  return SpherocylinderData<Scalar, OurTopology, CenterDataType, OrientationDataType, RadiusDataType, LengthDataType>{
      bulk_data, center_data, orientation_data, radius_data, length_data};
}

/// \brief A helper function to create a NgpSpherocylinderData object
/// See the discussion for create_spherocylinder_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename CenterDataType,                // deduced
          typename OrientationDataType,           // deduced
          typename RadiusDataType,                // deduced
          typename LengthDataType>                // deduced
auto create_ngp_spherocylinder_data(stk::mesh::NgpMesh ngp_mesh, CenterDataType& center_data,
                                    OrientationDataType& orientation_data, RadiusDataType& radius_data,
                                    LengthDataType& length_data) {
  return NgpSpherocylinderData<Scalar, OurTopology, CenterDataType, OrientationDataType, RadiusDataType,
                               LengthDataType>{ngp_mesh, center_data, orientation_data, radius_data, length_data};
}

/// \brief A concept to check if a type provides the same data as SpherocylinderData
template <typename Agg>
concept ValidSpherocylinderDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::orientation_data_t;
  typename Agg::radius_data_t;
  typename Agg::length_data_t;
  std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  mundy::math::is_quaternion_v<std::decay_t<typename Agg::orientation_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::orientation_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::radius_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::length_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::length_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
  { agg.bulk_data() } -> std::convertible_to<stk::mesh::BulkData&>;
  { agg.center_data() } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.orientation_data() } -> std::convertible_to<typename Agg::orientation_data_t&>;
  { agg.radius_data() } -> std::convertible_to<typename Agg::radius_data_t&>;
  { agg.length_data() } -> std::convertible_to<typename Agg::length_data_t&>;
};  // ValidSpherocylinderDataType

/// \brief A concept to check if a type provides the same data as NgpSpherocylinderData
template <typename Agg>
concept ValidNgpSpherocylinderDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::orientation_data_t;
  typename Agg::radius_data_t;
  typename Agg::length_data_t;
  std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  mundy::math::is_quaternion_v<std::decay_t<typename Agg::orientation_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::orientation_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::radius_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::length_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::length_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
  { agg.ngp_mesh() } -> std::convertible_to<stk::mesh::NgpMesh>;
  { agg.center_data() } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.orientation_data() } -> std::convertible_to<typename Agg::orientation_data_t&>;
  { agg.radius_data() } -> std::convertible_to<typename Agg::radius_data_t&>;
  { agg.length_data() } -> std::convertible_to<typename Agg::length_data_t&>;
};  // ValidNgpSpherocylinderDataType

static_assert(ValidSpherocylinderDataType<                            //
                  SpherocylinderData<float,                           //
                                     stk::topology::NODE,             //
                                     stk::mesh::Field<float>,         //
                                     mundy::math::Quaternion<float>,  //
                                     float,                           //
                                     float>> &&
                  ValidSpherocylinderDataType<                     //
                      SpherocylinderData<float,                    //
                                         stk::topology::PARTICLE,  //
                                         stk::mesh::Field<float>,  //
                                         stk::mesh::Field<float>,  //
                                         stk::mesh::Field<float>,  //
                                         stk::mesh::Field<float>>>,
              "SpherocylinderData must satisfy the ValidSpherocylinderDataType concept");

static_assert(ValidNgpSpherocylinderDataType<                            //
                  NgpSpherocylinderData<float,                           //
                                        stk::topology::NODE,             //
                                        stk::mesh::NgpField<float>,      //
                                        mundy::math::Quaternion<float>,  //
                                        float,                           //
                                        float>> &&
                  ValidNgpSpherocylinderDataType<                        //
                      NgpSpherocylinderData<float,                       //
                                            stk::topology::NODE,         //
                                            stk::mesh::NgpField<float>,  //
                                            stk::mesh::NgpField<float>,  //
                                            stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>,
              "NgpSpherocylinderData must satisfy the ValidNgpSpherocylinderDataType concept");

/// \brief A helper function to get an updated NgpSpherocylinderData object from a SpherocylinderData object
/// \param data The SpherocylinderData object to convert
template <ValidSpherocylinderDataType SpherocylinderDataType>
auto get_updated_ngp_data(SpherocylinderDataType data) {
  using scalar_t = typename SpherocylinderDataType::scalar_t;
  using orientation_data_t = typename SpherocylinderDataType::orientation_data_t;
  using radius_data_t = typename SpherocylinderDataType::radius_data_t;
  using length_data_t = typename SpherocylinderDataType::length_data_t;

  constexpr bool is_orientation_a_field = std::is_same_v<std::decay_t<orientation_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr bool is_length_a_field = std::is_same_v<std::decay_t<length_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr stk::topology::topology_t topology_t = SpherocylinderDataType::topology_t;
  if constexpr (is_radius_a_field && is_orientation_a_field && is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(              //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                    //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data()),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.length_data()));
  } else if constexpr (is_radius_a_field && is_orientation_a_field && !is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(              //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                    //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data()),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data()),       //
        data.length_data());
  } else if constexpr (is_radius_a_field && !is_orientation_a_field && is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(         //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),  //
        data.orientation_data(),                                         //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data()),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.length_data()));
  } else if constexpr (!is_radius_a_field && is_orientation_a_field && is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(              //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                    //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data()),  //
        data.radius_data(),                                                   //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.length_data()));
  } else if constexpr (is_radius_a_field && !is_orientation_a_field && !is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(         //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),  //
        data.orientation_data(),                                         //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data()),  //
        data.length_data());
  } else if constexpr (!is_radius_a_field && is_orientation_a_field && !is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(              //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                    //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data()),  //
        data.radius_data(),                                                   //
        data.length_data());
  } else if constexpr (!is_radius_a_field && !is_orientation_a_field && is_length_a_field) {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(         //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),  //
        data.orientation_data(),                                         //
        data.radius_data(),                                              //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.length_data()));
  } else {
    return create_ngp_spherocylinder_data<scalar_t, topology_t>(         //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),  //
        data.orientation_data(),                                         //
        data.radius_data(),                                              //
        data.length_data());
  }
}

/// \brief A traits class to provide abstracted access to a spherocylinder's data via an aggregate
///
/// By default, this class is compatible with SpherocylinderData or any class the meets the
/// ValidSpherocylinderDataType concept. Users can specialize this class to support other aggregate types.
template <typename Agg>
struct SpherocylinderDataTraits {
  static_assert(ValidSpherocylinderDataType<Agg>,
                "Agg must satisfy the ValidSpherocylinderDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderData but is free to "
                "extend it as "
                "needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using orientation_data_t = typename Agg::orientation_data_t;
  using radius_data_t = typename Agg::radius_data_t;
  using length_data_t = typename Agg::length_data_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static constexpr decltype(auto) has_shared_orientation() {
    return mundy::math::is_quaternion_v<std::decay_t<orientation_data_t>>;
  }

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static constexpr bool has_shared_length() {
    return std::is_same_v<std::decay_t<length_data_t>, scalar_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity spherocylinder_node) {
    return mundy::mesh::vector3_field_data(agg.center_data(), spherocylinder_node);
  }

  static decltype(auto) orientation(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_orientation()) {
      return agg.orientation_data();
    } else {
      return mundy::mesh::quaternion_field_data(agg.orientation_data(), spherocylinder);
    }
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return stk::mesh::field_data(agg.radius_data(), spherocylinder)[0];
    }
  }

  static decltype(auto) length(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_length()) {
      return agg.length_data();
    } else {
      return stk::mesh::field_data(agg.length_data(), spherocylinder)[0];
    }
  }
};  // SpherocylinderDataTraits

/// \brief A traits class to provide abstracted access to a spherocylinder's data via an NGP-compatible aggregate
/// See the discussion for SpherocylinderDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpSpherocylinderDataTraits {
  static_assert(ValidNgpSpherocylinderDataType<Agg>,
                "Agg must satisfy the ValidNgpSpherocylinderDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderData but is free to "
                "extend it as "
                "needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using orientation_data_t = typename Agg::orientation_data_t;
  using radius_data_t = typename Agg::radius_data_t;
  using length_data_t = typename Agg::length_data_t;

  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_orientation() {
    return mundy::math::is_quaternion_v<std::decay_t<orientation_data_t>>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_length() {
    return std::is_same_v<std::decay_t<length_data_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex spherocylinder_node_index) {
    return mundy::mesh::vector3_field_data(agg.center_data(), spherocylinder_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) orientation(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_orientation()) {
      return agg.orientation_data();
    } else {
      return mundy::mesh::quaternion_field_data(agg.orientation_data(), spherocylinder_index);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return agg.radius_data()(spherocylinder_index, 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) length(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_length()) {
      return agg.length_data();
    } else {
      return agg.length_data()(spherocylinder_index, 0);
    }
  }
};  // NgpSpherocylinderDataTraits

/// @brief A view of an STK entity meant to represent a spherocylinder
///
/// We type specialize this class based on the valid set of topologies for a spherocylinder entity.
///
/// Use \ref create_spherocylinder_entity_view to build an SpherocylinderEntityView object with automatic template
/// deduction.
template <stk::topology::topology_t OurTopology, typename SpherocylinderDataType>
class SpherocylinderEntityView;

/// @brief A view of a NODE STK entity meant to represent a spherocylinder
template <typename SpherocylinderDataType>
class SpherocylinderEntityView<stk::topology::NODE, SpherocylinderDataType> {
  static_assert(SpherocylinderDataType::topology_t == stk::topology::NODE,
                "The topology of the spherocylinder data must match the view");

 public:
  using data_access_t = SpherocylinderDataTraits<SpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));
  using orientation_t =
      decltype(data_access_t::orientation(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  SpherocylinderEntityView(SpherocylinderDataType data, stk::mesh::Entity spherocylinder)
      : data_(data), spherocylinder_(spherocylinder) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(spherocylinder_) == stk::topology::NODE_RANK,
                       std::invalid_argument, "The spherocylinder entity rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(spherocylinder_), std::invalid_argument,
                       "The given spherocylinder entity is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, spherocylinder_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, spherocylinder_);
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data_, spherocylinder_);
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_);
  }

  decltype(auto) length() {
    return data_access_t::length(data_, spherocylinder_);
  }

  decltype(auto) length() const {
    return data_access_t::length(data_, spherocylinder_);
  }

 private:
  SpherocylinderDataType data_;
  stk::mesh::Entity spherocylinder_;
};  // SpherocylinderEntityView<stk::topology::NODE, SpherocylinderDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a spherocylinder
template <typename SpherocylinderDataType>
class SpherocylinderEntityView<stk::topology::PARTICLE, SpherocylinderDataType> {
  static_assert(SpherocylinderDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the spherocylinder data must match the view");

 public:
  using data_access_t = SpherocylinderDataTraits<SpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));
  using orientation_t =
      decltype(data_access_t::orientation(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  SpherocylinderEntityView(SpherocylinderDataType data, stk::mesh::Entity spherocylinder)
      : data_(data), spherocylinder_(spherocylinder), node_(data_.bulk_data().begin_nodes(spherocylinder_)[0]) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(spherocylinder_) == stk::topology::ELEM_RANK,
                       std::invalid_argument, "The spherocylinder entity rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(spherocylinder_), std::invalid_argument,
                       "The given spherocylinder entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().num_nodes(spherocylinder_) >= 1, std::invalid_argument,
                       "The given spherocylinder entity must have at least one node");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(node_), std::invalid_argument,
                       "The node entity associated with the spherocylinder is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_);
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_);
  }

  decltype(auto) length() {
    return data_access_t::length(data_, spherocylinder_);
  }

  decltype(auto) length() const {
    return data_access_t::length(data_, spherocylinder_);
  }

 private:
  SpherocylinderDataType data_;
  stk::mesh::Entity spherocylinder_;
  stk::mesh::Entity node_;
};  // SpherocylinderEntityView<stk::topology::PARTICLE, SpherocylinderDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a spherocylinder
/// See the discussion for SpherocylinderEntityView for more information. The only difference is ngp-compatible data
/// access.
template <stk::topology::topology_t OurTopology, typename NgpSpherocylinderDataType>
class NgpSpherocylinderEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a spherocylinder
template <typename NgpSpherocylinderDataType>
class NgpSpherocylinderEntityView<stk::topology::NODE, NgpSpherocylinderDataType> {
  static_assert(NgpSpherocylinderDataType::topology_t == stk::topology::NODE,
                "The topology of the spherocylinder data must match the view");

 public:
  using data_access_t = NgpSpherocylinderDataTraits<NgpSpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpSpherocylinderDataType>(),
                                                 std::declval<stk::mesh::FastMeshIndex>()));
  using orientation_t = decltype(data_access_t::orientation(std::declval<NgpSpherocylinderDataType>(),
                                                            std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderEntityView(NgpSpherocylinderDataType data, stk::mesh::FastMeshIndex spherocylinder_index)
      : data_(data), spherocylinder_index_(spherocylinder_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() {
    return data_access_t::length(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() const {
    return data_access_t::length(data_, spherocylinder_index_);
  }

 private:
  NgpSpherocylinderDataType data_;
  stk::mesh::FastMeshIndex spherocylinder_index_;
};  // NgpSpherocylinderEntityView<stk::topology::NODE, NgpSpherocylinderDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a spherocylinder
template <typename NgpSpherocylinderDataType>
class NgpSpherocylinderEntityView<stk::topology::PARTICLE, NgpSpherocylinderDataType> {
  static_assert(NgpSpherocylinderDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the spherocylinder data must match the view");

 public:
  using data_access_t = NgpSpherocylinderDataTraits<NgpSpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpSpherocylinderDataType>(),
                                                 std::declval<stk::mesh::FastMeshIndex>()));
  using orientation_t = decltype(data_access_t::orientation(std::declval<NgpSpherocylinderDataType>(),
                                                            std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderEntityView(NgpSpherocylinderDataType data, stk::mesh::FastMeshIndex spherocylinder_index)
      : data_(data),
        spherocylinder_index_(spherocylinder_index),
        node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, spherocylinder_index_)[0])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() {
    return data_access_t::length(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() const {
    return data_access_t::length(data_, spherocylinder_index_);
  }

 private:
  NgpSpherocylinderDataType data_;
  stk::mesh::FastMeshIndex spherocylinder_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpSpherocylinderEntityView<stk::topology::PARTICLE, NgpSpherocylinderDataType>

static_assert(ValidSpherocylinderType<  //
                  SpherocylinderEntityView<stk::topology::NODE,
                                           SpherocylinderData<float,                           //
                                                              stk::topology::NODE,             //
                                                              stk::mesh::Field<float>,         //
                                                              mundy::math::Quaternion<float>,  //
                                                              float,                           //
                                                              float>>> &&
                  ValidSpherocylinderType<  //
                      SpherocylinderEntityView<stk::topology::PARTICLE,
                                               SpherocylinderData<float,                    //
                                                                  stk::topology::PARTICLE,  //
                                                                  stk::mesh::Field<float>,  //
                                                                  stk::mesh::Field<float>,  //
                                                                  stk::mesh::Field<float>,  //
                                                                  stk::mesh::Field<float>>>> &&
                  ValidSpherocylinderType<  //
                      NgpSpherocylinderEntityView<stk::topology::NODE,
                                                  NgpSpherocylinderData<float,                           //
                                                                        stk::topology::NODE,             //
                                                                        stk::mesh::NgpField<float>,      //
                                                                        mundy::math::Quaternion<float>,  //
                                                                        float,                           //
                                                                        float>>> &&
                  ValidSpherocylinderType<  //
                      NgpSpherocylinderEntityView<stk::topology::PARTICLE,
                                                  NgpSpherocylinderData<float,                       //
                                                                        stk::topology::PARTICLE,     //
                                                                        stk::mesh::NgpField<float>,  //
                                                                        stk::mesh::NgpField<float>,  //
                                                                        stk::mesh::NgpField<float>,  //
                                                                        stk::mesh::NgpField<float>>>>,
              "SpherocylinderEntityView and NgpSpherocylinderEntityView must be valid Spherocylinder types");

/// \brief A helper function to create a SpherocylinderEntityView object with type deduction
template <typename SpherocylinderDataType>  // deduced
auto create_spherocylinder_entity_view(SpherocylinderDataType& data, stk::mesh::Entity spherocylinder) {
  return SpherocylinderEntityView<SpherocylinderDataType::topology_t, SpherocylinderDataType>(data, spherocylinder);
}

/// \brief A helper function to create a NgpSpherocylinderEntityView object with type deduction
template <typename NgpSpherocylinderDataType>  // deduced
auto create_ngp_spherocylinder_entity_view(NgpSpherocylinderDataType data,
                                           stk::mesh::FastMeshIndex spherocylinder_index) {
  return NgpSpherocylinderEntityView<NgpSpherocylinderDataType::topology_t, NgpSpherocylinderDataType>(
      data, spherocylinder_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDER_HPP_
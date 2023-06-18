// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
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

#ifndef MUNDY_METHODS_COMPUTE_BOUNDING_RADIUS_KERNELS_SPHERE_HPP_
#define MUNDY_METHODS_COMPUTE_BOUNDING_RADIUS_KERNELS_SPHERE_HPP_

/// \file Sphere.hpp
/// \brief Declaration of the ComputeBoundingRadius's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                  // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                  // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>         // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>               // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaRegistry.hpp>              // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>          // for mundy::meta::PartRequirements
#include <mundy_methods/ComputeBoundingRadius.hpp>  // for mundy::methods::ComputeBoundingRadius

namespace mundy {

namespace methods {

namespace compute_bounding_radius {

namespace kernels {

/// \class Sphere
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Sphere : public mundy::meta::MetaKernel<void, Sphere>, public ComputeBoundingRadius::OurKernelRegistry<Sphere> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this kernel imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> details_static_get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    std::shared_ptr<mundy::meta::PartRequirements> required_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_part_params->set_part_topology(stk::topology::PARTICLE);
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_radius_field_name_), stk::topology::ELEMENT_RANK, 1, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_bounding_radius_field_name_), stk::topology::ELEMENT_RANK, 1, 1));
    return required_part_params;
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_params;
    default_fixed_params.set(
        "bounding_sphere_field_name", std::string(default_bounding_radius_field_name_),
        "Name of the element field within which the output bounding radius will be written.");
    default_fixed_params.set("radius_field_name", std::string(default_radius_field_name_),
                                     "Name of the element field containing the sphere radius.");
    return default_fixed_params;
  }

  /// \brief Get the default mutable parameters for this class (those that do not impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_mutable_params() {
    static Teuchos::ParameterList default_mutable_params;
    default_mutable_params.set("buffer_distance", default_buffer_distance_,
                                         "Buffer distance to be added to the axis-aligned boundary box.");
    return default_mutable_params;
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernelBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<Sphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param element [in] The element acted on by the kernel.
  void execute(const stk::mesh::Entity &element) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_bounding_radius_field_name_ = "BOUNDING_RADIUS";
  static constexpr std::string_view default_radius_field_name_ = "RADIUS";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static const std::string class_identifier_ = "SPHERE";

  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Buffer distance to be added to the axis-aligned boundary box.
  ///
  /// For example, if the original axis-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_;

  /// \brief Name of the element field within which the output bounding radius will be written.
  std::string bounding_radius_field_name_;

  /// \brief Name of the element field containing the sphere radius.
  std::string radius_field_name_;

  /// \brief Element field within which the output bounding radius will be written.
  stk::mesh::Field<double> *bounding_radius_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *radius_field_ptr_;
};  // Sphere

}  // namespace kernels

}  // namespace compute_bounding_radius

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_BOUNDING_RADIUS_KERNELS_SPHERE_HPP_

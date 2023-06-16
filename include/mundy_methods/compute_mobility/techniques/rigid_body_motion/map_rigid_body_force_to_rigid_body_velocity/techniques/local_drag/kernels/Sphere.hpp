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

#ifndef MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCAL_DRAG_KERNELS_SPHERE_HPP_
#define MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCAL_DRAG_KERNELS_SPHERE_HPP_

/// \file Sphere.hpp
/// \brief Declaration of the LocalDrag's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>        // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_methods/compute_mobility/techniques/LocalDrag.hpp>  // for mundy::methods::compute_mobility::techniques::LocalDrag
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace local_drag {

namespace kernels {

/// \class Sphere
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Sphere : public mundy::meta::MetaMultibodyKernel<void, Sphere>,
               public mundy::meta::MetaMultibodyKernelRegistry<void, Sphere, LocalDrag> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this kernel imposes upon each particle and/or constraint.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::vector<std::shared_ptr<mundy::meta::PartRequirements>> details_static_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_parameter_list) {
    std::shared_ptr<mundy::meta::PartRequirements> required_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_part_params->set_part_topology(stk::topology::PARTICLE);
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_force_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_torque_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_velocity_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_omega_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_element_radius_field_name_), stk::topology::ELEMENT_RANK, 1, 1));
    return required_part_params;
  }

  /// \brief Get the default transient parameters for this class (those that impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_parameter_list;
    default_fixed_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                                     "Name of the node field containing the force on the sphere's center.");
    default_fixed_parameter_list.set("node_torque_field_name", std::string(default_node_torque_field_name_),
                                     "Name of the node field containing the torque on the sphere's center.");
    default_fixed_parameter_list.set(
        "node_velocity_field_name", std::string(default_node_velocity_field_name_),
        "Name of the node field containing the translational velocity of the sphere's center.");
    default_fixed_parameter_list.set(
        "node_omega_field_name", std::string(default_node_omega_field_name_),
        "Name of the node field containing the rotational velocity of the sphere's center.");
    default_fixed_parameter_list.set("element_radius_field_name", std::string(default_element_radius_field_name_),
                                     "Name of the element field containing the sphere's radius.");
    return default_fixed_parameter_list;
  }

  /// \brief Get the default transient parameters for this class (those that do not impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_transient_params() {
    static Teuchos::ParameterList default_transient_parameter_list;
    default_transient_parameter_list.set("time_step_size", default_time_step_size_, "The numerical timestep size.");
    return default_transient_parameter_list;
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernelBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return std::make_shared<Sphere>(bulk_data_ptr, fixed_parameter_list);
  }
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

  static constexpr double default_time_step_size_ = -1;
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_torque_field_name_ = "NODE_TORQUE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_ = "NODE_OMEGA";
  static constexpr std::string_view default_element_radius_field_name_name_ = "ELEMENT_RADIUS";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static const mundy::multibody::Factory::FastIdType class_identifier_ =
      mundy::multibody::Factory::get_fast_id("SPHERE");

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief Name of the node field containing the force on the sphere's center.
  std::string node_force_field_name_;

  /// \brief Name of the node field containing the torque on the sphere's center.
  std::string node_torque_field_name_;

  /// \brief Name of the node field containing the translational velocity of the sphere's center.
  std::string node_velocity_field_name_;

  /// \brief Name of the node field containing the rotational velocity of the sphere's center.
  std::string node_omega_field_name_;

  /// \brief Name of the element field containing the sphere's radius.
  std::string element_radius_field_name_;

  /// \brief Node field containing the force on the sphere's center.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Node field containing the torque on the sphere's center..
  stk::mesh::Field<double> *node_torque_field_ptr_ = nullptr;

  /// \brief Node field containing the translational velocity of the sphere's center.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the rotational velocity of the sphere's center.
  stk::mesh::Field<double> *node_omega_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // Sphere

}  // namespace kernels

}  // namespace local_drag

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCAL_DRAG_KERNELS_SPHERE_HPP_

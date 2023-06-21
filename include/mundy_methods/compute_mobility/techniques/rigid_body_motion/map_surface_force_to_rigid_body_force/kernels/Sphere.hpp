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

#ifndef MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_SURFACE_FORCE_TO_RIGID_BODY_FORCE_KERNELS_SPHERE_HPP_
#define MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_SURFACE_FORCE_TO_RIGID_BODY_FORCE_KERNELS_SPHERE_HPP_

/// \file Sphere.hpp
/// \brief Declaration of the MapSurfaceForceToRigidBodyForce's Sphere kernel.

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
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>           // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>        // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements
#include <mundy_methods/compute_mobility/techniques/rigid_body_motion/MapSurfaceForceToRigidBodyForce.hpp>  // for mundy::methods::...::MapSurfaceForceToRigidBodyForce

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

namespace map_surface_force_to_rigid_body_force {

namespace kernels {

/// \class Sphere
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Sphere : public mundy::meta::MetaKernel<void, Sphere>,
               public MapSurfaceForceToRigidBodyForce::OurKernelRegistry<Sphere> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> details_static_get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    std::string node_coord_field_name = valid_fixed_params.get<std::string>("node_coord_field_name");
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string node_torque_field_name = valid_fixed_params.get<std::string>("node_torque_field_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name("SPHERE");
    part_reqs->set_part_topology(stk::topology::PARTICLE);
    part_reqs->put_multibody_part_attribute(mundy::muntibody::Factory::get_fast_id("SPEHRE"));
    part_reqs->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(node_coord_field_name,
                                                                                      stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(node_force_field_name,
                                                                                      stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(node_torque_field_name,
                                                                                      stk::topology::NODE_RANK, 3, 1));

    DECLARE THE LINKER AND GIVE IT A SHERE MULTIBODY TYPE

        auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_req(part_reqs);
    return multibody_part_params;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_coordinate_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_coordinate_field_name");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'node_coordinate_field_name' but "
                                 "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_coordinate_field_name", std::string(default_node_coord_field_name_),
                            "Name of the node field containing the coordinate of the sphere's center.");
    }

    if (fixed_params_ptr->isParameter("node_force_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_force_field_name");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'node_force_field_name' but "
                                 "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_force_field_name", std::string(default_node_force_field_name_),
                            "Name of the node field containing the surface and body force.");
    }

    if (fixed_params_ptr->isParameter("node_torque_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_torque_field_name");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'node_torque_field_name' but "
                                 "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_torque_field_name", std::string(default_node_torque_field_name_),
                            "Name of the node field containing the surface and body torque.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isParameter("alpha")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("alpha");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'alpha' but "
                                 "with a type other than double");
    } else {
      mutable_params_ptr->set("alpha", default_alpha_,
                              "Scale for the force and torque such that F = beta * F0 + alpha * Fnew.");
    }

    if (mutable_params_ptr->isParameter("beta")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("beta");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'beta' but "
                                 "with a type other than double");
    } else {
      mutable_params_ptr->set("beta", default_beta_,
                              "Scale for the force and torque such that F = beta * F0 + alpha * Fnew.");
    }
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

  /// \brief Setup the kernel's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  void setup() override;

  /// \brief Run the kernel's core calculation.
  /// \param linker [in] The linker acted on by the kernel.
  void execute(const stk::mesh::Entity &linker) override;

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform redictions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_alpha_ = 1.0;
  static constexpr double default_beta_ = 0.0;
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_torque_field_name_ = "NODE_TORQUE";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static const std::string class_identifier_ = "SPHERE";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Name of the node field containing the coordinate of the sphere's center.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the surface and body force.
  std::string node_force_field_name_;

  /// \brief Name of the node field containing the surface and body torque.
  std::string node_torque_field_name_;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the surface and body force.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Node field containing the surface and body torque.
  stk::mesh::Field<double> *node_torque_field_ptr_ = nullptr;
  //@}
};  // Sphere

}  // namespace kernels

}  // namespace map_surface_force_to_rigid_body_force

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_SURFACE_FORCE_TO_RIGID_BODY_FORCE_KERNELS_SPHERE_HPP_

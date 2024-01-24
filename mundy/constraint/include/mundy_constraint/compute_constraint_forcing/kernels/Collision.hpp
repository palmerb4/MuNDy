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

#ifndef MUNDY_CONSTRAINT_COMPUTE_CONSTRAINT_FORCING_KERNELS_COLLISION_HPP_
#define MUNDY_CONSTRAINT_COMPUTE_CONSTRAINT_FORCING_KERNELS_COLLISION_HPP_

/// \file Collision.hpp
/// \brief Declaration of the ComputeConstraintForcing's Collision kernel.

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
#include <mundy_mesh/BulkData.hpp>                        // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                        // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>               // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>                     // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                      // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>                    // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>                // for mundy::meta::PartRequirements

namespace mundy {

namespace constraint {

namespace compute_constraint_forcing {

namespace kernels {

/// \class Collision
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Collision : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Collision(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    std::string node_normal_field_name = valid_fixed_params.get<std::string>("node_normal_field_name");
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string element_lagrange_multiplier_field_name =
        valid_fixed_params.get<std::string>("element_lagrange_multiplier_field_name");
    std::string associated_part_name = valid_fixed_params.get<std::string>("part_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name(associated_part_name);
    part_reqs->set_part_topology(stk::topology::BEAM_2);
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(node_normal_field_name,
                                                                                       stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(node_force_field_name,
                                                                                       stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        element_lagrange_multiplier_field_name, stk::topology::ELEMENT_RANK, 1, 1));

    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(part_reqs);
    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_normal_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_normal_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Collision: Type error. Given a parameter with name 'node_normal_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_normal_field_name", std::string(default_node_normal_field_name_),
                            "Name of the node field containing the node's normal.");
    }

    if (fixed_params_ptr->isParameter("node_force_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_force_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Collision: Type error. Given a parameter with name 'node_force_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_force_field_name", std::string(default_node_force_field_name_),
                            "Name of the node field containing force on the constraint's endpoints.");
    }

    if (fixed_params_ptr->isParameter("element_lagrange_multiplier_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("element_lagrange_multiplier_field_name");
      MUNDY_THROW_ASSERT(
          valid_type, std::invalid_argument,
          "Collision: Type error. Given a parameter with name 'element_lagrange_multiplier_field_name' but "
          "with a type other than std::string");
    } else {
      fixed_params_ptr->set("element_lagrange_multiplier_field_name",
                            std::string(default_element_lagrange_multiplier_field_name_),
                            "Name of the element field containing the constraint's Lagrange multiplier.");
    }

    if (fixed_params_ptr->isParameter("part_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("part_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Collision: Type error. Given a parameter with name 'part_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("part_name", std::string(default_part_name_),
                            "Name of the part associated with this kernel.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaTwoWayKernelRegistry.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernel<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<Collision>(bulk_data_ptr, fixed_params);
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
  /// \param collision_node [in] A node connected to a collision element.
  void execute(const stk::mesh::Entity &collision_node) override;

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_part_name_ = "COLLISION";
  static constexpr std::string_view default_node_normal_field_name_ = "NODE_NORMAL";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_element_lagrange_multiplier_field_name_ = "ELEMENT_LAGRANGE_MULTIPLIER";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "COLLISION";

  /// \brief The name of the part associated with this kernel.
  std::string associated_part_name_;

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Mesh ordinal for the part containing all collision constraints.
  size_t collision_part_ordinal_;

  /// \brief Minimum allowable signed separation distance between colliding bodies.
  double minimum_allowable_separation_;

  /// \brief Name of the node field containing the normal of each node.
  std::string node_normal_field_name_;

  /// \brief Name of the node field containing the force on each node.
  std::string node_force_field_name_;

  /// \brief Name of the element field containing the constraint's Lagrange multiplier.
  std::string element_lagrange_multiplier_field_name_;

  /// \brief Node field containing the surface normal at the attachment points.
  stk::mesh::Field<double> *node_normal_field_ptr_ = nullptr;

  /// \brief Node field containing the force at the attachment points.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Element field containing the lagrange multiplier associated with the collision constraint.
  stk::mesh::Field<double> *element_lagrange_multiplier_field_ptr_ = nullptr;
  //@}
};  // Collision

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace constraint

}  // namespace mundy

#endif  // MUNDY_CONSTRAINT_COMPUTE_CONSTRAINT_FORCING_KERNELS_COLLISION_HPP_

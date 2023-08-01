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

#ifndef MUNDY_METHODS_COMPUTE_AABB_KERNELS_SPHEROCYLINDER_HPP_
#define MUNDY_METHODS_COMPUTE_AABB_KERNELS_SPHEROCYLINDER_HPP_

/// \file Spherocylinder.hpp
/// \brief Declaration of the ComputeAABB's Spherocylinder kernel.

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
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements
#include <mundy_multibody/MultibodyFactory.hpp>       // for mundy::multibody::MultibodyFactory
#include <mundy_methods/ComputeAABB.hpp>                          // for mundy::methods::ComputeAABB

namespace mundy {

namespace methods {

namespace compute_aabb {

namespace kernels {

/// \class Spherocylinder
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of Spherocylinders.
class Spherocylinder : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Spherocylinder(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    std::string node_coord_field_name = valid_fixed_params.get<std::string>("node_coord_field_name");
    std::string element_radius_field_name = valid_fixed_params.get<std::string>("element_radius_field_name");
    std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name("SPHEROCYLINDER");
    part_reqs->set_part_topology(stk::topology::BEAM_3);
    part_reqs->put_multibody_part_attribute(mundy::multibody::MultibodyFactory::get_multibody_type("SPHEROCYLINDER"));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(node_coord_field_name,
                                                                                       stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        element_radius_field_name, stk::topology::ELEMENT_RANK, 1, 1));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        element_aabb_field_name, stk::topology::ELEMENT_RANK, 6, 1));

    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(part_reqs);
    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_coord_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_coord_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Spherocylinder: Type error. Given a parameter with name 'node_coord_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_coord_field_name", std::string(default_node_coord_field_name_),
                            "Name of the node field containing the coordinate of the Spherocylinder's nodes.");
    }

    if (fixed_params_ptr->isParameter("element_radius_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("element_radius_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Spherocylinder: Type error. Given a parameter with name 'element_length_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("element_radius_field_name", std::string(default_element_radius_field_name_),
                            "Name of the element field containing the Spherocylinder's radius.");
    }

    if (fixed_params_ptr->isParameter("element_aabb_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("element_aabb_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Spherocylinder: Type error. Given a parameter with name 'element_aabb_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set(
          "element_aabb_field_name", std::string(default_element_aabb_field_name_),
          "Name of the element field within which the output axis-aligned boundary boxes will be written.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isParameter("buffer_distance")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("buffer_distance");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "Spherocylinder: Type error. Given a parameter with name 'buffer_distance' but "
                             << "with a type other than unsigned");
    } else {
      mutable_params_ptr->set("buffer_distance", default_buffer_distance_,
                              "Buffer distance to be added to the axis-aligned boundary box.");
    }
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernel<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<Spherocylinder>(bulk_data_ptr, fixed_params);
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
  /// \param Spherocylinder_element [in] The Spherocylinder element acted on by the kernel.
  void execute(const stk::mesh::Entity &Spherocylinder_element) override;

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_element_radius_field_name_ = "ELEMENT_RADIUS";
  static constexpr std::string_view default_element_aabb_field_name_ = "ELEMENT_AABB";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "SPHEROCYLINDER";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Buffer distance to be added to the axis-aligned boundary box.
  ///
  /// For example, if the original axis-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_;

  /// \brief Name of the node field containing the coordinate of the Spherocylinder's nodes.
  std::string node_coord_field_name_;

  /// \brief Name of the element field containing the Spherocylinder's radius.
  std::string element_radius_field_name_;

  /// \brief Name of the element field within which the output axis-aligned boundary boxes will be written.
  std::string element_aabb_field_name_;

  /// \brief Node field containing the coordinate of the Spherocylinder's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Element field containing the Spherocylinder's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;

  /// \brief Element field within which the output axis-aligned boundary boxes will be written.
  stk::mesh::Field<double> *element_aabb_field_ptr_ = nullptr;
  //@}
};  // Spherocylinder

}  // namespace kernels

}  // namespace compute_aabb

}  // namespace methods

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register ComputeAABB with the global MetaMethodFactory.
MUNDY_REGISTER_METACLASS(mundy::methods::compute_aabb::kernels::Spherocylinder,
                         mundy::methods::ComputeAABB::OurKernelFactory)
//}

#endif  // MUNDY_METHODS_COMPUTE_AABB_KERNELS_SPHEROCYLINDER_HPP_

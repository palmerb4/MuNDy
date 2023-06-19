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

#ifndef MUNDY_METHODS_GENERATE_COLLISION_CONSTRAINTS_KERNELS_COLLISIONSPHERESPHERE_HPP_
#define MUNDY_METHODS_GENERATE_COLLISION_CONSTRAINTS_KERNELS_COLLISIONSPHERESPHERE_HPP_

/// \file CollisionSphereSphere.hpp
/// \brief Declaration of the GenerateCollisionConstraints' CollisionSphereSphere kernel.

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <string>   // for std::string
#include <utility>  // for std::make_pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                         // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                         // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>                // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>                      // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                       // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaRegistry.hpp>                     // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>                 // for mundy::meta::PartRequirements
#include <mundy_methods/GenerateCollisionConstraints.hpp>  // for mundy::methods::GenerateCollisionConstraints

namespace mundy {

namespace methods {

namespace generate_collision_constraints {

namespace kernels {

/// \class CollisionSphereSphere
/// \brief Concrete implementation of a \c MetaMultibodyTwoWayKernel for generating a collision constraint between two
/// spheres.
class CollisionSphereSphere : public mundy::meta::MetaKernel<void, CollisionSphereSphere>,
                              public GenerateCollisionConstraints::OurKernelRegistry<CollisionSphereSphere> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit CollisionSphereSphere(mundy::mesh::BulkData *const bulk_data_ptr,
                                 const Teuchos::ParameterList &fixed_params);
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
    std::string radius_field_name = valid_fixed_params.get<std::string>("radius_field_name");
    std::string aabb_field_name = valid_fixed_params.get<std::string>("aabb_field_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name("SPHERE");
    part_reqs->set_part_topology(stk::topology::PARTICLE);
    part_reqs->put_multibody_part_attribute(mundy::muntibody::Factory::get_fast_id("SPEHRE"));
    part_reqs->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(node_coord_field_name,
                                                                                      stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_req(
        std::make_shared<mundy::meta::FieldRequirements<double>>(radius_field_name, stk::topology::ELEMENT_RANK, 1, 1));
    part_reqs->add_field_req(
        std::make_shared<mundy::meta::FieldRequirements<double>>(aabb_field_name, stk::topology::ELEMENT_RANK, 4, 1));

    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_req(part_reqs);
    return multibody_part_params;
  }

  /// \brief Get the requirements that this kernel imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c
  /// PartRequirements will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements>([[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    std::shared_ptr<mundy::meta::PartRequirements> required_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_part_params->set_part_topology(stk::topology::PARTICLE);
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_coord_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_radius_field_name_), stk::topology::ELEMENT_RANK, 1, 1));
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_aabb_field_name_), stk::topology::ELEMENT_RANK, 4, 1));
    return required_part_params;
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_params;
    default_fixed_params.set(
        "aabb_field_name", std::string(default_aabb_field_name_),
        "Name of the element field within which the output axis-aligned boundary boxes will be written.");
    default_fixed_params.set("radius_field_name", std::string(default_radius_field_name_),
                             "Name of the element field containing the CollisionSphereSphere radius.");
    default_fixed_params.set("node_coordinate_field_name", std::string(default_node_coord_field_name_),
                             "Name of the node field containing the coordinate of the CollisionSphereSphere's center.");
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
    return std::make_shared<CollisionSphereSphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  Teuchos::ParameterList set_mutable_params(const Teuchos::ParameterList &mutable_params) const override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param collision [in] A collision element connected to the left and right spheres.
  /// \param left_sphere [in] The left sphere element attached to the collision element.
  /// \param right_sphere [in] The right sphere element attached to the collision element.
  void execute(const stk::mesh::Entity &collision, const stk::mesh::Entity &left_sphere,
               const stk::mesh::Entity &right_sphere) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_node_normal_field_name_ = "NODE_NORMAL";
  static constexpr std::string_view default_element_signed_separation_dist_field_name_ =
      "ELEMENT_SIGNED_SEPARATION_DIST";
  static constexpr std::string_view default_element_radius_field_name_ = "ELEMENT_RADIUS";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static constexpr std::string_view class_identifier_ = "COLLISIONSPHERESPHERE";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Name of the node field containing the coordinate of each Sphere's center and the contact points.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the normal at the point of contact.
  std::string node_normal_field_name_;

  /// \brief Name of the element field containing the signed separation distance between the spheres.
  std::string element_signed_separation_dist_field_name_;

  /// \brief Name of the element field containing the radius of the spheres.
  std::string element_radius_field_name_;

  /// \brief Node field containing the coordinate of each Sphere's center and the contact points.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the normal at the point of contact.
  stk::mesh::Field<double> *node_normal_field_ptr_ = nullptr;

  /// \brief Element field containing the signed separation distance between the spheres.
  stk::mesh::Field<double> *element_signed_separation_dist_field_ptr_ = nullptr;

  /// \brief Element field containing the radius of the spheres.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // CollisionSphereSphere

}  // namespace kernels

}  // namespace generate_collision_constraints

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_GENERATE_COLLISION_CONSTRAINTS_KERNELS_COLLISIONSPHERESPHERE_HPP_

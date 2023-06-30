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

#ifndef MUNDY_METHODS_COMPUTE_OBB_KERNELS_SPHERE_HPP_
#define MUNDY_METHODS_COMPUTE_OBB_KERNELS_SPHERE_HPP_

/// \file Sphere.hpp
/// \brief Declaration of the ComputeOBB's Sphere kernel.

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
#include <mundy_methods/ComputeOBB.hpp>      // for mundy::methods::ComputeOBB

namespace mundy {

namespace methods {

namespace compute_obb {

namespace kernels {

/// \class Sphere
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Sphere : public mundy::meta::MetaKernel<void, Sphere>, public ComputeOBB::OurKernelRegistry<Sphere> {
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
    std::string obb_field_name = valid_fixed_params.get<std::string>("obb_field_name");
    std::string radius_field_name = valid_fixed_params.get<std::string>("radius_field_name");
    std::string node_coord_field_name = valid_fixed_params.get<std::string>("node_coord_field_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name("SPHERE");
    part_reqs->set_part_topology(stk::topology::PARTICLE);
    part_reqs->put_multibody_part_attribute(mundy::muntibody::Factory::get_fast_id("SPHERE"));
    part_reqs->add_field_req(
        std::make_shared<mundy::meta::FieldRequirements<double>>(obb_field_name, stk::topology::ELEMENT_RANK, 4, 1));
    part_reqs->add_field_req(
        std::make_shared<mundy::meta::FieldRequirements<double>>(radius_field_name, stk::topology::ELEMENT_RANK, 1, 1));
    part_reqs->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(node_coord_field_name,
                                                                                      stk::topology::NODE_RANK, 3, 1));

    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_req(part_reqs);
    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("obb_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("obb_field_name");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'obb_field_name' but "
                                     << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("obb_field_name", std::string(default_obb_field_name_),
                            "Element field within which the output object-aligned boundary boxes will be written.");
    }

    if (fixed_params_ptr->isParameter("radius_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("radius_field_name");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'radius_field_name' but "
                                 "with a type other than std::string");
    } else {
      fixed_params_ptr->set("radius_field_name", std::string(default_radius_field_name_),
                            "Name of the element field containing the sphere radius.");
    }

    if (fixed_params_ptr->isParameter("node_coord_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_coord_field_name");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'node_coord_field_name' but "
                                     << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_coord_field_name", std::string(default_node_coord_field_name_),
                            "Name of the node field containing the coordinate of the sphere's center.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isParameter("buffer_distance")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("buffer_distance");
      TEUCHOS_TEST_FOR_EXCEPTION(valid_type, std::invalid_argument,
                                 "Sphere: Type error. Given a parameter with name 'buffer_distance' but "
                                     << "with a type other than unsigned");
    } else {
      mutable_params_ptr->set("buffer_distance", default_buffer_distance_,
                              "Buffer distance to be added to the axis-aligned boundary box.");
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

  /// \brief Setup the method's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  void setup() override;

  /// \brief Run the kernel's core calculation.
  /// \param sphere_element [in] The sphere element acted on by the kernel.
  void execute(const stk::mesh::Entity &sphere_element) override;

  /// \brief Finalize the method's core calculations.
  /// For example, communicate between ghosts, perform redictions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_obb_field_name_ = "OBB";
  static constexpr std::string_view default_radius_field_name_ = "RADIUS";
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static const std::string class_identifier_ = "SPHERE";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Buffer distance to be added to the object-aligned boundary box.
  ///
  /// For example, if the original object-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_;

  /// \brief Name of the element field within which the output object-aligned boundary boxes will be written.
  std::string obb_field_name_;

  /// \brief Name of the element field containing the sphere radius.
  std::string radius_field_name_;

  /// \brief Name of the node field containing the coordinate of the sphere's center.
  std::string node_coord_field_name_;

  /// \brief Element field within which the output object-aligned boundary boxes will be written.
  stk::mesh::Field<double> *obb_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *radius_field_ptr_;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_;
  //@}
};  // Sphere

}  // namespace kernels

}  // namespace compute_obb

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_OBB_KERNELS_SPHERE_HPP_

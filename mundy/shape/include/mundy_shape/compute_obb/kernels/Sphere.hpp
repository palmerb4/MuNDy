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

#ifndef MUNDY_SHAPE_COMPUTE_OBB_KERNELS_SPHERE_HPP_
#define MUNDY_SHAPE_COMPUTE_OBB_KERNELS_SPHERE_HPP_

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
#include <mundy_agent/AgentHierarchy.hpp>    // for mundy::agent::AgentHierarchy
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>           // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>        // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace shape {

namespace compute_obb {

namespace kernels {

/// \class Sphere
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Sphere : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

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
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>();
    std::string element_obb_field_name = valid_fixed_params.get<std::string>("element_obb_field_name");
    Teuchos::Array<std::string> input_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("input_part_names");
    const int num_parts = input_part_names.size();
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = input_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
          element_obb_field_name, stk::topology::ELEMENT_RANK, 6, 1));

      if (part_name == "SPHERES") {
        // Add the requirements directly to spheres part.
        const std::string parent_part_name = "SHAPES";
        mundy::agent::AgentHierarchy::add_part_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      } else {
        // Add the associated part as a subset of the spheres part.
        const std::string parent_part_name = "SPHERES";
        mundy::agent::AgentHierarchy::add_subpart_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      }
    }
    return mesh_reqs_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr, mundy::meta::ParamConfig<Teuchos::Array<std::string>>{
                              .name = "input_part_names",
                              .default_value = Teuchos::tuple<std::string>(std::string(default_part_name_)),
                              .doc_string = "Name of the parts associated with this kernel."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr,
        mundy::meta::ParamConfig<std::string>{
            .name = "element_obb_field_name",
            .default_value = std::string(default_element_obb_field_name_),
            .doc_string = "Element field within which the output object-aligned boundary boxes will be written."});
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    mundy::meta::check_parameter_and_set_default(
        mutable_params_ptr,
        mundy::meta::ParamConfig<double>{.name = "buffer_distance",
                            .default_value = default_buffer_distance_,
                            .doc_string = "Buffer distance to be added to the axis-aligned boundary box."});
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
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_element_obb_field_name_ = "ELEMENT_OBB";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "SPHERES";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Buffer distance to be added to the object-aligned boundary box.
  ///
  /// For example, if the original object-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_ = default_buffer_distance_;

  /// \brief Element field within which the output object-aligned boundary boxes will be written.
  stk::mesh::Field<double> *element_obb_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *element_radius_field_ptr_;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_;
  //@}
};  // Sphere

}  // namespace kernels

}  // namespace compute_obb

}  // namespace shape

}  // namespace mundy

#endif  // MUNDY_SHAPE_COMPUTE_OBB_KERNELS_SPHERE_HPP_

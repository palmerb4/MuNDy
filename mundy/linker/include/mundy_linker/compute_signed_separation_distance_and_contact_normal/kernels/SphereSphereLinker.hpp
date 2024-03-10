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

#ifndef MUNDY_LINKER_COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL_SPHERESPHERELINKER_HPP_
#define MUNDY_LINKER_COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL_SPHERESPHERELINKER_HPP_

/// \file SphereSphereLinker.hpp
/// \brief Declaration of the ComputeSignedSeparationDistanceAndContactNormal's SphereSphereLinker kernel.

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
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace linker {

namespace compute_signed_separation_distance_and_contact_normal {

namespace kernels {

/// \class SphereSphereLinker
/// \brief Concrete implementation of \c MetaKernel for computing the signed separation distance and contact normal
/// between two spheres.
class SphereSphereLinker : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit SphereSphereLinker(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    valid_fixed_params.validateParametersAndSetDefaults(SphereSphereLinker::get_valid_fixed_params());

    valid_fixed_params.print(std::cout, Teuchos::ParameterList::PrintOptions().showDoc(true).indent(2).showTypes(true));

    // Add the requirements for the linker.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>();
    std::string linker_signed_separation_distance_field_name =
        valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
    std::string linker_contact_normal_field_name =
        valid_fixed_params.get<std::string>("linker_contact_normal_field_name");

    Teuchos::Array<std::string> valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_linker_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_linker_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
          linker_signed_separation_distance_field_name, stk::topology::CONSTRAINT_RANK, 1, 1));
      part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
          linker_contact_normal_field_name, stk::topology::CONSTRAINT_RANK, 3, 1));

      if (part_name == "SPHERE_SPHERE_LINKERS") {
        // Add the requirements directly to sphere sphere linker part.
        const std::string parent_part_name = "NEIGHBOR_LINKERS";
        mundy::agent::AgentHierarchy::add_part_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      } else {
        // Add the associated part as a subset of the spheres part.
        const std::string parent_part_name = "SPHERE_SPHERE_LINKERS";
        mundy::agent::AgentHierarchy::add_subpart_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      }
    }

    // Add the requirements for the connected spheres linker.
    // We don't have any requirements for the connected spheres not already specified by the sphere agent (center
    // position and radius).
    Teuchos::Array<std::string> valid_sphere_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");
    const int num_sphere_parts = static_cast<int>(valid_sphere_part_names.size());
    for (int i = 0; i < num_sphere_parts; i++) {
      const std::string part_name = valid_sphere_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
      part_reqs->set_part_name(part_name);
      if (part_name == "SPHERES") {
        // Add the requirements directly to sphere sphere linker part.
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

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set<Teuchos::Array<std::string>>("valid_entity_part_names",
                                                            Teuchos::tuple<std::string>("SPHERE_SPHERE_LINKERS"),
                                                            "List of valid entity part names for the kernel.");

    // Soap box: Why do we allow the user to specify valid_sphere_part_names rather than simply applying requirements
    // to the SPHERES part? Well, users might not want to apply the requirements to the SPHERES part. They might want to
    // apply the requirements to a subset of the SPHERES part. Why apply the requirements of this class to all spheres
    // if it'll only be used on a subset of them?
    default_parameter_list.set<Teuchos::Array<std::string>>("valid_sphere_part_names",
                                                            Teuchos::tuple<std::string>("SPHERES"),
                                                            "List of valid sphere part names for the kernel.");
    default_parameter_list.set(
        "linker_signed_separation_distance_field_name",
        std::string(default_linker_signed_separation_distance_field_name_),
        "Name of the constraint-rank field within which the signed separation distance will be written.");
    default_parameter_list.set("linker_contact_normal_field_name",
                               std::string(default_linker_contact_normal_field_name_),
                               "Name of the constraint-rank field within which the contact normal (pointing from left "
                               "entity to right entity) will be written.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override;

  /// \brief Get the entity rank that the kernel acts on.
  stk::topology::rank_t get_entity_rank() const override;

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernel<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<SphereSphereLinker>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Setup the kernel's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  void setup() override;

  /// \brief Run the kernel's core calculation.
  /// \param sphere_sphere_linker [in] The linker acted on by this kernel.
  void execute(const stk::mesh::Entity &sphere_sphere_linker) override;

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_linker_signed_separation_distance_field_name_ =
      "LINKER_SIGNED_SEPARATION_DISTANCE";
  static constexpr std::string_view default_linker_contact_normal_field_name_ = "LINKER_CONTACT_NORMAL";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The valid sphere parts.
  std::vector<stk::mesh::Part *> valid_sphere_parts_;

  /// \brief Node coordinate field.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Element radius field.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;

  /// \brief Linker signed separation distance field.
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr_ = nullptr;

  /// \brief Linker contact normal field.
  stk::mesh::Field<double> *linker_contact_normal_field_ptr_ = nullptr;
  //@}
};  // SphereSphereLinker

}  // namespace kernels

}  // namespace compute_signed_separation_distance_and_contact_normal

}  // namespace linker

}  // namespace mundy

#endif  // MUNDY_LINKER_COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL_SPHERESPHERE_HPP_

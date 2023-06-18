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

#ifndef MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_
#define MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_

/// \file MapRigidBodyVelocityToSurfaceVelocity.hpp
/// \brief Declaration of the MapRigidBodyVelocityToSurfaceVelocity class

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <string>   // for std::string
#include <utility>  // for std::pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Entity.hpp>      // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>    // for stk::mesh::Selector
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaTwoWayKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_methods/compute_mobility/techniques/RigidBodyMotion.hpp>  // for mundy::methods::...::RigidBodyMotion

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

/// \class MapRigidBodyVelocityToSurfaceVelocity
/// \brief Method for using rigid body motion about a known body point to compute the velocity at all surface points.
class MapRigidBodyVelocityToSurfaceVelocity
    : public mundy::meta::MetaMethod<void, MapRigidBodyVelocityToSurfaceVelocity>,
      public RigidBodyMotion::OurMethodRegistry<MapRigidBodyVelocityToSurfaceVelocity> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MapRigidBodyVelocityToSurfaceVelocity() = delete;

  /// \brief Constructor
  MapRigidBodyVelocityToSurfaceVelocity(mundy::mesh::BulkData *const bulk_data_ptr,
                                        const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name Typedefs
  //@{

  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, MapRigidBodyVelocityToSurfaceVelocity>;

  template <typename ClassToRegister>
  using OurKernelRegistry = mundy::meta::MetaKernelRegistry<void, ClassToRegister, MapRigidBodyVelocityToSurfaceVelocity>;
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements>(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_fixed_parameter_list = fixed_params;
    valid_fixed_parameter_list.validateParametersAndSetDefaults(static_get_valid_fixed_params());

    // Create and store the required part params. One per input part.
    Teuchos::ParameterList &parts_parameter_list = valid_fixed_parameter_list.sublist("input_part_pairs");
    const unsigned num_part_pairs = parts_parameter_list.get<unsigned>("count");
    std::vector<std::shared_ptr<mundy::meta::PartRequirements>> part_requirements;
    for (size_t i = 0; i < num_part_pairs; i++) {
      // Create a new part requirement.
      part_requirements.emplace_back(std::make_shared<mundy::meta::PartRequirements>());
      part_requirements.emplace_back(std::make_shared<mundy::meta::PartRequirements>());

      // Fetch the i'th part parameters.
      Teuchos::ParameterList &part_pair_parameter_list =
          parts_parameter_list.sublist("input_part_pair_" + std::to_string(i));
      const Teuchos::Array<std::string> pair_names = part_pair_parameter_list.get<Teuchos::Array<std::string>>("name");

      // Add method-specific requirements.
      part_requirements[i - 1]->set_part_name(pair_names[0]);
      part_requirements[i]->set_part_name(pair_names[1]);
      part_requirements[i - 1]->set_part_rank(stk::topology::CONSTRAINT_RANK);
      part_requirements[i]->set_part_rank(stk::topology::ELEMENT_RANK);

      // Fetch the parameters for this part's kernel.
      Teuchos::ParameterList &part_kernel_parameter_list =
          part_parameter_list.sublist("kernels").sublist("map_rigid_body_velocity_to_surface_velocity");

      // Validate the kernel params and fill in defaults.
      const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
      part_kernel_parameter_list.validateParametersAndSetDefaults(
          mundy::meta::MetaTwoWayKernelFactory<void, MapRigidBodyVelocityToSurfaceVelocity>::get_valid_params(
              kernel_name));

      // Merge the kernel requirements.
      std::pair<std::shared_ptr<mundy::meta::PartRequirements>, std::shared_ptr<mundy::meta::PartRequirements>>
          pair_requirements =
              mundy::meta::MetaTwoWayKernelFactory<void, MapRigidBodyVelocityToSurfaceVelocity>::get_part_requirements(
                  kernel_name, part_kernel_parameter_list);
      part_requirements[i - 1]->merge(pair_requirements.first);
      part_requirements[i]->merge(pair_requirements.second);
    }

    return part_requirements;
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_parameter_list;
    Teuchos::ParameterList &kernel_params = default_fixed_parameter_list.sublist(
        "kernels", false, "Sublist that defines the kernels and their parameters.");
    kernel_params.sublist("map_rigid_body_velocity_to_surface_velocity", false,
                          "Sublist that defines the map's kernel parameters.");
    return default_fixed_parameter_list;
  }

  /// \brief Get the default mutable parameters for this class (those that do not impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_mutable_params() {
    static Teuchos::ParameterList default_mutable_parameter_list;
    return default_mutable_parameter_list;
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<MapRigidBodyVelocityToSurfaceVelocity>(bulk_data_ptr, parameter_list);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view class_identifier_ = "MAP_RIGID_BODY_VELOCITY_TO_SURFACE_VELOCITY";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of part pairs that this method acts on.
  size_t num_part_pairs_ = 0;

  /// \brief Vector of pointers to the part pairs that this class will act upon.
  std::vector<std::pair<stk::mesh::Part *>> part_pair_ptr_vector_;

  /// \brief Kernels corresponding to each of the specified part pairs.
  std::vector<std::shared_ptr<mundy::meta::MetaTwoWayKernelBase<void>>> kernel_ptrs_;
  //@}
};  // MapRigidBodyVelocityToSurfaceVelocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_

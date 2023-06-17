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

#ifndef MUNDY_METHODS_COMPUTEBOUNDINGRADIUS_HPP_
#define MUNDY_METHODS_COMPUTEBOUNDINGRADIUS_HPP_

/// \file ComputeBoundingRadius.hpp
/// \brief Declaration of the ComputeBoundingRadius class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

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
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace methods {

/// \class ComputeBoundingRadius
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeBoundingRadius : public mundy::meta::MetaMethod<void, ComputeBoundingRadius>,
                              public mundy::meta::MetaMethodRegistry<void, ComputeBoundingRadius> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeBoundingRadius() = delete;

  /// \brief Constructor
  ComputeBoundingRadius(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list);
  //@}

  //! \name Typedefs
  //@{

  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, std::string, ComputeBoundingRadius>;

  template<typename ClassToRegister>
  using OurKernelRegistry = mundy::meta::MetaKernelRegistry<void, ClassToRegister, std::string, ComputeBoundingRadius>;
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> details_static_get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_parameter_list) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_fixed_params = fixed_parameter_list;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // This method itself does not impose requirements, but the kernels might.
    Teuchos::Array &enabled_multibody_type_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("enabled_multibody_type_names");
    Teuchos::ParameterList &kernel_params = valid_fixed_params.sublist("kernels").sublist("compute_bounding_radius");
    for (const auto enabled_multibody_type_name : enabled_multibody_type_names) {
      Teuchos::ParameterList &multibody_params = kernel_params.sublist(enabled_multibody_type_name);
      mundy::multibody::Factory::declare_mesh_requirements(enabled_multibody_type_name, meta_data_ptr,
                                                           multibody_params);
    }
  }
  /// \brief Validate the default fixed parameters for this class (those that impact the mesh requirements) and set
  /// their defaults.
  ///
  /// The only required parameter is "enabled_multibody_type_names" which must specify the name of at least one
  /// multibody type to enable. The compute_bounding_radius kernel associated with this type must be registered with our
  /// kernel factory.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *fixed_parameter_list_ptr) {
    Teuchos::ParameterList params = &fixed_parameter_list_ptr;
    TEUCHOS_TEST_FOR_EXCEPTION(
        params.isParameter("enabled_multibody_type_names"), std::invalid_argument,
        "ComputeBoundingRadius: The provided parameter list must include the set of enabled multibody type names.");
    Teuchos::Array &enabled_multibody_type_names =
        params.get<Teuchos::Array<std::string>>("enabled_multibody_type_names");
    TEUCHOS_TEST_FOR_EXCEPTION(enabled_multibody_type_names.size() != 0, std::invalid_argument,
                               "ComputeBoundingRadius: The enabled multibody type names must not be empty.");

    Teuchos::ParameterList &kernel_params =
        fixed_parameter_list_ptr->sublist("kernels", false).sublist("compute_bounding_radius", false);
    for (const auto enabled_multibody_type_name : enabled_multibody_type_names) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          mundy::multibody::Factory::is_valid(enabled_multibody_type_name), std::invalid_argument,
          "ComputeBoundingRadius: Failed to find a multibody type with name (" << enabled_multibody_type_name << ").");
      TEUCHOS_TEST_FOR_EXCEPTION(
          OurKernelFactory::is_valid_key(enabled_multibody_type_name), std::invalid_argument,
          "ComputeBoundingRadius: Failed to find a compute_bounding_radius kernel associated with the "
          "provided multibody type name ("
              << enabled_multibody_type_name << ").");
      Teuchos::ParameterList &multibody_params = kernel_params.sublist(enabled_multibody_type_name, false);
      OurKernelFactory::validate_fixed_parameters_and_set_defaults(enabled_multibody_type_name, multibody_params);
    }
  }

  /// \brief Get the default mutable parameters for this class (those that do not impact the mesh requirements) and
  /// set their defaults.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *mutable_parameter_list_ptr) {
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return std::make_shared<ComputeBoundingRadius>(bulk_data_ptr, fixed_parameter_list);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_parameter_list) override;
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
  static constexpr std::string_view class_identifier_ = "COMPUTE_BOUNDING_SPHERE";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of parts that this method acts on.
  size_t num_parts_ = 0;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  std::vector<std::shared_ptr<mundy::meta::MetaKernelBase<void>>> compute_bounding_sphere_kernel_ptrs_;
  //@}
};  // ComputeBoundingRadius

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEBOUNDINGRADIUS_HPP_

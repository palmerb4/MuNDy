// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

#ifndef MUNDY_LINKERS_LINKERPOTENTIALFORCEMAGNITUDEREDUCTION_HPP_
#define MUNDY_LINKERS_LINKERPOTENTIALFORCEMAGNITUDEREDUCTION_HPP_

/// \file LinkerPotentialForceMagnitudeReduction.hpp
/// \brief Declaration of the LinkerPotentialForceMagnitudeReduction class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_linkers/linker_potential_force_magnitude_reduction/kernels/Sphere.hpp>  // for mundy::linkers::...::kernels::Sphere
#include <mundy_linkers/linker_potential_force_magnitude_reduction/kernels/Spherocylinder.hpp>  // for mundy::linkers::...::kernels::Spherocylinder
#include <mundy_linkers/linker_potential_force_magnitude_reduction/kernels/SpherocylinderSegment.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegment
#include <mundy_mesh/BulkData.hpp>                                                      // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace linkers {

/// \class LinkerPotentialForceMagnitudeReduction
/// \brief Method for summing the effect of multiple linker potentials.
class LinkerPotentialForceMagnitudeReduction
    : public mundy::meta::MetaKernelDispatcher<LinkerPotentialForceMagnitudeReduction,
                                               mundy::meta::make_registration_string(
                                                   "LINKER_POTENTIAL_FORCE_MAGNITUDE_REDUCTION")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  LinkerPotentialForceMagnitudeReduction() = delete;

  /// \brief Constructor
  LinkerPotentialForceMagnitudeReduction(mundy::mesh::BulkData *const bulk_data_ptr,
                                         const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<LinkerPotentialForceMagnitudeReduction,
                                          mundy::meta::make_registration_string(
                                              "LINKER_POTENTIAL_FORCE_MAGNITUDE_REDUCTION")>(bulk_data_ptr,
                                                                                             fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("name_of_linker_part_to_reduce_over",
                               std::string(default_name_of_linker_part_to_reduce_over_),
                               "The name of the linker part that we will reduce over.");
    default_parameter_list.set<std::string>(
        "linker_potential_force_magnitude_field_name",
        std::string(default_linker_potential_force_magnitude_field_name_),
        "The field name of the linker potential force magnitude field that we will use for the reduction.");
    default_parameter_list.set("linker_contact_normal_field_name",
                               std::string(default_linker_contact_normal_field_name_),
                               "The field name of the linker contact normal along which the force potentials act.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_name_of_linker_part_to_reduce_over_ = "NEIGHBOR_LINKERS";
  static constexpr std::string_view default_linker_contact_normal_field_name_ = "LINKER_CONTACT_NORMAL";
  static constexpr std::string_view default_linker_potential_force_magnitude_field_name_ =
      "LINKER_POTENTIAL_FORCE_MAGNITUDE";
  //@}

};  // LinkerPotentialForceMagnitudeReduction

}  // namespace linkers

}  // namespace mundy

//! \name Registration
//@{
/// @brief Register our default kernels
MUNDY_REGISTER_METACLASS("SPHERE", mundy::linkers::linker_potential_force_magnitude_reduction::kernels::Sphere,
                         mundy::linkers::LinkerPotentialForceMagnitudeReduction::OurKernelFactory)
MUNDY_REGISTER_METACLASS("SPHEROCYLINDER",
                          mundy::linkers::linker_potential_force_magnitude_reduction::kernels::Spherocylinder,
                          mundy::linkers::LinkerPotentialForceMagnitudeReduction::OurKernelFactory)
MUNDY_REGISTER_METACLASS("SPHEROCYLINDER_SEGMENT",
                          mundy::linkers::linker_potential_force_magnitude_reduction::kernels::SpherocylinderSegment,
                          mundy::linkers::LinkerPotentialForceMagnitudeReduction::OurKernelFactory)
//@}

#endif  // MUNDY_LINKERS_LINKERPOTENTIALFORCEMAGNITUDEREDUCTION_HPP_

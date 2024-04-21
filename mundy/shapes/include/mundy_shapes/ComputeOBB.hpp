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

#ifndef MUNDY_SHAPES_COMPUTEOBB_HPP_
#define MUNDY_SHAPES_COMPUTEOBB_HPP_

/// \file ComputeOBB.hpp
/// \brief Declaration of the ComputeOBB class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS
#include <mundy_shapes/compute_obb/kernels/Sphere.hpp>          // for mundy::shapes::compute_obb::kernels::Sphere
#include <mundy_shapes/compute_obb/kernels/Spherocylinder.hpp>  // for mundy::shapes::compute_obb::kernels::Spherocylinder
#include <mundy_shapes/compute_obb/kernels/SpherocylinderSegment.hpp>  // for mundy::shapes::compute_obb::kernels::SpherocylinderSegment

namespace mundy {

namespace shapes {

/// \class ComputeOBB
/// \brief Method for computing the object aligned boundary box of different parts.
class ComputeOBB
    : public mundy::meta::MetaKernelDispatcher<ComputeOBB, mundy::meta::make_registration_string("COMPUTE_OBB")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeOBB() = delete;

  /// \brief Constructor
  ComputeOBB(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<ComputeOBB, mundy::meta::make_registration_string("COMPUTE_OBB")>(
            bulk_data_ptr, fixed_params) {
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
    default_parameter_list.set("element_obb_field_name", std::string(default_element_obb_field_name_),
                               "Element field within which the output object-aligned boundary boxes will be written.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("buffer_distance", default_buffer_distance_,
                               "Buffer distance to be added to the object-aligned boundary box.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_element_obb_field_name_ = "ELEMENT_OBB";
  //@}
};  // ComputeOBB

}  // namespace shapes

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register our default kernels
MUNDY_REGISTER_METACLASS("SPHERE", mundy::shapes::compute_obb::kernels::Sphere,
                         mundy::shapes::ComputeOBB::OurKernelFactory)
MUNDY_REGISTER_METACLASS("SPHEROCYLINDER", mundy::shapes::compute_obb::kernels::Spherocylinder,
                         mundy::shapes::ComputeOBB::OurKernelFactory)
MUNDY_REGISTER_METACLASS("SPHEROCYLINDER_SEGMENT", mundy::shapes::compute_obb::kernels::SpherocylinderSegment,
                         mundy::shapes::ComputeOBB::OurKernelFactory)
//@}

#endif  // MUNDY_SHAPES_COMPUTEOBB_HPP_

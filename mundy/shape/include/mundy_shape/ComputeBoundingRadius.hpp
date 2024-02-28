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

#ifndef MUNDY_SHAPE_COMPUTEBOUNDINGRADIUS_HPP_
#define MUNDY_SHAPE_COMPUTEBOUNDINGRADIUS_HPP_

/// \file ComputeBoundingRadius.hpp
/// \brief Declaration of the ComputeBoundingRadius class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>           // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_shape/compute_bounding_radius/kernels/Sphere.hpp>  // for mundy::shape::compute_bounding_radius::kernels::Sphere

namespace mundy {

namespace shape {

/// \class ComputeBoundingRadius
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeBoundingRadius
    : public mundy::meta::MetaKernelDispatcher<ComputeBoundingRadius,
                                               mundy::meta::make_registration_string("COMPUTE_BOUNDING_SPHERE")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeBoundingRadius() = delete;

  /// \brief Constructor
  ComputeBoundingRadius(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<ComputeBoundingRadius,
                                          mundy::meta::make_registration_string("COMPUTE_BOUNDING_SPHERE")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require all kernels registered with our kernel factory to have.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("element_bounding_radius_field_name",
                               std::string(default_element_bounding_radius_field_name_),
                               "Name of the element field within which the output bounding radius will be written.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require all kernels registered with our kernel factory to have.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("buffer_distance", default_buffer_distance_,
                               "Buffer distance to be added to the bounding radius.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_element_bounding_radius_field_name_ = "ELEMENT_BOUNDING_RADIUS";
  //@}
};  // ComputeBoundingRadius

}  // namespace shape

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register our default kernels
MUNDY_REGISTER_METACLASS("SPHERE", mundy::shape::compute_bounding_radius::kernels::Sphere,
                         mundy::shape::ComputeBoundingRadius::OurKernelFactory)
//@}

#endif  // MUNDY_SHAPE_COMPUTEBOUNDINGRADIUS_HPP_

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

/// \file ComputeBoundingRadius.cpp
/// \brief Definition of the ComputeBoundingRadius class

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                       // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>                           // for mundy::meta::MetaKernelDispatcher
#include <mundy_shape/ComputeBoundingRadius.hpp>                       // for mundy::shape::ComputeBoundingRadius
#include <mundy_shape/compute_bounding_radius/kernels/AllKernels.hpp>  // performs the registration of all kernels

namespace mundy {

namespace shape {

// \name Constructors and destructor
//{

ComputeBoundingRadius::ComputeBoundingRadius(mundy::mesh::BulkData *const bulk_data_ptr,
                                             const Teuchos::ParameterList &fixed_params)
    : mundy::meta::MetaKernelDispatcher<ComputeBoundingRadius>(bulk_data_ptr, fixed_params) {
}
//}

}  // namespace shape

}  // namespace mundy

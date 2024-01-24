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

/// \file ComputeConstraintForcing.cpp
/// \brief Definition of the ComputeConstraintForcing class

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_constraint/ComputeConstraintForcing.hpp>  // for mundy::constraint::ComputeConstraintForcing
#include <mundy_mesh/BulkData.hpp>                                             // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>                                 // for mundy::meta::MetaKernelDispatcher

namespace mundy {

namespace constraint {

// \name Constructors and destructor
//{

// TODO(palmerb4): The following is incorrect because we do never reset the constraint force field.
// We need to add in alpha and beta to let the user choose.
ComputeConstraintForcing::ComputeConstraintForcing(mundy::mesh::BulkData *const bulk_data_ptr,
                                                   const Teuchos::ParameterList &fixed_params)
    : mundy::meta::MetaKernelDispatcher<ComputeConstraintForcing>(bulk_data_ptr, fixed_params) {
}
//}

}  // namespace constraint

}  // namespace mundy

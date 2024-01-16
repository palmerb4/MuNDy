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

/// \file ComputeMobility.cpp
/// \brief Definition of the ComputeMobility class

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy/StringLiteral.hpp>                                     // for mundy::make_string_literal
#include <mundy_mesh/BulkData.hpp>                                     // for mundy::mesh::BulkData
#include <mundy_meta/MetaTechniqueDispatcher.hpp>                      // for mundy::meta::MetaTechniqueDispatcher
#include <mundy_motion/ComputeMobility.hpp>                            // for mundy::motion::ComputeMobility
#include <mundy_motion/compute_mobility/techniques/AllTechniques.hpp>  // performs the registration of all techniques

namespace mundy {

namespace motion {

// \name Constructors and destructor
//{

ComputeMobility::ComputeMobility(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : mundy::meta::MetaTechniqueDispatcher<ComputeMobility, mundy::make_string_literal("RIGID_BODY_MOTION")>(
          bulk_data_ptr, fixed_params) {
}
//}

}  // namespace motion

}  // namespace mundy

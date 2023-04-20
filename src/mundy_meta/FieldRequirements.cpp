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

#ifndef MUNDY_META_FIELDREQUIREMENTS_HPP_
#define MUNDY_META_FIELDREQUIREMENTS_HPP_

/// \file FieldRequirements.hpp
/// \brief Declaration of the FieldRequirements class

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>          // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsRegistry.hpp>  // for mundy::meta::FieldRequirementsRegistry

namespace mundy {

namespace meta {

/// @brief Perform the static registration of the desired FieldRequirements FieldTypes.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
// clang-format off
const bool FieldRequirementsRegistry<float>::is_registered = FieldRequirementsRegistry<float>::register_type("FLOAT");
const bool FieldRequirementsRegistry<double>::is_registered =FieldRequirementsRegistry<double>::register_type("DOUBLE");
const bool FieldRequirementsRegistry<int>::is_registered = FieldRequirementsRegistry<int>::register_type("INT");
const bool FieldRequirementsRegistry<int64_t>::is_registered =FieldRequirementsRegistry<int64_t>::register_type("INT64");
const bool FieldRequirementsRegistry<unsigned>::is_registered =FieldRequirementsRegistry<unsigned>::register_type("UNSIGNED");
// clang-format on

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQUIREMENTS_HPP_

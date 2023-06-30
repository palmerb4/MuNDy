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

#ifndef MUNDY_MULTIBODY_MULTIBODYATTRIBUTE_HPP_
#define MUNDY_MULTIBODY_MULTIBODYATTRIBUTE_HPP_

/// \file MultibodyAttribute.hpp
/// \brief Declaration of the MultibodyAttribute struct

// Mundy libs
#include <mundy_multibody/Multibody.hpp>  // for mundy::multibody::Multibody

namespace mundy {

namespace multibody {

/// \struct MultibodyAttribute
/// \brief A simple struct that uses value semantics to store a multibody's fast ID. This is compatible with STK's
/// mesh/part/field attribute design.
struct MultibodyAttribute {
  multibody_t value;
};  // MultibodyAttribute

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_MULTIBODYATTRIBUTE_HPP_

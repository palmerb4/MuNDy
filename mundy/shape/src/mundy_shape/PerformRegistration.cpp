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

/// \file PerformRegistration.cpp
/// \brief Perform all registrations within MundyAgent.

// Mundy libs
#include <mundy_agent/AgentHierarchy.hpp>          // for mundy::agent::AgentHierarchy
#include <mundy_shape/Shapes.hpp>                  // for mundy::shape::Shapes
#include <mundy_shape/shapes/Spheres.hpp>          // for mundy::shape::shapes::Spheres
#include <mundy_shape/shapes/Spherocylinders.hpp>  // for mundy::shape::shapes::Spherocylinders
#include <mundy_agent/PerformRegistration.hpp>     // for mundy::agent::perform_registration

namespace mundy {

namespace shape {

void perform_registration() {
  mundy::agent::perform_registration();

  mundy::agent::AgentHierarchy::register_new_class<Shapes>();
  mundy::agent::AgentHierarchy::register_new_class<shapes::Spheres>();
  mundy::agent::AgentHierarchy::register_new_class<shapes::Spherocylinders>();
}

}  // namespace shape

}  // namespace mundy

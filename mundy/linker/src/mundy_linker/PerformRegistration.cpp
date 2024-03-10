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
/// \brief Perform all registrations within MundyLinker.

// Mundy libs
#include <MundyLinker_config.hpp>                                   // for HAVE_MUNDYLINKER_MUNDYSHAPE
#include <mundy_agent/AgentHierarchy.hpp>                           // for mundy::agent::AgentHierarchy
#include <mundy_agent/PerformRegistration.hpp>                      // for mundy::agent::perform_registration
#include <mundy_linker/Linkers.hpp>                                 // for mundy::linker::Linkers
#include <mundy_linker/linkers/NeighborLinkers.hpp>                 // for mundy::linker::linkers::NeighborLinkers
#include <mundy_linker/linkers/neighbor_linkers/SphereSpheres.hpp>  // for mundy::linker::linkers::neighbor_linkers::SphereSpheres

#ifdef HAVE_MUNDYLINKER_MUNDYSHAPE
#include <mundy_shape/PerformRegistration.hpp>  // for mundy::shape::perform_registration
#endif

#ifdef HAVE_MUNDYLINKER_MUNDYCONSTRAINT
#include <mundy_constraint/PerformRegistration.hpp>  // for mundy::constraint::perform_registration
#endif

namespace mundy {

namespace linker {

void perform_registration() {
  mundy::agent::perform_registration();

#ifdef HAVE_MUNDYLINKER_MUNDYSHAPE
  mundy::shape::perform_registration();
#endif

#ifdef HAVE_MUNDYLINKER_MUNDYCONSTRAINT
  mundy::constraint::perform_registration();
#endif

  mundy::agent::AgentHierarchy::register_new_class<Linkers>();
  mundy::agent::AgentHierarchy::register_new_class<linkers::NeighborLinkers>();
  mundy::agent::AgentHierarchy::register_new_class<linkers::neighbor_linkers::SphereSpheres>();
}

}  // namespace linker

}  // namespace mundy

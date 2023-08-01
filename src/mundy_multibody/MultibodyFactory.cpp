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

/// \file MultibodyFactory.cpp
/// \brief Definition of the MultibodyFactory class

// Mundy libs
#include <mundy_multibody/Multibody.hpp>         // for mundy::multibody::multibody_t
#include <mundy_multibody/MultibodyFactory.hpp>  // for mundy::multibody::MultibodyFactory

// While not directly used by this class, this header is included here to ensure that all multibody types are
// registered with the factory. If you are a user and wish to register your own multibody type, make sure that
// you include the header file containing MUNDY_REGISTER_MULTIBODYTYPE(YourType) somewhere in your code. Otherwise,
// registration will never occur and your type will not be recognized by the factory. To avoid cyclic dependencies,
// follow our example and use an include within a cpp file.
#include <mundy_multibody/type/AllTypes.hpp>  // performs registration of all multibody types

namespace mundy {

namespace multibody {

// Static member initialization
// Note for devs: If this static member initialization is removed, the linker will optimize out the include of
// AllTypes.hpp and the registration of our multibody types will not occur.
multibody_t MultibodyFactory::number_of_registered_types_ = 0;

}  // namespace multibody

}  // namespace mundy

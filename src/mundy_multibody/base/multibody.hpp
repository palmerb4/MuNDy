// @HEADER
// **********************************************************************************************************************
//
//                                          MuNDy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// MuNDy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// MuNDy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with MuNDy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_MULTIBODY_HPP_
#define MUNDY_MULTIBODY_HPP_

/// \file multibody.hpp
/// \brief Declaration of the multibody class

namespace mundy {

/// \class multibody
/// \brief An enumerator defining the various multibody objects—both particles and constraints.
///
/// To add new particles or constraints to MuNDy, contact the design team.
enum class multibody {
  //! \name Particles
  //@{
  SPHERE,
  SPHEROCYLINDER,
  SUPERELLIPSOID,
  POLYTOPE,
  //@}

  //! \name Constraints
  //@{
  COLLISION,
  SPRING,
  TORSIONALSPRING,
  JOINT,
  HINGE
  //@}
};  // class multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_HPP_

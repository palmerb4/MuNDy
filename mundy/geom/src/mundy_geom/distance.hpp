// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

#ifndef MUNDY_MATH_DISTANCE_HPP_
#define MUNDY_MATH_DISTANCE_HPP_

// All of our headers for computing separation distances between shape primitives

// For now, we are skipping planes


// Point to <shape> distance headers
#include <mundy_math/distance/PointPoint.hpp>
#include <mundy_math/distance/PointLine.hpp>
#include <mundy_math/distance/PointLineSegment.hpp>
// #include <mundy_math/distance/PointPlane.hpp>
#include <mundy_math/distance/PointSphere.hpp>
#include <mundy_math/distance/PointEllipsoid.hpp>

// Line to <shape> distance headers
#include <mundy_math/distance/LineLine.hpp>
#include <mundy_math/distance/LineLineSegment.hpp>
// #include <mundy_math/distance/LinePlane.hpp>
#include <mundy_math/distance/LineSphere.hpp>
#include <mundy_math/distance/LineEllipsoid.hpp>

// Line segment to <shape> distance headers
#include <mundy_math/distance/LineSegmentLineSegment.hpp>
// #include <mundy_math/distance/LineSegmentPlane.hpp>
#include <mundy_math/distance/LineSegmentSphere.hpp>
#include <mundy_math/distance/LineSegmentEllipsoid.hpp>

// Plane to <shape> distance headers
// #include <mundy_math/distance/PlanePlane.hpp>
// #include <mundy_math/distance/PlaneSphere.hpp>
// #include <mundy_math/distance/PlaneEllipsoid.hpp>

// Sphere to <shape> distance headers
#include <mundy_math/distance/SphereSphere.hpp>
#include <mundy_math/distance/SphereEllipsoid.hpp>

// Ellipsoid to <shape> distance headers
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>

#endif  // MUNDY_MATH_DISTANCE_HPP_
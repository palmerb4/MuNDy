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

#ifndef MUNDY_DRIVER_CONFIGURATOR_HPP_
#define MUNDY_DRIVER_CONFIGURATOR_HPP_

/// \file Configurator.hpp
/// \brief Declaration of the Configurator class

// C++ core libs

// Trilinos libs

// Mundy libs
#include <MundyDriver_config.hpp>                                     // for HAVE_MUNDYLINKER_*
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::StringBasedMetaFactory
#include <mundy_meta/MetaMethodExecutionInterface.hpp>                // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS

// Mundy libs to register with the ConfigurableMetaMethodFactory
#ifdef HAVE_MUNDYLINKER_MUNDYCONSTRAINTS
#include <mundy_constraints/ComputeConstraintForcing.hpp>     // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/ComputeConstraintProjection.hpp>  // for mundy::constraints::ComputeConstraintProjection
#include <mundy_constraints/ComputeConstraintResidual.hpp>    // for mundy::constraints::ComputeConstraintResidual
#include <mundy_constraints/ComputeConstraintViolation.hpp>   // for mundy::constraints::ComputeConstraintViolation
#endif                                                        // HAVE_MUNDYLINKER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYLINKER_MUNDYIO
#include <mundy_io/IOBroker.hpp>  // for mundy::io::IOBroker
#endif                            // HAVE_MUNDYLINKER_MUNDYIO

#ifdef HAVE_MUNDYLINKER_MUNDYLINKERS
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                  // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>                // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>                 // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceMagnitudeReduction.hpp>  // for mundy::linkers::LinkerPotentialForceMagnitudeReduction
#endif                                                               // HAVE_MUNDYLINKER_MUNDYLINKERS

#ifdef HAVE_MUNDYLINKER_MUNDYSHAPES
#include <mundy_shapes/ComputeAABB.hpp>            // for mundy::shapes::ComputeAABB
#include <mundy_shapes/ComputeBoundingRadius.hpp>  // for mundy::shapes::ComputeBoundingRadius
#include <mundy_shapes/ComputeOBB.hpp>             // for mundy::shapes::ComputeOBB
#include <mundy_shapes/DeclareAndInitShapes.hpp>   // for mundy::shapes::DeclareAndInitShapes
#endif                                             // HAVE_MUNDYLINKER_MUNDYSHAPES

namespace mundy {

namespace driver {

template <typename PolymorphicBaseType>
using ConfigurableMetaMethodFactory =
    mundy::meta::StringBasedMetaFactory<PolymorphicBaseType,
                                        mundy::meta::make_registration_string("CONFIGURABLE_META_METHODS")>;

}  // namespace driver

}  // namespace mundy

#ifdef HAVE_MUNDYLINKER_MUNDYCONSTRAINTS
MUNDY_REGISTER_METACLASS("COMPUTE_CONSTRAINT_FORCING", mundy::constraints::ComputeConstraintForcing,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("COMPUTE_CONSTRAINT_PROJECTION", mundy::constraints::ComputeConstraintProjection,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("COMPUTE_CONSTRAINT_RESIDUAL", mundy::constraints::ComputeConstraintResidual,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("COMPUTE_CONSTRAINT_VIOLATION", mundy::constraints::ComputeConstraintViolation,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
#endif  // HAVE_MUNDYLINKER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYLINKER_MUNDYLINKERS
MUNDY_REGISTER_METACLASS("COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL",
                         mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("DESTROY_NEIGHBOR_LINKERS", mundy::linkers::DestroyNeighborLinkers,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("EVALUATE_LINKER_POTENTIALS", mundy::linkers::EvaluateLinkerPotentials,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS(
    "GENERATE_NEIGHBOR_LINKERS", mundy::linkers::GenerateNeighborLinkers,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("LINKER_POTENTIAL_FORCE_MAGNITUDE_REDUCTION",
                         mundy::linkers::LinkerPotentialForceMagnitudeReduction,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
#endif  // HAVE_MUNDYLINKER_MUNDYLINKERS

#ifdef HAVE_MUNDYLINKER_MUNDYSHAPES
MUNDY_REGISTER_METACLASS("COMPUTE_AABB", mundy::shapes::ComputeAABB,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("COMPUTE_BOUNDING_RADIUS", mundy::shapes::ComputeBoundingRadius,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("COMPUTE_OBB", mundy::shapes::ComputeOBB,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface>)
MUNDY_REGISTER_METACLASS("DECLARE_AND_INIT_SHAPES", mundy::shapes::DeclareAndInitShapes,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodExecutionDispatcher>)
#endif  // HAVE_MUNDYLINKER_MUNDYSHAPES

#endif  // MUNDY_DRIVER_CONFIGURATOR_HPP_

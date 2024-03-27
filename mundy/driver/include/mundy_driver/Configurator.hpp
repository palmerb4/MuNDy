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

// Mundy includes
#include <MundyDriver_config.hpp>                                     // for HAVE_MUNDYDRIVER_*
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::StringBasedMetaFactory
#include <mundy_meta/MetaMethodExecutionInterface.hpp>                // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS

#ifdef HAVE_MUNDYDRIVER_MUNDYAGENTS
#include <mundy_agents/Agents.hpp>             // for mundy::agents::Agents
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents
#include <mundy_agents/RegisterAgents.hpp>     // MUNDY_REGISTER_AGENTS
#endif                                         // HAVE_MUNDYDRIVER_MUNDYAGENTS

#ifdef HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS
#include <mundy_constraints/ComputeConstraintForcing.hpp>     // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/ComputeConstraintProjection.hpp>  // for mundy::constraints::ComputeConstraintProjection
#include <mundy_constraints/ComputeConstraintResidual.hpp>    // for mundy::constraints::ComputeConstraintResidual
#include <mundy_constraints/ComputeConstraintViolation.hpp>   // for mundy::constraints::ComputeConstraintViolation
#include <mundy_constraints/Springs.hpp>                      // for mundy::constraints::Springs
#endif                                                        // HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYDRIVER_MUNDYIO
#include <mundy_io/IOBroker.hpp>  // for mundy::io::IOBroker
#endif                            // HAVE_MUNDYDRIVER_MUNDYIO

#ifdef HAVE_MUNDYDRIVER_MUNDYLINKERS
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                  // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>                // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>                 // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceMagnitudeReduction.hpp>  // for mundy::linkers::LinkerPotentialForceMagnitudeReduction
#include <mundy_linkers/Linkers.hpp>                                 // for mundy::linkers::Linkers
#include <mundy_linkers/NeighborLinkers.hpp>                         // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#endif                                                             // HAVE_MUNDYDRIVER_MUNDYLINKERS

#ifdef HAVE_MUNDYDRIVER_MUNDYSHAPES
#include <mundy_shapes/ComputeAABB.hpp>            // for mundy::shapes::ComputeAABB
#include <mundy_shapes/ComputeBoundingRadius.hpp>  // for mundy::shapes::ComputeBoundingRadius
#include <mundy_shapes/ComputeOBB.hpp>             // for mundy::shapes::ComputeOBB
#include <mundy_shapes/DeclareAndInitShapes.hpp>   // for mundy::shapes::DeclareAndInitShapes
#include <mundy_shapes/Spheres.hpp>                // for mundy::shapes::Spheres
#include <mundy_shapes/Spherocylinders.hpp>        // for mundy::shapes::Spherocylinders
#endif                                             // HAVE_MUNDYDRIVER_MUNDYSHAPES

namespace mundy {

namespace driver {

/// \brief A factory for creating meta methods that are configurable via the Configurator.
/// The core requirements for these classes is that they have a void execute function which the Configurator knows how
/// to call.
template <typename PolymorphicBaseType>
using ConfigurableMetaMethodFactory =
    mundy::meta::StringBasedMetaFactory<PolymorphicBaseType,
                                        mundy::meta::make_registration_string("CONFIGURABLE_META_METHODS")>;

}  // namespace driver

}  // namespace mundy

#ifdef HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS
MUNDY_REGISTER_AGENTS(mundy::constraints::Springs)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_FORCING", mundy::constraints::ComputeConstraintForcing,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_PROJECTION", mundy::constraints::ComputeConstraintProjection,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_RESIDUAL", mundy::constraints::ComputeConstraintResidual,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_VIOLATION", mundy::constraints::ComputeConstraintViolation,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
#endif  // HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYDRIVER_MUNDYLINKERS
MUNDY_REGISTER_AGENTS(mundy::linkers::Linkers)
MUNDY_REGISTER_AGENTS(mundy::linkers::NeighborLinkers)
MUNDY_REGISTER_AGENTS(mundy::linkers::neighbor_linkers::SphereSphereLinkers)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL",
    mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "DESTROY_NEIGHBOR_LINKERS", mundy::linkers::DestroyNeighborLinkers,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "EVALUATE_LINKER_POTENTIALS", mundy::linkers::EvaluateLinkerPotentials,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "GENERATE_NEIGHBOR_LINKERS", mundy::linkers::GenerateNeighborLinkers,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "LINKER_POTENTIAL_FORCE_MAGNITUDE_REDUCTION", mundy::linkers::LinkerPotentialForceMagnitudeReduction,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
#endif  // HAVE_MUNDYDRIVER_MUNDYLINKERS

#ifdef HAVE_MUNDYDRIVER_MUNDYSHAPES
MUNDY_REGISTER_AGENTS(mundy::shapes::Spheres)
MUNDY_REGISTER_AGENTS(mundy::shapes::Spherocylinders)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_AABB", mundy::shapes::ComputeAABB,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_BOUNDING_RADIUS", mundy::shapes::ComputeBoundingRadius,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "COMPUTE_OBB", mundy::shapes::ComputeOBB,
    mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("DECLARE_AND_INIT_SHAPES", mundy::shapes::DeclareAndInitShapes,
                         mundy::driver::ConfigurableMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>)
#endif  // HAVE_MUNDYDRIVER_MUNDYSHAPES

#endif  // MUNDY_DRIVER_CONFIGURATOR_HPP_

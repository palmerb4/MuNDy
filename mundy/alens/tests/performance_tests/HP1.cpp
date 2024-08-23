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

#pragma TODO "Add a brief description of the file here."
/* Notes:

Brief note:
We are trying to simulate the HP1 project with Hossein.

Here is the generic form of a single chromatin chain. In this example, we have 3 chromatin 'repeats', which means 2
heterochromatin sections, and 1 euchromatin section in the middle.

E : euchromatin spheres
H : heterochromatin spheres
| : HP1 crosslinker
--- : backbone spring

|   |                           |   |
H---H---E---E---E---E---E---E---H---H

Chromatin backbone:
Backbone segments are modeled as spherocylinder segments, and can have a different spring constants that determine the
separation between adjacent spheres.

Interactions:

1. Backbone segment --- Backbone segment
   - Interactions are via a hertzian contact potential.
2. Backbone segment -- itself
   - Interactions are via a harmonic spring between adjacent spheres, but done along the spherocylinder segment
(backbone).


*/

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>            // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>   // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_alens/actions_crosslinkers.hpp>             // for mundy::alens::crosslinkers...
#include <mundy_alens/periphery/Periphery.hpp>              // for gen_sphere_quadrature
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                   // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_io/IOBroker.hpp>         // for mundy::io::IOBroker
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers
#include <mundy_math/Hilbert.hpp>                           // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>                           // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>       // for mundy::math::distance::ellipsoid_ellipsoid
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>    // for mundy::mesh::MetaData
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>        // for mundy::mesh::utils::destroy_flagged_entities
#include <mundy_mesh/utils/FillFieldWithValue.hpp>            // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

namespace mundy {

namespace alens {

namespace crosslinkers {

class HP1 {
 public:
  enum BINDING_STATE_CHANGE : unsigned { NONE = 0u, LEFT_TO_DOUBLY, RIGHT_TO_DOUBLY, DOUBLY_TO_LEFT, DOUBLY_TO_RIGHT };
  enum BOND_TYPE : unsigned { HARMONIC = 0u, FENE };

  using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
  using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

  HP1() = default;

  void print_rank0(auto thing_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << thing_to_print << std::endl;
    }
  }

  void parse_user_inputs(int argc, char **argv) {
    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, true);
    std::string chromatin_spring_type;
    std::string crosslinker_spring_type;

    // Simulation parameters:
    cmdp.setOption("num_time_steps", &num_time_steps_, "Number of time steps.");
    cmdp.setOption("num_time_steps_equilibrate", &num_time_steps_equilibrate_, "Number of equilibration time steps.");
    cmdp.setOption("timestep_size", &timestep_size_, "Time step size.");
    cmdp.setOption("kt_brownian", &kt_brownian_, "Temperature kT for Brownian Motion.");
    cmdp.setOption("kt_kmc", &kt_kmc_, "Temperature kT for KMC.");
    cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");
    cmdp.setOption("log_frequency", &log_frequency_, "Number of timesteps between logging.");
    cmdp.setOption("initial_loadbalance", "no_initial_loadbalance", &initial_loadbalance_, "Initial loadbalance.");
    cmdp.setOption("do_equilibrate", "no_do_equilibrate", &do_equilibrate_, "Do equilibrate.");
    cmdp.setOption("initialization_type", &initialization_type_, "Initialization_type.");
    cmdp.setOption("viscosity", &viscosity_, "Viscosity.");

    // Periphery or unit cell:
#pragma TODO This will be replaced with the periphery radius and values eventually
    cmdp.setOption("hydrodynamic_periphery_radius", &hydrodynamic_periphery_radius_, "Hydrodynamic periphery radius.");
    cmdp.setOption("collision_periphery_radius", &collision_periphery_radius_, "Collision periphery radius.");
    cmdp.setOption("collision_periphery_radius_start", &collision_periphery_radius_start_,
                   "Collision periphery radius start (equilibration).");
    cmdp.setOption("periphery_spring_constant", &periphery_spring_constant_, "Periphery spring constant.");
    cmdp.setOption("periphery_spectral_order", &periphery_spectral_order_, "Periphery spectral order.");

    // Chromatin chains:
    cmdp.setOption("num_chromosomes", &num_chromosomes_, "Number of chromosomes (chromatin chains).");
    cmdp.setOption("num_chromatin_repeats", &num_chromatin_repeats_, "Number of chromatin repeats per chain.");
    cmdp.setOption("num_euchromatin_per_repeat", &num_euchromatin_per_repeat_,
                   "Number of euchromatin beads per repeat.");
    cmdp.setOption("num_heterochromatin_per_repeat", &num_heterochromatin_per_repeat_,
                   "Number of heterochromatin beads per repeat.");
    cmdp.setOption("backbone_excluded_volume_radius", &backbone_excluded_volume_radius_,
                   "Backbone excluded volume radius (segments).");
    cmdp.setOption("sphere_hydrodynamic_radius", &sphere_hydrodynamic_radius_, "Hydrodynamic sphere radius.");
    cmdp.setOption("initial_sphere_separation", &initial_sphere_separation_, "Initial backbone sphere separation.");
    cmdp.setOption("backbone_youngs_modulus", &backbone_youngs_modulus_, "Backbone Youngs modulus.");
    cmdp.setOption("backbone_poissons_ratio", &backbone_poissons_ratio_, "Backbone poissons ratio.");
    // cmdp.setOption("sphere_drag_coeff", &sphere_drag_coeff_, "Backbone sphere drag coefficient.");

    //  Chromatin spring:
    cmdp.setOption("chromatin_spring_type", &chromatin_spring_type, "Chromatin spring type.");
    cmdp.setOption("chromatin_spring_constant", &chromatin_spring_constant_, "Chromatin spring constant.");
    cmdp.setOption("chromatin_spring_rest_length", &chromatin_spring_rest_length_, "Chromatin rest length.");

    //  Crosslinker (spring and other):
    cmdp.setOption("crosslinker_spring_type", &crosslinker_spring_type, "Crosslinker spring type.");
    cmdp.setOption("crosslinker_spring_constant", &crosslinker_spring_constant_, "Crosslinker spring constant.");
    cmdp.setOption("crosslinker_rest_length", &crosslinker_rest_length_, "Crosslinker rest length.");
    cmdp.setOption("crosslinker_left_binding_rate", &crosslinker_left_binding_rate_, "Crosslinker left binding rate.");
    cmdp.setOption("crosslinker_right_binding_rate", &crosslinker_right_binding_rate_,
                   "Crosslinker right binding rate.");
    cmdp.setOption("crosslinker_left_unbinding_rate", &crosslinker_left_unbinding_rate_,
                   "Crosslinker left unbinding rate.");
    cmdp.setOption("crosslinker_right_unbinding_rate", &crosslinker_right_unbinding_rate_,
                   "Crosslinker right unbinding rate.");

    // Neighbor list
    cmdp.setOption("skin_distance", &skin_distance_, "Neighbor list skin distance.");
    cmdp.setOption("force_neighborlist_update", "no_force_update_neighbor_list", &force_neighborlist_update_,
                   "Force update of the neighbor list.");
    cmdp.setOption("force_neighborlist_update_nsteps", &force_neighborlist_update_nsteps_,
                   "Number of timesteps between updating the neighbor list (if forced).");
    cmdp.setOption("print_neighborlist_statistics", "no_print_neighborlist_statistics", &print_neighborlist_statistics_,
                   "Print neighbor list statistics.");

    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_ASSERT(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");

    MUNDY_THROW_ASSERT(num_chromosomes_ > 0, std::invalid_argument, "num_chromosomes_ must be greater than 0.");
    MUNDY_THROW_ASSERT(num_chromatin_repeats_ > 0, std::invalid_argument,
                       "num_chromatin_repeats_ must be greater than 0.");
    MUNDY_THROW_ASSERT(num_euchromatin_per_repeat_ > 0, std::invalid_argument,
                       "num_euchromatin_per_repeat_ must be greater than 0.");
    MUNDY_THROW_ASSERT(num_heterochromatin_per_repeat_ > 0, std::invalid_argument,
                       "num_heterochromatin_per_repeat_ must be greater than 0.");
    MUNDY_THROW_ASSERT(backbone_excluded_volume_radius_ > 0, std::invalid_argument,
                       "backbone_excluded_volume_radius_ must be greater than 0.");

    MUNDY_THROW_ASSERT(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_ASSERT(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_ASSERT(io_frequency_ > 0, std::invalid_argument, "io_frequency_ must be greater than 0.");

    // If we are equilibrating, set the unit cell to the periphery equilibrate radius, if not, then use the normal
    // radius
    if (do_equilibrate_) {
      unit_cell_length_ = 2.0 * collision_periphery_radius_start_;
      collision_periphery_radius_current_ = collision_periphery_radius_start_;
    } else {
      unit_cell_length_ = 2.0 * collision_periphery_radius_;
      collision_periphery_radius_current_ = collision_periphery_radius_;
    }

    // Set the hydrodynamic drag on the spheres to 6 * pi * eta * r
    sphere_drag_coeff_ = 6.0 * M_PI * viscosity_ * sphere_hydrodynamic_radius_;
    // Modify any variables into their final form
    skin_distance2_over4_ = skin_distance_ * skin_distance_ / 4.0;
    // Compute the cutoff radius for the crosslinker
    crosslinker_rcut_ = crosslinker_rest_length_ + 5.0 * std::sqrt(1.0 / kt_kmc_ / crosslinker_spring_constant_);
    // Set the type of springs we are using (hookean or fene)
    if (crosslinker_spring_type == "harmonic") {
      crosslinker_spring_type_ = BOND_TYPE::HARMONIC;
    } else if (crosslinker_spring_type == "fene") {
      crosslinker_spring_type_ = BOND_TYPE::FENE;
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "FENE bonds not currently implemented for crosslinkers.");
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid crosslinker spring type.");
    }
    if (chromatin_spring_type == "harmonic") {
      chromatin_spring_type_ = BOND_TYPE::HARMONIC;
    } else if (chromatin_spring_type == "fene") {
      chromatin_spring_type_ = BOND_TYPE::FENE;
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "FENE bonds not currently implemented for chromatin chains.");
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid backbone spring type.");
    }
  }

  void dump_user_inputs() {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "" << std::endl;

      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps:       " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size:        " << timestep_size_ << std::endl;
      std::cout << "  io_frequency:         " << io_frequency_ << std::endl;
      std::cout << "  log_frequency:        " << log_frequency_ << std::endl;
      std::cout << "  kT (Brownian):        " << kt_brownian_ << std::endl;
      std::cout << "  kT (KMC):             " << kt_kmc_ << std::endl;
      std::cout << "  initialization_type:  " << initialization_type_ << std::endl;
      std::cout << "  do_equilibrate:       " << do_equilibrate_ << std::endl;
      std::cout << "  viscosity:            " << viscosity_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "UNIT CELL:" << std::endl;
      std::cout << "  box_length: " << unit_cell_length_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "NEIGHBOR LIST:" << std::endl;
      std::cout << "  force_update_neighborlist: " << force_neighborlist_update_ << std::endl;
      std::cout << "  force_update_neighborlist_nsteps: " << force_neighborlist_update_nsteps_ << std::endl;
      std::cout << "  skin_distance: " << 2.0 * std::sqrt(skin_distance2_over4_) << std::endl;
      std::cout << "    (skin_distance2_over4): " << skin_distance2_over4_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "CHROMATIN CHAINS:" << std::endl;
      std::cout << "  num_chromosomes: " << num_chromosomes_ << std::endl;
      std::cout << "  num_chromatin_repeats: " << num_chromatin_repeats_ << std::endl;
      std::cout << "  num_euchromatin_per_repeat: " << num_euchromatin_per_repeat_ << std::endl;
      std::cout << "  num_heterochromatin_per_repeat: " << num_heterochromatin_per_repeat_ << std::endl;
      std::cout << "  backbone_excluded_volume_radius: " << backbone_excluded_volume_radius_ << std::endl;
      std::cout << "  sphere_hydrodynamic_radius: " << sphere_hydrodynamic_radius_ << std::endl;
      std::cout << "  initial_sphere_separation: " << initial_sphere_separation_ << std::endl;
      std::cout << "  youngs_modulus: " << backbone_youngs_modulus_ << std::endl;
      std::cout << "  poissons_ratio: " << backbone_poissons_ratio_ << std::endl;
      std::cout << "  sphere_drag_coeff (calc): " << sphere_drag_coeff_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "CHROMATIN SPRINGS:" << std::endl;
      std::string chromatin_spring_type = chromatin_spring_type_ == BOND_TYPE::HARMONIC ? "harmonic" : "fene";
      std::cout << "  chromatin_spring_type: " << chromatin_spring_type << std::endl;
      std::cout << "  chromatin_spring_constant: " << chromatin_spring_constant_ << std::endl;
      std::cout << "  chromatin_spring_rest_length: " << chromatin_spring_rest_length_ << std::endl;
      std::cout << "" << std::endl;

      std::string crosslinker_spring_type = crosslinker_spring_type_ == BOND_TYPE::HARMONIC ? "harmonic" : "fene";
      std::cout << "CROSSLINKERS:" << std::endl;
      std::cout << "  crosslinker_spring_type: " << crosslinker_spring_type << std::endl;
      std::cout << "  crosslinker_spring_constant: " << crosslinker_spring_constant_ << std::endl;
      std::cout << "  crosslinker_rest_length: " << crosslinker_rest_length_ << std::endl;
      std::cout << "  crosslinker_rcut: " << crosslinker_rcut_ << std::endl;
      std::cout << "  crosslinker_left_binding_rate: " << crosslinker_left_binding_rate_ << std::endl;
      std::cout << "  crosslinker_right_binding_rate: " << crosslinker_right_binding_rate_ << std::endl;
      std::cout << "  crosslinker_left_unbinding_rate: " << crosslinker_left_unbinding_rate_ << std::endl;
      std::cout << "  crosslinker_right_unbinding_rate: " << crosslinker_right_unbinding_rate_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "PERIPHERY:" << std::endl;
      std::cout << "  hydrodynamic_periphery_radius: " << hydrodynamic_periphery_radius_ << std::endl;
      std::cout << "  collision_periphery_radius: " << collision_periphery_radius_ << std::endl;
      std::cout << "  collision_periphery_radius_start: " << collision_periphery_radius_start_ << std::endl;
      std::cout << "  periphery_spring_constant: " << periphery_spring_constant_ << std::endl;
      std::cout << "  periphery_spectral_order: " << periphery_spectral_order_ << std::endl;

      std::cout << "##################################################" << std::endl;
    }
  }

  template <typename FieldDataType, size_t field_size>
  void print_field(const stk::mesh::Field<FieldDataType> &field) {
    stk::mesh::BulkData &bulk_data = field.get_mesh();
    stk::mesh::Selector selector = stk::mesh::Selector(field);

    stk::mesh::EntityVector entities;
    stk::mesh::get_selected_entities(selector, bulk_data_ptr_->buckets(field.entity_rank()), entities);

    for (const stk::mesh::Entity &entity : entities) {
      const FieldDataType *field_data = stk::mesh::field_data(field, entity);
      std::cout << "Entity " << bulk_data.identifier(entity) << " field data: ";
      for (size_t i = 0; i < field_size; ++i) {
        std::cout << field_data[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  void assert_invariant(const std::string &message = std::string()) {
#ifdef DEBUG
    Kokkos::Profiling::pushRegion("HP1::assert_invariant");
#pragma TODO CJE Remove the mesh dump so that we can see the metadata
    std::cout << "############################################" << std::endl;
    std::cout << "Mesh at message " << message << std::endl;
    stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
    std::cout << "############################################" << std::endl;
    Kokkos::Profiling::popRegion();
#endif
  }

  void build_our_mesh_and_method_instances() {
    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements to the sphere part for this example. These are requirements that exceed those of the
    // enabled methods and allow us to extend the functionality offered natively by Mundy.

    // Spheres need to be modified to contain the subparts for E, H, and BindSite.
    auto custom_sphere_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_part_reqs->add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<unsigned>("NODE_RNG_COUNTER", node_rank_, 1, 1)
        .add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", node_rank_, 3, 1)
        .add_subpart_reqs("E", stk::topology::PARTICLE)
        .add_subpart_reqs("H", stk::topology::PARTICLE)
        .add_subpart_reqs("BS", stk::topology::PARTICLE);
    mundy::shapes::Spheres::add_and_sync_part_reqs(custom_sphere_part_reqs);
    mesh_reqs_ptr_->sync(mundy::shapes::Spheres::get_mesh_requirements());

    // HP1 needs to be added to the mesh. This includes the subparts for the states of HP1. It will be added to the
    // SpherocylinderSegment part the same was as StickySettings.
    auto custom_hp1_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_part_reqs->set_part_name("HP1S")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ELEMENT_REALIZED_UNBINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_REALIZED_BINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<unsigned>("ELEMENT_RNG_COUNTER", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("ELEMENT_PERFORM_STATE_CHANGE", element_rank_, 1, 1)
        .add_subpart_reqs("LEFT_HP1", stk::topology::BEAM_2)
        .add_subpart_reqs("DOUBLY_HP1_H", stk::topology::BEAM_2)
        .add_subpart_reqs("DOUBLY_HP1_BS", stk::topology::BEAM_2);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_hp1_part_reqs);

    // Create the backbone segments.
    auto custom_backbone_segments_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_backbone_segments_part_reqs->set_part_name("BACKBONE_SEGMENTS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_subpart_reqs("EESPRINGS", stk::topology::BEAM_2)
        .add_subpart_reqs("EHSPRINGS", stk::topology::BEAM_2)
        .add_subpart_reqs("HHSPRINGS", stk::topology::BEAM_2);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_backbone_segments_part_reqs);

    // Create the generalized interaction entities that connect HP1 and (H)eterochromatin
    //   This entity "knows" how to compute the binding probability between a crosslinker and a H and how to
    //   perform binding between a crosslinker and a H. It is a constraint rank entitiy because it must connect
    //   element rank entities.
    auto custom_hp1_h_genx_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_h_genx_part_reqs->set_part_name("HP1_H_NEIGHBOR_GENXS")
        .set_part_rank(constraint_rank_)
        .add_field_reqs<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_and_sync_subpart_reqs(custom_hp1_h_genx_part_reqs);
    mesh_reqs_ptr_->sync(mundy::linkers::NeighborLinkers::get_mesh_requirements());
    // Create the same type of entity to the binding sites on the periphery
    auto custom_hp1_bs_genx_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_bs_genx_part_reqs->set_part_name("HP1_BS_NEIGHBOR_GENXS")
        .set_part_rank(constraint_rank_)
        .add_field_reqs<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_and_sync_subpart_reqs(custom_hp1_bs_genx_part_reqs);
    mesh_reqs_ptr_->sync(mundy::linkers::NeighborLinkers::get_mesh_requirements());

    // Add the custom neighbor list implementation to the axis-aligned bounding box. In theory, we only care if the
    // corners of the AABB move more than skin distance/2. Set the AABB field to have an additional state that it keeps
    // track of at the end so that we can do dr calculations trivially. This create the two element aabb fields, but
    // also adds a second entire set of fields to the mesh.
    //
    // We also need an accumulator for summing up the total distance traveled by the AABB corners.
#pragma TODO right now this puts the ELEMENT_AABB on the entire mesh(elements)
    mesh_reqs_ptr_->add_field_reqs<double>("ELEMENT_AABB", element_rank_, 6, 2);
    auto custom_sphere_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_aabb_accumulator_reqs->set_part_name("SPHERES")
        .set_part_topology(stk::topology::PARTICLE)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    auto custom_backbone_segment_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_backbone_segment_aabb_accumulator_reqs->set_part_name("BACKBONE_SEGMENTS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    auto custom_hp1_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_aabb_accumulator_reqs->set_part_name("HP1S")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_sphere_aabb_accumulator_reqs);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_backbone_segment_aabb_accumulator_reqs);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_hp1_aabb_accumulator_reqs);

    // Setup our fixed parameters for any of methods that we intend to use
    // When we eventually switch to the configurator, these individual fixed params will become sublists within a single
    // master parameter list. Note, sublist will return a reference to the sublist with the given name.
    //
    // Compute constraint (bonded) forces for the the BACKBONE_SEGMENTS and HP1S parts
    compute_constraint_forcing_fixed_params_ =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"));
    compute_constraint_forcing_fixed_params_.sublist("HOOKEAN_SPRINGS")
        .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS", "HP1S"));

    // Compute the minimum distance for the SCS-SCS, HP1-H, HP1-BS interactions (SCS-SCS, S-SCS, S-SCS)
    // Try to be as explicit as possible with the parts that are associated with each of the interactions.
    compute_ssd_and_cn_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER",
                                                               "SPHERE_SPHEROCYLINDER_SEGMENT_LINKER"));
    compute_ssd_and_cn_fixed_params_.sublist("SPHERE_SPHEROCYLINDER_SEGMENT_LINKER")
        .set("valid_entity_part_names", mundy::core::make_string_array("HP1_H_NEIGHBOR_GENXS", "HP1_BS_NEIGHBOR_GENXS"))
        .set("valid_sphere_part_names", mundy::core::make_string_array("H", "BS"))
        .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("HP1S"));
    compute_ssd_and_cn_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER")
        .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_BACKBONE_NEIGHBOR_GENXS"))
        .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));

    // Set up the AABB for the system
    compute_aabb_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE", "SPHEROCYLINDER_SEGMENT"));
    compute_aabb_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("HP1S", "BACKBONE_SEGMENTS"));

    // Generate the GENX neighbor linkers between spherocylinder segments
    generate_scs_scs_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("BACKBONE_BACKBONE_NEIGHBOR_GENXS"));
    generate_scs_scs_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));

    // Generate the GENX neighbor linkers between HP1 and H
    generate_hp1_h_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("HP1_H_NEIGHBOR_GENXS"));
    generate_hp1_h_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("HP1S")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("H"));

    // Generate the GENX neighbor linkers between HP1 and BS
    generate_hp1_bs_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("HP1_BS_NEIGHBOR_GENXS"));
    generate_hp1_bs_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("HP1S")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("BS"));

    // Evaluate the scs-scs hertzian contacts
    evaluate_linker_potentials_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names",
        mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
    evaluate_linker_potentials_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT")
        .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));

    // Reduce the forces on the spherocylinder segments
    linker_potential_force_reduction_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"))
            .set("name_of_linker_part_to_reduce_over", "BACKBONE_BACKBONE_NEIGHBOR_GENXS");

    // Destroy the distant neighbors over time
    destroy_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    destroy_neighbor_linkers_fixed_params_.sublist("DESTROY_DISTANT_NEIGHBORS")
        .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
        .set("valid_connected_source_and_target_part_names",
             mundy::core::make_string_array(std::string("SPHEROCYLINDER_SEGMENTS"), std::string("HP1S")));

    // Destroy bound linkers to prevent pathological behavior along a chain
    destroy_bound_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_BOUND_NEIGHBORS");

    // Synchronize (merge and rectify differences) the requirements for each method based on the fixed parameters.
    // For now, we will directly use the types that each method corresponds to. The configurator will
    // fetch the static members of these methods using the configurable method factory.
    mesh_reqs_ptr_->sync(
        mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_scs_scs_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_hp1_h_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_hp1_bs_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::LinkerPotentialForceReduction::get_mesh_requirements(
        linker_potential_force_reduction_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_bound_neighbor_linkers_fixed_params_));

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDS");
    meta_data_ptr_->commit();
  }

  template <typename FieldType>
  stk::mesh::Field<FieldType> *fetch_field(const std::string &field_name, stk::topology::rank_t rank) {
    auto field_ptr = meta_data_ptr_->get_field<FieldType>(rank, field_name);
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "Field " << field_name << " not found in the mesh meta data.");
    return field_ptr;
  }

  stk::mesh::Part *fetch_part(const std::string &part_name) {
    auto part_ptr = meta_data_ptr_->get_part(part_name);
    MUNDY_THROW_ASSERT(part_ptr != nullptr, std::invalid_argument,
                       "Part " << part_name << " not found in the mesh meta data.");
    return part_ptr;
  }

  void fetch_fields_and_parts() {
    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDS", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_rng_field_ptr_ = fetch_field<unsigned>("NODE_RNG_COUNTER", node_rank_);

    element_rng_field_ptr_ = fetch_field<unsigned>("ELEMENT_RNG_COUNTER", element_rank_);
    element_hookean_spring_constant_field_ptr_ = fetch_field<double>("ELEMENT_HOOKEAN_SPRING_CONSTANT", element_rank_);
    element_hookean_spring_rest_length_field_ptr_ =
        fetch_field<double>("ELEMENT_HOOKEAN_SPRING_REST_LENGTH", element_rank_);
    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    element_aabb_field_ptr_ = fetch_field<double>("ELEMENT_AABB", element_rank_);
    element_corner_displacement_field_ptr_ = fetch_field<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_);
    element_binding_rates_field_ptr_ = fetch_field<double>("ELEMENT_REALIZED_BINDING_RATES", element_rank_);
    element_unbinding_rates_field_ptr_ = fetch_field<double>("ELEMENT_REALIZED_UNBINDING_RATES", element_rank_);
    element_perform_state_change_field_ptr_ = fetch_field<unsigned>("ELEMENT_PERFORM_STATE_CHANGE", element_rank_);

    constraint_potential_force_field_ptr_ = fetch_field<double>("LINKER_POTENTIAL_FORCE", constraint_rank_);
    constraint_state_change_rate_field_ptr_ =
        fetch_field<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_);
    constraint_perform_state_change_field_ptr_ =
        fetch_field<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_);
    constraint_linked_entities_field_ptr_ =
        fetch_field<mundy::linkers::LinkedEntitiesFieldType::value_type>("LINKED_NEIGHBOR_ENTITIES", constraint_rank_);
    constraint_linked_entity_owners_field_ptr_ = fetch_field<int>("LINKED_NEIGHBOR_ENTITY_OWNERS", constraint_rank_);

    // Fetch the parts
    spheres_part_ptr_ = fetch_part("SPHERES");
    e_part_ptr_ = fetch_part("E");
    h_part_ptr_ = fetch_part("H");
    bs_part_ptr_ = fetch_part("BS");

    hp1_part_ptr_ = fetch_part("HP1S");
    left_hp1_part_ptr_ = fetch_part("LEFT_HP1");
    doubly_hp1_h_part_ptr_ = fetch_part("DOUBLY_HP1_H");
    doubly_hp1_bs_part_ptr_ = fetch_part("DOUBLY_HP1_BS");

    backbone_segments_part_ptr_ = fetch_part("BACKBONE_SEGMENTS");
    ee_springs_part_ptr_ = fetch_part("EESPRINGS");
    eh_springs_part_ptr_ = fetch_part("EHSPRINGS");
    hh_springs_part_ptr_ = fetch_part("HHSPRINGS");

    backbone_backbone_neighbor_genx_part_ptr_ = fetch_part("BACKBONE_BACKBONE_NEIGHBOR_GENXS");
    hp1_h_neighbor_genx_part_ptr_ = fetch_part("HP1_H_NEIGHBOR_GENXS");
    hp1_bs_neighbor_genx_part_ptr_ = fetch_part("HP1_BS_NEIGHBOR_GENXS");
  }

  void instantiate_metamethods() {
    // Create the non-custom MetaMethods
    // MetaMethodExecutionInterface

    // MetaMethodSubsetExecutionInterface
    compute_constraint_forcing_ptr_ = mundy::constraints::ComputeConstraintForcing::create_new_instance(
        bulk_data_ptr_.get(), compute_constraint_forcing_fixed_params_);
    compute_ssd_and_cn_ptr_ = mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
        bulk_data_ptr_.get(), compute_ssd_and_cn_fixed_params_);
    compute_aabb_ptr_ =
        mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr_.get(), compute_aabb_fixed_params_);
    evaluate_linker_potentials_ptr_ = mundy::linkers::EvaluateLinkerPotentials::create_new_instance(
        bulk_data_ptr_.get(), evaluate_linker_potentials_fixed_params_);
    linker_potential_force_reduction_ptr_ = mundy::linkers::LinkerPotentialForceReduction::create_new_instance(
        bulk_data_ptr_.get(), linker_potential_force_reduction_fixed_params_);
    destroy_neighbor_linkers_ptr_ = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), destroy_neighbor_linkers_fixed_params_);
    destroy_bound_neighbor_linkers_ptr_ = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), destroy_bound_neighbor_linkers_fixed_params_);

    // MetaMethodPairwiseSubsetExecutionInterface
    generate_scs_scs_genx_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_scs_scs_neighbor_linkers_fixed_params_);
    generate_hp1_h_genx_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_hp1_h_neighbor_linkers_fixed_params_);
    generate_hp1_bs_genx_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_hp1_bs_neighbor_linkers_fixed_params_);
  }

  void set_mutable_parameters() {
    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", skin_distance_);
    compute_aabb_ptr_->set_mutable_params(compute_aabb_mutable_params);
  }

  void setup_io_mundy() {
    // Create a mundy io broker via it's fixed parameters
    // Dump everything for now
    auto fixed_params_iobroker =
        Teuchos::ParameterList()
            .set("enabled_io_parts",
                 mundy::core::make_string_array("E", "H", "BS", "EESPRINGS", "EHSPRINGS", "HHSPRINGS", "LEFT_HP1",
                                                "DOUBLY_HP1_H", "DOUBLY_HP1_BS"))
            .set("enabled_io_fields_node_rank",
                 mundy::core::make_string_array("NODE_VELOCITY", "NODE_FORCE", "NODE_RNG_COUNTER"))
            .set("enabled_io_fields_element_rank",
                 mundy::core::make_string_array("ELEMENT_RADIUS", "ELEMENT_RNG_COUNTER",
                                                "ELEMENT_REALIZED_BINDING_RATES", "ELEMENT_REALIZED_UNBINDING_RATES",
                                                "ELEMENT_PERFORM_STATE_CHANGE"))
            .set("coordinate_field_name", "NODE_COORDS")
            .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
            .set("exodus_database_output_filename", "HP.exo")
            .set("parallel_io_mode", "hdf5")
            .set("database_purpose", "results");
    // Create the IO broker
    io_broker_ptr_ = mundy::io::IOBroker::create_new_instance(bulk_data_ptr_.get(), fixed_params_iobroker);
  }

  void ghost_linked_entities() {
    bulk_data_ptr_->modification_begin();
    const stk::mesh::Selector linker_parts_selector = stk::mesh::selectUnion(stk::mesh::ConstPartVector{
        backbone_backbone_neighbor_genx_part_ptr_, hp1_h_neighbor_genx_part_ptr_, hp1_bs_neighbor_genx_part_ptr_});
    mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr_, *constraint_linked_entities_field_ptr_,
                                                 *constraint_linked_entity_owners_field_ptr_, linker_parts_selector);
    bulk_data_ptr_->modification_end();
  }

  void loadbalance() {
    stk::balance::balanceStkMesh(balance_settings_, *bulk_data_ptr_);
    ghost_linked_entities();
  }

  void rotate_field_states() {
    bulk_data_ptr_->update_field_data_states();
  }

  // Create the chromatin backbone and HP1 crosslinkers
  void create_chromatin_backbone_and_hp1() {
    // Calculate some constants, like the total number of spheres or segments per chromosome
    const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                               num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
    const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
    const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
    const size_t num_spheres_per_chromosome = num_nodes_per_chromosome;
    const size_t num_segments_per_chromosome = num_nodes_per_chromosome - 1;

    std::cout << "Per chromosome:\n";
    std::cout << "num_heterochromatin_spheres: " << num_heterochromatin_spheres << std::endl;
    std::cout << "num_euchromatin_spheres:     " << num_euchromatin_spheres << std::endl;
    std::cout << "num_nodes_per_chromosome:    " << num_nodes_per_chromosome << std::endl;
    std::cout << "num_spheres_per_chromosome:  " << num_spheres_per_chromosome << std::endl;
    std::cout << "num_segments_per_chromosome: " << num_segments_per_chromosome << std::endl;

    bulk_data_ptr_->modification_begin();

    // Rank 0: Declare N chromatin chains randomly in space
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        std::cout << "Creating chromosome " << j << std::endl;

        // Figure out the starting indices of the nodes and elements
        size_t start_node_id = num_nodes_per_chromosome * j + 1u;
        size_t start_element_id =
            (num_spheres_per_chromosome + num_segments_per_chromosome + num_heterochromatin_spheres) * j + 1u;

        // Helper functions for getting the IDs of various objects
        auto get_node_id = [start_node_id](const size_t &seq_node_index) { return start_node_id + seq_node_index; };

        auto get_sphere_id = [start_element_id](const size_t &seq_sphere_index) {
          return start_element_id + seq_sphere_index;
        };

        auto get_segment_id = [start_element_id, num_spheres_per_chromosome](const size_t &seq_segment_index) {
          return start_element_id + num_spheres_per_chromosome + seq_segment_index;
        };

        auto get_crosslinker_id = [start_element_id, num_spheres_per_chromosome,
                                   num_segments_per_chromosome](const size_t &seq_crosslinker_index) {
          return start_element_id + num_spheres_per_chromosome + num_segments_per_chromosome + seq_crosslinker_index;
        };

        // Try to use modulo math to determine region
        const size_t num_heterochromatin_per_repeat = num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_per_repeat = num_euchromatin_per_repeat_;
        auto get_region_by_id = [num_heterochromatin_per_repeat,
                                 num_euchromatin_per_repeat](const size_t &seq_sphere_id) {
          auto local_idx = seq_sphere_id % (num_heterochromatin_per_repeat + num_euchromatin_per_repeat);
          return local_idx < num_heterochromatin_per_repeat ? std::string("H") : std::string("E");
        };

        // Show what a single chromatin chain would be in terms of membership
        std::cout << "Regional map:" << std::endl;
        for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
          std::cout << get_region_by_id(i);
        }
        std::cout << std::endl;

        // Temporary/scratch variables
        stk::mesh::PartVector empty;

        // Logically, it makes the most sense to march down the segments in a single chromosome, adjusting their part
        // membership as we go. Do this across the elements of the chromatin backbone.
        // Initialize the backbone such that we have different sphere types.
        //  E : euchromatin spheres
        //  H : heterochromatin spheres
        // ---: backbone springs (EE, EH, or HH depending on attached spheres)
        //
        //  H---H---E---E---E---E---E---E---H---H
        //
        for (size_t segment_local_idx = 0; segment_local_idx < num_segments_per_chromosome; segment_local_idx++) {
          // Keep track of the vertex IDs for part memebership (local index into array)
          const size_t vertex_left_idx = segment_local_idx;
          const size_t vertex_right_idx = segment_local_idx + 1;
          // Process the nodes for this segment
          stk::mesh::EntityId left_node_id = get_node_id(segment_local_idx);
          stk::mesh::EntityId right_node_id = get_node_id(segment_local_idx + 1);

          stk::mesh::Entity left_node = bulk_data_ptr_->get_entity(node_rank_, left_node_id);
          stk::mesh::Entity right_node = bulk_data_ptr_->get_entity(node_rank_, right_node_id);
          if (!bulk_data_ptr_->is_valid(left_node)) {
            left_node = bulk_data_ptr_->declare_node(left_node_id, empty);
          }
          if (!bulk_data_ptr_->is_valid(right_node)) {
            right_node = bulk_data_ptr_->declare_node(right_node_id, empty);
          }

          // Each node is attached to a sphere that is (H)eterochromatin, (E)uchromatin, or (BS)BindingSite
          stk::mesh::EntityId left_sphere_id = get_sphere_id(segment_local_idx);
          stk::mesh::EntityId right_sphere_id = get_sphere_id(segment_local_idx + 1);
          stk::mesh::Entity left_sphere = bulk_data_ptr_->get_entity(element_rank_, left_sphere_id);
          stk::mesh::Entity right_sphere = bulk_data_ptr_->get_entity(element_rank_, right_sphere_id);
          if (!bulk_data_ptr_->is_valid(left_sphere)) {
            // Figure out the part we belong to
            stk::mesh::PartVector pvector;
            if (get_region_by_id(vertex_left_idx) == "H") {
              pvector.push_back(h_part_ptr_);
            } else if (get_region_by_id(vertex_left_idx) == "E") {
              pvector.push_back(e_part_ptr_);
            }
            // Declare the sphere and connect to it's node
            left_sphere = bulk_data_ptr_->declare_element(left_sphere_id, pvector);
            bulk_data_ptr_->declare_relation(left_sphere, left_node, 0);
          }
          if (!bulk_data_ptr_->is_valid(right_sphere)) {
            // Figure out the part we belong to
            stk::mesh::PartVector pvector;
            if (get_region_by_id(vertex_right_idx) == "H") {
              pvector.push_back(h_part_ptr_);
            } else if (get_region_by_id(vertex_right_idx) == "E") {
              pvector.push_back(e_part_ptr_);
            }
            // Declare the sphere and connect to it's node
            right_sphere = bulk_data_ptr_->declare_element(right_sphere_id, pvector);
            bulk_data_ptr_->declare_relation(right_sphere, right_node, 0);
          }

          // Figure out how to do the spherocylinder segments along the edges now
          stk::mesh::Entity segment = bulk_data_ptr_->get_entity(element_rank_, get_segment_id(segment_local_idx));
          if (!bulk_data_ptr_->is_valid(segment)) {
            stk::mesh::PartVector pvector;
            pvector.push_back(backbone_segments_part_ptr_);
            if (get_region_by_id(vertex_left_idx) == "E" && get_region_by_id(vertex_right_idx) == "E") {
              pvector.push_back(ee_springs_part_ptr_);
            } else if (get_region_by_id(vertex_left_idx) == "E" && get_region_by_id(vertex_right_idx) == "H") {
              pvector.push_back(eh_springs_part_ptr_);
            } else if (get_region_by_id(vertex_left_idx) == "H" && get_region_by_id(vertex_right_idx) == "E") {
              pvector.push_back(eh_springs_part_ptr_);
            } else if (get_region_by_id(vertex_left_idx) == "H" && get_region_by_id(vertex_right_idx) == "H") {
              pvector.push_back(hh_springs_part_ptr_);
            }
            segment = bulk_data_ptr_->declare_element(get_segment_id(segment_local_idx), pvector);
            bulk_data_ptr_->declare_relation(segment, left_node, 0);
            bulk_data_ptr_->declare_relation(segment, right_node, 1);
          }
        }

        // Declare the crosslinkers along the backbone
        // Every sphere gets a left bound crosslinker
        //  E : euchromatin spheres
        //  H : heterochromatin spheres
        //  | : crosslinkers
        // ---: backbone springs
        //
        //  |   |                           |   |
        //  H---H---E---E---E---E---E---E---H---H

        // March down the chain of spheres, adding crosslinkers as we go. We just want to add to the heterochromatin
        // spheres, and so keep track of a running hp1_sphere_index.
        size_t hp1_sphere_index = 0;
        for (size_t sphere_local_idx = 0; sphere_local_idx < num_spheres_per_chromosome; sphere_local_idx++) {
          stk::mesh::Entity sphere_node = bulk_data_ptr_->get_entity(node_rank_, get_node_id(sphere_local_idx));
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(sphere_node), std::invalid_argument,
                             "Node " << sphere_local_idx << " is not valid.");
          // Check if we are a heterochromatin sphere
          if (get_region_by_id(sphere_local_idx) == "H") {
            // Bind left and right nodes to the same node to start simulation (everybody is left bound)
            // Create the HP1 crosslinker
            auto left_bound_hp1_part_vector = stk::mesh::PartVector{left_hp1_part_ptr_};
            stk::mesh::EntityId hp1_crosslinker_id = get_crosslinker_id(hp1_sphere_index);
            stk::mesh::Entity hp1_crosslinker =
                bulk_data_ptr_->declare_element(hp1_crosslinker_id, left_bound_hp1_part_vector);
            stk::mesh::Permutation invalid_perm = stk::mesh::Permutation::INVALID_PERMUTATION;
            bulk_data_ptr_->declare_relation(hp1_crosslinker, sphere_node, 0, invalid_perm);
            bulk_data_ptr_->declare_relation(hp1_crosslinker, sphere_node, 1, invalid_perm);
            MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(hp1_crosslinker).topology() != stk::topology::INVALID_TOPOLOGY,
                               std::logic_error,
                               "The crosslinker with id " << hp1_crosslinker_id << " has an invalid topology.");

            hp1_sphere_index++;
          }
        }
      }
    }
    bulk_data_ptr_->modification_end();
  }

  // Initialize a part for excluded volume interactions (Hertzian)
  void initialize_excluded_volume_part_from_selector(const stk::mesh::Selector &local_selector,
                                                     stk::mesh::Field<double> *youngs_modulus_field_ptr,
                                                     stk::mesh::Field<double> *poissons_ratio_field_ptr,
                                                     const double &youngs_modulus, const double &poissons_ratio) {
    // Alias the fields for the foreach lambda
    const stk::mesh::Field<double> &youngs_modulus_field = *youngs_modulus_field_ptr;
    const stk::mesh::Field<double> &poissons_ratio_field = *poissons_ratio_field_ptr;
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, local_selector,
        [&youngs_modulus_field, &poissons_ratio_field, &youngs_modulus, &poissons_ratio](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &local_wca) {
          // Assign the hertzian contact parameters to the backbone segments
          stk::mesh::field_data(youngs_modulus_field, local_wca)[0] = youngs_modulus;
          stk::mesh::field_data(poissons_ratio_field, local_wca)[0] = poissons_ratio;
        });  // for_each_entity_run
  }

  // Initialize a spring part to a common spring constant and rest length
  void initialize_spring_part_from_selector(const stk::mesh::Selector &local_selector,
                                            stk::mesh::Field<double> *spring_constant_field_ptr,
                                            stk::mesh::Field<double> *spring_rest_length_field_ptr,
                                            const double &spring_constant, const double &spring_rest_length) {
    // Initialize the spring constants on the backbone for every EE spring
    const stk::mesh::Field<double> &spring_constant_field = *spring_constant_field_ptr;
    const stk::mesh::Field<double> &spring_rest_length_field = *spring_rest_length_field_ptr;

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, local_selector,
        [&spring_constant_field, &spring_rest_length_field, &spring_constant, &spring_rest_length](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &local_spring) {
          // Assign the hertzian contact parameters to the backbone segments
          stk::mesh::field_data(spring_constant_field, local_spring)[0] = spring_constant;
          stk::mesh::field_data(spring_rest_length_field, local_spring)[0] = spring_rest_length;
        });  // for_each_entity_run
  }

#pragma TODO all of the initialization should become part of the chain of springs - like initialization
  // Initialize the chromsomes on a grid
  void initialize_chromosomes_grid() {
    std::cout << "Initializating chromosomes on a grid" << std::endl;
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        std::cout << "Initializing chromosome " << j << std::endl;
        openrand::Philox rng(j, 0);
        double jdouble = static_cast<double>(j);
        mundy::math::Vector3<double> r_start(2.0 * jdouble, 0.0, 0.0);
        // Add a tiny random change in X to make sure we don't wind up in perfectly parallel pathological states
        mundy::math::Vector3<double> u_hat(rng.uniform<double>(0.0, 0.001), 0.0, 1.0);
        u_hat = u_hat / mundy::math::two_norm(u_hat);

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument, "Node " << i << " is not valid.");

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_sphere_separation_ * u_hat;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
        }
      }
    }
  }

  // Initialize the chromosomes randomly in the unit cell
  void initialize_chromosomes_random_unit_cell() {
    std::cout << "Initializating chromosomes randomly in unit cell" << std::endl;
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        std::cout << "Initializing chromosome " << j << std::endl;

        // Find a random place within the unit cell with a random orientation for the chain.
        openrand::Philox rng(j, 0);
        mundy::math::Vector3<double> r_start(rng.uniform<double>(0.0, unit_cell_length_),
                                             rng.uniform<double>(0.0, unit_cell_length_),
                                             rng.uniform<double>(0.0, unit_cell_length_));
        // Find a random unit vector direction
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument, "Node " << i << " is not valid.");

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_sphere_separation_ * u_hat;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
        }
      }
    }
  }

  // Initialize for the overlap test
  void initialize_chromosomes_overlap_test() {
    std::cout << "Initializing chromosomes for the overlap test\n";
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        std::cout << "Initializing chromosome " << j << std::endl;

        // Start like we are pretending to be on a grid
        double jdouble = static_cast<double>(j);
        mundy::math::Vector3<double> r_start(2.0 * jdouble, 0.0, 0.0);
        mundy::math::Vector3<double> u_hat(0.0, 0.0, 1.0);

        // If num_chromosomes == 2, then try to do the crosshatch for a timestep?
        if (num_chromosomes_ == 2) {
          if (j == 0) {
            r_start = mundy::math::Vector3<double>(0.0, 0.0, 0.0);
            u_hat = mundy::math::Vector3<double>(0.0, 0.0, 1.0);
          } else if (j == 1) {
            r_start = mundy::math::Vector3<double>(-5.0, 0.25, 5.0);
            u_hat = mundy::math::Vector3<double>(1.0, 0.0, 0.0);
          }
        }

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument, "Node " << i << " is not valid.");

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_sphere_separation_ * u_hat;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
        }
      }
    }
  }

  // Initialize the chromosomes randomly in the unit cell
  //
  // If we want to initialize uniformly inside a sphere packing, here are the coordinates for a given number of spheres
  // within a bigger sphere.
  // http://hydra.nat.uni-magdeburg.de/packing/ssp/ssp.html
  void initialize_chromosomes_hilbert_random_unitcell() {
    std::cout << "Initializating chromosomes as hilbert curves randomly within the unit cell as a sphere" << std::endl;
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      std::vector<mundy::math::Vector3<double>> chromosome_centers_array;
      std::vector<double> chromosome_radii_array;
      for (size_t ichromosome = 0; ichromosome < num_chromosomes_; ichromosome++) {
        std::cout << "Initializing chromosome " << ichromosome << std::endl;

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * ichromosome + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (ichromosome + 1) + 1u;

        // Generate a random unit vector (will be used for creating the locatino of the nodes, the random position in
        // the unit cell will be handled later).
        openrand::Philox rng(ichromosome, 0);
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

        // Once we have the number of chromosome spheres we can get the hilbert curve set up. This will be at some
        // orientation and then have sides with a length of initial_sphere_separation_.
        auto [hilbert_position_array, hilbert_directors] = mundy::math::create_hilbert_positions_and_directors(
            num_nodes_per_chromosome, u_hat, initial_sphere_separation_);

        // Create the local positions of the spheres
        std::vector<mundy::math::Vector3<double>> sphere_position_array;
        for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
          sphere_position_array.push_back(hilbert_position_array[isphere]);
        }

        // Figure out where the center of the chromosome is, and its radius, in its own local space
        mundy::math::Vector3<double> r_chromosome_center_local(0.0, 0.0, 0.0);
        double r_max = 0.0;
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          r_chromosome_center_local += sphere_position_array[i];
        }
        r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
        }

        // Do max_trials number of insertion attempts to get a random position and orientation within the unit cell that
        // doesn't overlap with exiting chromosomes.
        const size_t max_trials = 1000;
        size_t itrial = 0;
        bool chromosome_inserted = false;
        while (itrial <= max_trials) {
          // Generate a random position and orientation within the unit cell as a sphere.

          // Create an adjusted length to keep the chromosome away from the wall, and inside a sphere. There is a
          // non-sampling way to do this, but meh.
          double init_periphery_radius = 0.0;
          if (do_equilibrate_) {
            init_periphery_radius = collision_periphery_radius_start_;
          } else {
            init_periphery_radius = collision_periphery_radius_;
          }
          double adjusted_radius = init_periphery_radius - r_max;
          mundy::math::Vector3<double> r_start(init_periphery_radius, init_periphery_radius, init_periphery_radius);
          while (mundy::math::two_norm(r_start) > adjusted_radius) {
            r_start = mundy::math::Vector3<double>(rng.uniform<double>(-1.0 * adjusted_radius, adjusted_radius),
                                                   rng.uniform<double>(-1.0 * adjusted_radius, adjusted_radius),
                                                   rng.uniform<double>(-1.0 * adjusted_radius, adjusted_radius));
          }

          // Check for overlaps with existing chromosomes
          bool found_overlap = false;
          for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
            double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
            if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
              found_overlap = true;
              break;
            }
          }
          if (found_overlap) {
            itrial++;
          } else {
            chromosome_inserted = true;
            chromosome_centers_array.push_back(r_start);
            chromosome_radii_array.push_back(r_max);
            break;
          }
        }
        MUNDY_THROW_ASSERT(chromosome_inserted, std::runtime_error,
                           "Failed to insert chromosome after " << max_trials << " trials.");

        // Generate all the positions along the curve due to the placement in the global space
        std::vector<mundy::math::Vector3<double>> new_position_array;
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          new_position_array.push_back(chromosome_centers_array.back() + r_chromosome_center_local -
                                       sphere_position_array[i]);
        }

        // Update the coordinates for this chromosome
        for (size_t i = start_node_index, idx = 0; i < end_node_index; ++i, ++idx) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument, "Node " << i << " is not valid.");

          // Assign the node coordinates
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = new_position_array[idx][0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = new_position_array[idx][1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = new_position_array[idx][2];
        }
      }
    }
  }

  // Initialize the chromatin backbone and HP1 linkers based on part membership
  //
  // The part membership should already be set up, which makes this much easier to do, as we can just loop over the
  // parts.
  void initialize_chromatin_backbone_and_hp1() {
    // Get parts for composed selectors we are using
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Part &eh_springs_part = *eh_springs_part_ptr_;
    stk::mesh::Part &hh_springs_part = *hh_springs_part_ptr_;
    stk::mesh::Part &hp1_part = *hp1_part_ptr_;

    // Initialize the excluded volume interaction for backbone segments (hertzian)
    const stk::mesh::Selector local_backbone_segments =
        (ee_springs_part | eh_springs_part | hh_springs_part) & bulk_data_ptr_->mesh_meta_data().locally_owned_part();
    initialize_excluded_volume_part_from_selector(local_backbone_segments, element_youngs_modulus_field_ptr_,
                                                  element_poissons_ratio_field_ptr_, backbone_youngs_modulus_,
                                                  backbone_poissons_ratio_);

    // Initialize the backbone springs (EE, EH, HH)
    initialize_spring_part_from_selector(ee_springs_part, element_hookean_spring_constant_field_ptr_,
                                         element_hookean_spring_rest_length_field_ptr_, chromatin_spring_constant_,
                                         chromatin_spring_rest_length_);
    initialize_spring_part_from_selector(eh_springs_part, element_hookean_spring_constant_field_ptr_,
                                         element_hookean_spring_rest_length_field_ptr_, chromatin_spring_constant_,
                                         chromatin_spring_rest_length_);
    initialize_spring_part_from_selector(hh_springs_part, element_hookean_spring_constant_field_ptr_,
                                         element_hookean_spring_rest_length_field_ptr_, chromatin_spring_constant_,
                                         chromatin_spring_rest_length_);
    // Also initialize the HP1 springs
    initialize_spring_part_from_selector(hp1_part, element_hookean_spring_constant_field_ptr_,
                                         element_hookean_spring_rest_length_field_ptr_, crosslinker_spring_constant_,
                                         crosslinker_rest_length_);

    // Initialize leftover HP1 variables
    //
    // This includes the RNG field, and the cutoff radius (not the rest length, set above)
    {
      const stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
      const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
      double &crosslinker_rcut = crosslinker_rcut_;
      stk::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK,
          hp1_part & bulk_data_ptr_->mesh_meta_data().locally_owned_part(),
          [&element_rng_field, &element_radius_field, &crosslinker_rcut](
              [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &local_hp1) {
            // Assign RNG counter and the cutoff radius to the HP1 crosslinkers
            stk::mesh::field_data(element_rng_field, local_hp1)[0] = 0;
            stk::mesh::field_data(element_radius_field, local_hp1)[0] = crosslinker_rcut;
          });  // for_each_entity_run
    }

    // Initialize leftover backbone variables
    //
    // This includes the hertzian radius
    {
      const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
      double radius_cutoff = backbone_excluded_volume_radius_;
      stk::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK, local_backbone_segments,
          [&element_radius_field, &radius_cutoff]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                  const stk::mesh::Entity &local_backbone) {
            // Set the radius
            stk::mesh::field_data(element_radius_field, local_backbone)[0] = radius_cutoff;
          });  // for_each_entity_run

      // Initialize leftover hydrodynamic sphere variables
      //
      // This includes the hydrodynamic radius of the spheres
      {
        const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
        double hydrodynamic_radius = sphere_hydrodynamic_radius_;
        stk::mesh::for_each_entity_run(
            *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_part,
            [&element_radius_field, &hydrodynamic_radius]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                          const stk::mesh::Entity &local_sphere) {
              // Set the radius
              stk::mesh::field_data(element_radius_field, local_sphere)[0] = hydrodynamic_radius;
            });  // for_each_entity_run
      }
    }

    // Initialize node positions for each chromosome
    if (initialization_type_ == "grid") {
      initialize_chromosomes_grid();
    } else if (initialization_type_ == "random_unit_cell") {
      initialize_chromosomes_random_unit_cell();
    } else if (initialization_type_ == "overlap_test") {
      initialize_chromosomes_overlap_test();
    } else if (initialization_type_ == "hilbertrandomunitcell") {
      initialize_chromosomes_hilbert_random_unitcell();
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown initialization type: " << initialization_type_);
    }
  }

  void declare_and_initialize_hp1() {
    ////////////////////////////////////////
    // Create the mesh nodes and elements //
    ////////////////////////////////////////
    create_chromatin_backbone_and_hp1();

    ////////////////////////////////////////
    // Initilize                          //
    ////////////////////////////////////////
    initialize_chromatin_backbone_and_hp1();
  }

  void initialize_periphery() {
    std::cout << "Initializing Periphery" << std::endl;
    // Setup the periphery
    const double viscosity = viscosity_;
    const bool invert = true;
    const bool include_poles = false;
    const size_t spectral_order = periphery_spectral_order_;
    const double periphery_radius = hydrodynamic_periphery_radius_;  // use the hydrodynamic radius)
    std::vector<double> points_vec;
    std::vector<double> weights_vec;
    std::vector<double> normals_vec;
    mundy::alens::periphery::gen_sphere_quadrature(spectral_order, periphery_radius, &points_vec, &weights_vec,
                                                   &normals_vec, include_poles, invert);
    const size_t num_surface_nodes = weights_vec.size();

    std::cout << "  Periphery has " << num_surface_nodes << " surface nodes" << std::endl;

    periphery_ptr_ = std::make_shared<mundy::alens::periphery::Periphery>(num_surface_nodes, viscosity);
    periphery_ptr_->set_surface_positions(points_vec.data())
        .set_quadrature_weights(weights_vec.data())
        .set_surface_normals(normals_vec.data());
    const bool write_to_file = false;
    periphery_ptr_->build_inverse_self_interaction_matrix(write_to_file);
  }

  void debug_print_meta_data() {
    std::cout << "############################################" << std::endl;
    const std::vector<std::string> &rank_names = meta_data_ptr_->entity_rank_names();
    for (size_t i = 0, e = rank_names.size(); i < e; ++i) {
      stk::mesh::EntityRank rank = static_cast<stk::mesh::EntityRank>(i);
      std::cout << "  All " << rank_names[i] << " entities:" << std::endl;

      const stk::mesh::BucketVector &buckets = bulk_data_ptr_->buckets(rank);
      for (stk::mesh::Bucket *bucket : buckets) {
        std::cout << "    bucket " << bucket->bucket_id() << " parts: { ";
        const stk::mesh::PartVector &supersets = bucket->supersets();
        for (const stk::mesh::Part *part : supersets) {
          std::cout << part->name() << " ";
        }
        std::cout << "}" << std::endl;
      }
    }

    auto print_super_and_subsets([](const stk::mesh::Part *part) {
      std::cout << "    part " << part->name() << " supersets: { ";
      const stk::mesh::PartVector &supersets = part->supersets();
      for (const stk::mesh::Part *part : supersets) {
        std::cout << part->name() << " ";
      }
      std::cout << "}" << std::endl;
      std::cout << "    part " << part->name() << " subsets: { ";
      const stk::mesh::PartVector &subsets = part->subsets();
      for (const stk::mesh::Part *part : subsets) {
        std::cout << part->name() << " ";
      }
      std::cout << "}" << std::endl;
    });

    // print_super_and_subsets(spheres_part_ptr_);
    // print_super_and_subsets(agents_part_ptr_);
    // print_super_and_subsets(springs_part_ptr_);
    // print_super_and_subsets(sphere_sphere_linkers_part_ptr_);
    // print_super_and_subsets(crosslinker_sphere_linkers_part_ptr_);
    // print_super_and_subsets(crosslinkers_part_ptr_);
    // print_super_and_subsets(left_bound_crosslinkers_part_ptr_);
    // print_super_and_subsets(right_bound_crosslinkers_part_ptr_);
    // print_super_and_subsets(doubly_bound_crosslinkers_part_ptr_);
    std::cout << "############################################" << std::endl;
  }

  void zero_out_transient_node_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
  }

  void zero_out_transient_element_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*element_binding_rates_field_ptr_,
                                                      std::array<double, 2>{0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*element_unbinding_rates_field_ptr_,
                                                      std::array<double, 2>{0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<unsigned>(*element_perform_state_change_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
  }

  void zero_out_transient_constraint_fields() {
    mundy::mesh::utils::fill_field_with_value<unsigned>(*constraint_perform_state_change_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<double>(*constraint_state_change_rate_field_ptr_,
                                                      std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*constraint_potential_force_field_ptr_,
                                                      std::array<double, 3>{0.0, 0.0, 0.0});
  }

  void zero_out_accumulator_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*element_corner_displacement_field_ptr_,
                                                      std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  }

  void detect_neighbors_initial() {
    Kokkos::Profiling::pushRegion("HP1::detect_neighbors_initial");

    last_neighborlist_update_step_ = 0;
    neighborlist_update_timer_.reset();

    // ComputeAABB for everyone (assume same buffer distance)
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto h_selector = stk::mesh::Selector(*h_part_ptr_);
    auto bs_selector = stk::mesh::Selector(*bs_part_ptr_);

    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);
    auto hp1_h_neighbor_genx_selector = stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_);
    auto hp1_bs_neighbor_genx_selector = stk::mesh::Selector(*hp1_bs_neighbor_genx_part_ptr_);

    compute_aabb_ptr_->execute(spheres_selector | backbone_segments_selector | hp1_selector);
    destroy_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector | hp1_h_neighbor_genx_selector |
                                        hp1_bs_neighbor_genx_selector);

    // Generate the GENX neighbor linkers
    generate_scs_scs_genx_ptr_->execute(backbone_segments_selector, backbone_segments_selector);
    ghost_linked_entities();
    generate_hp1_h_genx_ptr_->execute(hp1_selector, h_selector);
    ghost_linked_entities();
    generate_hp1_bs_genx_ptr_->execute(hp1_selector, bs_selector);
    ghost_linked_entities();

    // Destroy linkers along backbone chains
    destroy_bound_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    ghost_linked_entities();
    Kokkos::Profiling::popRegion();
  }

  void detect_neighbors() {
    Kokkos::Profiling::pushRegion("HP1::detect_neighbors");

    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto h_selector = stk::mesh::Selector(*h_part_ptr_);
    auto bs_selector = stk::mesh::Selector(*bs_part_ptr_);

    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);
    auto hp1_h_neighbor_genx_selector = stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_);
    auto hp1_bs_neighbor_genx_selector = stk::mesh::Selector(*hp1_bs_neighbor_genx_part_ptr_);

    // ComputeAABB for everybody at each time step. The accumulator then always uses this updated information to
    // calculate if we need to update the entire neighbor list.
    compute_aabb_ptr_->execute(spheres_selector | backbone_segments_selector | hp1_selector);
    update_accumulators();

    // Check if we need to update the neighbor list. Eventually this will be replaced with a mesh attribute to
    // synchronize across multiple tasks. For now, make sure that the default is to not update neighbor lists.
    check_update_neighbor_list();

    // Now do a check to see if we need to update the neighbor list.
    if (((force_neighborlist_update_) && (timestep_index_ % force_neighborlist_update_nsteps_ == 0)) ||
        update_neighbor_list_) {
      // Read off the timing information before doing anything else and reset it
      auto elapsed_steps = timestep_index_ - last_neighborlist_update_step_;
      auto elapsed_time = neighborlist_update_timer_.seconds();
      neighborlist_update_steps_times_.push_back(std::make_tuple(timestep_index_, elapsed_steps, elapsed_time));
      last_neighborlist_update_step_ = timestep_index_;
      neighborlist_update_timer_.reset();

      // Reset the accumulators
      zero_out_accumulator_fields();

      // Update the neighbor list
      destroy_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector | hp1_h_neighbor_genx_selector |
                                             hp1_bs_neighbor_genx_selector);
      ghost_linked_entities();
      
      // Generate the GENX neighbor linkers
      generate_scs_scs_genx_ptr_->execute(backbone_segments_selector, backbone_segments_selector);
      ghost_linked_entities();
      generate_hp1_h_genx_ptr_->execute(hp1_selector, h_selector);
      ghost_linked_entities();
      generate_hp1_bs_genx_ptr_->execute(hp1_selector, bs_selector);
      ghost_linked_entities();

      // Destroy linkers along backbone chains
      destroy_bound_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector);
      ghost_linked_entities();
    }

    Kokkos::Profiling::popRegion();
  }

  void update_accumulators() {
#pragma TODO This will be at the mercy of periodic boundary condition calculations.
    Kokkos::Profiling::pushRegion("HP1::update_accumulators");

    // Selectors and aliases
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);

    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_aabb_field_old = element_aabb_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;

    stk::mesh::Selector combined_selector = spheres_selector | backbone_segments_selector | hp1_selector;

    // Update the accumulators based on the difference to the previous state
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
        [&element_aabb_field, &element_aabb_field_old, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_aabb = stk::mesh::field_data(element_aabb_field, aabb_entity);
          double *element_aabb_old = stk::mesh::field_data(element_aabb_field_old, aabb_entity);
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Add the (new_aabb - old_aabb) to the corner displacement
          element_corner_displacement[0] += element_aabb[0] - element_aabb_old[0];
          element_corner_displacement[1] += element_aabb[1] - element_aabb_old[1];
          element_corner_displacement[2] += element_aabb[2] - element_aabb_old[2];
          element_corner_displacement[3] += element_aabb[3] - element_aabb_old[3];
          element_corner_displacement[4] += element_aabb[4] - element_aabb_old[4];
          element_corner_displacement[5] += element_aabb[5] - element_aabb_old[5];
        });

    Kokkos::Profiling::popRegion();
  }

  void check_update_neighbor_list() {
    Kokkos::Profiling::pushRegion("HP1::check_update_neighbor_list");

    // Local variable for if we should update the neighbor list (do as an integer for now because MPI doesn't like
    // bools)
    int local_update_neighbor_list_int = 0;

    // Selectors and aliases
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);

    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;
    const double &skin_distance2_over4 = skin_distance2_over4_;

    stk::mesh::Selector combined_selector = spheres_selector | backbone_segments_selector | hp1_selector;

    // Check if each corner has moved skin_distance/2. Or, if dr_mag2 >= skin_distance^2/4
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
        [&local_update_neighbor_list_int, &skin_distance2_over4, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Compute dr2 for each corner
          double dr2_corner0 = element_corner_displacement[0] * element_corner_displacement[0] +
                               element_corner_displacement[1] * element_corner_displacement[1] +
                               element_corner_displacement[2] * element_corner_displacement[2];
          double dr2_corner1 = element_corner_displacement[3] * element_corner_displacement[3] +
                               element_corner_displacement[4] * element_corner_displacement[4] +
                               element_corner_displacement[5] * element_corner_displacement[5];

          if (dr2_corner0 >= skin_distance2_over4 || dr2_corner1 >= skin_distance2_over4) {
            local_update_neighbor_list_int = 1;
          }
        });

    // Communicate local_update_neighbor_list to all ranks. Convert to an integer first (MPI doesn't handle booleans
    // well).
    int global_update_neighbor_list_int = 0;
    MPI_Allreduce(&local_update_neighbor_list_int, &global_update_neighbor_list_int, 1, MPI_INT, MPI_LOR,
                  MPI_COMM_WORLD);
    // Convert back to the boolean for the global version and or it with the original value (in case somebody else set
    // the neighbor list update 'signal').
    update_neighbor_list_ = update_neighbor_list_ || (global_update_neighbor_list_int == 1);

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for left-bound crosslinkers
  void compute_z_partition_left_bound_harmonic() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition_left_bound_harmonic");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &constraint_state_change_probability = *constraint_state_change_rate_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_constant = *element_hookean_spring_constant_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_rest_length = *element_hookean_spring_rest_length_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;
    stk::mesh::Part &hp1_h_neighbor_genx_part = *hp1_h_neighbor_genx_part_ptr_;
    const double inv_kt = 1.0 / kt_kmc_;
    const double &crosslinker_right_binding_rate = crosslinker_right_binding_rate_;

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, hp1_h_neighbor_genx_part,
        [&node_coord_field, &constraint_linked_entities_field, &constraint_state_change_probability,
         &crosslinker_spring_constant, &crosslinker_spring_rest_length, &left_hp1_part, &inv_kt,
         &crosslinker_right_binding_rate]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                          const stk::mesh::Entity &neighbor_genx) {
          // Get the sphere and crosslinker attached to the linker.
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(constraint_linked_entities_field, neighbor_genx));
          const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), std::invalid_argument,
                             "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
          MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), std::invalid_argument,
                             "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

          // We need to figure out if this is a self-interaction or not. Since we are a left-bound crosslinker.
          const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
          bool is_self_interaction = false;
          if (bulk_data.bucket(crosslinker).member(left_hp1_part)) {
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
          }

          // Only act on the left-bound crosslinkers
          if (bulk_data.bucket(crosslinker).member(left_hp1_part) && !is_self_interaction) {
            const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                            mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
            const double dr_mag = mundy::math::norm(dr);

            // Compute the Z-partition score
            // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
            // A = crosslinker_binding_rates
            // k = crosslinker_spring_constant
            // r0 = crosslinker_spring_rest_length
            const double A = crosslinker_right_binding_rate;
            const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
            const double r0 = stk::mesh::field_data(crosslinker_spring_rest_length, crosslinker)[0];
            double Z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
            stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
          }
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for doubly_bound crosslinkers
  void compute_z_partition_doubly_bound_harmonic() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition_doubly_bound_harmonic");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;
    const double &crosslinker_right_unbinding_rate = crosslinker_right_unbinding_rate_;

    // Loop over the neighbor list of the crosslinkers, then select down to the ones that are left-bound only.
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_hp1_h_part,
        [&node_coord_field, &crosslinker_unbinding_rates, &doubly_hp1_h_part, &crosslinker_right_unbinding_rate](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          // This is a left-bound crosslinker, so just calculate the right unbinding rate and store on the crosslinker
          // itself in the correct position.
          stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1] = crosslinker_right_unbinding_rate;
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function for everybody
  void compute_z_partition() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition");

    // Compute the left-bound to doubly-bound score
    compute_z_partition_left_bound_harmonic();

    // Compute the doubly-bound to left-bound score
    compute_z_partition_doubly_bound_harmonic();

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_left_to_doubly() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_left_to_doubly");

    // Selectors and aliases
    stk::mesh::Part &hp1_h_neighbor_genx_part = *hp1_h_neighbor_genx_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    stk::mesh::Field<unsigned> &constraint_perform_state_change_field = *constraint_perform_state_change_field_ptr_;
    stk::mesh::Field<double> &constraint_state_change_rate_field = *constraint_state_change_rate_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    const double &timestep_size = timestep_size_;
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;

    // Loop over left-bound crosslinkers and decide if they bind or not
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, left_hp1_part,
        [&hp1_h_neighbor_genx_part, &element_rng_field, &constraint_perform_state_change_field,
         &element_perform_state_change_field, &constraint_state_change_rate_field, &constraint_linked_entities_field,
         &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          // Get all of my associated crosslinker_sphere_linkers
          const stk::mesh::Entity &any_arbitrary_crosslinker_node = bulk_data.begin_nodes(crosslinker)[0];
          const stk::mesh::Entity *neighbor_genx_linkers =
              bulk_data.begin(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);
          const unsigned num_neighbor_genx_linkers =
              bulk_data.num_connectivity(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);

          // Loop over the attached crosslinker_sphere_linkers and bind if the rqng falls in their range.
          double z_tot = 0.0;
          for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
            const auto &constraint_rank_entity = neighbor_genx_linkers[j];
            const bool is_hp1_h_neighbor_genx =
                bulk_data.bucket(constraint_rank_entity).member(hp1_h_neighbor_genx_part);
            if (is_hp1_h_neighbor_genx) {
              const double z_i =
                  timestep_size * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
              z_tot += z_i;
            }
          }

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, crosslinker);
          const stk::mesh::EntityId crosslinker_gid = bulk_data.identifier(crosslinker);
          openrand::Philox rng(crosslinker_gid, element_rng_counter[0]);
          const double randu01 = rng.rand<double>();
          element_rng_counter[0]++;

          // Notice that the sum of all probabilities is 1.
          // The probability of nothing happening is
          //   std::exp(-z_tot)
          // The probability of an individual event happening is
          //   z_i / z_tot * (1 - std::exp(-z_tot))
          //
          // This is (by construction) true since
          //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
          //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
          //
          // This means that binding only happens if randu01 < (1 - std::exp(-z_tot))
          const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
          const double scale_factor = probability_of_no_state_change * timestep_size / z_tot;
          if (randu01 < (1.0 - std::exp(-z_tot))) {
            // Binding occurs.
            // Loop back over the neighbor linkers to see if one of them binds in the running sum

            double cumsum = 0.0;
            for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
              auto &constraint_rank_entity = neighbor_genx_linkers[j];
              bool is_hp1_h_neighbor_genx = bulk_data.bucket(constraint_rank_entity).member(hp1_h_neighbor_genx_part);
              if (is_hp1_h_neighbor_genx) {
                const double binding_probability =
                    scale_factor * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
                cumsum += binding_probability;
                if (randu01 < cumsum) {
                  // We have a binding event, set this, then bail on the for loop
                  // Store the state change on both the genx and the crosslinker
                  stk::mesh::field_data(constraint_perform_state_change_field, constraint_rank_entity)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  stk::mesh::field_data(element_perform_state_change_field, crosslinker)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  break;
                }
              }
            }
          }
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Perform the binding of a crosslinker to a sphere (doubly to left)
  void kmc_crosslinker_doubly_to_left() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_doubly_to_left");

    // Selectors and aliases
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    const double &timestep_size = timestep_size_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;

    // This is just a loop over the doubly bound crosslinkers, since we know that the right head in is [1].
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_hp1_h_part,
        [&element_rng_field, &element_perform_state_change_field, &crosslinker_unbinding_rates, &timestep_size](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          // We only have a single node, our right node, that is bound that we can unbind.
          // TODO(cje): Right now this is coded to have a loop wrapping it, maybe not needed?
          const double unbinding_probability =
              timestep_size * stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1];
          double Z_tot = unbinding_probability;
          const double unbind_scale_factor = (1.0 - exp(-Z_tot)) * timestep_size;

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, crosslinker);
          const stk::mesh::EntityId crosslinker_gid = bulk_data.identifier(crosslinker);
          openrand::Philox rng(crosslinker_gid, element_rng_counter[0]);
          double randZ = rng.rand<double>() * Z_tot;
          double cumsum = 0.0;
          element_rng_counter[0]++;

          // Now check the cummulative sum and if less than perform the unbinding
          cumsum += unbind_scale_factor * stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1];
          if (randZ < cumsum) {
            // Set the state change on the element
            stk::mesh::field_data(element_perform_state_change_field, crosslinker)[0] =
                static_cast<unsigned>(BINDING_STATE_CHANGE::DOUBLY_TO_LEFT);
          }
        });

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_sphere_linker_sampling() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_sphere_linker_sampling");

    // Perform the left to doubly bound crosslinker binding calc
    kmc_crosslinker_left_to_doubly();

    // Perform the doubly to left bound crosslinker binding calc
    kmc_crosslinker_doubly_to_left();

    // At this point, constraint_state_change_rate_field is only up-to-date for locally-owned entities. We need
    // to communicate this information to all other processors.
    stk::mesh::communicate_field_data(
        *bulk_data_ptr_, {element_perform_state_change_field_ptr_, constraint_perform_state_change_field_ptr_});

    Kokkos::Profiling::popRegion();
  }

  /// \brief Perform the state change of the crosslinkers
  void state_change_crosslinkers() {
    Kokkos::Profiling::pushRegion("HP1::state_change_crosslinkers");

    // Loop over both the CROSSLINKER_SPHERE_LINKERS and the CROSSLINKERS to perform the state changes.
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;

    // Get the vector of entities to modify
    stk::mesh::EntityVector hp1_h_neighbor_genxs;
    stk::mesh::EntityVector doubly_bound_hp1s;
    stk::mesh::get_selected_entities(stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_),
                                     bulk_data_ptr_->buckets(constraint_rank_), hp1_h_neighbor_genxs);
    stk::mesh::get_selected_entities(stk::mesh::Selector(*doubly_hp1_h_part_ptr_),
                                     bulk_data_ptr_->buckets(element_rank_), doubly_bound_hp1s);

    // TODO(cje): It might be worth checking to see if we have any state changes in any threads before we crack open the
    // modification section, as even doing that is slightly expensive.

    bulk_data_ptr_->modification_begin();

    // Perform L->D
    for (const stk::mesh::Entity &hp1_h_neighbor_genx : hp1_h_neighbor_genxs) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*constraint_perform_state_change_field_ptr_, hp1_h_neighbor_genx)[0]);
      const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
      if (perform_state_change) {
        // Get our connections (as the genx)
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(*constraint_linked_entities_field_ptr_, hp1_h_neighbor_genx));
        const stk::mesh::Entity &crosslinker_hp1 = bulk_data_ptr_->get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &target_sphere = bulk_data_ptr_->get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(crosslinker_hp1), std::invalid_argument,
                           "Encountered invalid crosslinker entity in state_change_crosslinkers.");
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(target_sphere), std::invalid_argument,
                           "Encountered invalid sphere entity in state_change_crosslinkers.");

        const stk::mesh::Entity &target_sphere_node = bulk_data_ptr_->begin_nodes(target_sphere)[0];
        // Call the binding function
        if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
          // Unbind the right side of the crosslinker from the left node and bind it to the target node
          const bool bind_worked =
              bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, crosslinker_hp1, target_sphere_node, 1);
          MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

          std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Binding crosslinker "
                    << bulk_data_ptr_->identifier(crosslinker_hp1) << " to node "
                    << bulk_data_ptr_->identifier(target_sphere_node) << std::endl;

          // Now change the part from left to doubly bound.
          const bool is_crosslinker_locally_owned =
              bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
          if (is_crosslinker_locally_owned) {
            auto add_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
            auto remove_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
            bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
          }
        }
      }
    }

    // Perform D->L
    for (const stk::mesh::Entity &crosslinker_hp1 : doubly_bound_hp1s) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*element_perform_state_change_field_ptr_, crosslinker_hp1)[0]);
      if (state_change_action == BINDING_STATE_CHANGE::DOUBLY_TO_LEFT) {
        // Unbind the right side of the crosslinker from the current node and bind it to the left crosslinker node
        const stk::mesh::Entity &left_node = bulk_data_ptr_->begin_nodes(crosslinker_hp1)[0];
        const bool unbind_worked =
            bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, crosslinker_hp1, left_node, 1);
        MUNDY_THROW_ASSERT(unbind_worked, std::logic_error, "Failed to unbind crosslinker from node.");

        std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Unbinding crosslinker "
                  << bulk_data_ptr_->identifier(crosslinker_hp1) << " from node "
                  << bulk_data_ptr_->identifier(bulk_data_ptr_->begin_nodes(crosslinker_hp1)[1]) << std::endl;

        // Now change the part from doubly to left bound.
        const bool is_crosslinker_locally_owned =
            bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
        if (is_crosslinker_locally_owned) {
          auto add_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
          auto remove_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
          bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
        }
      }
    }

    bulk_data_ptr_->modification_end();

    // The above may have invalidated the ghosting for our genx ghosting, so we need to reghost the linked entities to
    // any process that owns any of the other linked entities.
    ghost_linked_entities();

    Kokkos::Profiling::popRegion();
  }

  void update_crosslinker_state() {
    Kokkos::Profiling::pushRegion("HP1::update_crosslinker_state");

    // We want to loop over all LEFT_BOUND_CROSSLINKERS, RIGHT_BOUND_CROSSLINKERS, and DOUBLY_BOUND_CROSSLINKERS to
    // generate state changes. This is done to build up a list of actions that we will take later during a mesh
    // modification step.
    {
      compute_z_partition();
      kmc_crosslinker_sphere_linker_sampling();
    }

    // Loop over the different crosslinkers, look at their actions, and enforce the state change.
    {
      // Call the global state change function
      state_change_crosslinkers();
    }

    Kokkos::Profiling::popRegion();
  }

  void compute_rpy_hydro_with_no_slip_periphery() {
    Kokkos::Profiling::pushRegion("HP1::compute_rpy_hydro_with_no_slip_periphery");
    MUNDY_THROW_ASSERT(periphery_ptr_ != nullptr, std::runtime_error, "Periphery not initialized.");
    const double viscosity = viscosity_;

    // TODO(palmerb4): Uncertain what the most efficient way to achieve this is. We use KokkosBlas to perform the matrix
    // inversion and matrix-vector multiplication required for periphery initialization and evaluation. This is
    // efficient, but it requires copying some of our STK field data to Kokkos views.

    // Fetch the bucket of spheres to act on.
    stk::mesh::EntityVector sphere_elements;
    stk::mesh::Selector spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    stk::mesh::get_selected_entities(spheres_selector, bulk_data_ptr_->buckets(stk::topology::ELEMENT_RANK),
                                     sphere_elements);
    const size_t num_spheres = sphere_elements.size();

    // Copy the sphere positions, radii, forces, and velocities to Kokkos views
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions("sphere_positions", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii("sphere_radii", num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces("sphere_forces", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities("sphere_velocities",
                                                                                    num_spheres * 3);

#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      const double *sphere_position = stk::mesh::field_data(*node_coord_field_ptr_, sphere_node);
      const double *sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);
      const double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      const double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_positions(i * 3 + j) = sphere_position[j];
        sphere_forces(i * 3 + j) = sphere_force[j];
        sphere_velocities(i * 3 + j) = sphere_velocity[j];
      }
      sphere_radii(i) = *sphere_radius;
    }

    // Setup the periphery information
    const size_t num_surface_nodes = periphery_ptr_->get_num_nodes();
    auto surface_positions = periphery_ptr_->get_surface_positions();
    auto surface_weights = periphery_ptr_->get_quadrature_weights();
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_radii("surface_radii", num_surface_nodes);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_velocities("surface_velocities",
                                                                                     3 * num_surface_nodes);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_forces("surface_forces",
                                                                                 3 * num_surface_nodes);
    Kokkos::deep_copy(surface_radii, 0.0);

    // Apply the RPY kernel from spheres to spheres
    mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                              sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

    // Apply the RPY kernel from spheres to periphery
    mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, surface_positions,
                                              sphere_radii, surface_radii, sphere_forces, surface_velocities);

    // Apply no-slip boundary conditions
    // This is done in two steps: first, we compute the forces on the periphery necessary to enforce no-slip
    // Then we evaluate the flow these forces induce on the spheres.
    periphery_ptr_->compute_surface_forces(surface_velocities, surface_forces);

    // // If we evaluate the flow these forces induce on the periphery, do they satisfy no-slip?
    // Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> M("Mnew", 3 * num_surface_nodes,
    //                                                                  3 * num_surface_nodes);
    // fill_skfie_matrix(DeviceExecutionSpace(), viscosity, num_surface_nodes, num_surface_nodes, surface_positions,
    //                   surface_positions, surface_normals, surface_weights, M);
    // KokkosBlas::gemv(DeviceExecutionSpace(), "N", 1.0, M, surface_forces, 1.0, surface_velocities);
    // EXPECT_NEAR(max_speed(surface_velocities), 0.0, 1.0e-10);

    mundy::alens::periphery::apply_weighted_stokes_kernel(DeviceExecutionSpace(), viscosity, surface_positions,
                                                          sphere_positions, surface_forces, surface_weights,
                                                          sphere_velocities);

    // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
    mundy::alens::periphery::apply_local_drag(DeviceExecutionSpace(), viscosity, sphere_velocities, sphere_forces,
                                              sphere_radii);

    // Copy the sphere forces and velocities back to STK fields
#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_force[j] = sphere_forces(i * 3 + j);
        sphere_velocity[j] = sphere_velocities(i * 3 + j);
      }
    }
    Kokkos::Profiling::popRegion();
  }

  void compute_periphery_collision_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_periphery_collision_forces");
    const double spring_constant = periphery_spring_constant_;
    const double a = collision_periphery_radius_current_;
    const double b = collision_periphery_radius_current_;
    const double c = collision_periphery_radius_current_;
    const double inv_a2 = 1.0 / (a * a);
    const double inv_b2 = 1.0 / (b * b);
    const double inv_c2 = 1.0 / (c * c);
    const mundy::math::Vector3<double> center(0.0, 0.0, 0.0);
    const auto orientation = mundy::math::Quaternion<double>::identity();
    auto level_set = [&inv_a2, &inv_b2, &inv_c2, &center,
                      &orientation](const mundy::math::Vector3<double> &point) -> double {
      const auto body_frame_point = conjugate(orientation) * (point - center);
      return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
              body_frame_point[2] * body_frame_point[2] * inv_c2) -
             1;
    };

    // Fetch loc al references to the fields
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, *spheres_part_ptr_,
        [&node_coord_field, &node_force_field, &element_aabb_field, &level_set, &center, &orientation, &a, &b, &c,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // For our coarse search, we check if the coners of the sphere's aabb lie inside the ellipsoidal periphery
          // This can be done via the (body frame) inside outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2 + z^2/c^2)
          // This is possible due to the convexity of the ellipsoid
          const double *sphere_aabb = stk::mesh::field_data(element_aabb_field, sphere_element);
          const double x0 = sphere_aabb[0];
          const double y0 = sphere_aabb[1];
          const double z0 = sphere_aabb[2];
          const double x1 = sphere_aabb[3];
          const double y1 = sphere_aabb[4];
          const double z1 = sphere_aabb[5];

          // Compute all 8 corners of the AABB
          const auto bottom_left_front = mundy::math::Vector3<double>(x0, y0, z0);
          const auto bottom_right_front = mundy::math::Vector3<double>(x1, y0, z0);
          const auto top_left_front = mundy::math::Vector3<double>(x0, y1, z0);
          const auto top_right_front = mundy::math::Vector3<double>(x1, y1, z0);
          const auto bottom_left_back = mundy::math::Vector3<double>(x0, y0, z1);
          const auto bottom_right_back = mundy::math::Vector3<double>(x1, y0, z1);
          const auto top_left_back = mundy::math::Vector3<double>(x0, y1, z1);
          const auto top_right_back = mundy::math::Vector3<double>(x1, y1, z1);
          const double all_points_inside_periphery =
              level_set(bottom_left_front) < 0.0 && level_set(bottom_right_front) < 0.0 &&
              level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 &&
              level_set(bottom_left_back) < 0.0 && level_set(bottom_right_back) < 0.0 &&
              level_set(top_left_back) < 0.0 && level_set(top_right_back) < 0.0;

          if (!all_points_inside_periphery) {
            // std::cout << "Sphere element " << bulk_data.identifier(sphere_element)
            //           << " is not entirely inside the periphery." << std::endl;
            // We might have a collision, perform the more expensive check
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);

            mundy::math::Vector3<double> contact_point;
            const double shared_normal_ssd = mundy::math::distance::shared_normal_ssd_between_ellipsoid_and_point(
                center, orientation, a, b, c, node_coords, &contact_point);

            // Note, the ellipsoid for the ssd calc has outward normal, whereas the periphery has inward normal
            // As a result, overlap occurs when the shared_normal_ssd is positive.
            if (shared_normal_ssd > 0.0) {
              // We have a collision, compute the force
              auto inward_normal = contact_point - node_coords;
              inward_normal /= mundy::math::two_norm(inward_normal);
#pragma omp atomic
              node_force[0] += spring_constant * inward_normal[0] * shared_normal_ssd;
#pragma omp atomic
              node_force[1] += spring_constant * inward_normal[1] * shared_normal_ssd;
#pragma omp atomic
              node_force[2] += spring_constant * inward_normal[2] * shared_normal_ssd;
            }
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void compute_hertzian_contact_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_hertzian_contact_forces");

    // Potential evaluation (Hertzian contact)
    auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);

    compute_ssd_and_cn_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    evaluate_linker_potentials_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    linker_potential_force_reduction_ptr_->execute(backbone_selector);

    Kokkos::Profiling::popRegion();
  }

  void compute_harmonic_bond_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_harmonic_bond_forces");

    // Need to select the active springs in the system, so backbone springs, and active HP1 springs. Do this by
    // selecting down from all HP1 springs.
    auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto left_hp1_selector = stk::mesh::Selector(*left_hp1_part_ptr_);
    auto actively_bound_springs = backbone_selector | (hp1_selector - left_hp1_selector);

    // Potentials
    compute_constraint_forcing_ptr_->execute(actively_bound_springs);

    Kokkos::Profiling::popRegion();
  }

  void compute_brownian_velocity() {
    // Compute the velocity due to brownian motion
    Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

    // Selectors and aliases
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Field<unsigned> &node_rng_field = *node_rng_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double &sphere_drag_coeff = sphere_drag_coeff_;
    double &kt = kt_brownian_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
        [&node_velocity_field, &node_force_field, &node_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff,
         &kt](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_field, sphere_node);

          // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * kt * sphere_drag_coeff / timestep_size) * inv_drag_coeff;
          node_velocity[0] += coeff * rng.randn<double>();
          node_velocity[1] += coeff * rng.randn<double>();
          node_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });

    Kokkos::Profiling::popRegion();
  }

  void compute_external_velocity() {
    // Compute both the velocity due to external forces
    Kokkos::Profiling::pushRegion("HP1::compute_external_velocity");

    // Selectors and aliases
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double &sphere_drag_coeff = sphere_drag_coeff_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
        [&node_velocity_field, &node_force_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          double *node_force = stk::mesh::field_data(node_force_field, sphere_node);

          // Uext = Fext * inv_drag_coeff
          node_velocity[0] += node_force[0] * inv_drag_coeff;
          node_velocity[1] += node_force[1] * inv_drag_coeff;
          node_velocity[2] += node_force[2] * inv_drag_coeff;
        });

    Kokkos::Profiling::popRegion();
  }

  void update_positions() {
    Kokkos::Profiling::pushRegion("HP1::update_positions");

    // Selectors and aliases
    size_t &timestep_index = timestep_index_;
    double &timestep_size = timestep_size_;
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;

    // Update the positions for all spheres based on velocity
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
        [&node_coord_field, &node_velocity_field, &timestep_size, &timestep_index](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_coord = stk::mesh::field_data(node_coord_field, sphere_node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);

          // Check to see if we've moved 1/10 of the diameter of one of the hydrodynamic beads in this timestep
          const auto dr = mundy::math::Vector3<double>(
              timestep_size * node_velocity[0], timestep_size * node_velocity[1], timestep_size * node_velocity[2]);
          const double dr_mag = mundy::math::norm(dr);
          if (dr_mag > 1.0e-1) {
            std::cout << "Step: " << timestep_index << ", large movement detected\n";
            std::cout << "  dr: " << dr << ", dr_mag: " << dr_mag << std::endl;
            std::cout << "  node_velocity: " << node_velocity[0] << ", " << node_velocity[1] << ", " << node_velocity[2]
                      << std::endl;
            stk::mesh::impl::dump_mesh_per_proc(bulk_data, "meshfailure");
            MUNDY_THROW_ASSERT(false, std::runtime_error, "Large movement due to timestep detected.");
          }

          // x(t+dt) = x(t) + dt * v(t)
          node_coord[0] += timestep_size * node_velocity[0];
          node_coord[1] += timestep_size * node_velocity[1];
          node_coord[2] += timestep_size * node_velocity[2];
        });

    Kokkos::Profiling::popRegion();
  }

  void run(int argc, char **argv) {
    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    Kokkos::Profiling::pushRegion("HP1::Setup");
    build_our_mesh_and_method_instances();

    fetch_fields_and_parts();
    instantiate_metamethods();
    set_mutable_parameters();
    setup_io_mundy();
    declare_and_initialize_hp1();
    initialize_periphery();
    Kokkos::Profiling::popRegion();

    // Loadbalance?
    Kokkos::Profiling::pushRegion("HP1::Loadbalance");
    if (initial_loadbalance_) {
      loadbalance();
    }
    Kokkos::Profiling::popRegion();

    // Equilibrate the system. This runs brownian dynamics on the chains with the periphery, but no hydrodynamics or
    // crosslinker activity.
    if (do_equilibrate_) {
      const double dr_periphery = (collision_periphery_radius_start_ - collision_periphery_radius_) /
                                  static_cast<double>(num_time_steps_equilibrate_);
      detect_neighbors_initial();
      print_rank0(std::string("Equilibrating the simulation for ") + std::to_string(num_time_steps_equilibrate_) +
                  " time steps.");

      Kokkos::Timer equilibrate_timer;
      Kokkos::Profiling::pushRegion("HP1::Equilibrate");
      for (timestep_index_ = 0; timestep_index_ < num_time_steps_equilibrate_; timestep_index_++) {
        // Prepare the current configuration.
        Kokkos::Profiling::pushRegion("HP1::PrepareStep");
        zero_out_transient_node_fields();
        zero_out_transient_element_fields();
        zero_out_transient_constraint_fields();
        rotate_field_states();
        Kokkos::Profiling::popRegion();

        // Update the periphery to shrink it
        collision_periphery_radius_current_ -= dr_periphery;

        // Detect sphere-sphere and crosslinker-sphere neighbors
        update_neighbor_list_ = false;
        detect_neighbors();

        // Evaluate forces f(x(t)).
        compute_hertzian_contact_forces();
        compute_harmonic_bond_forces();
        compute_periphery_collision_forces();

        // Compute velocities.
        compute_brownian_velocity();
        compute_external_velocity();

        // Logging, if desired, write to console
        Kokkos::Profiling::pushRegion("HP1::Logging");
        if (timestep_index_ % log_frequency_ == 0) {
          if (bulk_data_ptr_->parallel_rank() == 0) {
            double tps = static_cast<double>(timestep_index_) / static_cast<double>(equilibrate_timer.seconds());
            std::cout << "Equilibration Step: " << std::setw(15) << timestep_index_
                      << ", tps: " << std::setprecision(15) << tps << std::endl;
          }
        }
        Kokkos::Profiling::popRegion();

        // Update positions. x(t + dt) = x(t) + dt * v(t).
        update_positions();
      }
      Kokkos::Profiling::popRegion();

      // Do a synchronize to force everybody to stop here
      stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    }

    // Reset simulation control variables
    timestep_index_ = 0;
    collision_periphery_radius_current_ = collision_periphery_radius_;  // Set to the master collision radius
    std::cout << "  Reset collision periphery radius to " << collision_periphery_radius_current_ << std::endl;
    // Make sure that all of the spheres are within the collision radius of the periphery
    {
      stk::mesh::Part &spheres_part = *spheres_part_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      double &hydrodynamic_periphery_radius = hydrodynamic_periphery_radius_;

      stk::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
          [&node_coord_field, &hydrodynamic_periphery_radius](const stk::mesh::BulkData &bulk_data,
                                                              const stk::mesh::Entity &sphere_node) {
            // Get the coordinates of the sphere
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            if (mundy::math::norm(node_coords) > hydrodynamic_periphery_radius) {
              std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                        << " is outside the hydrodynamic periphery." << std::endl;
              std::cout << "  node_coords: " << node_coords << std::endl;
              std::cout << "  norm(node_coords): " << mundy::math::norm(node_coords) << std::endl;
              MUNDY_THROW_ASSERT(false, std::runtime_error, "Sphere node outside hydrodynamic periphery.");
            }
          });
    }

    // Run an initial detection of neighbors not using an adaptive neighborlist to set the initial state, and calculate
    // the initial AABB for everybody.
    detect_neighbors_initial();

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("MainLoop");
    for (timestep_index_ = 0; timestep_index_ < num_time_steps_; timestep_index_++) {
      // Prepare the current configuration.
      Kokkos::Profiling::pushRegion("HP1::PrepareStep");
      zero_out_transient_node_fields();
      zero_out_transient_element_fields();
      zero_out_transient_constraint_fields();
      rotate_field_states();
      Kokkos::Profiling::popRegion();

      // Detect sphere-sphere and crosslinker-sphere neighbors
      update_neighbor_list_ = false;
      detect_neighbors();

      // Update the state changes in the system s(t).;
      update_crosslinker_state();

      // Evaluate forces f(x(t)).
      compute_hertzian_contact_forces();
      compute_harmonic_bond_forces();
      compute_periphery_collision_forces();

      // Compute velocities.
      compute_brownian_velocity();
      compute_rpy_hydro_with_no_slip_periphery();
      // compute_external_velocity();

      // Logging, if desired, write to console
      Kokkos::Profiling::pushRegion("HP1::Logging");
      if (timestep_index_ % log_frequency_ == 0) {
        if (bulk_data_ptr_->parallel_rank() == 0) {
          double tps = static_cast<double>(timestep_index_) / static_cast<double>(timer.seconds());
          std::cout << "Step: " << std::setw(15) << timestep_index_ << ", tps: " << std::setprecision(15) << tps
                    << std::endl;
        }
      }
      Kokkos::Profiling::popRegion();

      // IO. If desired, write out the data for time t (STK or mundy)
      Kokkos::Profiling::pushRegion("HP1::IO");
      if (timestep_index_ % io_frequency_ == 0) {
        io_broker_ptr_->write_io_broker_timestep(static_cast<int>(timestep_index_),
                                                 static_cast<double>(timestep_index_));
      }
      Kokkos::Profiling::popRegion();

      // Update positions. x(t + dt) = x(t) + dt * v(t).
      update_positions();
    }
    Kokkos::Profiling::popRegion();

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
      double tps = static_cast<double>(timestep_index_) / static_cast<double>(timer.seconds());
      std::cout << "******************Final statistics (Rank 0)**************\n";
      if (print_neighborlist_statistics_) {
        std::cout << "****************\n";
        std::cout << "Neighbor list statistics\n";
        for (auto &neighborlist_entry : neighborlist_update_steps_times_) {
          auto [timestep, elasped_step, elapsed_time] = neighborlist_entry;
          auto tps_nl = static_cast<double>(elasped_step) / elapsed_time;
          std::cout << "  Rebuild timestep: " << timestep << ", elapsed_steps: " << elasped_step
                    << ", elapsed_time: " << elapsed_time << ", tps: " << tps_nl << std::endl;
        }
      }
      std::cout << "****************\n";
      std::cout << "Simulation statistics\n";
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
      std::cout << "Timesteps per second: " << std::setprecision(15) << tps << std::endl;
    }
  }

 private:
  //! \name Useful aliases
  //@{

  static constexpr auto node_rank_ = stk::topology::NODE_RANK;
  static constexpr auto edge_rank_ = stk::topology::EDGE_RANK;
  static constexpr auto element_rank_ = stk::topology::ELEMENT_RANK;
  static constexpr auto constraint_rank_ = stk::topology::CONSTRAINT_RANK;
  //@}

  //! \name Internal state
  //@{

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_;
  std::shared_ptr<mundy::io::IOBroker> io_broker_ptr_ = nullptr;
  size_t output_file_index_;
  size_t timestep_index_;
  std::shared_ptr<mundy::alens::periphery::Periphery> periphery_ptr_;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<unsigned> *node_rng_field_ptr_;
  stk::mesh::Field<double> *node_coord_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;

  stk::mesh::Field<unsigned> *element_rng_field_ptr_;
  stk::mesh::Field<double> *element_radius_field_ptr_;
  stk::mesh::Field<double> *element_hookean_spring_constant_field_ptr_;
  stk::mesh::Field<double> *element_hookean_spring_rest_length_field_ptr_;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr_;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr_;
  stk::mesh::Field<double> *element_aabb_field_ptr_;
  stk::mesh::Field<double> *element_corner_displacement_field_ptr_;
  stk::mesh::Field<double> *element_binding_rates_field_ptr_;
  stk::mesh::Field<double> *element_unbinding_rates_field_ptr_;
  stk::mesh::Field<unsigned> *element_perform_state_change_field_ptr_;

  stk::mesh::Field<double> *constraint_potential_force_field_ptr_;
  stk::mesh::Field<double> *constraint_state_change_rate_field_ptr_;
  stk::mesh::Field<unsigned> *constraint_perform_state_change_field_ptr_;
  stk::mesh::Field<int> *constraint_linked_entity_owners_field_ptr_;
  mundy::linkers::LinkedEntitiesFieldType *constraint_linked_entities_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *spheres_part_ptr_ = nullptr;
  stk::mesh::Part *e_part_ptr_ = nullptr;
  stk::mesh::Part *h_part_ptr_ = nullptr;
  stk::mesh::Part *bs_part_ptr_ = nullptr;

  stk::mesh::Part *hp1_part_ptr_ = nullptr;
  stk::mesh::Part *left_hp1_part_ptr_ = nullptr;
  stk::mesh::Part *doubly_hp1_h_part_ptr_ = nullptr;
  stk::mesh::Part *doubly_hp1_bs_part_ptr_ = nullptr;

  stk::mesh::Part *backbone_segments_part_ptr_ = nullptr;
  stk::mesh::Part *ee_springs_part_ptr_ = nullptr;
  stk::mesh::Part *eh_springs_part_ptr_ = nullptr;
  stk::mesh::Part *hh_springs_part_ptr_ = nullptr;

  stk::mesh::Part *backbone_backbone_neighbor_genx_part_ptr_ = nullptr;
  stk::mesh::Part *hp1_h_neighbor_genx_part_ptr_ = nullptr;
  stk::mesh::Part *hp1_bs_neighbor_genx_part_ptr_ = nullptr;

  //@}

  //! \name MetaMethod instances
  //@{

  std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>> declare_and_init_constraints_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_aabb_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_constraint_forcing_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_ssd_and_cn_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> evaluate_linker_potentials_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> linker_potential_force_reduction_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> destroy_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> destroy_bound_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> generate_scs_scs_genx_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> generate_hp1_h_genx_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> generate_hp1_bs_genx_ptr_;

  //@}

  //! \name Fixed params for MetaMethods
  //@{

  Teuchos::ParameterList compute_constraint_forcing_fixed_params_;
  Teuchos::ParameterList compute_ssd_and_cn_fixed_params_;
  Teuchos::ParameterList compute_aabb_fixed_params_;
  Teuchos::ParameterList generate_scs_scs_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList generate_hp1_h_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList generate_hp1_bs_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList evaluate_linker_potentials_fixed_params_;
  Teuchos::ParameterList linker_potential_force_reduction_fixed_params_;
  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList destroy_bound_neighbor_linkers_fixed_params_;

  //@}

  //! \name Partitioning settings
  //@{

  class RcbSettings : public stk::balance::BalanceSettings {
   public:
    RcbSettings() {
    }
    virtual ~RcbSettings() {
    }

    virtual bool isIncrementalRebalance() const {
      return false;
    }
    virtual std::string getDecompMethod() const {
      return std::string("rcb");
    }
    virtual std::string getCoordinateFieldName() const {
      return std::string("NODE_COORDS");
    }
    virtual bool shouldPrintMetrics() const {
      return false;
    }
  };  // RcbSettings

  RcbSettings balance_settings_;
  //@}

  //! \name User parameters
  //@{

  // Simulation params
  bool initial_loadbalance_ = false;
  bool do_equilibrate_ = false;
  size_t num_time_steps_ = 100;
  size_t num_time_steps_equilibrate_ = 10000;
  size_t io_frequency_ = 10;
  size_t log_frequency_ = 10;
  double timestep_size_ = 0.001;
  double kt_brownian_ = 1.0;
  double kt_kmc_ = 1.0;
  double viscosity_ = 1.0;
  std::string initialization_type_ = "grid";

  // Unit cell/periodicity params
  double unit_cell_length_ = 10.0;

  // Chromatin params
  size_t num_chromosomes_ = 1;
  size_t num_chromatin_repeats_ = 2;
  size_t num_euchromatin_per_repeat_ = 1;
  size_t num_heterochromatin_per_repeat_ = 1;
  double backbone_excluded_volume_radius_ = 0.5;
  double sphere_hydrodynamic_radius_ = 0.05;
  double initial_sphere_separation_ = 1.0;
  double backbone_youngs_modulus_ = 1000.0;
  double backbone_poissons_ratio_ = 0.3;
  double sphere_drag_coeff_ = 1.0;

  // Chromatin spring params
  BOND_TYPE chromatin_spring_type_ = BOND_TYPE::HARMONIC;
  double chromatin_spring_constant_ = 100.0;
  double chromatin_spring_rest_length_ = 1.0;

  // Crosslinker params
  BOND_TYPE crosslinker_spring_type_ = BOND_TYPE::HARMONIC;
  double crosslinker_spring_constant_ = 10.0;
  double crosslinker_rest_length_ = 2.5;
  double crosslinker_rcut_ = 1.0;
  double crosslinker_left_binding_rate_ = 1.0;
  double crosslinker_right_binding_rate_ = 1.0;
  double crosslinker_left_unbinding_rate_ = 1.0;
  double crosslinker_right_unbinding_rate_ = 1.0;

  // Periphery params
  double hydrodynamic_periphery_radius_ = 5.0;
  double collision_periphery_radius_ = 5.0;
  double collision_periphery_radius_start_ = 5.0;
  double collision_periphery_radius_current_ = 0.0;
  double periphery_spring_constant_ = 1000.0;
  size_t periphery_spectral_order_ = 32;

  // Flags for simulation control
  // Neighbor list
  double skin_distance_ = 1.0;
  double skin_distance2_over4_ = 1.0;
  bool update_neighbor_list_ = false;
  bool force_neighborlist_update_ = false;
  size_t force_neighborlist_update_nsteps_ = 10;

  // Neighborlist rebuild information
  size_t last_neighborlist_update_step_ = 0;
  Kokkos::Timer neighborlist_update_timer_;

  // [timestep, elapsed_timesteps, elapsed_time]
  std::vector<std::tuple<size_t, size_t, double>> neighborlist_update_steps_times_;
  bool print_neighborlist_statistics_ = false;

  //@}
};  // class HP1

}  // namespace crosslinkers

}  // namespace alens

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  mundy::alens::crosslinkers::HP1().run(argc, argv);

  // Before exiting, sleep for some amount of time to force Kokkos to print better at the end.
  std::this_thread::sleep_for(std::chrono::milliseconds(stk::parallel_machine_rank(MPI_COMM_WORLD)));

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

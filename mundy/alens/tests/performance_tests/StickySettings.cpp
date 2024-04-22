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

/* Notes:

Our goal is to simulate a chain of N Brownian diffusing spheres in a 3D domain. Sphere-sphere collision will be modeled
via Hertzian contact. Neighboring spheres in the chain will be connected by Hookean springs. Each sphere will be
attached to a crosslinker that can bind to (and unbind from) other nearby spheres. The crosslinker is also modeled as a
hookean spring, although it will only induce force when doubly bound.


Crosslinkers have the following states: unbound, left bound, right bound, and doubly bound.
Associated with these states are the following rates: left/right binding/unbinding rates.

Given the current state of the crosslinker, we have to use these rates to decide which state the crosslinker will
transition to next.

              0------------------------------------Normalized probability-----------------------------------------1
Unbound:      |--prob of becoming left bound--|--prob of becoming right bound--|--prob of remaining unbound-------|
Left bound:   |--prob of unbinding left node--|--prob of binding right node----|--prob of remaining left bound----|
Right bound:  |--prob of binding left node----|--prob of unbinding right node--|--prob of remaining right bound---|
Doubly bound: |--prob of unbinding left node--|--prob of unbinding right node--|--prob of remaining doubly bound--|

Parts: spheres, spherocylinders, hookean_springs, crosslinkers, doubly_bound_crosslinkers left_bound_crosslinkers,
right_bound_crosslinkers, unbound_crosslinkers doubly_bound_crosslinkers, left_bound_crosslinkers,
right_bound_crosslinkers, and unbound_crosslinkers will be subparts of crosslinkers, which is, in turn a subpart of
hookean_springs. For now, we add crosslinkers as a subpart of spherocylinders. This is a temporary workaround until we
implement a planned refactor of the agent hierarchy and merely serves to give our crosslinkers access to shape-based
methods. The element radius should be the crosslinker search radius. In the long term, we'll still use the same
shape-based methods but the crosslinker radius will be stored in SEARCH_RADIUS and the crosslinkers won't need to be a
subpart of spherocylinders.


Nearby spheres are connected by sphere-sphere neighbor linkers.
Adjacent spheres in the chain are connected by hookean springs.
Attached to each sphere is
               __________________________________________
              /    ____________________________________  \
             /    /                                    \  \
CONSTRAINT  |    |  neighbor_linker1          ___neighbor_linker2___
            |    | /  |    |    |   \        /           |           \
ELEMENT     | sphere1 | spring1 | sphere2   / left_bound_crosslinker1 \
            |    |    |   / \   |    |     / /                       \ \
FACE        |    |    |  /   \  |    |    / /                         \ \
            |    |    | /     \ |    |   / /                           \ \
EDGE         \   |   / /       \ \   |  / /                             \ \
              \  |  / /         \ \  | / /                               \ \
NODE           node1               node2                                 node3

      |
      x
      |
o-----o------o

NeighborLinkers are designed such that the lower rank connections (nodes, edges, faces) of neighbors are shared between
processes. This means that a process that has access to one of the spheres will have (at the very least) ghosted access
to the neighboring spheres and shared access to their nodes, making it perfectly valid for a crosslinker to fetch the
constraint rank entities attached to its node... Using the node to fetch the neighbor linker is nice but it does lose
information about connectivity order. This information can be determined


If a crosslinker is doubly bound, how do we decide which of the two sides to unbind? They both unbind at independent
rates.

If a crosslinker completely unbinds from both nodes, it will be deleted, as all entities must connect to nodes.

If some crosslinkers are to float around in space without being connected to the nodes of another element, then we
should replace the lost nodes of a crosslinker when they unbind. This, in turn, means that we cannot rely on the lack of
a connected node to determine the bound state of a crosslinker. Instead, we can break the singly bound crosslinkers part
into left bound and right bound. This makes logical sense, as these two types of crosslinkers have distinct behavior.

Can a crosslinker attached to a sphere along the backbone attach to one of the adjacent spheres?

Crosslinkers needs to store a rate for which their left and right nodes bind and unbind. This can differ from
crosslinker-to-crosslinker. The crosslinkers part itself con't be used to unbind the left/right nodes, since the
existence of connected nodes doesn't indicate that the crosslinker is bound to another object at that point. We won't,
however, restrict the left/right unbind rate to the respective singly bound parts, as external factors may change these
rates independent of the current bind status.

Unbinding the left node of a crosslinker can be done by looping over all left bound and doubly bound crosslinkers. The
same works for right nodes. Binding is harder. I don't think the correct solution is to rely on the
sphere-sphere neighbor list to determine if a singly bound crosslinker attached to a sphere binds to one of the
neighbors of said sphere. Separation of concerns seems to demand that we introduce crosslinker sphere neighbor linkers.
Actually, without crosslinker sphere linkers how would we handle the binding of unbound crosslinkers?


What is the easiest way to exclude certain types of neighbors? If a crosslinker is connected to a sphere, delete the
linker from the crosslinker to that sphere. If a crosslinker connected to a sphere, delete the linker from the
crosslinker to the adjacent spheres.





We need to modify our existing spring constraint forcing functions to account for the possibility that some springs will
lack some lower rank entities. Basically, if num connected nodes is less than the max amount, continue.

Order of operations:

// Preprocess
- Parse user parameter

// Setup
- Declare the fixed and mutable params for the desired MetaMethods
- Construct the mesh and the method instances
- Declare and initialize the chain of spheres, their connecting springs, and the crosslinkers along their backbone

// Timeloop
- Run the timeloop for t in range(0, T):
    // IO.
    - If desired, write out the data

    // Setup the current configuration. Think of this step as computing values for t + dt
    - Zero the node forces and velocities

    // Neighbor detection sphere-sphere
    - Check if the sphere-sphere neighbor list needs updated or not
        - Compute the AABB for the spheres and the neighbor linkers
        - Delete sphere-sphere and crosslinker-sphere neighbor linkers that are too far apart
        - Generate neighbor linkers between nearby spheres
        - Compute the signed separation distance and contact normal between neighboring spheres

    // Neighbor detection sphere-crosslinker

    // Pairwise potentials
    - Evaluate the Hertzian contact potential between neighboring spheres
    - Sum the linker potential force magnitude to get the induced node force on each sphere

    // Motion
    - Compute the velocity induced by the node forces using local drag
    - Compute the brownian velocity for the nodes
    - Apply velocity/position constraints like no motion for particle 1
    - Update the node positions using a first order Euler method
*/

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>   // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                   // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                  // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>                // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>                 // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceMagnitudeReduction.hpp>  // for mundy::linkers::LinkerPotentialForceMagnitudeReduction
#include <mundy_linkers/NeighborLinkers.hpp>                         // for mundy::linkers::NeighborLinkers
#include <mundy_mesh/BulkData.hpp>                                   // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                                   // for mundy::mesh::MetaData
#include <mundy_mesh/utils/FillFieldWithValue.hpp>                   // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaFactory.hpp>                                // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                                 // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                       // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

#define DEBUG

///////////////////////////
// StickySettings        //
///////////////////////////
class StickySettings {
 public:
  StickySettings() = default;

  void print_rank0(auto thing_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << thing_to_print << std::endl;
    }
  }

  void debug_print([[maybe_unused]] auto thing_to_print, [[maybe_unused]] int indent_level = 0) {
#ifdef DEBUG
    print_rank0(thing_to_print, indent_level);
#endif
  }

  void parse_user_inputs(int argc, char **argv) {
    debug_print("Parsing user inputs.");

    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, false);

    // If we should accept the parameters directly from the command line or from a file
    bool use_input_file = false;
    cmdp.setOption("use_input_file", "no_use_input_file", &use_input_file, "Use an input file.");
    bool use_input_file_found = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_ASSERT(use_input_file_found, std::invalid_argument, "Failed to parse the command line arguments.");

    // Switch to requiring that all options must be recognized.
    cmdp.recogniseAllOptions(true);

    if (!use_input_file) {
      // Parse the command line options.

      //   Backbone initialization:
      cmdp.setOption("num_spheres", &num_spheres_, "Number of spheres in backbone.");
      cmdp.setOption("sphere_radius", &sphere_radius_, "Backbone sphere radius.");
      cmdp.setOption("sphere_youngs_modulus", &sphere_youngs_modulus_, "Backbone sphere Youngs modulus.");
      cmdp.setOption("sphere_poissons_ratio", &sphere_poissons_ratio_, "Backbone sphere poissons ratio.");
      cmdp.setOption("sphere_drag_coeff", &sphere_drag_coeff_, "Backbone sphere drag coefficient.");

      //   Backbone spring:
      cmdp.setOption("backbone_spring_constant", &backbone_spring_constant_, "Backbone spring constant.");
      cmdp.setOption("backbone_spring_rest_length", &backbone_spring_rest_length_, "Backbone rest length.");

      //   Crosslinker params:
      //   cmdp.setOption("crosslinker_spring_constant", &crosslinker_spring_constant_, "Crosslinker spring constant.");
      //   cmdp.setOption("crosslinker_spring_rest_length", &crosslinker_spring_rest_length_,
      //                  "Crosslinker spring rest length.");
      //   cmdp.setOption("crosslinker_left_binding_rate", &crosslinker_left_binding_rate_,
      //                  "Crosslinkler left binding rate.");
      //   cmdp.setOption("crosslinker_right_binding_rate", &crosslinker_right_binding_rate_,
      //                  "Crosslinker right binding rate.");
      //   cmdp.setOption("crosslinker_left_unbinding_rate", &crosslinker_left_unbinding_rate_,
      //                  "Crosslinkler left unbinding rate.");
      //   cmdp.setOption("crosslinker_right_unbinding_rate", &crosslinker_right_unbinding_rate_,
      //                  "Crosslinker right unbinding rate.");

      //   The simulation:
      cmdp.setOption("num_time_steps", &num_time_steps_, "Number of time steps.");
      cmdp.setOption("timestep_size", &timestep_size_, "Time step size.");
      cmdp.setOption("kt", &kt_, "Temperature kT.");
      cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");

      bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
      MUNDY_THROW_ASSERT(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");
    } else {
      cmdp.setOption("input_file", &input_file_name_, "The name of the input file.");
      bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
      MUNDY_THROW_ASSERT(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");

      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Should not be reading in from YAML yet.");

      //   // Read in the parameters from the parameter list.
      //   Teuchos::ParameterList param_list_ = *Teuchos::getParametersFromYamlFile(input_file_name_);

      //   num_sperm_ = param_list_.get<int>("num_sperm");
      //   num_nodes_per_sperm_ = param_list_.get<int>("num_nodes_per_sperm");
      //   sperm_radius_ = param_list_.get<double>("sperm_radius");
      //   sperm_initial_segment_length_ = param_list_.get<double>("sperm_initial_segment_length");
      //   sperm_rest_segment_length_ = param_list_.get<double>("sperm_rest_segment_length");
      //   sperm_rest_curvature_twist_ = param_list_.get<double>("sperm_rest_curvature_twist");
      //   sperm_rest_curvature_bend1_ = param_list_.get<double>("sperm_rest_curvature_bend1");
      //   sperm_rest_curvature_bend2_ = param_list_.get<double>("sperm_rest_curvature_bend2");

      //   sperm_density_ = param_list_.get<double>("sperm_density");
      //   sperm_youngs_modulus_ = param_list_.get<double>("sperm_youngs_modulus");
      //   sperm_poissons_ratio_ = param_list_.get<double>("sperm_poissons_ratio");
      //   sperm_poissons_ratio_ = param_list_.get<double>("sperm_poissons_ratio");

      //   num_time_steps_ = param_list_.get<int>("num_time_steps");
      //   timestep_size_ = param_list_.get<double>("timestep_size");
    }

    check_input_parameters();
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_ASSERT(num_spheres_ > 0, std::invalid_argument, "num_spheres_ must be greater than 0.");
    MUNDY_THROW_ASSERT(sphere_radius_ > 0, std::invalid_argument, "sphere_radius_ must be greater than 0.");

    MUNDY_THROW_ASSERT(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_ASSERT(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_ASSERT(io_frequency_ > 0, std::invalid_argument, "io_frequency_ must be greater than 0.");
  }

  void dump_user_inputs() {
    debug_print("Dumping user inputs.");
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps: " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size: " << timestep_size_ << std::endl;
      std::cout << "  io_frequency: " << io_frequency_ << std::endl;
      std::cout << "  kT: " << kt_ << std::endl;
      std::cout << "BACKBONE SPHERES:" << std::endl;
      std::cout << "  num_spheres: " << num_spheres_ << std::endl;
      std::cout << "  sphere_radius: " << sphere_radius_ << std::endl;
      std::cout << "  youngs_modulus: " << sphere_youngs_modulus_ << std::endl;
      std::cout << "  poissons_ratio: " << sphere_poissons_ratio_ << std::endl;
      std::cout << "  drag_coeff: " << sphere_drag_coeff_ << std::endl;
      std::cout << "BACKBONE SPRINGS:" << std::endl;
      std::cout << "  backbone_spring_constant: " << backbone_spring_constant_ << std::endl;
      std::cout << "  backbone_spring_rest_length: " << backbone_spring_rest_length_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  void build_our_mesh_and_method_instances() {
    debug_print("Building our mesh and method instances.");

    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshRequirements>(MPI_COMM_WORLD);
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and
    // allow us to extend the functionality offered natively by Mundy.
    //
    // We add the following methods to act on the spheres agent. We directly apply these requirements to all spheres.
    //   1. A node euler timestep method for the nonorientable spheres.
    //     Requirements: NODE_VELOCITY
    //   2. A method to compute the mobility of the nonorientable spheres. In this case, we use a local drag method.
    //     Requirements: NODE_FORCE, NODE_VELOCITY
    //   3. A method to compute the brownian velocity of the nonorientable spheres.
    //     Requirements: NODE_VELOCITY, NODE_RNG_COUNTER
    auto custom_sphere_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    custom_sphere_part_reqs
        ->add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 1)
        // Add the node fields
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<unsigned>("NODE_RNG_COUNTER", node_rank_, 1, 1);
    // Add to the spheres part
    mundy::shapes::Spheres::add_part_reqs(custom_sphere_part_reqs);
    mesh_reqs_ptr_->merge(mundy::shapes::Spheres::get_mesh_requirements());

    // We add the following methods to act on the crosslinkers agent. We directly apply these requirements to all
    // crosslinkers. These methods use multi-staged approaches to reduce branching and improve performance.
    //   1. A method for computing the unbinding probability of left/right bound crosslinkers.
    //     Requirements: ELEMENT_UNBIND_RATES (size 2), ELEMENT_UNBINDING_PROBABILITY (size 2)
    //   2. A method for computing the binding probability (z-score?) for each crosslinker-sphere pair.
    //     Requirements: CONSTRAINT_BINDING_PROBABILITY, ELEMENT_BINDING_RATES (size 2)
    //   3. A method for determining unbound-to-singly bound transitions,
    //      another for singly bound-to-doubly bound transitions, and
    //      another for doubly bound-to-singly bound transitions.
    //     Requirements: CONSTRAINT_BINDING_PROBABILITY (on each crosslinker sphere linker),
    //                   CONSTRAINT_PERFORM_BINDING (on each crosslinker sphere linker type bool),
    //                   ELEMENT_UNBINDING_PROBABILITY (size 2),
    //                   ELEMENT_PERFORM_UNBINDING (size 2 type bool)
    //   4. A single method for performing staged state changes within a modification cycle.
    //    Requirements: ELEMENT_CROSSLINKER_STATE_CHANGE (int cast to from state change enum)
    //   5. A method for updating the center node of the spring. If doubly bound, the center node is the midpoint of the
    //      two bound nodes. If singly bound, the center node is the bound node. If unbound, the center node is owned by
    //      the crosslinker and will diffuse randomly in space.
    auto custom_crosslinkers_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    custom_crosslinkers_part_reqs->set_part_name("CROSSLINKERS")
        .set_part_topology(stk::topology::BEAM_2)
        // Add element fields
        .add_field_reqs<double>("ELEMENT_UNBIND_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_BINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_UNBINDING_PROBABILITY", element_rank_, 2, 1)
        .add_field_reqs<int>("ELEMENT_CROSSLINKER_STATE_CHANGE", element_rank_, 1, 1);
    // Add the subpart requirements to spherocylindersegments
    mundy::shapes::SpherocylinderSegments::add_subpart_reqs(custom_crosslinkers_part_reqs);
    mesh_reqs_ptr_->merge(mundy::shapes::SpherocylinderSegments::get_mesh_requirements());

    // Create the linkers/interactions between crosslinkers and spheres
    auto custom_crosslinker_sphere_linkers_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    custom_crosslinker_sphere_linkers_part_reqs->set_part_name("CROSSLINKER_SPHERE_LINKERS")
        .set_part_rank(constraint_rank_)
        // Constraint fields
        .add_field_reqs<double>("CONSTRAINT_BINDING_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<double>("CONSTRAINT_PERFORM_BINDING", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_subpart_reqs(custom_crosslinker_sphere_linkers_part_reqs);
    mesh_reqs_ptr_->merge(mundy::linkers::NeighborLinkers::get_mesh_requirements());

    // ComputeConstraintForcing fixed parameters
    auto compute_constraint_forcing_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"));
    compute_constraint_forcing_fixed_params.sublist("HOOKEAN_SPRINGS")
        .set("valid_entity_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"));
    mesh_reqs_ptr_->merge(
        mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params));

    // ComputeSignedSeparationDistanceAndContactNormal fixed parameters
    auto compute_ssd_and_cn_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKER"));
    mesh_reqs_ptr_->merge(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params));

    // ComputeAABB fixed parameters
    auto compute_aabb_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE", "SPHEROCYLINDER_SEGMENT"));
    compute_aabb_fixed_params.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("CROSSLINKERS"));
    mesh_reqs_ptr_->merge(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params));

    // GenerateNeighborLinkers fixed parameters
    // First, the parameters for generating the sphere-sphere neighbor linkers
    auto generate_sphere_sphere_neighbor_linkers_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"));
    generate_sphere_sphere_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHERES"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));
    mesh_reqs_ptr_->merge(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_sphere_sphere_neighbor_linkers_fixed_params));

    // Next, the parameters for generating the crosslinker-sphere neighbor linkers
    auto generate_crosslinker_sphere_neighbor_linkers_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("CROSSLINKER_SPHERE_LINKERS"));
    generate_crosslinker_sphere_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("CROSSLINKERS")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));
    mesh_reqs_ptr_->merge(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_crosslinker_sphere_neighbor_linkers_fixed_params));

    // EvaluateLinkerPotentials fixed parameters
    auto evaluate_linker_potentials_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT"));
    mesh_reqs_ptr_->merge(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params));

    // LinkerPotentialForceMagnitudeReduction fixed parameters
    auto linker_potential_force_magnitude_reduction_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
            .set("name_of_linker_part_to_reduce_over", "SPHERE_SPHERE_LINKERS");
    mesh_reqs_ptr_->merge(mundy::linkers::LinkerPotentialForceMagnitudeReduction::get_mesh_requirements(
        linker_potential_force_magnitude_reduction_fixed_params));

    // DestroyNeighborLinkers fixed parameters
    auto destroy_neighbor_linkers_fixed_params =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    destroy_neighbor_linkers_fixed_params.sublist("DESTROY_DISTANT_NEIGHBORS")
        .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
        .set("valid_connected_source_and_target_part_names",
             mundy::core::make_string_array(std::string("SPHERES"), std::string("CROSSLINKERS")));
    mesh_reqs_ptr_->merge(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params));

    // DeclareAndInitConstraints fixed parameters
    auto declare_and_init_constraints_fixed_params =
        Teuchos::ParameterList().set("enabled_technique_name", "CHAIN_OF_SPRINGS");
    declare_and_init_constraints_fixed_params.sublist("CHAIN_OF_SPRINGS")
        .set("hookean_springs_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"))
        .set("sphere_part_names", mundy::core::make_string_array("SPHERES"))
        .set<bool>("generate_hookean_springs", true)
        .set<bool>("generate_spheres_at_nodes", true);
    mesh_reqs_ptr_->merge(mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(
        declare_and_init_constraints_fixed_params));

    // Print the mesh requirements out
    mesh_reqs_ptr_->print_reqs();

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDINATES");
    meta_data_ptr_->commit();

    debug_print("Mesh contents after commit.");
#ifdef DEBUG
    // Dump the mesh info as it exists now (with fields)
    stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
#endif
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
    debug_print("Fetching fields and parts.");

    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDINATES", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_rng_field_ptr_ = fetch_field<unsigned>("NODE_RNG_COUNTER", node_rank_);
    // node_acceleration_field_ptr_ = fetch_field<double>("NODE_ACCELERATION", node_rank_);
    // node_twist_field_ptr_ = fetch_field<double>("NODE_TWIST", node_rank_);
    // node_twist_velocity_field_ptr_ = fetch_field<double>("NODE_TWIST_VELOCITY", node_rank_);
    // node_twist_torque_field_ptr_ = fetch_field<double>("NODE_TWIST_TORQUE", node_rank_);
    // node_twist_acceleration_field_ptr_ = fetch_field<double>("NODE_TWIST_ACCELERATION", node_rank_);
    // node_curvature_field_ptr_ = fetch_field<double>("NODE_CURVATURE", node_rank_);
    // node_rest_curvature_field_ptr_ = fetch_field<double>("NODE_REST_CURVATURE", node_rank_);
    // node_rotation_gradient_field_ptr_ = fetch_field<double>("NODE_ROTATION_GRADIENT", node_rank_);
    // node_radius_field_ptr_ = fetch_field<double>("NODE_RADIUS", node_rank_);
    // node_archlength_field_ptr_ = fetch_field<double>("NODE_ARCHLENGTH", node_rank_);

    // edge_orientation_field_ptr_ = fetch_field<double>("EDGE_ORIENTATION", edge_rank_);
    // edge_tangent_field_ptr_ = fetch_field<double>("EDGE_TANGENT", edge_rank_);
    // edge_binormal_field_ptr_ = fetch_field<double>("EDGE_BINORMAL", edge_rank_);
    // edge_length_field_ptr_ = fetch_field<double>("EDGE_LENGTH", edge_rank_);

    // element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    // element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    // element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    // element_rest_length_field_ptr_ = fetch_field<double>("ELEMENT_REST_LENGTH", element_rank_);

    // Fetch the parts
    // centerline_twist_springs_part_ptr_ = fetch_part("CENTERLINE_TWIST_SPRINGS");
    // MUNDY_THROW_ASSERT(centerline_twist_springs_part_ptr_->topology() == stk::topology::SHELL_TRI_3,
    // std::logic_error,
    //                    "CENTERLINE_TWIST_SPRINGS part must have SHELL_TRI_3 topology.");
  }

  void run(int argc, char **argv) {
    debug_print("Running the simulation.");

    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    build_our_mesh_and_method_instances();
    fetch_fields_and_parts();
    // declare_and_initialize_sperm();
    // setup_io();

    // // Time loop
    // print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    // Kokkos::Timer timer;
    // for (size_t i = 0; i < num_time_steps_; i++) {
    //   debug_print(std::string("Time step ") + std::to_string(i) + " of " + std::to_string(num_time_steps_));

    //   // IO. If desired, write out the data for time t.
    //   if (i % io_frequency_ == 0) {
    //     stk_io_broker_.begin_output_step(output_file_index_, static_cast<double>(i));
    //     stk_io_broker_.write_defined_output_fields(output_file_index_);
    //     stk_io_broker_.end_output_step(output_file_index_);
    //     stk_io_broker_.flush_output();
    //   }

    //   // Prepare the current configuration.
    //   {
    //     // Rotate the field states.
    //     rotate_field_states();

    //     // Motion from t -> t + dt.
    //     // x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2.
    //     update_position_and_twist();

    //     // Zero the node velocities, accelerations, and forces/torques for time t + dt.
    //     zero_out_transient_node_fields();
    //   }

    //   // Evaluate forces f(x(t + dt)).
    //   {
    //     // Centerline twist rod model
    //     compute_centerline_twist_force_and_torque();
    //   }

    //   // Compute velocity and acceleration.
    //   {
    //     // Evaluate a(t + dt) = M^{-1} f(x(t + dt)).
    //     // Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2.
    //     compute_velocity_and_acceleration();
    //   }
    // }

    // // Do a synchronize to force everybody to stop here, then write the time
    // stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    // if (bulk_data_ptr_->parallel_rank() == 0) {
    //   double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
    //   std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    // }
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
  std::shared_ptr<mundy::meta::MeshRequirements> mesh_reqs_ptr_;
  stk::io::StkMeshIoBroker stk_io_broker_;
  size_t output_file_index_;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<unsigned> *node_rng_field_ptr_;

  stk::mesh::Field<double> *node_coord_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;

  //@}

  //! \name Parts
  //@{

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
      return std::string("NODE_COORDINATES");
    }
    virtual bool shouldPrintMetrics() const {
      return false;
    }
  };  // RcbSettings

  RcbSettings balance_settings_;
  //@}

  //! \name User parameters
  //@{
  std::string input_file_name_ = "input.yaml";

  // Sphere params
  size_t num_spheres_ = 10;
  double sphere_radius_ = 0.5;
  double sphere_youngs_modulus_ = 1000.0;
  double sphere_poissons_ratio_ = 0.3;
  double sphere_drag_coeff_ = 1.0;

  // Backbone spring params
  double backbone_spring_constant_ = 100.0;
  double backbone_spring_rest_length_ = 1.0;

  // Crosslinker params
  double crosslinker_spring_constant_ = 100.0;
  double crosslinker_spring_rest_length_ = 1.0;
  double crosslinker_left_binding_rate_ = 1.0;
  double crosslinker_right_binding_rate_ = 1.0;
  double crosslinker_left_unbinding_rate_ = 1.0;
  double crosslinker_right_unbinding_rate_ = 1.0;

  // Simulation params
  size_t num_time_steps_ = 100;
  double timestep_size_ = 0.01;
  double kt_ = 1.0;
  size_t io_frequency_ = 10;
  //@}
};

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  StickySettings().run(argc, argv);

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

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
#include <stk_mesh/base/Comm.hpp>            // for stk::mesh::comm_mesh_counts
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
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

// #define DEBUG

///////////////////////////
// StickySettings        //
///////////////////////////
class StickySettings {
 public:
  enum BINDING_STATE_CHANGE : unsigned { NONE = 0u, LEFT_TO_DOUBLY, RIGHT_TO_DOUBLY };

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

  //  // Blocking debug print with mpi. This will also dump the mesh
  //  void debug_print_state_mpi([[maybe_unused]] auto thing_to_print, [[maybe_unused]] int indent_level = 0) {
  // #ifdef DEBUG
  //    // Create the blocking call to MPI
  //    int trap_key = 0;
  //    auto mrank = stk::parallel_machine_rank(MPI_COMM_WORLD);
  //    auto msize = stk::parallel_machine_size(MPI_COMM_WORLD);
  //    MPI_Status mstatus;
  //    // This should allow only rank 0 to escape through for now...
  //    if (mrank != trap_key) {
  //      MPI_Recv(&trap_key, 1, MPI_INT, mrank - 1, 0, MPI_COMM_WORLD, &mstatus);
  //    }
  //
  //    // This should be a critical section
  //    {
  //      std::cout << "I am process " << mrank << " of " << msize << std::endl;
  //      stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
  //    }
  //
  //    // This is how we get out of this
  //    if (mrank != msize - 1) {
  //      MPI_Send(&trap_key, 1, MPI_INT, mrank + 1, 0, MPI_COMM_WORLD);
  //    }
  // #endif
  //  }

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
      cmdp.setOption("initial_sphere_separation", &initial_sphere_separation_, "Initial backbone sphere separation.");
      cmdp.setOption("sphere_youngs_modulus", &sphere_youngs_modulus_, "Backbone sphere Youngs modulus.");
      cmdp.setOption("sphere_poissons_ratio", &sphere_poissons_ratio_, "Backbone sphere poissons ratio.");
      cmdp.setOption("sphere_drag_coeff", &sphere_drag_coeff_, "Backbone sphere drag coefficient.");

      //   Backbone spring:
      cmdp.setOption("backbone_spring_constant", &backbone_spring_constant_, "Backbone spring constant.");
      cmdp.setOption("backbone_spring_rest_length", &backbone_spring_rest_length_, "Backbone rest length.");

      //   Crosslinker spring:
      cmdp.setOption("crosslinker_spring_constant", &crosslinker_spring_constant_, "Crosslinker spring constant.");
      cmdp.setOption("crosslinker_rest_length", &crosslinker_rest_length_, "Crosslinker rest length.");

      //   The simulation:
      cmdp.setOption("num_time_steps", &num_time_steps_, "Number of time steps.");
      cmdp.setOption("timestep_size", &timestep_size_, "Time step size.");
      cmdp.setOption("kt", &kt_, "Temperature kT.");
      cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");
      cmdp.setOption("initial_loadbalance", "no_initial_loadbalance", &initial_loadbalance_, "Initial loadbalance.");

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
      std::cout << "" << std::endl;

      std::cout << "BACKBONE SPHERES:" << std::endl;
      std::cout << "  num_spheres: " << num_spheres_ << std::endl;
      std::cout << "  sphere_radius: " << sphere_radius_ << std::endl;
      std::cout << "  initial_sphere_separation: " << initial_sphere_separation_ << std::endl;
      std::cout << "  youngs_modulus: " << sphere_youngs_modulus_ << std::endl;
      std::cout << "  poissons_ratio: " << sphere_poissons_ratio_ << std::endl;
      std::cout << "  drag_coeff: " << sphere_drag_coeff_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "BACKBONE SPRINGS:" << std::endl;
      std::cout << "  backbone_spring_constant: " << backbone_spring_constant_ << std::endl;
      std::cout << "  backbone_spring_rest_length: " << backbone_spring_rest_length_ << std::endl;
      std::cout << "" << std::endl;

      std::cout << "CROSSLINKER SPRINGS:" << std::endl;
      std::cout << "  crosslinker_spring_constant: " << crosslinker_spring_constant_ << std::endl;
      std::cout << "  crosslinker_rest_length: " << crosslinker_rest_length_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  void build_our_mesh_and_method_instances() {
    debug_print("Building our mesh and method instances.");

    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
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
    auto custom_sphere_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_part_reqs
        ->add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 1)
        // Add the node fields
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<unsigned>("NODE_RNG_COUNTER", node_rank_, 1, 1);

    // Add to the spheres part
    mundy::shapes::Spheres::add_and_sync_part_reqs(custom_sphere_part_reqs);
    mesh_reqs_ptr_->sync(mundy::shapes::Spheres::get_mesh_requirements());

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
    auto custom_crosslinkers_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_crosslinkers_part_reqs->set_part_name("CROSSLINKERS")
        .set_part_topology(stk::topology::BEAM_2)
        // Add element fields
        .add_field_reqs<double>("ELEMENT_UNBIND_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_BINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_UNBINDING_PROBABILITY", element_rank_, 2, 1)
        .add_field_reqs<int>("ELEMENT_CROSSLINKER_STATE_CHANGE", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("ELEMENT_RNG_COUNTER", element_rank_, 1, 1)
        // Add subparts for left and right bound crosslinkers
        .add_subpart_reqs("LEFT_BOUND_CROSSLINKERS", stk::topology::BEAM_2)
        .add_subpart_reqs("RIGHT_BOUND_CROSSLINKERS", stk::topology::BEAM_2)
        .add_subpart_reqs("DOUBLY_BOUND_CROSSLINKERS", stk::topology::BEAM_2);

    // Add this part to the mesh directly
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_crosslinkers_part_reqs);

    // Create the generalized interaction entities that connect crosslinkers and spheres
    //   This entity "knows" how to compute the binding probability between a crosslinker and a sphere and how to
    //   perform binding between a crosslinker and a sphere. It is a constraint rank entitiy because itc must connect element rank entities.
    auto custom_crosslinker_sphere_linkers_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_crosslinker_sphere_linkers_part_reqs->set_part_name("CROSSLINKER_SPHERE_LINKERS")
        .set_part_rank(constraint_rank_)
        // Constraint fields
        .add_field_reqs<double>("CONSTRAINT_BINDING_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<unsigned>("CONSTRAINT_PERFORM_BINDING", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_and_sync_subpart_reqs(custom_crosslinker_sphere_linkers_part_reqs);
    mesh_reqs_ptr_->sync(mundy::linkers::NeighborLinkers::get_mesh_requirements());

    // Setup our fixed parameters for any of methods that we intend to use
    // When we eventually switch to the configurator, these individual fixed params will become sublists within a single
    // master parameter list. Note, sublist will return a reference to the sublist with the given name.
    compute_constraint_forcing_fixed_params_ =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"));
    compute_constraint_forcing_fixed_params_.sublist("HOOKEAN_SPRINGS")
        .set("valid_entity_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS", "CROSSLINKERS"));

    compute_ssd_and_cn_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names",
        mundy::core::make_string_array("SPHERE_SPHERE_LINKER", "SPHERE_SPHEROCYLINDER_SEGMENT_LINKER"));
    
    compute_aabb_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE", "SPHEROCYLINDER_SEGMENT"));
    compute_aabb_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("CROSSLINKERS"));

    generate_sphere_sphere_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"));
    generate_sphere_sphere_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHERES"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));

    generate_crosslinker_sphere_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("CROSSLINKER_SPHERE_LINKERS"));
    generate_crosslinker_sphere_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("CROSSLINKERS")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));

    evaluate_linker_potentials_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT"));

    linker_potential_force_magnitude_reduction_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
            .set("name_of_linker_part_to_reduce_over", "SPHERE_SPHERE_LINKERS");

    destroy_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    destroy_neighbor_linkers_fixed_params_.sublist("DESTROY_DISTANT_NEIGHBORS")
        .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
        .set("valid_connected_source_and_target_part_names",
             mundy::core::make_string_array(std::string("SPHERES"), std::string("CROSSLINKERS")));

    declare_and_init_constraints_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "CHAIN_OF_SPRINGS");
    declare_and_init_constraints_fixed_params_.sublist("CHAIN_OF_SPRINGS")
        .set("hookean_springs_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"))
        .set("sphere_part_names", mundy::core::make_string_array("SPHERES"))
        .set<bool>("generate_hookean_springs", true)
        .set<bool>("generate_spheres_at_nodes", true);


    // Synchronize (merge and rectify differences) the requirements for each method based on the fixed parameters.
    // For now, we will directly use the types that each method corresponds to. The configurator will
    // fetch the static members of these methods using the configurable method factory.
    mesh_reqs_ptr_->sync(
        mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_sphere_sphere_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_crosslinker_sphere_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::LinkerPotentialForceMagnitudeReduction::get_mesh_requirements(
        linker_potential_force_magnitude_reduction_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(
        declare_and_init_constraints_fixed_params_));

#ifdef DEBUG
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      mesh_reqs_ptr_->print();
    }
#endif

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
    debug_print("Fetching fields and parts.");

    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDS", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_rng_field_ptr_ = fetch_field<unsigned>("NODE_RNG_COUNTER", node_rank_);

    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    element_hookean_spring_constant_field_ptr_ = fetch_field<double>("ELEMENT_HOOKEAN_SPRING_CONSTANT", element_rank_);
    element_hookean_spring_rest_length_field_ptr_ =
        fetch_field<double>("ELEMENT_HOOKEAN_SPRING_REST_LENGTH", element_rank_);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    element_rng_field_ptr_ = fetch_field<unsigned>("ELEMENT_RNG_COUNTER", element_rank_);

    constraint_perform_binding_field_ptr_ = fetch_field<unsigned>("CONSTRAINT_PERFORM_BINDING", constraint_rank_);

    // Fetch the parts
    spheres_part_ptr_ = fetch_part("SPHERES");
    crosslinkers_part_ptr_ = fetch_part("CROSSLINKERS");
    agents_part_ptr_ = fetch_part("AGENTS");
    springs_part_ptr_ = fetch_part("HOOKEAN_SPRINGS");
    sphere_sphere_linkers_part_ptr_ = fetch_part("SPHERE_SPHERE_LINKERS");
    crosslinker_sphere_linkers_part_ptr_ = fetch_part("CROSSLINKER_SPHERE_LINKERS");

    left_bound_crosslinkers_part_ptr_ = fetch_part("LEFT_BOUND_CROSSLINKERS");
    right_bound_crosslinkers_part_ptr_ = fetch_part("RIGHT_BOUND_CROSSLINKERS");
    doubly_bound_crosslinkers_part_ptr_ = fetch_part("DOUBLY_BOUND_CROSSLINKERS");
  }

  void instantiate_metamethods() {
    debug_print("Instantiating MetaMethods.");

    // Create the non-custom MetaMethods
    // MetaMethodExecutionInterface
    declare_and_init_constraints_ptr_ = mundy::constraints::DeclareAndInitConstraints::create_new_instance(
        bulk_data_ptr_.get(), declare_and_init_constraints_fixed_params_);

    // MetaMethodSubsetExecutionInterface
    compute_constraint_forcing_ptr_ = mundy::constraints::ComputeConstraintForcing::create_new_instance(
        bulk_data_ptr_.get(), compute_constraint_forcing_fixed_params_);
    compute_ssd_and_cn_ptr_ = mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
        bulk_data_ptr_.get(), compute_ssd_and_cn_fixed_params_);
    compute_aabb_ptr_ =
        mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr_.get(), compute_aabb_fixed_params_);
    evaluate_linker_potentials_ptr_ = mundy::linkers::EvaluateLinkerPotentials::create_new_instance(
        bulk_data_ptr_.get(), evaluate_linker_potentials_fixed_params_);
    linker_potential_force_magnitude_reduction_ptr_ =
        mundy::linkers::LinkerPotentialForceMagnitudeReduction::create_new_instance(
            bulk_data_ptr_.get(), linker_potential_force_magnitude_reduction_fixed_params_);
    destroy_neighbor_linkers_ptr_ = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), destroy_neighbor_linkers_fixed_params_);

    // MetaMethodPairwiseSubsetExecutionInterface
    generate_sphere_sphere_neighbor_linkers_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_sphere_sphere_neighbor_linkers_fixed_params_);
    generate_crosslinker_sphere_neighbor_linkers_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_crosslinker_sphere_neighbor_linkers_fixed_params_);
  }

  void set_mutable_parameters() {
    debug_print("Setting mutable parameters.");

    // The NodeEuler type updates, and the compute mobility, etc, are not set here, so just do the following.

    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", 0.0);
    compute_aabb_ptr_->set_mutable_params(compute_aabb_mutable_params);

    // DeclareAndInitConstraints mutable parameters
    Teuchos::ParameterList declare_and_init_constraints_mutable_params;
    using CoordinateMappingType =
        mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
    using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::StraightLine;

    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    double orientation_x = 1.0;
    double orientation_y = 0.0;
    double orientation_z = 0.0;
    auto coord_mapping_ptr = std::make_shared<OurCoordinateMappingType>(
        num_spheres_, center_x, center_y, center_z,
        (static_cast<double>(num_spheres_) - 1) * 2.0 * initial_sphere_separation_, orientation_x, orientation_y,
        orientation_z);
    declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
        .set<size_t>("num_nodes", num_spheres_)
        .set<size_t>("node_id_start", 1u)
        .set<size_t>("element_id_start", 1u)
        .set("hookean_spring_constant", backbone_spring_constant_)
        .set("hookean_spring_rest_length", backbone_spring_rest_length_)
        .set("sphere_radius", sphere_radius_)
        .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", coord_mapping_ptr);
    declare_and_init_constraints_ptr_->set_mutable_params(declare_and_init_constraints_mutable_params);
  }

  void setup_io() {
    debug_print("Setting up IO.");

    // Declare each part as an IO part
    //
    // Note. This can be a problem. If you add multiple parts (at the moment) with the same information, it will cause a
    // crash in the IO routines due to multiple objects writing. The errors are also somewhat incomprehensible.
    stk::io::put_io_part_attribute(*spheres_part_ptr_);
    stk::io::put_io_part_attribute(*left_bound_crosslinkers_part_ptr_);
    stk::io::put_io_part_attribute(*right_bound_crosslinkers_part_ptr_);
    stk::io::put_io_part_attribute(*doubly_bound_crosslinkers_part_ptr_);

    // Setup the IO broker
    stk_io_broker_.use_simple_fields();
    stk_io_broker_.set_bulk_data(*bulk_data_ptr_);

    output_file_index_ = stk_io_broker_.create_output_mesh("Sticky.exo", stk::io::WRITE_RESULTS);
    stk_io_broker_.add_field(output_file_index_, *node_coord_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_velocity_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_force_field_ptr_);
  }

  void loadbalance() {
    debug_print("Load balancing the mesh.");
    stk::balance::balanceStkMesh(balance_settings_, *bulk_data_ptr_);
  }

  void declare_and_initialize_sticky() {
    //////////////////////////////////////
    // Initialize the spheres and nodes //
    //////////////////////////////////////
    debug_print("Declaring and initializing the StickySettings.");

    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;

    // Initialize the backbone and its springs
    declare_and_init_constraints_ptr_->execute();

    // Initialize the parameters on each particle needed for the Hertzian contacts. We need to initialize the poisson
    // ratio and youngs modulus on every sphere as well. Scope this so that things dont leak out.
    {
      // Set up some aliases and the selector
      stk::mesh::Part &spheres_part = *spheres_part_ptr_;
      const stk::mesh::Field<double> &element_youngs_modulus_field = *element_youngs_modulus_field_ptr_;
      const stk::mesh::Field<double> &element_poissons_ratio_field = *element_poissons_ratio_field_ptr_;
      const stk::mesh::Selector locally_owned_spheres =
          spheres_part & bulk_data_ptr_->mesh_meta_data().locally_owned_part();

      double &youngs_modulus = sphere_youngs_modulus_;
      double &poissons_ratio = sphere_poissons_ratio_;

      stk::mesh::for_each_entity_run(
          *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()), stk::topology::ELEMENT_RANK, locally_owned_spheres,
          [&element_youngs_modulus_field, &element_poissons_ratio_field, &youngs_modulus, &poissons_ratio](
              [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere) {
            // const stk::mesh::Entity &node = bulk_data.begin_nodes(sphere)[0];

            // stk::mesh::field_data(node_rng_counter_field, node)[0] = 0;
            stk::mesh::field_data(element_youngs_modulus_field, sphere)[0] = youngs_modulus;
            stk::mesh::field_data(element_poissons_ratio_field, sphere)[0] = poissons_ratio;
          });  // for_each_entity_run
    }

    // Declare the crosslinkers along the backbone
    // Every sphere gets a left bound crosslinker
    //  o : spheres
    //  | : crosslinkers
    // ---: backbone springs
    //
    //  |   |   |   |   |   |   |   |   |   |
    //  o---o---o---o---o---o---o---o---o---o
    //
    // One of the design features that we're working on is better composition/extension for declaring and initializing
    // entities. The following will be replaced with a more elegant solution in the future that enforcing a certain
    // structure on the chain of springs. Specifically, we want to enforce the node ordering and partitioning to be the
    // following:
    //
    // o---o---o---o---o---o---o---o---o---o
    // 1   2   3   4   5   6   7   8   9   10 <- node ids
    // <--- rank 0 ---> <--- rank 1 ---> <--- rank 2 --->
    //
    // What follows is the same decomposition strategy used by the DeclareAndInitConstraints' ChainOfSprings technique.
    // To enforce consistency, we should make ChainOfSpheres take in a decomposition strategy (or something along those
    // lines) such that we can use the same strategy here when declaring the crosslinkers.

    // Because we are creating multiple sperm, we need to determine the node and element index ranges for each sperm.
    size_t start_node_id = 1u;
    // Shift by number of spheres, then numbers of spheres-1 to get through the already declared hookean springs for the
    // backbone.
    size_t start_crosslinker_id = num_spheres_ + num_spheres_ - 1u + 1u;

    auto get_node_id = [start_node_id](const size_t &seq_node_index) {
      return start_node_id + seq_node_index;
    };

    auto get_node = [get_node_id, &bulk_data](const size_t &seq_node_index) {
      return bulk_data.get_entity(stk::topology::NODE_RANK, get_node_id(seq_node_index));
    };

    auto get_crosslinker_id = [start_crosslinker_id](const size_t &seq_crosslinker_index) {
      return start_crosslinker_id + seq_crosslinker_index;
    };

    auto get_crosslinker = [get_crosslinker_id, &bulk_data](const size_t &seq_crosslinker_index) {
      return bulk_data.get_entity(stk::topology::ELEMENT_RANK, get_crosslinker_id(seq_crosslinker_index));
    };

    // Create the springs and their connected nodes, distributing the work across the ranks.
    const size_t rank = bulk_data_ptr_->parallel_rank();
    const size_t nodes_per_rank = num_spheres_ / bulk_data_ptr_->parallel_size();
    const size_t remainder = num_spheres_ % bulk_data_ptr_->parallel_size();
    const size_t start_seq_node_index = rank * nodes_per_rank + std::min(rank, remainder);
    const size_t end_seq_node_index = start_seq_node_index + nodes_per_rank + (rank < remainder ? 1 : 0);

    bulk_data_ptr_->modification_begin();
    // Create the elements for the crosslinkers
    const size_t start_element_chain_index = start_seq_node_index;
    const size_t end_start_element_chain_index = end_seq_node_index;
    for (size_t i = start_element_chain_index; i < end_start_element_chain_index; ++i) {
      debug_print("Adding crosslinker " + std::to_string(i));
      // Temporary/scatch variables
      stk::mesh::PartVector empty;
      stk::mesh::Permutation perm = stk::mesh::Permutation::INVALID_PERMUTATION;
      stk::mesh::OrdinalVector scratch1, scratch2, scratch3;
      auto left_bound_crosslinker_part_vector = stk::mesh::PartVector{left_bound_crosslinkers_part_ptr_};

      // Bind left and right nodes to the same node to start simulation (everybody is left bound)
      stk::mesh::EntityId left_node_id = get_node_id(i);
      stk::mesh::Entity left_node = bulk_data_ptr_->get_entity(node_rank_, left_node_id);

      // Just assert that the node exists
      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(left_node), std::invalid_argument, "Node " << i << " is not valid.");

      // Fetch the centerline twist spring and connect it to the nodes/edges
      stk::mesh::EntityId crosslinker_id = get_crosslinker_id(i);
      stk::mesh::Entity crosslinker =
          bulk_data_ptr_->declare_element(crosslinker_id, left_bound_crosslinker_part_vector);

      // Connect back onto the same node for now, as it is a left bound crosslinker
      bulk_data_ptr_->declare_relation(crosslinker, left_node, 0, perm, scratch1, scratch2, scratch3);
      bulk_data_ptr_->declare_relation(crosslinker, left_node, 1, perm, scratch1, scratch2, scratch3);
      MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(crosslinker).topology() != stk::topology::INVALID_TOPOLOGY,
                         std::logic_error, "The crosslinker with id " << crosslinker_id << " has an invalid topology.");

      // Set the crosslinker rng counter
      stk::mesh::field_data(*element_rng_field_ptr_, crosslinker)[0] = 0;

      // Set the crosslinker spring constant and rest length
      stk::mesh::field_data(*element_hookean_spring_constant_field_ptr_, crosslinker)[0] = crosslinker_spring_constant_;
      stk::mesh::field_data(*element_hookean_spring_rest_length_field_ptr_, crosslinker)[0] = crosslinker_rest_length_;
      // Search radius of springs
      stk::mesh::field_data(*element_radius_field_ptr_, crosslinker)[0] = crosslinker_rest_length_;
    }
    bulk_data_ptr_->modification_end();

#ifdef DEBUG
    for (size_t i = start_element_chain_index; i < end_start_element_chain_index; ++i) {
      stk::mesh::Entity spring = get_crosslinker(i);
      MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(spring).member(*crosslinkers_part_ptr_), std::logic_error,
                         "The crosslinker must be a member of the crosslinker part.");
      MUNDY_THROW_ASSERT(crosslinkers_part_ptr_->topology() == stk::topology::BEAM_2, std::logic_error,
                         "The crosslinker part must have BEAM_2 topology. Instead, it has topology "
                             << crosslinkers_part_ptr_->topology());
      MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(spring).entity_rank() == stk::topology::ELEMENT_RANK, std::logic_error,
                         "The crosslinker must have element rank. Instead, it has rank "
                             << bulk_data_ptr_->bucket(spring).entity_rank());
      MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(spring).topology() == stk::topology::BEAM_2, std::logic_error,
                         "The crosslinker must have BEAM_2 topology. Instead, it has topology "
                             << bulk_data_ptr_->bucket(spring).topology());
    }

    {
      std::vector<size_t> entity_counts;
      stk::mesh::comm_mesh_counts(*bulk_data_ptr_, entity_counts);
      debug_print(std::string("Num nodes: ") + std::to_string(entity_counts[stk::topology::NODE_RANK]));
      debug_print(std::string("Num elements: ") + std::to_string(entity_counts[stk::topology::ELEMENT_RANK]));
    }
#endif
  }

  void zero_out_transient_node_fields() {
    debug_print("Zeroing out the transient node fields.");
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
  }

  void zero_out_transient_constraint_fields() {
    debug_print("Zeroing out the transient constraint fields.");
    mundy::mesh::utils::fill_field_with_value<unsigned>(*constraint_perform_binding_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
  }

  void detect_neighbors() {
    // Neighbor detection
    // We'll use the same skin distance for both spheres and crosslinkers and rebuild both neighbor lists any time
    // the a sphere moves more than the skin distance.
    //
    // Right now this also counts the crosslinker_spheres for doubly bound crosslinkers, even though we don't use that
    // information. This is because if we go through an unbindng event, we will need that information. Right now this is
    // a choice we are making to not re-calculate the quantities when we go through an unbindng reaction, so might need
    // to change in the future.
    if (timestep_index_ % 100 == 0) {
      // ComputeAABB for everyone (assume same buffer distance...)
      auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
      auto crosslinkers_selector = stk::mesh::Selector(*crosslinkers_part_ptr_);
      auto sphere_sphere_linkers_selector = stk::mesh::Selector(*sphere_sphere_linkers_part_ptr_);
      auto crosslinker_sphere_linkers_selector = stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_);

      compute_aabb_ptr_->execute(spheres_selector | crosslinkers_selector);
      destroy_neighbor_linkers_ptr_->execute(sphere_sphere_linkers_selector | crosslinker_sphere_linkers_selector);
      generate_sphere_sphere_neighbor_linkers_ptr_->execute(spheres_selector, spheres_selector);
      generate_crosslinker_sphere_neighbor_linkers_ptr_->execute(crosslinkers_selector, spheres_selector);
    }
  }

  /// \brief Connect a crosslinker to a new node.
  ///
  /// A parallel-local mesh modification operation.
  ///
  /// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
  /// modification cycle.
  ///
  /// This is on an actual crosslinker, not the neighbor. This generically binds to a node in the conn_ordinal position
  /// (0 or 1).
  void bind_crosslinker_to_node(mundy::mesh::BulkData *const bulk_data_ptr, const stk::mesh::Entity &crosslinker,
                                const stk::mesh::Entity &new_node, const int &conn_ordinal) {
    MUNDY_THROW_ASSERT(bulk_data_ptr->in_modifiable_state(), std::logic_error,
                       "bind_crosslinker_to_node_right: The mesh must be in a modification cycle.");
    MUNDY_THROW_ASSERT(bulk_data_ptr->bucket(crosslinker).topology().base() == stk::topology::BEAM_2, std::logic_error,
                       "bind_crosslinker_to_node_right: The crosslinker must have BEAM_2 as a base topology.");
    MUNDY_THROW_ASSERT(bulk_data_ptr->entity_rank(new_node) == stk::topology::NODE_RANK, std::logic_error,
                       "bind_crosslinker_to_node_right: The right node must have NODE_RANK.");

    // The crosslinker is currently connected to a temporary node. We'll destroy this relation and replace it with
    // the new one.
    MUNDY_THROW_ASSERT(bulk_data_ptr->num_nodes(crosslinker) == 2, std::logic_error,
                       "bind_crosslinker_to_node_right: The crosslinker must be connected to exactly two nodes.");
    const stk::mesh::Entity &current_node = bulk_data_ptr->begin_nodes(crosslinker)[conn_ordinal];
    bulk_data_ptr->destroy_relation(crosslinker, current_node, conn_ordinal);
    bulk_data_ptr->declare_relation(crosslinker, new_node, conn_ordinal);

    // Resolve sharing of the new right node.
    const bool is_crosslinker_locally_owned = bulk_data_ptr->bucket(crosslinker).owned();
    const bool is_new_node_locally_owned = bulk_data_ptr->parallel_owner_rank(new_node);

    if (is_crosslinker_locally_owned && !is_new_node_locally_owned) {
      // We own the crosslinker but not the right node.
      const int rank_that_we_share_with = bulk_data_ptr->parallel_owner_rank(new_node);
      bulk_data_ptr->add_node_sharing(new_node, rank_that_we_share_with);
    } else if (!is_crosslinker_locally_owned && is_new_node_locally_owned) {
      // We don't own the crosslinker but we own the right node.
      const int rank_that_we_share_with = bulk_data_ptr->parallel_owner_rank(crosslinker);
      bulk_data_ptr->add_node_sharing(new_node, rank_that_we_share_with);
    }
  }

  void force_crosslinker_sphere_linker_binding() {
    debug_print("Forcing a crosslinker to bind (change state)");

    // Selectors and aliases
    const size_t &timestep_index = timestep_index_;
    const size_t &num_spheres = num_spheres_;

    auto crosslinker_sphere_linkers_selector = stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_);
    stk::mesh::Field<unsigned> &constraint_perform_binding_field = *constraint_perform_binding_field_ptr_;

    // Get the selector for the crosslinker_sphere_linkers
    auto locally_owned_selector =
        stk::mesh::Selector(crosslinker_sphere_linkers_selector) & meta_data_ptr_->locally_owned_part();
    // Loop over the entities and set the binding field
    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()), stk::topology::CONSTRAINT_RANK,
        locally_owned_selector,
        [&constraint_perform_binding_field, &num_spheres, &timestep_index](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker_constraint) {
          // Get the specific values for each crosslinker
          unsigned *constraint_perform_binding =
              stk::mesh::field_data(constraint_perform_binding_field, crosslinker_constraint);

          // XXX: Force the connection if the first linker between node 1 and node 2 from node 1
          // Get the 0th element (crosslinker), and 1st element (sphere)
          const stk::mesh::Entity &crosslinker_entity = bulk_data.begin_elements(crosslinker_constraint)[0];
          const stk::mesh::Entity &sphere_entity = bulk_data.begin_elements(crosslinker_constraint)[1];

      // Get the entity that I want to compare to
#ifdef DEBUG
          std::cout << "My constraint identifier (me): " << bulk_data.identifier(crosslinker_constraint) << std::endl;
          std::cout << "  My crosslinker identifier: " << bulk_data.identifier(crosslinker_entity) << std::endl;
          std::cout << "  My sphere identifier: " << bulk_data.identifier(sphere_entity) << std::endl;
#endif

          // FORCE THE BINDING OF crosslinker_sphere_linker[3] to crosslinker[3] and sphere[4]
          // Force the binding of the first crosslinker to the second sphere in the system
          // crosslinker_id = 2*num_spheres_;
          // sphere_id = num_hookean_springs + desired_number_of_spheres
          // if (bulk_data.identifier(crosslinker_constraint) == 3 && bulk_data.identifier(crosslinker_entity) == 4 &&
          //    bulk_data.identifier(sphere_entity) == 3 && timestep_index == 1) {
          if ((bulk_data.identifier(crosslinker_entity) == 2 * num_spheres) &&
              (bulk_data.identifier(sphere_entity) == (num_spheres - 1 + 2)) && (timestep_index == 1)) {
            constraint_perform_binding[0] = static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
          }
        });
  }

  void state_change_crosslinkers() {
    debug_print("Applying the state change of crosslinkers to the mesh.");

    // Loop over CROSSLINKER_SPHERE_LINKERS to look for a state change. Use an entity vector since this is done
    // within a modification.
    //
    // In many cases this looks similar to mundy_mesh::destroy_flagged_entities
    bulk_data_ptr_->modification_begin();

    // Setup aliases
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    auto crosslinker_sphere_linkers_selector = stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_);
    stk::mesh::Field<unsigned> &constraint_perform_binding_field = *constraint_perform_binding_field_ptr_;

    // Get the vector of entities to modify
    stk::mesh::EntityVector entities_to_modify;
    stk::mesh::get_selected_entities(crosslinker_sphere_linkers_selector, bulk_data_ptr_->buckets(constraint_rank_),
                                     entities_to_modify);
    // Iterate over the entities
    for (const stk::mesh::Entity &entity : entities_to_modify) {
      // Decode the binding type enum for this entity
      auto binding_action =
          static_cast<BINDING_STATE_CHANGE>(stk::mesh::field_data(constraint_perform_binding_field, entity)[0]);
      const bool perform_binding = binding_action != BINDING_STATE_CHANGE::NONE;
      if (perform_binding) {
        // Get the associated crosslinker that I'm performing the binding on
        const stk::mesh::Entity &crosslinker_entity = bulk_data.begin_elements(entity)[0];
        const stk::mesh::Entity &sphere_entity = bulk_data.begin_elements(entity)[1];
        // Grab the sphere's node
        const stk::mesh::Entity &sphere_node_entity = bulk_data.begin_nodes(sphere_entity)[0];

        // Call the binding function, note that the connectivity ordinal changes depending on right_to_left or
        // left_to_right.
        if (binding_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
          bind_crosslinker_to_node(bulk_data_ptr_.get(), crosslinker_entity, sphere_node_entity, 1);
          // Now change the part
          auto add_parts = stk::mesh::PartVector{doubly_bound_crosslinkers_part_ptr_};
          auto remove_parts = stk::mesh::PartVector{left_bound_crosslinkers_part_ptr_};
          bulk_data.change_entity_parts(crosslinker_entity, add_parts, remove_parts);
        } else if (binding_action == BINDING_STATE_CHANGE::RIGHT_TO_DOUBLY) {
          bind_crosslinker_to_node(bulk_data_ptr_.get(), crosslinker_entity, sphere_node_entity, 0);
          // Now change the part
          auto add_parts = stk::mesh::PartVector{doubly_bound_crosslinkers_part_ptr_};
          auto remove_parts = stk::mesh::PartVector{right_bound_crosslinkers_part_ptr_};
          bulk_data.change_entity_parts(crosslinker_entity, add_parts, remove_parts);
        }
      }
    }

    bulk_data_ptr_->modification_end();
  }

  void update_crosslinker_state() {
    debug_print("Updating crosslinker state.");

    // TODO(adam):
    // Here is where all of the kmc magic will happen. This will first mimic what is done in the
    // evaluate_linker_potentials (until we have a kernel for doing the Z-calc for the binding). This is then put onto
    // the crosslinker_sphere_neighbor_linkers. These can be though of as our possible generalized interactions. Once
    // these are done, we can use the information in the crosslinkers themselves when looping.

    // Selectors and aliases
    auto crosslinker_sphere_linkers_selector = stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_);
    // stk::mesh::Field<unsigned> &constraint_perform_binding_field = *constraint_perform_binding_field_ptr_;

    // Compute the signed separation distance on these
    compute_ssd_and_cn_ptr_->execute(crosslinker_sphere_linkers_selector);
    // TODO(cje): Figure out how to get the signed seapration distance for use later...

    // We want to loop over all LEFT_BOUND_CROSSLINKERS, RIGHT_BOUND_CROSSLINKERS, and DOUBLY_BOUND_CROSSLINKERS to
    // generate state changes. This is done to build up a list of actions that we will take later during a mesh
    // modification step.

    // TODO(cje): Here is where we would compute what agents change state.
    // XXX Force a linkage to exist for a single test crosslinker.
    {
      // Force a linkage in timtestep 1
      force_crosslinker_sphere_linker_binding();
    }

    // Loop over the different crosslinkers, look at their actions, and enforce the state change.
    {
      // Call the global state change function
      state_change_crosslinkers();
    }
  }

  void compute_hertzian_contact_forces() {
    debug_print("Computing Hertzian contact forces.");

    // Potential evaluation (Hertzian contact)
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto sphere_sphere_linkers_selector = stk::mesh::Selector(*sphere_sphere_linkers_part_ptr_);

    compute_ssd_and_cn_ptr_->execute(sphere_sphere_linkers_selector);
    evaluate_linker_potentials_ptr_->execute(sphere_sphere_linkers_selector);
    linker_potential_force_magnitude_reduction_ptr_->execute(spheres_selector);
  }

  void compute_harmonic_bond_forces() {
    debug_print("Computing harmonic bond forces.");

    // Need a compound selector for all springs, including those that are not unbound of singly bound crosslinkers.
    auto actively_bound_springs = stk::mesh::Selector(*springs_part_ptr_) - *left_bound_crosslinkers_part_ptr_ -
                                  *right_bound_crosslinkers_part_ptr_;

    // Potentials
    compute_constraint_forcing_ptr_->execute(actively_bound_springs);
  }

  void compute_velocity() {
    // Compute both the velocity due to brownian motion and the velocity due to force in the system at the same
    // time.
    debug_print("Computing velocity.");

    // Alias various quantities
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;

    stk::mesh::Field<unsigned> &node_rng_field = *node_rng_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    double &timestep_size = timestep_size_;
    double &sphere_drag_coeff = sphere_drag_coeff_;
    double &kt = kt_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Get the selector for the spheres
    auto locally_owned_selector = stk::mesh::Selector(spheres_part) & meta_data_ptr_->locally_owned_part();

    // Compute the total velocity of the nonorientable spheres
    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()), stk::topology::NODE_RANK, locally_owned_selector,
        [&node_velocity_field, &node_force_field, &node_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff,
         &kt]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          double *node_force = stk::mesh::field_data(node_force_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_field, sphere_node);

          // F = (Fext + Fbr) / gamma
          // F = (Fext + Fbr) * inv_drag_coeff

          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * kt * sphere_drag_coeff / timestep_size);
          node_velocity[0] += (node_force[0] + coeff * rng.randn<double>()) * inv_drag_coeff;
          node_velocity[1] += (node_force[1] + coeff * rng.randn<double>()) * inv_drag_coeff;
          node_velocity[2] += (node_force[2] + coeff * rng.randn<double>()) * inv_drag_coeff;
          node_rng_counter[0]++;
        });
  }

  void update_positions() {
    debug_print("Updating positions.");

    // Set aliases
    double &timestep_size = timestep_size_;

    stk::mesh::Part &spheres_part = *spheres_part_ptr_;

    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;

    // Get the selector for the spheres
    auto locally_owned_selector = stk::mesh::Selector(spheres_part) & meta_data_ptr_->locally_owned_part();

    // Update the positions for all spheres based on velocity
    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()), stk::topology::NODE_RANK, locally_owned_selector,
        [&node_coord_field, &node_velocity_field, &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                                  const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_coord = stk::mesh::field_data(node_coord_field, sphere_node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);

          // x(t+dt) = x(t) + dt * v(t)

          node_coord[0] += timestep_size * node_velocity[0];
          node_coord[1] += timestep_size * node_velocity[1];
          node_coord[2] += timestep_size * node_velocity[2];
        });
  }

  void run(int argc, char **argv) {
    debug_print("Running the simulation.");

    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    build_our_mesh_and_method_instances();
    debug_print("Mesh contents after build_our_mesh_and_method_instances.");
#ifdef DEBUG
    // Dump the mesh info as it exists now (with fields)
    stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
#endif
    fetch_fields_and_parts();
    instantiate_metamethods();
    set_mutable_parameters();
    setup_io();
    declare_and_initialize_sticky();
    debug_print("Mesh contents after declare_and_initialize_sticy.");
#ifdef DEBUG
    // Dump the mesh info as it exists now (with fields)
    stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);

    // Dump the mesh per processor as it exists right now
    stk::mesh::impl::dump_mesh_per_proc(*bulk_data_ptr_, "init");
#endif

    // Loadbalance?
    if (initial_loadbalance_) {
      loadbalance();
#ifdef DEBUG
      // Dump the mesh per processor as it exists right now
      stk::mesh::impl::dump_mesh_per_proc(*bulk_data_ptr_, "loadbalance");
#endif
    }

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer timer;
    for (timestep_index_ = 0; timestep_index_ < num_time_steps_; timestep_index_++) {
      debug_print(std::string("Time step ") + std::to_string(timestep_index_) + " of " +
                  std::to_string(num_time_steps_));

      // Prepare the current configuration.
      {
        // Zero the node velocities, and forces/torques for time t.
        zero_out_transient_node_fields();

        // Zero out the constraint binding state changes
        zero_out_transient_constraint_fields();
      }

      // Detect all possible neighbors in the system
      {
        // Detect neighbors of spheres-spheres and crosslinkers-spheres
        detect_neighbors();
      }

      // Update the state changes in the system s(t).
      {
        // State change of every crosslinker
        update_crosslinker_state();

        debug_print("Mesh contents after update_crosslinker_state.");
#ifdef DEBUG
        // Dump the mesh info as it exists now (with fields)
        stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
#endif
      }

      // Evaluate forces f(x(t)).
      {
        // Hertzian forces
        compute_hertzian_contact_forces();

        // Compute harmonic bond forces
        compute_harmonic_bond_forces();

        // Write out the mesh information
        debug_print("Mesh contents after compute_forces.");
#ifdef DEBUG
        stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
#endif
      }

      // Compute velocity.
      {
        // Evaluate v(t) = Mf(t).
        compute_velocity();
        debug_print("Mesh contents after compute_velocity.");
#ifdef DEBUG
        // Dump the mesh info as it exists now (with fields)
        stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
#endif
      }

      // IO. If desired, write out the data for time t.
      if (timestep_index_ % io_frequency_ == 0) {
        // Also write out a 'log'
        if (bulk_data_ptr_->parallel_rank() == 0) {
          double tps = static_cast<double>(timestep_index_) / static_cast<double>(timer.seconds());
          std::cout << "Step: " << std::setw(15) << timestep_index_ << ", tps: " << std::setprecision(15) << tps
                    << std::endl;
        }
        stk_io_broker_.begin_output_step(output_file_index_, static_cast<double>(timestep_index_));
        stk_io_broker_.write_defined_output_fields(output_file_index_);
        stk_io_broker_.end_output_step(output_file_index_);
        stk_io_broker_.flush_output();
#ifdef DEBUG
        std::ostringstream mstream;
        mstream << "step" << timestep_index_;
        // Dump the mesh per processor as it exists right now
        stk::mesh::impl::dump_mesh_per_proc(*bulk_data_ptr_, mstream.str());
#endif
      }

      // Update positions.
      {
        // Evaluate x(t + dt) = x(t) + dt * v(t).
        update_positions();
        debug_print("Mesh contents after update_positions.");
#ifdef DEBUG
        // Dump the mesh info as it exists now (with fields)
        stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
#endif
      }
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
      double tps = static_cast<double>(timestep_index_) / static_cast<double>(timer.seconds());
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
  stk::io::StkMeshIoBroker stk_io_broker_;
  size_t output_file_index_;
  size_t timestep_index_;
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

  stk::mesh::Field<unsigned> *constraint_perform_binding_field_ptr_;

  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *spheres_part_ptr_ = nullptr;
  stk::mesh::Part *crosslinkers_part_ptr_ = nullptr;
  stk::mesh::Part *sphere_sphere_linkers_part_ptr_ = nullptr;
  stk::mesh::Part *crosslinker_sphere_linkers_part_ptr_ = nullptr;
  stk::mesh::Part *springs_part_ptr_ = nullptr;
  stk::mesh::Part *agents_part_ptr_ = nullptr;

  stk::mesh::Part *left_bound_crosslinkers_part_ptr_ = nullptr;
  stk::mesh::Part *right_bound_crosslinkers_part_ptr_ = nullptr;
  stk::mesh::Part *doubly_bound_crosslinkers_part_ptr_ = nullptr;

  //@}

  //! \name MetaMethod instances
  //@{
  std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>> declare_and_init_constraints_ptr_;

  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_aabb_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_constraint_forcing_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_ssd_and_cn_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> evaluate_linker_potentials_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>>
      linker_potential_force_magnitude_reduction_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> destroy_neighbor_linkers_ptr_;

  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>
      generate_sphere_sphere_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>
      generate_crosslinker_sphere_neighbor_linkers_ptr_;

  //@}

  //! \name Fixed params for MetaMethods
  //@{

  Teuchos::ParameterList compute_constraint_forcing_fixed_params_;
  Teuchos::ParameterList compute_aabb_fixed_params_;
  Teuchos::ParameterList compute_ssd_and_cn_fixed_params_;
  Teuchos::ParameterList generate_sphere_sphere_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList generate_crosslinker_sphere_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList evaluate_linker_potentials_fixed_params_;
  Teuchos::ParameterList linker_potential_force_magnitude_reduction_fixed_params_;
  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList declare_and_init_constraints_fixed_params_;

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
  std::string input_file_name_ = "input.yaml";

  // Sphere params
  size_t num_spheres_ = 10;
  double initial_sphere_separation_ = 0.5;
  double sphere_radius_ = 0.5;
  double sphere_youngs_modulus_ = 1000.0;
  double sphere_poissons_ratio_ = 0.3;
  double sphere_drag_coeff_ = 1.0;

  // Backbone spring params
  double backbone_spring_constant_ = 100.0;
  double backbone_spring_rest_length_ = 1.0;

  // Crosslinker params
  double crosslinker_spring_constant_ = 100.0;
  double crosslinker_rest_length_ = 1.0;
  double crosslinker_left_binding_rate_ = 1.0;
  double crosslinker_right_binding_rate_ = 1.0;
  double crosslinker_left_unbinding_rate_ = 1.0;
  double crosslinker_right_unbinding_rate_ = 1.0;

  // Simulation params
  bool initial_loadbalance_ = false;
  size_t num_time_steps_ = 100;
  size_t io_frequency_ = 10;
  double timestep_size_ = 0.01;
  double kt_ = 1.0;
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

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

///////////////////////////
// Partitioning settings //
///////////////////////////
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

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  ///////////////////////////
  // Parse user parameters //
  ///////////////////////////
  // Default values for the inputs

  // Simulation parameters
  double timestep_size = 0.01;
  size_t num_time_steps = 100;
  double viscosity = 1.0;

  // Sphere params
  size_t num_spheres = 10;
  double sphere_radius = 0.6;
  double sphere_youngs_modulus = 1000.0;
  double sphere_poissons_ratio = 0.3;
  double sphere_diffusion_coeff = 1.0;  // Does this get replaced with KBT?

  // Backbone spring params
  double backbone_spring_constant = 100.0;
  double backbone_spring_rest_length = 2 * sphere_radius;

  // Crosslinker params
  double crosslinker_spring_constant = 100.0;
  double crosslinker_spring_rest_length = 2 * sphere_radius;
  double crosslinker_left_binding_rate = 1.0;
  double crosslinker_right_binding_rate = 1.0;
  double crosslinker_left_unbinding_rate = 1.0;
  double crosslinker_right_unbinding_rate = 1.0;

  // Parse the command line options.
  Teuchos::CommandLineProcessor cmdp(false, true);

  // Optional command line arguments for controlling sphere initialization:
  cmdp.setOption("num_spheres", &num_spheres, "Number of spheres.");
  cmdp.setOption("sphere_radius", &sphere_radius, "The radius of the spheres.");
  cmdp.setOption("initial_segment_length", &initial_segment_length, "Initial segment length.");
  cmdp.setOption("rest_length", &rest_length, "Rest length of the spring.");
  cmdp.setOption("num_time_steps", &num_time_steps, "Number of time steps.");
  cmdp.setOption("timestep_size", &timestep_size, "Time step size.");
  cmdp.setOption("diffusion_coeff", &diffusion_coeff, "Diffusion coefficient.");
  cmdp.setOption("viscosity", &viscosity, "Viscosity.");
  cmdp.setOption("youngs_modulus", &youngs_modulus, "Young's modulus.");
  cmdp.setOption("poissons_ratio", &poissons_ratio, "Poisson's ratio.");
  cmdp.setOption("spring_constant", &spring_constant, "Spring constant.");

  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    Kokkos::finalize();
    stk::parallel_machine_finalize();
    return EXIT_FAILURE;
  }

  MUNDY_THROW_ASSERT(timestep_size > 0, std::invalid_argument, "Time step size must be greater than zero.");

  // Dump the parameters to screen on rank 0
  if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
    std::cout << "##################################################" << std::endl;
    std::cout << "INPUT PARAMETERS:" << std::endl;
    std::cout << "  num_spheres: " << num_spheres << std::endl;
    std::cout << "  sphere_radius: " << sphere_radius << std::endl;
    std::cout << "  initial_segment_length: " << initial_segment_length << std::endl;
    std::cout << "  rest_length: " << rest_length << std::endl;
    std::cout << "  num_time_steps: " << num_time_steps << std::endl;
    std::cout << "  timestep_size: " << timestep_size << std::endl;
    std::cout << "  diffusion_coeff: " << diffusion_coeff << std::endl;
    std::cout << "  viscosity: " << viscosity << std::endl;
    std::cout << "  youngs_modulus: " << youngs_modulus << std::endl;
    std::cout << "  poissons_ratio: " << poissons_ratio << std::endl;
    std::cout << "  spring_constant: " << spring_constant << std::endl;
    std::cout << "##################################################" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // Setup the fixed parameters and generate the corresponding class instances and mesh //
  ////////////////////////////////////////////////////////////////////////////////////////
  using DoubleFieldReqs = mundy::meta::FieldRequirements<double>;
  using UIntFieldReqs = mundy::meta::FieldRequirements<unsigned>;
  using BoolFieldReqs = mundy::meta::FieldRequirements<bool>;

  // Setup the mesh requirements.
  // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
  // In the future, all of this will be done via the Configurator.
  auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and allow
  // us to extend the functionality offered natively by Mundy.
  //
  // We add the following methods to act on the spheres agent. We directly apply these requirements to all spheres.
  //   1. A node euler timestep method for the nonorientable spheres.
  //     Requirements: NODE_VELOCITY
  //   2. A method to compute the mobility of the nonorientable spheres. In this case, we use a local drag method.
  //     Requirements: NODE_FORCE, NODE_VELOCITY
  //   3. A method to compute the brownian velocity of the nonorientable spheres.
  //     Requirements: NODE_VELOCITY, NODE_RNG_COUNTER
  auto custom_spheres_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
  custom_spheres_part_reqs->add_field_reqs(std::make_shared<DoubleFieldReqs>(
      "NODE_VELOCITY", stk::topology::NODE_RANK, /* num coords = */ 3, /* num states = */ 1));
  custom_spheres_part_reqs->add_field_reqs(
      std::make_shared<UIntFieldReqs>("NODE_FORCE", stk::topology::NODE_RANK, 3, 1));
  custom_spheres_part_reqs->add_field_reqs(
      std::make_shared<UIntFieldReqs>("NODE_RNG_COUNTER", stk::topology::NODE_RANK, 1, 1));
  mundy::shapes::Spheres::add_part_reqs(custom_spheres_part_reqs);
  mesh_reqs_ptr->merge(mundy::shapes::Spheres::get_mesh_requirements());

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
  custom_crosslinkers_part_reqs->set_part_name("CROSSLINKERS");
  custom_crosslinkers_part_reqs->set_part_topology(stk::topology::BEAM_3);
  custom_crosslinkers_part_reqs->add_field_reqs(
      std::make_shared<DoubleFieldReqs>("ELEMENT_UNBIND_RATES", stk::topology::ELEMENT_RANK, 2, 1));
  custom_crosslinkers_part_reqs->add_field_reqs(
      std::make_shared<DoubleFieldReqs>("ELEMENT_BINDING_RATES", stk::topology::ELEMENT_RANK, 2, 1));
  custom_crosslinkers_part_reqs->add_field_reqs(
      std::make_shared<DoubleFieldReqs>("ELEMENT_UNBINDING_PROBABILITY", stk::topology::ELEMENT_RANK, 2, 1));
  custom_crosslinkers_part_reqs->add_field_reqs<int>("ELEMENT_CROSSLINKER_STATE_CHANGE", stk::topology::ELEMENT_RANK, 1,
                                                     1);
  mundy::shapes::Spherocylinders::add_subpart_reqs(custom_crosslinkers_part_reqs);
  mesh_reqs_ptr->merge(mundy::shapes::Spherocylinders::get_mesh_requirements());

  auto custom_crosslinker_sphere_linkers_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
  custom_crosslinker_sphere_linkers_part_reqs->set_part_name("CROSSLINKER_SPHERE_LINKERS");
  custom_crosslinker_sphere_linkers_part_reqs->set_part_rank(stk::topology::CONSTRAINT_RANK);
  custom_crosslinker_sphere_linkers_part_reqs->add_field_reqs(
      std::make_shared<DoubleFieldReqs>("CONSTRAINT_BINDING_PROBABILITY", stk::topology::CONSTRAINT_RANK, 1, 1));
  custom_crosslinker_sphere_linkers_part_reqs->add_field_reqs(
      std::make_shared<BoolFieldReqs>("CONSTRAINT_PERFORM_BINDING", stk::topology::CONSTRAINT_RANK, 1, 1));
  mundy::linkers::NeighborLinkers::add_subpart_reqs(custom_crosslinker_sphere_linkers_part_reqs);
  mesh_reqs_ptr->merge(mundy::linkers::NeighborLinkers::get_mesh_requirements());

  // ComputeConstraintForcing fixed parameters
  auto compute_constraint_forcing_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"))
          .sublist("HOOKEAN_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS", "CROSSLINKERS"));
  mesh_reqs_ptr->merge(
      mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params));

  // ComputeSignedSeparationDistanceAndContactNormal fixed parameters
  auto compute_ssd_and_cn_fixed_params =
      Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKER"));
  mesh_reqs_ptr->merge(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
      compute_ssd_and_cn_fixed_params));

  // ComputeAABB fixed parameters
  auto compute_aabb_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_kernel_names", mundy::core::make_string_array("SPHERE", "SPHEROCYLINDER"))
          .sublist("SPHEROCYLINDER")
          .set("valid_entity_part_names", mundy::core::make_string_array("CROSSLINKERS"));
  mesh_reqs_ptr->merge(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params));

  // GenerateNeighborLinkers fixed parameters
  // First, the parameters for generating the sphere-sphere neighbor linkers
  auto generate_sphere_sphere_neighbor_linkers_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_technique_name", "STK_SEARCH")
          .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"))
          .sublist("STK_SEARCH")
          .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHERES"))
          .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));
  mesh_reqs_ptr->merge(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
      generate_sphere_sphere_neighbor_linkers_fixed_params));

  // Next, the parameters for generating the crosslinker-sphere neighbor linkers
  auto generate_crosslinker_sphere_neighbor_linkers_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_technique_name", "STK_SEARCH")
          .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("CROSSLINKER_SPHERE_LINKERS"))
          .sublist("STK_SEARCH")
          .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("CROSSLINKERS")))
          .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));
  mesh_reqs_ptr->merge(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
      generate_crosslinker_sphere_neighbor_linkers_fixed_params));

  // EvaluateLinkerPotentials fixed parameters
  auto evaluate_linker_potentials_fixed_params = Teuchos::ParameterList().set(
      "enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT"));
  mesh_reqs_ptr->merge(
      mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params));

  // LinkerPotentialForceMagnitudeReduction fixed parameters
  auto linker_potential_force_magnitude_reduction_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
          .set("name_of_linker_part_to_reduce_over", "SPHERE_SPHERE_LINKERS");
  mesh_reqs_ptr->merge(mundy::linkers::LinkerPotentialForceMagnitudeReduction::get_mesh_requirements(
      linker_potential_force_magnitude_reduction_fixed_params));

  // DestroyNeighborLinkers fixed parameters
  auto destroy_neighbor_linkers_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
          .sublist("DESTROY_DISTANT_NEIGHBORS")
          .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
          .set("valid_connected_source_and_target_part_names",
               mundy::core::make_string_array(std::string("SPHERES"), std::string("CROSSLINKERS")));
  mesh_reqs_ptr->merge(
      mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params));

  // DeclareAndInitConstraints fixed parameters
  auto declare_and_init_constraints_fixed_params =
      Teuchos::ParameterList()
          .set("enabled_technique_name", "CHAIN_OF_SPRINGS")
          .sublist("CHAIN_OF_SPRINGS")
          .set("hookean_springs_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"))
          .set("sphere_part_names", mundy::core::make_string_array("SPHERES"))
          .set<bool>("generate_hookean_springs", true)
          .set<bool>("generate_spheres_at_nodes", true);
  mesh_reqs_ptr->merge(
      mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(declare_and_init_constraints_fixed_params));

  // The mesh requirements are now set up, so we solidify the mesh structure.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  meta_data_ptr->set_coordinate_field_name("NODE_COORDINATES");
  meta_data_ptr->commit();

  //////////////////////////////////////////////////////////////////////
  // Create the class instances and populate their mutable parameters //
  //////////////////////////////////////////////////////////////////////
  auto node_euler_ptr = NodeEuler::create_new_instance(bulk_data_ptr.get(), node_euler_fixed_params);
  auto compute_mobility_ptr = ComputeMobility::create_new_instance(bulk_data_ptr.get(), compute_mobility_fixed_params);
  auto compute_constraint_forcing_ptr = mundy::constraints::ComputeConstraintForcing::create_new_instance(
      bulk_data_ptr.get(), compute_constraint_forcing_fixed_params);
  auto compute_ssd_and_cn_ptr = mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
      bulk_data_ptr.get(), compute_ssd_and_cn_fixed_params);
  auto compute_aabb_ptr =
      mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr.get(), compute_aabb_fixed_params);
  auto generate_sphere_sphere_neighbor_linkers_ptr = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
      bulk_data_ptr.get(), generate_sphere_sphere_neighbor_linkers_fixed_params);
  auto generate_crosslinker_sphere_neighbor_linkers_ptr = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
      bulk_data_ptr.get(), generate_crosslinker_sphere_neighbor_linkers_fixed_params);
  auto evaluate_linker_potentials_ptr = mundy::linkers::EvaluateLinkerPotentials::create_new_instance(
      bulk_data_ptr.get(), evaluate_linker_potentials_fixed_params);
  auto linker_potential_force_magnitude_reduction_ptr =
      mundy::linkers::LinkerPotentialForceMagnitudeReduction::create_new_instance(
          bulk_data_ptr.get(), linker_potential_force_magnitude_reduction_fixed_params);
  auto destroy_neighbor_linkers_ptr = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
      bulk_data_ptr.get(), destroy_neighbor_linkers_fixed_params);
  auto declare_and_init_constraints_ptr = mundy::constraints::DeclareAndInitConstraints::create_new_instance(
      bulk_data_ptr.get(), declare_and_init_constraints_fixed_params);

  // If a class doesn't have mutable parameters, we can skip setting them.

  // ComputeBrownianVelocity mutable parameters
  auto compute_brownian_velocity_mutable_params = Teuchos::ParameterList()
                                                      .set("timestep_size", timestep_size)
                                                      .sublist("SPHERE")
                                                      .set("diffusion_coeff", diffusion_coeff);
  compute_brownian_velocity_ptr->set_mutable_params(compute_brownian_velocity_mutable_params);

  // NodeEuler mutable parameters
  auto node_euler_mutable_params = Teuchos::ParameterList().set("timestep_size", timestep_size);
  node_euler_ptr->set_mutable_params(node_euler_mutable_params);

  // ComputeMobility mutable parameters
  auto compute_mobility_mutable_params = Teuchos::ParameterList().sublist("LOCAL_DRAG").set("viscosity", viscosity);
  compute_mobility_ptr->set_mutable_params(compute_mobility_mutable_params);

  // ComputeAABB mutable parameters
  auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", 0.0);
  compute_aabb_ptr->set_mutable_params(compute_aabb_mutable_params);

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
  auto coord_mapping_ptr = std::make_shared<OurCoordinateMappingType>(num_spheres, center_x, center_y, center_z,
                                                                      orientation_x, orientation_y, orientation_z);
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
      .set<size_t>("num_nodes", num_spheres)
      .set<size_t>("node_id_start", i * num_spheres + +1)
      .set<size_t>("element_id_start", i * (num_spheres + (num_spheres - 1) * generate_hookean_springs +
                                            (num_spheres - 2) * generate_angular_springs) +
                                           1)
      .set("hookean_spring_constant", spring_constant)
      .set("hookean_spring_rest_length", rest_length)
      .set("sphere_radius", sphere_radius)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", coord_mapping_ptr);
  declare_and_init_constraints_ptr->set_mutable_params(declare_and_init_constraints_mutable_params);

  ////////////////////////////////
  // Fetch the fields and parts //
  ////////////////////////////////
  // These fields/parts are all things that we know should exist given the set of enabled kernels and techniques.
  // We will throw an exception if they don't exist. We're fetching them for initialization and IO purposes.

  // Node rank fields
  auto node_coordinates_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  auto node_rng_counter_field_ptr = meta_data_ptr->get_field<int>(stk::topology::NODE_RANK, "NODE_RNG_COUNTER");
  auto node_velocity_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_VELOCITY");
  auto node_force_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");

  // Element rank fields
  auto element_radius_field_ptr = meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  auto element_rest_length_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_HOOKEAN_SPRING_REST_LENGTH");
  auto element_spring_constant_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_HOOKEAN_SPRING_CONSTANT");

  // Validate that the fields exist
  auto check_if_exists = [](const stk::mesh::FieldBase *const field_ptr, const std::string &name) {
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       name + "cannot be a nullptr. Check that the field exists.");
  };

  check_if_exists(node_coordinates_field_ptr, "NODE_COORDINATES");
  check_if_exists(node_rng_counter_field_ptr, "NODE_RNG_COUNTER");
  check_if_exists(node_velocity_field_ptr, "NODE_VELOCITY");
  check_if_exists(node_force_field_ptr, "NODE_FORCE");
  check_if_exists(element_radius_field_ptr, "ELEMENT_RADIUS");
  check_if_exists(element_rest_length_field_ptr, "ELEMENT_REST_LENGTH");
  check_if_exists(element_spring_constant_field_ptr, "ELEMENT_SPRING_CONSTANT");

  // Fetch the parts
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  MUNDY_THROW_ASSERT(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");
  stk::mesh::Part &spheres_part = *spheres_part_ptr;
  stk::io::put_io_part_attribute(spheres_part);

  stk::mesh::Part *crosslinkers_part_ptr = meta_data_ptr->get_part("CROSSLINKERS");
  MUNDY_THROW_ASSERT(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");
  stk::mesh::Part &spheres_part = *spheres_part_ptr;
  stk::io::put_io_part_attribute(spheres_part);

  stk::mesh::Part *sphere_sphere_linkers_part_ptr = meta_data_ptr->get_part("SPHERE_SPHERE_LINKERS");
  MUNDY_THROW_ASSERT(sphere_sphere_linkers_part_ptr != nullptr, std::invalid_argument,
                     "SPHERE_SPHERE_LINKERS part not found.");
  stk::mesh::Part &sphere_sphere_linkers_part = *sphere_sphere_linkers_part_ptr;
  stk::io::put_io_part_attribute(sphere_sphere_linkers_part);

  stk::mesh::Part *crosslinker_sphere_linkers_part_ptr = meta_data_ptr->get_part("CROSSLINKER_SPHERE_LINKERS");
  MUNDY_THROW_ASSERT(crosslinker_sphere_linkers_part_ptr != nullptr, std::invalid_argument,
                     "CROSSLINKER_SPHERE_LINKERS part not found.");
  stk::mesh::Part &crosslinker_sphere_linkers_part = *crosslinker_sphere_linkers_part_ptr;
  stk::io::put_io_part_attribute(crosslinker_sphere_linkers_part);

  stk::mesh::Part *springs_part_ptr = meta_data_ptr->get_part("HOOKEAN_SPRINGS");
  MUNDY_THROW_ASSERT(springs_part_ptr != nullptr, std::invalid_argument, "HOOKEAN_SPRINGS part not found.");
  stk::mesh::Part &springs_part = *springs_part_ptr;
  stk::io::put_io_part_attribute(springs_part);

  ///////////////////
  // Setup our IO  //
  ///////////////////
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

  size_t output_file_index = stk_io_broker.create_output_mesh("Springs.exo", stk::io::WRITE_RESULTS);
  stk_io_broker.add_field(output_file_index, *node_coordinates_field_ptr);
  stk_io_broker.add_field(output_file_index, *node_velocity_field_ptr);
  stk_io_broker.add_field(output_file_index, *node_force_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_radius_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_rest_length_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_spring_constant_field_ptr);

  //////////////////////////////////////
  // Initialize the spheres and nodes //
  //////////////////////////////////////

  // Declare and initialize the spring chain
  declare_and_init_constraints_ptr->execute();

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
  bulk_data_ptr->modification_begin();
  const size_t rank = bulk_data_ptr->parallel_rank();
  const size_t nodes_per_rank = num_spheres / bulk_data_ptr->parallel_size();
  const size_t remainder = num_spheres % bulk_data_ptr->parallel_size();
  const size_t node_id_start = rank * nodes_per_rank + std::min(rank, remainder) + 1;
  const size_t node_id_end = node_id_start + nodes_per_rank + (rank < remainder ? 1 : 0);
  for (size_t i = node_id_start; i < sphere_id_end; ++i) {
    stk::mesh::Entity &node_i = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, i);
    MUNDY_THROW_ASSERT(bulk_data_ptr->is_valid(node_i), std::invalid_argument, "Node " << i << " is not valid.");
    stk::mesh::Entity &crosslinker = bulk_data_ptr->declare_entity(stk::topology::ELEMENT_RANK, i + num_spheres);
    bulk_data_ptr->change_entity_parts(crosslinker, stk::mesh::ConstPartVector({&crosslinkers_part}));
    bulk_data_ptr->declare_relation(crosslinker, node_i, 0);
    bulk_data_ptr->declare_relation(crosslinker, node_i, 1);
    bulk_data_ptr->declare_relation(crosslinker, node_i, 2);

    // Set the crosslinker's fields
    stk::mesh::field_data(element_rest_length_field_ptr, crosslinker)[0] = crosslinker_rest_length;
    stk::mesh::field_data(element_spring_constant_field_ptr, crosslinker)[0] = crosslinker_spring_constant;
    stk::mesh::field_data(element_binding_rates_field_ptr, crosslinker)[0] = crosslinker_left_binding_rate;
    stk::mesh::field_data(element_binding_rates_field_ptr, crosslinker)[1] = crosslinker_right_binding_rate;
    stk::mesh::field_data(element_unbind_rates_field_ptr, crosslinker)[0] = crosslinker_left_unbinding_rate;
    stk::mesh::field_data(element_unbind_rates_field_ptr, crosslinker)[1] = crosslinker_right_unbinding_rate;
  }
  bulk_data_ptr->modification_end();

  if (loadbalance_initial_config) {
    RcbSettings balanceSettings;
    stk::balance::balanceStkMesh(balanceSettings, *static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));
  }

  ////////////////////////
  // Run the simulation //
  ////////////////////////

  // // Write the initial mesh to file
  // stk_io_broker.begin_output_step(output_file_index, 0.0);
  // stk_io_broker.write_defined_output_fields(output_file_index);
  // stk_io_broker.end_output_step(output_file_index);
  // stk_io_broker.flush_output();

  if (bulk_data_ptr->parallel_rank() == 0) {
    std::cout << "Running the simulation for " << num_time_steps << " time steps." << std::endl;
  }

  Kokkos::Timer timer;
  mundy::mesh::utils::fill_field_with_value<unsigned>(*node_rng_counter_field_ptr, std::array<unsigned, 1>{0u});
  for (size_t i = 0; i < num_time_steps; i++) {
    // Setup
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr, std::array{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr, std::array{0.0, 0.0, 0.0});

    // Potentials
    compute_constraint_forcing_ptr->execute(stk::mesh::Selector(springs_part) |
                                            stk::mesh::Selector(angular_springs_part));

    // Neighbor detection
    // We'll use the same skin distance for both spheres and crosslinkers and rebuild both neighbor lists any time
    // the a sphere moves more than the skin distance.
    if (i % 100 == 0) {
      compute_aabb_ptr->execute(spheres_part | crosslinkers_part);
      destroy_neighbor_linkers_ptr->execute(sphere_sphere_linkers_part | crosslinker_sphere_linkers_part);
      generate_sphere_sphere_neighbor_linkers_ptr->execute(spheres_part, spheres_part);
      generate_crosslinker_sphere_neighbor_linkers_ptr->execute(crosslinkers_part, spheres_part);
    }

    // Potential evaluation (Hertzian contact)
    compute_ssd_and_cn_ptr->execute(sphere_sphere_linkers_part);
    evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part);
    linker_potential_force_magnitude_reduction_ptr->execute(spheres_part);

    // Computing the mobility of the nonorientable spheres
    stk::mesh::for_each_entity_run(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()),
                                   stk::topology::ELEMENT_RANK, locally_owned_intersection_with_valid_entity_parts,
                                   [&node_force_field, &node_velocity_field, &element_radius_field, &viscosity](
                                       const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
                                     const stk::mesh::Entity &node = bulk_data.begin_nodes(sphere_element)[0];

                                     const double *element_radius =
                                         stk::mesh::field_data(element_radius_field, sphere_element);
                                     const double *node_force = stk::mesh::field_data(node_force_field, node);
                                     double *node_velocity = stk::mesh::field_data(node_velocity_field, node);
                                     const double inv_drag_coeff = 1.0 / (6.0 * M_PI * viscosity * element_radius[0]);
                                     node_velocity[0] += inv_drag_coeff * node_force[0];
                                     node_velocity[1] += inv_drag_coeff * node_force[1];
                                     node_velocity[2] += inv_drag_coeff * node_force[2];
                                   });

    // Compute the brownian velocity of the nonorientable spheres
    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()), stk::topology::NODE_RANK,
        locally_owned_intersection_with_valid_entity_parts,
        [&node_brownian_velocity_field, &node_rng_counter_field, &timestep_size, &diffusion_coeff](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          double *node_brownian_velocity = stk::mesh::field_data(node_brownian_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_counter_field, sphere_node);

          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * diffusion_coeff / timestep_size);
          node_brownian_velocity[0] += coeff * rng.randn<double>();
          node_brownian_velocity[1] += coeff * rng.randn<double>();
          node_brownian_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });

    // I/O
    if (i % 10000 == 0) {
      stk_io_broker.begin_output_step(output_file_index, static_cast<double>(i));
      stk_io_broker.write_defined_output_fields(output_file_index);
      stk_io_broker.end_output_step(output_file_index);
      stk_io_broker.flush_output();
    }

    // Update the node positions using a first order Euler method
    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()), stk::topology::NODE_RANK,
        locally_owned_intersection_with_valid_entity_parts,
        [&node_coord_field, &node_velocity_field, &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                                  const stk::mesh::Entity &sphere_node) {
          double *node_coords = stk::mesh::field_data(node_coord_field, sphere_node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          node_coords[0] += timestep_size * node_velocity[0];
          node_coords[1] += timestep_size * node_velocity[1];
          node_coords[2] += timestep_size * node_velocity[2];
        });
  }

  // Do a synchronize to force everybody to stop here, then write the time
  stk::parallel_machine_barrier(bulk_data_ptr->parallel());

  if (bulk_data_ptr->parallel_rank() == 0) {
    double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps);
    std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    std::cout << "Timesteps per second: " << 1.0 / avg_time_per_timestep << std::endl;
  }

  // Write the final mesh to file
  // stk_io_broker.begin_output_step(output_file_index, static_cast<double>(num_time_steps));
  // stk_io_broker.write_defined_output_fields(output_file_index);
  // stk_io_broker.end_output_step(output_file_index);
  // stk_io_broker.flush_output();

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

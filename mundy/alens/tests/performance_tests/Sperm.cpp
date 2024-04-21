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

The goal of this example is to simulate the swimming motion of long sperm confined to a 2D plane. For the sake of
collisions, the sperm are a chain of spherocylinder segments. For the sake of particle motion, we use "mass lumping" to
represent the chain as a collection of spheres located at the endpoints of each segment. To model inextensibility, we
add distance-based Hookean springs between adjacent spheres. To model flexibility, we employ a finite difference
discretization of a Kirchhoff rod model with a centerline-twist parametrization of the rod. To model the sperm's
swimming motion, we vary the rest curvature with time and archlength along the chain.

We will initialize the sperm as straight lines within the x-y plane. They can either all be initialized in the same
direction or in alternating direction. You can also, optionally, enable, two boundary sperm at the top and bottom of the
domain. These sperm will be able to move, swim, and collide with the other sperm, BUT we'll only consider one way
collision forces. As in, the boundary perm can exert a collision force on the bulk sperm, but the bulk sperm can't exert
a collision force on the boundary sperm.
  x---x---x---x---x---x---x---x
    <-o---o---o---o---o---o
    <-o---o---o---o---o---o
    <-o---o---o---o---o---o
    <-o---o---o---o---o---o
  x---x---x---x---x---x---x---x
or
  x---x---x---x---x---x---x---x
    <-o---o---o---o---o---o
      o---o---o---o---o---o->
    <-o---o---o---o---o---o
      o---o---o---o---o---o->
  x---x---x---x---x---x---x---x
The coordinate mapping used to set the initial position of the sperm in the bulk may differ from the coordinate mapping
used to set the initial position of the sperm on the boundary. The two choices of coordinate mapping are straight and
sinusoidal.

To start, we won't employ Mundy's requirements encapsulation. That's supposed to come last in the development process.
This can't be emphasized more: the goal is to get something working first and then wrap it in a MetaMethod so it can be
integrated into Mundy and runnable via our Configurator/Driver system.
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
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::...::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_math/Matrix3.hpp>                             // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>                          // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>                             // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_mesh/utils/FillFieldWithValue.hpp>            // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

//////////////////////////////
// Finite element rod model //
//////////////////////////////

/* Section nodes:

Centerline-twist rods store the following information
  Element
    Radius | double | len 1 | states 2
  Edge
    Orientation | double | len 4 | states 2
    Tangent     | double | len 3 | states 2
    Binormal    | double | len 3 | states 1
    Length      | double | len 3 | states 1
  Node
    Coord              | double | len 3 | states 2
    Velocity           | double | len 3 | states 1
    Acceleration       | double | len 3 | states 2
    Twist              | double | len 1 | states 2
    Twist rate         | double | len 1 | states 2
    Twist acceleration | double | len 1 | states 2
    Curvature          | double | len 3 | states 1
    Rest curvature     | double | len 3 | states 1
    Rotation gradient  | double | len 4 | states 1
    Force              | double | len 3 | states 1
    Twist torque       | double | len 1 | states 1

We require the following constant material properties for the rods:
  - Young's modulus
  - Poisson's ratio
  - Shear modulus
  - Density

Other miscellaneous parameters:
  - Time step size

We need to be able to perform the following operations on these rods:
  - Compute edge information (length, tangent, and binormal).
  - Compute node curvature and rotation gradient
  - Compute node rest curvature
  - Compute internal force and torque
  - Compute stretch force
  - Compute collision force
  - Compute node motion using a multi-step scheme
  - Write the rods to disk

*/

/// \brief The main function for the sperm simulation broken down into digestible chunks.
/// 
/// This struct is in charge of setting up and executing the following tasks:
///   // Preprocess
///   - Parse user parameter
///   
///   // Setup
///   - Declare the fixed and mutable params for the desired MetaMethods
///   - Construct the mesh and the method instances
///   - Declare and initialize the chain of rods and their connecting springs, and the STL elements
///   
///   // Timeloop
///   - Run the timeloop for t in range(0, T):
///       // IO.
///       - If desired, write out the data for time t
///         (Using stk::io::StkMeshIoBroker)
///   
///       // Setup the current configuration.
///       - Rotate the field states
///         (Using BulkData's update_field_data_states function)
///       
///       - Zero the node forces and velocities for time t + dt
///         (Using mundy's fill_field_with_value function)
///   
///       // Motion from t -> t + dt:
///        - Apply velocity/acceleration constraints like no motion for particle 1
///         (By directly looping over all nodes and setting the velocity/acceleration to zero)
///        
///        - Evaluate x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2
///         (By looping over all nodes and updating the coordinates)
///   
///       // Evaluate forces f(x(t + dt))
///       {
///         // Neighbor detection rod-rod
///         - Check if the rod-rod neighbor list needs updated or not
///             - Compute the AABBs for the rods
///              (Using mundy's ComputeAABB function)
///             
///             - Delete rod-rod neighbor linkers that are too far apart
///              (Using the DestroyDistantNeighbors technique of mundy's DestroyNeighborLinkers function)
///             
///             - Generate neighbor linkers between nearby rods
///              (Using the GenerateNeighborLinkers function of mundy's GenerateNeighborLinkers function)
///   
///         // Hertzian contact
///         - Compute the signed separation distance and contact normal between neighboring rods
///          (Using mundy's ComputeSignedSeparationDistanceAndContactNormal function)
///         
///         - Evaluate the Hertzian contact potential between neighboring rods
///          (Using mundy's EvaluateLinkerPotentials function)
///         
///         - Sum the linker potential force magnitude to get the induced node force on each rod
///          (Using mundy's LinkerPotentialForceMagnitudeReduction function)
///   
///         // Senterline twist rod model
///         - Compute the edge information (length, tangent, and binormal)
///          (By looping over all edges and computing the edge length, tangent, binormal)
///         
///         - Compute the node curvature and rotation gradient
///          (By looping over all centerline twist spring elements and computing the curvature & rotation gradient at their
///           center node using the edge orientations)
///         
///         - Compute the internal force and twist torque
///          (By looping over all centerline twist spring elements, using the curvature to compute the induced torque, and
///           then using the torque to compute the force and twist-torque on the nodes)
///       }
///   
///       // Compute velocity and acceleration
///        - Evaluate a(t + dt) = M^{-1} f(x(t + dt))
///         (By looping over all nodes and computing the node acceleration and twist acceleration from the force and twist torque)
///   
///        - Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2
///         (By looping over all nodes and updating the node velocity and twist rate using the corresponding accelerations)
///
/// All user inputs are parsed from the command line. 
class SpermSimulation {
 public:
  SpermSimulation() = default;

  bool parse_user_inputs(int argc, char **argv) {
    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, true);

    // Optional command line arguments for controlling
    //   Sphere initialization:
    cmdp.setOption("num_spheres", &num_spheres, "Number of spheres.");
    cmdp.setOption("sphere_radius", &sphere_radius, "The radius of the spheres.");
    cmdp.setOption("initial_segment_length", &initial_segment_length, "Initial segment length.");
    cmdp.setOption("rest_length", &rest_length, "Rest length of the spring.");
    cmdp.setOption("loadbalance", "no_loadbalance", &loadbalance_initial_config,
                  "Load balance the initial configuration.");

    //   Spring initialization:
    cmdp.setOption("spring_constant", &spring_constant, "Spring constant.");
    cmdp.setOption("angular_spring_constant", &angular_spring_constant, "Angular spring constant.");
    cmdp.setOption("angular_spring_rest_angle", &angular_spring_rest_angle, "Angular spring rest angle.");

    //   The simulation:
    cmdp.setOption("num_time_steps", &num_time_steps, "Number of time steps.");
    cmdp.setOption("timestep_size", &timestep_size, "Time step size.");
    cmdp.setOption("diffusion_coeff", &diffusion_coeff, "Diffusion coefficient.");
    cmdp.setOption("viscosity", &viscosity, "Viscosity.");
    cmdp.setOption("youngs_modulus", &youngs_modulus, "Young's modulus.");
    cmdp.setOption("poissons_ratio", &poissons_ratio, "Poisson's ratio.");

    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    return was_parse_successful;
  }

  void check_input_parameters() {
    MUNDY_THROW_ASSERT(num_spheres > 0, std::invalid_argument, "num_spheres must be greater than 0.");
    MUNDY_THROW_ASSERT(sphere_radius > 0, std::invalid_argument, "sphere_radius must be greater than 0.");
    MUNDY_THROW_ASSERT(initial_segment_length > 0, std::invalid_argument, "initial_segment_length must be greater than 0.");
    MUNDY_THROW_ASSERT(rest_length > 0, std::invalid_argument, "rest_length must be greater than 0.");
    MUNDY_THROW_ASSERT(num_time_steps > 0, std::invalid_argument, "num_time_steps must be greater than 0.");
    MUNDY_THROW_ASSERT(timestep_size > 0, std::invalid_argument, "timestep_size must be greater than 0.");
    MUNDY_THROW_ASSERT(diffusion_coeff > 0, std::invalid_argument, "diffusion_coeff must be greater than 0.");
    MUNDY_THROW_ASSERT(viscosity > 0, std::invalid_argument, "viscosity must be greater than 0.");
    MUNDY_THROW_ASSERT(youngs_modulus > 0, std::invalid_argument, "youngs_modulus must be greater than 0.");
    MUNDY_THROW_ASSERT(poissons_ratio > 0, std::invalid_argument, "poissons_ratio must be greater than 0.");
    MUNDY_THROW_ASSERT(spring_constant > 0, std::invalid_argument, "spring_constant must be greater than 0.");
    MUNDY_THROW_ASSERT(angular_spring_constant > 0, std::invalid_argument, "angular_spring_constant must be greater than 0.");
    MUNDY_THROW_ASSERT(angular_spring_rest_angle > 0, std::invalid_argument, "angular_spring_rest_angle must be greater than 0.");
  }

  void dump_user_inputs() {
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
      std::cout << "  angular_spring_constant: " << angular_spring_constant << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }
  
  void build_our_mesh_and_method_instances() {
    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr->set_spatial_dimension(3);
    mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and allow
    // us to extend the functionality offered natively by Mundy.
    //
    // We require that the centerline twist springs part exists, has a BEAM_3 topology, and has the desired
    // node/edge/element fields.
    auto clt_part_reqs = std::make_shared<mundy::meta::PartRequirements>()
                            ->set_part_name("CENTERLINE_TWIST_SPRINGS")
                            .set_part_topology(stk::topology::BEAM_3)

                            // Add the node fields
                            .add_field_reqs<double>("NODE_COORDINATES", node_rank, 3, 2)
                            .add_field_reqs<double>("NODE_VELOCITY", node_rank, 3, 2)
                            .add_field_reqs<double>("NODE_FORCE", node_rank, 3, 1)
                            .add_field_reqs<double>("NODE_ACCELERATION", node_rank, 3, 2)

                            .add_field_reqs<double>("NODE_TWIST", node_rank, 1, 2)
                            .add_field_reqs<double>("NODE_TWIST_VELOCITY", node_rank, 1, 2)
                            .add_field_reqs<double>("NODE_TWIST_TORQUE", node_rank, 1, 1)
                            .add_field_reqs<double>("NODE_TWIST_ACCELERATION", node_rank, 1, 2)

                            .add_field_reqs<double>("NODE_CURVATURE", node_rank, 3, 1)
                            .add_field_reqs<double>("NODE_REST_CURVATURE", node_rank, 3, 1)
                            .add_field_reqs<double>("NODE_ROTATION_GRADIENT", node_rank, 4, 1)

                            // Add the edge fields
                            .add_field_reqs<double>("EDGE_ORIENTATION", edge_rank, 4, 2)
                            .add_field_reqs<double>("EDGE_TANGENT", edge_rank, 3, 2)
                            .add_field_reqs<double>("EDGE_BINORMAL", edge_rank, 3, 1)
                            .add_field_reqs<double>("EDGE_LENGTH", edge_rank, 1, 1);

    // Create the mesh requirements
    mesh_reqs_ptr->add_part_reqs(clt_part_reqs);

    // ComputeConstraintForcing fixed parameters
    auto compute_constraint_forcing_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name()));
    mesh_reqs_ptr->merge(
        mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params));

    // ComputeSignedSeparationDistanceAndContactNormal fixed parameters
    auto compute_ssd_and_cn_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER"));
    mesh_reqs_ptr->merge(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params));

    // ComputeAABB fixed parameters
    auto compute_aabb_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));
    mesh_reqs_ptr->merge(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params));

    // GenerateNeighborLinkers fixed parameters
    auto generate_neighbor_linkers_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS"))
            .sublist("STK_SEARCH")
            .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"))
            .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"));
    mesh_reqs_ptr->merge(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_neighbor_linkers_fixed_params));

    // EvaluateLinkerPotentials fixed parameters
    auto evaluate_linker_potentials_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names",
        mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
    mesh_reqs_ptr->merge(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params));

    // LinkerPotentialForceMagnitudeReduction fixed parameters
    auto linker_potential_force_magnitude_reduction_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));
    mesh_reqs_ptr->merge(mundy::linkers::LinkerPotentialForceMagnitudeReduction::get_mesh_requirements(
        linker_potential_force_magnitude_reduction_fixed_params));

    // DestroyNeighborLinkers fixed parameters
    auto destroy_neighbor_linkers_fixed_params =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    mesh_reqs_ptr->merge(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params));

    // DeclareAndInitConstraints fixed parameters
    auto declare_and_init_constraints_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "CHAIN_OF_SPRINGS")
            .sublist("CHAIN_OF_SPRINGS")
            .set("hookean_springs_part_names",
                mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name()))
            .set("angular_springs_part_names",
                mundy::core::make_string_array(mundy::constraints::AngularSprings::get_name()))
            .set("sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()))
            .set("spherocylinder_segment_part_names",
                mundy::core::make_string_array(mundy::shapes::SpherocylinderSegments::get_name()))
            .set<bool>("generate_hookean_springs", true)
            .set<bool>("generate_angular_springs", false)
            .set<bool>("generate_spheres_at_nodes", false)
            .set<bool>("generate_spherocylinder_segments_along_edges", true);
    mesh_reqs_ptr->merge(
        mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(declare_and_init_constraints_fixed_params));
 
    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
    meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
    meta_data_ptr->set_coordinate_field_name("NODE_COORDINATES");
    meta_data_ptr->commit();

    // Create the class instances and populate their mutable parameters
    node_euler_ptr = NodeEuler::create_new_instance(bulk_data_ptr.get(), node_euler_fixed_params);
    compute_centerline_twist_constraint_force_ptr = ComputeCenterlineTwistSpringConstraintForce::create_new_instance(
        bulk_data_ptr.get(), compute_centerline_twist_constraint_force_fixed_params);
    compute_constraint_forcing_ptr =
        ComputeConstraintForcing::create_new_instance(bulk_data_ptr.get(), compute_constraint_forcing_fixed_params);
    compute_ssd_and_cn_ptr = ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
        bulk_data_ptr.get(), compute_ssd_and_cn_fixed_params);
    compute_aabb_ptr = ComputeAABB::create_new_instance(bulk_data_ptr.get(), compute_aabb_fixed_params);
    generate_neighbor_linkers_ptr =
        GenerateNeighborLinkers::create_new_instance(bulk_data_ptr.get(), generate_neighbor_linkers_fixed_params);
    evaluate_linker_potentials_ptr =
        EvaluateLinkerPotentials::create_new_instance(bulk_data_ptr.get(), evaluate_linker_potentials_fixed_params);
    linker_potential_force_magnitude_reduction_ptr = LinkerPotentialForceMagnitudeReduction::create_new_instance(
        bulk_data_ptr.get(), linker_potential_force_magnitude_reduction_fixed_params);
    destroy_neighbor_linkers_ptr =
        DestroyNeighborLinkers::create_new_instance(bulk_data_ptr.get(), destroy_neighbor_linkers_fixed_params);
    declare_and_init_constraints_ptr =
        DeclareAndInitConstraints::create_new_instance(bulk_data_ptr.get(), declare_and_init_constraints_fixed_params);
  
    // Set up the mutable parameters for the classes 
    // If a class doesn't have mutable parameters, we can skip setting them.

    // ComputeCenterlineTwistSpringConstraintForce mutable parameters
    auto compute_centerline_twist_constraint_force_mutable_params =
        Teuchos::ParameterList()
            .set("youngs_modulus", youngs_modulus)
            .set("poissons_ratio", poissons_ratio)
            .set("shear_modulus", youngs_modulus / (2.0 * (1.0 + poissons_ratio)))
            .set("density", 1.0);
    compute_centerline_twist_constraint_force_ptr->set_mutable_params(
        compute_centerline_twist_constraint_force_mutable_params);

    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", 0.0);
    compute_aabb_ptr->set_mutable_params(compute_aabb_mutable_params);
  }

  template <typename FieldType>
  stk::mesh::Field<FieldType> *fetch_field(const std::string &field_name, stk::topology::rank_t rank) {
    auto field_ptr = meta_data_ptr->get_field<FieldType>(rank, field_name);
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "Field " << field_name << " not found in the mesh meta data.");
    return field_ptr;
  }

  stk::mesh::Part* fetch_part(const std::string &part_name) {
    auto part_ptr = meta_data_ptr->get_part(part_name);
    MUNDY_THROW_ASSERT(part_ptr != nullptr, std::invalid_argument,
                       "Part " << part_name << " not found in the mesh meta data.");
    return part_ptr;
  }

  void fetch_fields_and_parts() {
    // Fetch the fields
    node_coordinates_field_ptr = fetch_field<double>("NODE_COORDINATES", stk::topology::NODE_RANK);
    node_velocity_field_ptr = fetch_field<double>("NODE_VELOCITY", stk::topology::NODE_RANK);
    node_force_field_ptr = fetch_field<double>("NODE_FORCE", stk::topology::NODE_RANK);
    node_acceleration_field_ptr = fetch_field<double>("NODE_ACCELERATION", stk::topology::NODE_RANK);
    node_twist_field_ptr = fetch_field<double>("NODE_TWIST", stk::topology::NODE_RANK);
    node_twist_velocity_field_ptr = fetch_field<double>("NODE_TWIST_VELOCITY", stk::topology::NODE_RANK);
    node_twist_torque_field_ptr = fetch_field<double>("NODE_TWIST_TORQUE", stk::topology::NODE_RANK);
    node_twist_acceleration_field_ptr = fetch_field<double>("NODE_TWIST_ACCELERATION", stk::topology::NODE_RANK);
    node_curvature_field_ptr = fetch_field<double>("NODE_CURVATURE", stk::topology::NODE_RANK);
    node_rest_curvature_field_ptr = fetch_field<double>("NODE_REST_CURVATURE", stk::topology::NODE_RANK);
    node_rotation_gradient_field_ptr = fetch_field<double>("NODE_ROTATION_GRADIENT", stk::topology::NODE_RANK);
    node_radius_field_ptr = fetch_field<double>("NODE_RADIUS", stk::topology::NODE_RANK);

    edge_orientation_field_ptr = fetch_field<double>("EDGE_ORIENTATION", stk::topology::EDGE_RANK);
    edge_tangent_field_ptr = fetch_field<double>("EDGE_TANGENT", stk::topology::EDGE_RANK);
    edge_binormal_field_ptr = fetch_field<double>("EDGE_BINORMAL", stk::topology::EDGE_RANK);
    edge_length_field_ptr = fetch_field<double>("EDGE_LENGTH", stk::topology::EDGE_RANK);

    element_radius_field_ptr = fetch_field<double>("ELEMENT_RADIUS", stk::topology::ELEMENT_RANK);
    element_youngs_modulus_field_ptr = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", stk::topology::ELEMENT_RANK);
    element_poissons_ratio_field_ptr = fetch_field<double>("ELEMENT_POISSONS_RATIO", stk::topology::ELEMENT_RANK);
    element_rest_length_field_ptr = fetch_field<double>("ELEMENT_REST_LENGTH", stk::topology::ELEMENT_RANK);
    element_spring_constant_field_ptr = fetch_field<double>("ELEMENT_SPRING_CONSTANT", stk::topology::ELEMENT_RANK);

    // Fetch the parts
    centerline_twist_springs_part_ptr = fetch_part("CENTERLINE_TWIST_SPRINGS");
    spherocylinder_segments_part_ptr = fetch_part("SPHEROCYLINDER_SEGMENTS");
    hookian_springs_part_ptr = fetch_part("HOOKEAN_SPRINGS");
  }

  void setup_io() {
    // Declare each part as an IO part
    stk::io::put_io_part_attribute(*centerline_twist_springs_part_ptr);
    stk::io::put_io_part_attribute(*spherocylinder_segments_part_ptr);
    stk::io::put_io_part_attribute(*hookian_springs_part_ptr);

    // Setup the IO broker
    stk_io_broker.use_simple_fields();
    stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

    size_t output_file_index = stk_io_broker.create_output_mesh("Sperm.exo", stk::io::WRITE_RESULTS);
    stk_io_broker.add_field(output_file_index, *node_coordinates_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_velocity_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_force_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_acceleration_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_twist_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_twist_velocity_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_twist_torque_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_twist_acceleration_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_curvature_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_rest_curvature_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_rotation_gradient_field_ptr);
    stk_io_broker.add_field(output_file_index, *node_radius_field_ptr);

    stk_io_broker.add_field(output_file_index, *edge_orientation_field_ptr);
    stk_io_broker.add_field(output_file_index, *edge_tangent_field_ptr);
    stk_io_broker.add_field(output_file_index, *edge_binormal_field_ptr);
    stk_io_broker.add_field(output_file_index, *edge_length_field_ptr);

    stk_io_broker.add_field(output_file_index, *element_radius_field_ptr);
    stk_io_broker.add_field(output_file_index, *element_youngs_modulus_field_ptr);
    stk_io_broker.add_field(output_file_index, *element_poissons_ratio_field_ptr);
    stk_io_broker.add_field(output_file_index, *element_rest_length_field_ptr);
    stk_io_broker.add_field(output_file_index, *element_spring_constant_field_ptr);
  }

  void declare_and_initialize_the_sperm() {
    // Declare N spring chains with a slight shift to each chain
    const int num_chains = 10;
    for (int i = 0; i < num_chains; i++) {
      // DeclareAndInitConstraints mutable parameters
      Teuchos::ParameterList declare_and_init_constraints_mutable_params;
      using CoordinateMappingType =
          mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
      using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::StraightLine;

      double center_x = 0.0;
      double center_y = 2 * i * sphere_radius;
      double center_z = 0.0;
      double length = num_spheres * initial_segment_length;
      double axis_x = 1.0;
      double axis_y = 0.0;
      double axis_z = 0.0;
      auto straight_line_mapping_ptr = std::make_shared<OurCoordinateMappingType>(
          num_spheres, center_x, center_y, center_z, length, axis_x, axis_y, axis_z);
      declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
          .set<size_t>("num_nodes", num_spheres)
          .set<size_t>("node_id_start", i * num_spheres + 1)
          .set<size_t>("element_id_start", i * (4 * num_spheres - 4) + 1)
          .set("hookean_spring_constant", spring_constant)
          .set("hookean_spring_rest_length", rest_length)
          .set("angular_spring_constant", angular_spring_constant)
          .set("angular_spring_rest_angle", angular_spring_rest_angle)
          .set("sphere_radius", sphere_radius)
          .set("spherocylinder_segment_radius", sphere_radius)
          .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", straight_line_mapping_ptr);
      declare_and_init_constraints_ptr->set_mutable_params(declare_and_init_constraints_mutable_params);
      declare_and_init_constraints_ptr->execute();
    }

    mundy::mesh::utils::fill_field_with_value<double>(*element_youngs_modulus_field_ptr,
                                                      std::array<double, 1>{youngs_modulus});
    mundy::mesh::utils::fill_field_with_value<double>(*element_poissons_ratio_field_ptr,
                                                      std::array<double, 1>{poissons_ratio});
  }
  
  void loadbalance() {
    RcbSettings balanceSettings;
    stk::balance::balanceStkMesh(balanceSettings, *static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));
  }

  void run(int argc, char **argv) {
    // Preprocess
    if (!parse_user_inputs(argc, argv)) {
      std::cerr << "Failed to parse user inputs." << std::endl;
      return;
    }
    check_input_parameters();
    dump_user_inputs();

    // Setup
    build_our_mesh_and_method_instances();
    fetch_fields_and_parts();
    setup_io();
    declare_and_initialize_the_sperm();

    // // Write the initial mesh to file
    // stk_io_broker.begin_output_step(output_file_index, 0.0);
    // stk_io_broker.write_defined_output_fields(output_file_index);
    // stk_io_broker.end_output_step(output_file_index);
    // stk_io_broker.flush_output();

    // Time loop
    if (bulk_data_ptr->parallel_rank() == 0) {
      std::cout << "Running the simulation for " << num_time_steps << " time steps." << std::endl;
    }

    Kokkos::Timer timer;
    for (size_t i = 0; i < num_time_steps; i++) {
      // Rotate the field states.
      bulk_data.update_field_data_states();

      // Output
      if (i % 10000 == 0) {
        stk_io_broker.begin_output_step(output_file_index, static_cast<double>(i));
        stk_io_broker.write_defined_output_fields(output_file_index);
        stk_io_broker.end_output_step(output_file_index);
        stk_io_broker.flush_output();
      }

      // Setup
      mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr, std::array{0.0, 0.0, 0.0});
      mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr, std::array{0.0, 0.0, 0.0});

      // Potentials
      compute_constraint_forcing_ptr->execute(stk::mesh::Selector(springs_part) |
                                              stk::mesh::Selector(angular_springs_part));

      // Collisions
      if (i % 100 == 0) {
        compute_aabb_ptr->execute(spherocylinder_segments_part);
        destroy_neighbor_linkers_ptr->execute(spherocylinder_segment_spherocylinder_segment_linkers_part);
        generate_neighbor_linkers_ptr->execute(spherocylinder_segments_part, spherocylinder_segments_part);
      }
      compute_ssd_and_cn_ptr->execute(spherocylinder_segment_spherocylinder_segment_linkers_part);
      evaluate_linker_potentials_ptr->execute(spherocylinder_segment_spherocylinder_segment_linkers_part);
      linker_potential_force_magnitude_reduction_ptr->execute(spherocylinder_segments_part);

      // Motion
      compute_mobility_ptr->execute(spheres_part);
      node_euler_ptr->execute(spheres_part);
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr->parallel());

    if (bulk_data_ptr->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps);
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    }

    // Write the final mesh to file
    // stk_io_broker.begin_output_step(output_file_index, static_cast<double>(num_time_steps));
    // stk_io_broker.write_defined_output_fields(output_file_index);
    // stk_io_broker.end_output_step(output_file_index);
    // stk_io_broker.flush_output();
  }

 private:
  // Useful aliases
  constexpr auto node_rank = stk::topology::NODE_RANK;
  constexpr auto edge_rank = stk::topology::EDGE_RANK;
  constexpr auto element_rank = stk::topology::ELEMENT_RANK;

  // Internal state 
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr;
  std::shared_ptr<mundy::meta::MeshRequirements> mesh_reqs_ptr;
  stk::io::StkMeshIoBroker stk_io_broker;

  // Fields
  stk::mesh::Field<double> *node_coordinates_field_ptr;
  stk::mesh::Field<double> *node_velocity_field_ptr;
  stk::mesh::Field<double> *node_force_field_ptr;
  stk::mesh::Field<double> *node_acceleration_field_ptr;
  stk::mesh::Field<double> *node_twist_field_ptr;
  stk::mesh::Field<double> *node_twist_velocity_field_ptr;
  stk::mesh::Field<double> *node_twist_torque_field_ptr;
  stk::mesh::Field<double> *node_twist_acceleration_field_ptr;
  stk::mesh::Field<double> *node_curvature_field_ptr;
  stk::mesh::Field<double> *node_rest_curvature_field_ptr;
  stk::mesh::Field<double> *node_rotation_gradient_field_ptr;
  stk::mesh::Field<double> *node_radius_field_ptr;
  
  stk::mesh::Field<double> *edge_orientation_field_ptr;
  stk::mesh::Field<double> *edge_tangent_field_ptr;
  stk::mesh::Field<double> *edge_binormal_field_ptr;
  stk::mesh::Field<double> *edge_length_field_ptr;

  stk::mesh::Field<double> *element_radius_field_ptr;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr;
  stk::mesh::Field<double> *element_rest_length_field_ptr;
  stk::mesh::Field<double> *element_spring_constant_field_ptr;

  // Parts
  stk::mesh::Part *centerline_twist_springs_part_ptr;
  stk::mesh::Part *spherocylinder_segments_part_ptr;
  stk::mesh::Part *hookian_springs_part_ptr;

  // Class instances
  std::shared_ptr<NodeEuler> node_euler_ptr;
  std::shared_ptr<ComputeCenterlineTwistSpringConstraintForce> compute_centerline_twist_constraint_force_ptr;
  std::shared_ptr<ComputeConstraintForcing> compute_constraint_forcing_ptr;
  std::shared_ptr<ComputeSignedSeparationDistanceAndContactNormal> compute_ssd_and_cn_ptr;
  std::shared_ptr<ComputeAABB> compute_aabb_ptr;
  std::shared_ptr<GenerateNeighborLinkers> generate_neighbor_linkers_ptr;
  std::shared_ptr<EvaluateLinkerPotentials> evaluate_linker_potentials_ptr;
  std::shared_ptr<LinkerPotentialForceMagnitudeReduction> linker_potential_force_magnitude_reduction_ptr;
  std::shared_ptr<DestroyNeighborLinkers> destroy_neighbor_linkers_ptr;
  std::shared_ptr<DeclareAndInitConstraints> declare_and_init_constraints_ptr;

  // Partitioning settings
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

  // User parameters
  size_t num_spheres = 10;
  double sphere_radius = 0.6;
  double initial_segment_length = 1.0;
  double rest_length = 2 * sphere_radius;
  bool loadbalance_initial_config = true;
  size_t num_time_steps = 100;
  double timestep_size = 0.01;
  double diffusion_coeff = 1.0;
  double viscosity = 1.0;
  double youngs_modulus = 1000.0;
  double poissons_ratio = 0.3;
  double spring_constant = 1.0;
  double angular_spring_constant = 1.0;
  double angular_spring_rest_angle = M_PI;
};  // SpermSimulation


class ComputeCenterlineTwistSpringConstraintForce : public mundy::meta::MetaMethodExecutionInterface<void> {
 public:
  ComputeCenterlineTwistSpringConstraintForce(mundy::mesh::BulkData *const bulk_data_ptr,
                                              const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                       "ComputeCenterlineTwistSpringConstraintForce: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(
        ComputeCenterlineTwistSpringConstraintForce::get_valid_fixed_params());

    // Store the ComputeCenterlineTwistSpringConstraintForce part acton on by this method.,
    centerline_twist_part_ptr_ = meta_data_ptr_->get_part(default_part_name_);

    // Fetch the fields.
    node_coordinates_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_coordinates_field_name_);
    node_twist_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_twist_field_name_);
    node_curvature_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_curvature_field_name_);
    node_rest_curvature_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_rest_curvature_field_name_);
    node_rotation_gradient_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_rotation_gradient_field_name_);
    node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_force_field_name_);
    node_twist_torque_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_twist_torque_field_name_);

    edge_orientation_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_orientation_field_name_);
    edge_tangent_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_tangent_field_name_);
    edge_binormal_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_binormal_field_name_);
    edge_length_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_length_field_name_);

    element_radius_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, default_element_radius_field_name_);

    auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
      MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                         "ComputeCenterlineTwistSpringConstraintForce: Field "
                             << field_name << " cannot be a nullptr. Check that the field exists.");
    };  // field_exists

    field_exists(node_coordinates_field_ptr_, default_node_coordinates_field_name_);
    field_exists(node_twist_field_ptr_, default_node_twist_field_name_);
    field_exists(node_curvature_field_ptr_, default_node_curvature_field_name_);
    field_exists(node_rest_curvature_field_ptr_, default_node_rest_curvature_field_name_);
    field_exists(node_rotation_gradient_field_ptr_, default_node_rotation_gradient_field_name_);
    field_exists(node_force_field_ptr_, default_node_force_field_name_);
    field_exists(node_twist_torque_field_ptr_, default_node_twist_torque_field_name_);

    field_exists(edge_orientation_field_ptr_, default_edge_orientation_field_name_);
    field_exists(edge_tangent_field_ptr_, default_edge_tangent_field_name_);
    field_exists(edge_binormal_field_ptr_, default_edge_binormal_field_name_);
    field_exists(edge_length_field_ptr_, default_edge_length_field_name_);

    field_exists(element_radius_field_ptr_, default_element_radius_field_name_);
  }
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(
        ComputeCenterlineTwistSpringConstraintForce::get_valid_fixed_params());

    // We require that the CenterlineTwistSpring part exists, has a BEAM_3 topology, and has the desired
    // node/edge/element fields.
    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name(default_part_name_);
    part_reqs->set_part_topology(stk::topology::BEAM_3);

    // Add the node fields
    part_reqs->add_field_reqs<double>(default_node_coordinates_field_name_, stk::topology::NODE_RANK, 3, 2);
    part_reqs->add_field_reqs<double>(default_node_twist_field_name_, stk::topology::NODE_RANK, 1, 2);
    part_reqs->add_field_reqs<double>(default_node_curvature_field_name_, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_node_rest_curvature_field_name_, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_node_rotation_gradient_field_name_, stk::topology::NODE_RANK, 4, 1);
    part_reqs->add_field_reqs<double>(default_node_force_field_name_, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_node_twist_torque_field_name_, stk::topology::NODE_RANK, 1, 1);

    // Add the edge fields
    part_reqs->add_field_reqs<double>(default_edge_orientation_field_name_, stk::topology::EDGE_RANK, 4, 2);
    part_reqs->add_field_reqs<double>(default_edge_tangent_field_name_, stk::topology::EDGE_RANK, 3, 2);
    part_reqs->add_field_reqs<double>(default_edge_binormal_field_name_, stk::topology::EDGE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_edge_length_field_name_, stk::topology::EDGE_RANK, 1, 1);

    // Create the mesh requirements
    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(part_reqs);
    return mesh_reqs;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;

    // Material properties
    default_parameter_list.set("youngs_modulus", default_youngs_modulus_, "Young's modulus.");
    default_parameter_list.set("poissons_ratio", default_poissons_ratio_, "Poisson's ratio.");
    default_parameter_list.set("shear_modulus", default_shear_modulus_, "Shear modulus.");
    default_parameter_list.set("density", default_density_, "Density.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<ComputeCenterlineTwistSpringConstraintForce>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(
        ComputeCenterlineTwistSpringConstraintForce::get_valid_mutable_params());

    // Material properties
    youngs_modulus_ = valid_mutable_params.get<double>("youngs_modulus");
    poissons_ratio_ = valid_mutable_params.get<double>("poissons_ratio");
    shear_modulus_ = valid_mutable_params.get<double>("shear_modulus");
    density_ = valid_mutable_params.get<double>("density");

    // Check invariants: All material properties must be positive. Time step size must be positive.
    MUNDY_THROW_ASSERT(youngs_modulus_ > 0.0, std::invalid_argument,
                       "ComputeCenterlineTwistSpringConstraintForce: Young's modulus must be greater than zero.");
    MUNDY_THROW_ASSERT(poissons_ratio_ > 0.0, std::invalid_argument,
                       "ComputeCenterlineTwistSpringConstraintForce: Poisson's ratio must be greater than zero.");
    MUNDY_THROW_ASSERT(shear_modulus_ > 0.0, std::invalid_argument,
                       "ComputeCenterlineTwistSpringConstraintForce: Shear modulus must be greater than zero.");
    MUNDY_THROW_ASSERT(density_ > 0.0, std::invalid_argument,
                       "ComputeCenterlineTwistSpringConstraintForce: Density must be greater than zero.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return {centerline_twist_part_ptr_};
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute() {
    // - Compute edge information (length, tangent, and binormal).
    // - Compute node curvature
    // - Compute internal force and torque

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_coordinates_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rest_curvature_field = *node_rest_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;
    const double youngs_modulus = youngs_modulus_;
    const double poissons_ratio = poissons_ratio_;
    const double shear_modulus = shear_modulus_;
    const double density = density_;

    // For some of our fields, we need to know about the reference configuration of the CenterlineTwistSpring part.
    // This information is stored in StateN
    // A note about states: StateNone, StateNew, and StateNP1 all refer to the same default accessed field.
    // StateOld and StateN refer to the old state, which, in our case, is the reference configuration.
    //
    stk::mesh::Field<double> &node_coordinates_field_ref = node_coordinates_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_field_ref = node_twist_field.field_of_state(stk::mesh::StateN);
    node_twist_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_orientation_field_ref = edge_orientation_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_tangent_field_ref = edge_tangent_field.field_of_state(stk::mesh::StateN);

    // Most of our calculations will be done on the locally owned part of the CenterlineTwistSpring part.
    stk::mesh::Selector locally_owned_selector = centerline_twist_part & meta_data_ptr_->locally_owned_part();

    // Now, evaluate the internal force f(x(t + dt))
    // This requires updating all the fields that vary with the current configuration.

    // For each edge in the CenterlineTwistSpring part, compute the edge tangent, binormal, and length.
    // length^i = ||x_{i+1} - x_i||
    // edge_tangent^i = (x_{i+1} - x_i) / length
    // edge_binormal^i = (2 edge_tangent_ref^i x edge_tangent^i) / (1 + edge_tangent_ref^i dot edge_tangent^i)
    stk::mesh::for_each_entity_run(
        bulk_data, stk::topology::EDGE_RANK, locally_owned_selector,
        [&node_coordinates_field, &edge_orientation_field, &edge_tangent_field, &edge_tangent_field_ref,
         &edge_binormal_field,
         &edge_length_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &centerline_twist_edge) {
          // Get the nodes of the edge
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(centerline_twist_edge);
          stk::mesh::Entity const &node_i = edge_nodes[0];
          stk::mesh::Entity const &node_ip1 = edge_nodes[1];

          // Get the required input fields
          const auto node_i_coords =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_coordinates_field, node_i));
          const auto node_ip1_coords =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_coordinates_field, node_ip1));
          const auto edge_tangent_ref = mundy::math::get_vector3_view<double>(
              stk::mesh::field_data(edge_tangent_field_ref, centerline_twist_edge));

          // Get the output fields
          auto edge_tangent =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(edge_tangent_field, centerline_twist_edge));
          auto edge_binormal =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(edge_binormal_field, centerline_twist_edge));
          auto edge_length =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(edge_length_field, centerline_twist_edge));

          // Compute the un-normalized edge tangent
          edge_tangent = node_ip1_coords - node_i_coords;
          edge_length[0] = mundy::math::norm(edge_tangent);
          edge_tangent /= edge_length[0];

          // Compute the edge binormal
          edge_binormal = (2.0 * mundy::math::cross(edge_tangent_ref, edge_tangent)) /
                          (1.0 + mundy::math::dot(edge_tangent_ref, edge_tangent));
        });

    // For each element in the CenterlineTwistSpring part, compute the node curvature at the center node.
    // The curvature can be computed from the edge orientations using
    //   kappa^i = q_i - conj(q_i) = 2 * vec(q_i)
    // where
    //   q_i = conj(d^{i-1}) d^i is the Lagrangian rotation gradient.
    stk::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, locally_owned_selector,
        [&edge_orientation_field, &node_curvature_field, &node_rotation_gradient_field](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &centerline_twist_element) {
          // Curvature needs to "know" about the order of edges, so it's best to loop over the centerline_twist elements
          // and not the nodes. Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(centerline_twist_element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(centerline_twist_element);

          MUNDY_THROW_ASSERT(bulk_data.num_nodes(centerline_twist_element) == 3, std::logic_error,
                             "ComputeCenterlineTwistSpringConstraintForce: Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(centerline_twist_element) == 2, std::logic_error,
                             "ComputeCenterlineTwistSpringConstraintForce: Elements must have exactly 2 edges.");

          // Get the required input fields
          const auto edge_im1_orientation =
              mundy::math::get_quaternion_view<double>(stk::mesh::field_data(edge_orientation_field, element_edges[0]));
          const auto edge_i_orientation =
              mundy::math::get_quaternion_view<double>(stk::mesh::field_data(edge_orientation_field, element_edges[1]));

          // Get the output fields
          auto node_curvature =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_curvature_field, element_nodes[1]));
          auto node_rotation_gradient = mundy::math::get_quaternion_view<double>(
              stk::mesh::field_data(node_rotation_gradient_field, element_nodes[1]));

          // Compute the node curvature
          node_rotation_gradient = mundy::math::conj(edge_im1_orientation) * edge_i_orientation;
          node_curvature = 2.0 * node_rotation_gradient.vector();
        });

    // Compute internal force and torque
    stk::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, locally_owned_selector,
        [&node_force_field, &&node_curvature_field, &node_rest_curvature_field, &node_rotation_gradient_field,
         &edge_tangent_field, &edge_binormal_field, &edge_length_field, &edge_orientation_field](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &centerline_twist_element) {
          // Ok. This is a bit involved.
          // First, we need to use the node curvature to compute the induced lagrangian torque according to the
          // Kirchhoff rod model. Then, we need to use a convoluted map to take this torque to force and torque on the
          // nodes.
          //
          // The torque induced by the curvature is
          //  T = B (kappa - kappa_rest)
          // where B is the diagonal matrix of bending moduli and kappa_rest is the rest curvature.
          //
          // To start, we set B = I

          // Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(centerline_twist_element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(centerline_twist_element);

          MUNDY_THROW_ASSERT(bulk_data.num_nodes(centerline_twist_element) == 3, std::logic_error,
                             "ComputeCenterlineTwistSpringConstraintForce: Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(centerline_twist_element) == 2, std::logic_error,
                             "ComputeCenterlineTwistSpringConstraintForce: Elements must have exactly 2 edges.");

          // Get the required input fields
          const auto node_i_curvature =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_curvature_field, element_nodes[1]));
          const auto node_i_rest_curvature =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_rest_curvature_field, element_nodes[1]));
          const auto node_i_rotation_gradient = mundy::math::get_quaternion_view<double>(
              stk::mesh::field_data(node_rotation_gradient_field, element_nodes[1]));
          const auto edge_im1_tangent =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(edge_tangent_field, element_edges[0]));
          const auto edge_i_tangent =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(edge_tangent_field, element_edges[1]));
          const auto edge_im1_binormal =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(edge_binormal_field, element_edges[0]));
          const auto edge_i_binormal =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(edge_binormal_field, element_edges[1]));
          const auto edge_im1_length =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(edge_length_field, element_edges[0]));
          const auto edge_i_length =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(edge_length_field, element_edges[1]));
          const auto edge_im1_orientation =
              mundy::math::get_quaternion_view<double>(stk::mesh::field_data(edge_orientation_field, element_edges[0]));

          // Get the output fields
          auto node_im1_force =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_force_field, element_nodes[0]));
          auto node_i_force =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_force_field, element_nodes[1]));
          auto node_ip1_force =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_force_field, element_nodes[2]));
          double *node_im1_twist_torque = stk::mesh::field_data(node_twist_torque_field, element_nodes[0]);
          double *node_i_twist_torque = stk::mesh::field_data(node_twist_torque_field, element_nodes[1]);

          // Compute the torque induced by the curvature
          // To start, the torque is in the lagrangian frame.
          auto bending_modulus = mundy::math::Matrix3::identity();  // TODO(palmerb4): Replace with Kirchhoff rod model
          auto bending_torque = bending_modulus * (node_i_curvature - node_i_rest_curvature);

          // We'll use the bending torque as a placeholder for the rotated bending torque
          bending_torque =
              edge_im1_orientation * (node_i_rotation_gradient.w() * bending_torque +
                                      mundy::math::cross(node_i_rotation_gradient.vector(), bending_torque));

          // Compute the force and torque on the nodes
          const auto tmp_force_ip1 = 1.0 / edge_i_length[0] *
                                     (mundy::math::cross(bending_torque, edge_i_tangent) +
                                      0.5 * mundy::math::dot(edge_i_tangent, bending_torque) *
                                          (mundy::math::dot(edge_i_tangent, edge_i_binormal) * edge_i_tangent) -
                                      edge_i_binormal);
          const auto tmp_force_im1 =
              1.0 / edge_im1_length[0] *
              (mundy::math::cross(bending_torque, edge_im1_tangent) +
               0.5 * mundy::math::dot(edge_im1_tangent, bending_torque) *
                   (mundy::math::dot(edge_im1_tangent, edge_im1_binormal) * edge_im1_tangent - edge_im1_binormal));

#pragma omp atomic
          node_i_twist_torque[0] += edge_i_tangent.dot(bending_torque);
#pragma omp atomic
          node_im1_twist_torque[0] += -edge_im1_tangent.dot(bending_torque);
#pragma omp atomic
          node_ip1_force[0] += tmp_force_ip1[0];
#pragma omp atomic
          node_ip1_force[1] += tmp_force_ip1[1];
#pragma omp atomic
          node_ip1_force[2] += tmp_force_ip1[2];
#pragma omp atomic
          node_i_force[0] -= tmp_force_ip1[0] + tmp_force_im1[0];
#pragma omp atomic
          node_i_force[1] -= tmp_force_ip1[1] + tmp_force_im1[1];
#pragma omp atomic
          node_i_force[2] -= tmp_force_ip1[2] + tmp_force_im1[2];
#pragma omp atomic
          node_im1_force[0] += tmp_force_im1[0];
#pragma omp atomic
          node_im1_force[1] += tmp_force_im1[1];
#pragma omp atomic
          node_im1_force[2] += tmp_force_im1[2];
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_youngs_modulus_ = 1000.0;
  static inline double default_poissons_ratio_ = 0.3;
  static inline double default_shear_modulus_ = 1000.0;
  static inline double default_density_ = 1.0;
  static constexpr std::string_view default_part_name_ = "CENTERLINE_TWIST_SPRINGS";
  static constexpr std::string_view default_node_coordinates_field_name_ = "NODE_COORDINATES";
  static constexpr std::string_view default_node_twist_field_name_ = "NODE_TWIST";
  static constexpr std::string_view default_node_curvature_field_name_ = "NODE_CURVATURE";
  static constexpr std::string_view default_node_rest_curvature_field_name_ = "NODE_REST_CURVATURE";
  static constexpr std::string_view default_edge_orientation_field_name_ = "EDGE_ORIENTATION";
  static constexpr std::string_view default_edge_tangent_field_name_ = "EDGE_TANGENT";
  static constexpr std::string_view default_edge_binormal_field_name_ = "EDGE_BINORMAL";
  static constexpr std::string_view default_edge_length_field_name_ = "EDGE_LENGTH";
  //@}

  //! \name Internal members
  //@{

  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  // Material properties
  double youngs_modulus_;
  double poissons_ratio_;
  double shear_modulus_;
  double density_;

  // Node rank fields
  stk::mesh::Field<double> *node_coordinates_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_twist_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_curvature_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_rest_curvature_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_rotation_gradient_field_ptr_ = nullptr;

  // Edge rank fields
  stk::mesh::Field<double> *edge_orientation_field_ptr_ = nullptr;
  stk::mesh::Field<double> *edge_tangent_field_ptr_ = nullptr;
  stk::mesh::Field<double> *edge_binormal_field_ptr_ = nullptr;
  stk::mesh::Field<double> *edge_length_field_ptr_ = nullptr;

  // Element rank fields
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // ComputeSLTConstraintForce


int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  SpermSimulation().run(argc, argv);
  
  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

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

/// \file GenerateCollisionConstraints.cpp
/// \brief Definition of the GenerateCollisionConstraints class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>     // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_meta/MetaFactory.hpp>                      // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                       // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>                       // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>                     // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>                 // for mundy::meta::PartRequirements
#include <mundy_mesh/BulkData.hpp>                         // for mundy::mesh::BulkData
#include <mundy_methods/GenerateCollisionConstraints.hpp>  // for mundy::methods::GenerateCollisionConstraints

namespace mundy {

namespace methods {

// \name Constructors and destructor
//{

GenerateCollisionConstraints::GenerateCollisionConstraints(mundy::mesh::BulkData *const bulk_data_ptr,
                                                           const Teuchos::ParameterList &fixed_parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "GenerateCollisionConstraints: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_fixed_params());

  // Parse the parameters
  Teuchos::ParameterList &parts_parameter_list = valid_fixed_parameter_list.sublist("input_parts");
  num_parts_ = parts_parameter_list.get<unsigned>("count");
  part_ptr_vector_.resize(num_parts_);
  for (size_t i = 0; i < num_parts_; i++) {
    // Fetch the i'th part and its parameters
    Teuchos::ParameterList &part_parameter_list = parts_parameter_list.sublist("input_part_" + std::to_string(i));
    const std::string part_name = part_parameter_list.get<std::string>("name");
    part_ptr_vector_[i] = meta_data_ptr_->get_part(part_name);

    // Fetch the parameters for this part's kernel
    const Teuchos::ParameterList &part_kernel_parameter_list =
        part_parameter_list.sublist("kernels").sublist("compute_aabb");

    // Create the kernel instance.
    const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
    compute_aabb_kernel_ptrs_.push_back(
        mundy::meta::MetaKernelFactory<void, GenerateCollisionConstraints>::create_new_instance(
            kernel_name, bulk_data_ptr_, part_kernel_parameter_list));
  }

  // For this method, the parts cannot intersect, if they did the result could be non-deterministic.
  for (size_t i = 0; i < num_parts_; i++) {
    for (size_t j = 0; j < num_parts_; j++) {
      if (i != j) {
        const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector_[i], *part_ptr_vector_[j]);
        TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                   "GenerateCollisionConstraints: Part " << part_ptr_vector_[i]->name() << " and "
                                                                         << "Part " << part_ptr_vector_[j]->name()
                                                                         << "intersect.");
      }
    }
  }
}
//}

// \name MetaMethod interface implementation
//{

Teuchos::ParameterList GenerateCollisionConstraints::set_mutable_params(
    const Teuchos::ParameterList &mutable_parameter_list) const {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_mutable_parameter_list = mutable_parameter_list;
  valid_mutable_parameter_list.validateParametersAndSetDefaults(this->get_valid_mutable_params());
}
//}

// \name Actions
//{

void GenerateCollisionConstraints::execute(const stk::mesh::Selector &input_selector) {
  // Two words of word of warning:
  //   1. This method is programmed with care to avoid generating duplicative constraints. To do so, we only generate a
  //      collision constraint if the entity_key of the source particle is less than that of the target particle.
  //   2. Parts of this method require mesh modification and are, therefore, incompatible with the GPU. We our best to
  //      isolate these sections.

  // If we make some basic assumptions then it's possible to break this process into a (CPU-based) constraint generation
  // routine and a (potentially GPU-based) constraint update routine.
  //
  // The core assumption is this: every pair of neighbors in the neighbor list will induce 1 collision constraint and 2
  // nodes. The nodes will connect to the dynamic connectivity (linkers) of the colliding pair. Under these assumptions
  // GenerateCollisionConstraints should
  //   0. Copy and clean up the neighbor list.
  //   1. Ghost neighbors and their downward connectivity.
  //   2. Call begin_modification()
  //   3. (On the CPU) Generate a new collision entity and two new nodes for each element of the neighbor list.
  //   4. (On the CPU) Generate a relation between the collision entity and its nodes as well as those nodes and the
  //   linker for the pair of particles. This step is independent of the multibody type associated with the elements.
  //   5. Call end_modification()
  //   6. (On the CPU or GPU) Loops over the generated collision constraints and call this kernel.
  // This kernel should
  //   0. Take in a a collision constraint and its left and right spheres.
  //   1. Fetch the linker's connected nodes and any of its fields necessary to compute the contact locations, contact
  //   normal, and signed separation distance.

  // Some issues:
  // - Issue: If attributes are fetched via type then how are we given the neighbor list? We make strong assumptions
  //     about our neighbor list. For example, complementarity collision constraints want the neighbor detection to have
  //     a buffer distance but potential-based methods typically want the neighbor detection to be as tight as possible.
  //     This class can't be the one to generate the neighbors tho since other classes may wish to loop over the
  //     neighbors. Yeah, but will those methods want to loop over the neighbors with our specific buffer distance?
  //     That doesn't seem unreasonable.
  //   Solution: the neighbor list can be passed in as a temporary parameter since Teuchos::ParameterList can legit
  //     hold any variable type.
  // - Issue: Who gets linkers and will they only every store the surface connectivity? What if I wanted to connect a
  //     sphere to another sphere?
  //   Solution: For now, only elements will receive linkers to encode their
  //     dynamic surface connectivity. Once we build up a higher level of abstraction, we can break dynamic
  //     connectivity into subsets.
  // - Issue: If linkers are generated on the fly, then GenerateCollisionConstraints is one of the classes that should
  //     generate linkers. I really don't want to generate a linker for every edge, face, and element.
  //   Solution: Polytopes are either represented as a super-element with its own linker or as a collection of linked
  //     elements. Either way, we only consider element-to-element neighbor detection, and one linker per element. Maybe
  //     add a flag to the multibody types to get if they have dynamic connectivity or not.
  // - Issue: Setting up this kernel in such a way that the node positions and their fields can be updated in a way that
  //     satisfies sharing/ghosting is hard. If we loop over the collision constraints, then how do we choose the
  //     correct kernel. If we loop over the neighbors, how can we guarantee that the sphere, linker, and its nodes are
  //     ghosted or shared?
  //   Solution: In the current design, the process that generated the collision constraint should have access to the
  //     linkers and spheres but not the node of the ghosted sphere. We need to ghost the downward connectivity
  //     of our neighbors.
  // - Issue: Given two spheres, how do we fetch the collision constraint that links them?
  //   Solution: Once GenerateCollisionConstraints generates the collision constraints, it should store them with the
  //     neighbor list such that we can pass this kernel the constraint and the two spheres without needing to perform
  //     complicated lookups. This will require modifying mundy's data structors to better accommodate KWay kernels
  //     without code repetition.

  for (size_t i = 0; i < num_parts_; i++) {
    std::shared_ptr<mundy::meta::MetaKernelBase<void>> compute_aabb_kernel_ptr = compute_aabb_kernel_ptrs_[i];

    stk::mesh::Selector locally_owned_part = meta_data_ptr_->locally_owned_part() & *part_ptr_vector_[i];
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEM_RANK, locally_owned_part,
        [&compute_aabb_kernel_ptr]([[maybe_unused]] const mundy::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
          compute_aabb_kernel_ptr->execute(element);
        });
  }
}
//}

}  // namespace methods

}  // namespace mundy

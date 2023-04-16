#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert

class Coarse_Seach{
public:

	Coarse_Seach() = delete;

	Coarse_Seachconst stk::mesh::BulkData *bulk_data_ptr, const std::vector<*stk::mesh::Part> &part_ptr_vector,
              const Teuchos::ParameterList &parameter_list)
    	: bulk_data_ptr_(bulk_data_ptr), part_ptr_vector_(part_ptr_vector), num_parts_(part_ptr_vector_.size()) {
		
		TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                                "mundy::methods::ComputeAABB: bulk_data_ptr cannot be a nullptr.");

        // The parts cannot intersect.
        for (int i = 0; i < num_parts_; i++) {
            for (int j = 0; j < num_parts_; j++) {
                fi(i==j) continue;
                const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector[i], *part_ptr_vector[j]);
                TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                        "mundy::methods::ComputeAABB: Part " << part_ptr_vector[i]->name() << " and "
                                                                                << "Part " << part_ptr_vector[j]->name()
                                                                                << "intersect.");
            }
			total_elements += *part_ptr_vector_[i].size();
		}
			// Store the input parameters, use default parameters for any parameter not given.
		// Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
		parameter_list_ = parameter_list;
		parameter_list_.validateParametersAndSetDefaults(get_valid_params());

		aabb_field_name_ = parameter_list_.get<std::string>("aabb_field_name");
		aabb_field_ptr_ = bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, aabb_field_name_);
		aabb_field_ptr_device_ = stk::mesh::get_updated_ngp_field<double>(aabb_field_ptr_);
	
	}


  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set(
        "aabb_field_name", default_aabb_field_name_,
        "Name of the element field within which the output axis-aligned boundary boxes will be written.");
    return default_parameter_list;
  }
	
	
	void execute() {
		Kokkos::View<ArborX::Box *, stk::mesh::NgpMesh::MeshExecSpace> boxes("Search::boxes",total_elements);
		int elements_so_far = 0;

		for (int i = 0; i < num_parts_; i++) {
			int part_size = *part_ptr_vector_[i].size();
			stk::mesh::Selector locally_owned_part = meta_mesh.locally_owned_part() && *part_ptr_vector_[i];

			// For this parallel_for, Kernels are same for all parts. 
			mundy::utils::for_each_entity_run(
				mesh,
				locally_owned_part.primary_entity_rank(),
				locally_owned_part,
				KOKKOS_LAMBDA(stk::mesh::Entity elem){
					stk::mesh::FastMeshIndex elemIndex = mesh.fast_mesh_index(elem);

					ArborX::Point min_corner = {aabb_field_ptr_device_(index, 0) - R,
												aabb_field_ptr_device_(index, 1) - R,
												aabb_field_ptr_device_(index, 2) - R};
					ArborX::Point max_corner = {aabb_field_ptr_device_(index, 0) + R,
												aabb_field_ptr_device_(index, 1) + R,
												aabb_field_ptr_device_(index, 2) + R};

					int box_idx = elements_so_far+elem.local_offset();
					boxes(box_idx) = {min_corner, max_corner};
				}
			);
			elements_so_far += part_size;
		}

		ArborX::BVH<CudaMemorySpace> index(stk::mesh::execution_space, boxes);
		Kokkos::View<int *, stk::mesh::NgpMesh::MeshExecSpace> indices("Search::indices", 0);
		Kokkos::View<int *, stk::mesh::NgpMesh::MeshExecSpace> offsets("Search::offsets", 0);

		index.query(execution_space, boxes, ExcludeSelfCollision{}, indices, offsets);
		}
}

private:
size_t num_parts_;
size_t total_elements;
static constexpr std::string default_aabb_field_name_ = "AABB";
std::string aabb_field_name_;

stk::mesh::Field<double> *aabb_field_ptr_;
stk::mesh::NgpField<double> * aabb_field_ptr_device_;

stk::mesh::DeviceMesh &mesh;
std::vector<*stk::mesh::Part> part_ptr_vector_;
std::vector<MetaKernel> get_boxes_from_aabb_kernels_;


};
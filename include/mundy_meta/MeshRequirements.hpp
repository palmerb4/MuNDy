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

#ifndef MUNDY_META_MESHREQUIREMENTS_HPP_
#define MUNDY_META_MESHREQUIREMENTS_HPP_

/// \file MeshRequirements.hpp
/// \brief Declaration of the MeshRequirements class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements
#include <mundy_mesh/BulkData.hpp>   // for mundy::mesh::BulkData

namespace mundy {

namespace meta {

/// \class MeshRequirements
/// \brief A set requirements imposed upon a MetaMesh, its Parts, and its Fields.
class MeshRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  MeshRequirements() = default;

  /// \brief Construct from a parameter list.
  ///
  /// \param parameter_list [in] Optional list of parameters for specifying the mesh requirements. The set of valid
  /// parameters is accessible via \c get_valid_params.
  explicit MeshRequirements(const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the spatial dimension of the mash.
  /// \param spatial_dimension [in] The dimension of the space within which the parts and entities reside.
  void set_spatial_dimension(const unsigned spatial_dimension);

  /// \brief Set the names assigned to each rank.
  /// \param entity_rank_names [in] The names assigned to each rank.
  MeshBuilder &set_entity_rank_names(const std::vector<std::string> &entity_rank_names);

  /// \brief Set the MPI communicator to be used by STK.
  /// \param comm [in] The MPI communicator.
  MeshBuilder &set_communicator(const stk::ParallelMachine &comm);

  /// \brief Set the chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  /// \param aura_option [in] The chosen Aura option.
  MeshBuilder &set_aura_option(const mundy::mesh::BulkData::AutomaticAuraOption &aura_option);

  /// \brief Set the field data manager.
  /// \param field_data_manager_ptr [in] Pointer to an existing field data manager.
  MeshBuilder &set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr);

  /// \brief Set the upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param bucket_capacity [in] The bucket capacity.
  MeshBuilder &set_bucket_capacity(const unsigned bucket_capacity);

  /// \brief Set the flag specifying if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  MeshBuilder &set_upward_connectivity_flag(const bool enable_upward_connectivity);

  /// \brief Get if the spatial dimension is constrained or not.
  bool constrains_spatial_dimension() const;

  /// \brief Get if the entity rank names are constrained or not.
  bool constrains_entity_rank_names() const;

  /// \brief Get if the communicator is constrained or not.
  bool constrains_communicator() const;

  /// \brief Get if the aura option is constrained or not.
  bool constrains_aura_option() const;

  /// \brief Get if the field data manager is constrained or not.
  bool constrains_field_data_manager() const;

  /// \brief Get if the bucket capacity is constrained or not.
  bool constrains_bucket_capacity() const;

  /// \brief Get if the upward connectivity flag is constrained or not.
  bool constrains_upward_connectivity_flag() const;

  /// \brief Get if the given type is a constrained mesh attribute or not.
  template <class T>
  void constrains_field_attribute() const {
    return field_attributes_.template get<std::shared_ptr<T>>() != nullptr;
  }

  /// \brief Return the dimension of the space within which the parts and entities reside.
  unsigned get_spatial_dimension() const;

  /// \brief Return the names assigned to each rank.
  std::vector<std::string> get_entity_rank_names() const;

  /// \brief Return the MPI communicator to be used by STK.
  stk::ParallelMachine get_communicator() const;

  /// \brief Return the chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  mundy::mesh::BulkData::AutomaticAuraOption get_aura_option() const;

  /// \brief Return the field data manager.
  stk::mesh::FieldDataManager *const get_field_data_manager() const;

  /// \brief Return the upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned get_bucket_capacity() const;

  /// \brief Return if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  bool get_upward_connectivity_flag() const;

  /// \brief Return the mesh field map.
  /// \brief field_rank [in] Rank associated with the retrieved fields.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> get_mesh_field_map();

  /// \brief Get the default parameters for this class.
  static Teuchos::ParameterList get_valid_params() {
    // TODO(palmerb4): This is wrong. We have sub-parameters that specify the fields and parts.
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("spatial_dimension", default_spatial_dimension_,
                               "Dimension of the space within which the parts and entities reside.");
    default_parameter_list.set("entity_rank_names", default_entity_rank_names_,
                               "Vector of names assigned to each rank.");
    default_parameter_list.set("communicator", default_communicator_, "MPI communicator to be used by STK..");
    default_parameter_list.set("aura_option", default_aura_option_, "The chosen Aura option.");
    default_parameter_list.set("field_data_manager_ptr", default_field_data_manager_ptr_,
                               "A pointer to a preexisting field data manager.");
    default_parameter_list.set(
        "bucket_capacity", default_bucket_capacity_,
        "Upper bound on the number of mesh entities that may be associated with a single bucket.");
    default_parameter_list.set("upward_connectivity_flag", default_upward_connectivity_flag_,
                               "Flag specifying if upward connectivity will be enabled or not.");
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare the mesh that this class defines including any of its fields, parts, and their
  /// fields/parts/subparts.
  ///
  /// The only setting that must be specified before declaring the mesh is the MPI communicator; all other settings have
  /// default options which will be used if not set.
  std::shared_ptr<mundy::mesh::BulkData> &declare_mesh() const;

  /// \brief Delete the spatial dimension constraint (if it exists).
  void delete_spatial_dimension_constraint() const;

  /// \brief Delete the entity rank names constraint (if it exists).
  void delete_entity_rank_names_constraint() const;

  /// \brief Delete the communicator constraint (if it exists).
  void delete_communicator_constraint() const;

  /// \brief Delete the aura option constraint (if it exists).
  void delete_aura_option_constraint() const;

  /// \brief Delete the field data manager constraint (if it exists).
  void delete_field_data_manager_constraint() const;

  /// \brief Delete the bucket capacity constraint (if it exists).
  void delete_bucket_capacity_constraint() const;

  /// \brief Delete the upward connectivity flag constraint (if it exists).
  void delete_upward_connectivity_flag_constraint() const;

  /// \brief Delete the specified attribute constraint (if it exists).
  template <class T>
  void delete_mesh_attribute_constraint() const {
    auto value = mesh_attributes_.template get<std::shared_ptr<T>>();
    mesh_attributes_.template remove<std::shared_ptr<T>>(value);
  }

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   - TODO(palmerb4): What are the mesh invariants set by STK?
  void check_if_valid() const;

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  ///
  /// \param field_req_ptr [in] Pointer to the field parameters to add to the part.
  void add_field_req(std::shared_ptr<FieldRequirementsBase> field_req_ptr);

  /// \brief Add the provided part to the mesh, given that it is valid.
  ///
  /// TODO(palmerb4): Are there any restrictions on what can and cannot be a part? If so, encode them here.
  ///
  /// \param part_req_ptr [in] Pointer to the part requirements to add to the mesh.
  void add_part_req(std::shared_ptr<PartRequirements> part_req_ptr);

  /// \brief Require that the mesh have a specific mesh attribute with known type.
  ///
  /// \note Attributes are fetched from an mundy::mesh::MetaData via the get_attribute<T> routine. As a result, the
  /// identifying feature of an attribute is its type. If you attempt to add a new attribute requirement when an
  /// attribute of that type already exists, then the contents of the two attributes must match.
  template <class T>
  void add_mesh_attribute_req(const std::shared_ptr<T> some_attribute_ptr) {
    mesh_attributes_.template insert_with_no_delete<std::shared_ptr<T>>(some_attribute_ptr);
  }

  /// \brief Merge the current requirements with another \c MeshRequirements.
  ///
  /// \param part_req_ptr [in] An \c MeshRequirements object to merge with the current object.
  void merge(const std::shared_ptr<MeshRequirements> &mesh_req_ptr);

  /// \brief Merge the current requirements with any number of other \c MeshRequirements.
  ///
  /// \param vector_of_part_req_ptrs [in] A vector of pointers to other \c MeshRequirements objects to merge with the
  /// current object.
  void merge(const std::vector<std::shared_ptr<MeshRequirements>> &vector_of_mesh_req_ptrs);
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr unsigned default_spatial_dimension_ = 3;
  static std::vector<std::string> default_entity_rank_names_ = std::vector<std::string>();
  static constexpr stk::ParallelMachine default_communicator_ = MPI_COMM_NULL;
  static constexpr mundy::mesh::BulkData::AutomaticAuraOption default_aura_option_ = mundy::mesh::BulkData::AUTO_AURA;
  static constexpr stk::mesh::FieldDataManager *default_field_data_manager_ptr_ = nullptr;
  static constexpr unsigned default_bucket_capacity_ = stk::mesh::impl::BucketRepository::default_bucket_capacity;
  static constexpr bool default_upward_connectivity_flag_ = true;
  //@}

  /// @brief The dimension of the space within which the parts and entities reside.
  unsigned spatial_dimension_;

  /// @brief The names assigned to each rank.
  std::vector<std::string> entity_rank_names_;

  /// @brief The MPI communicator to be used by STK.
  stk::ParallelMachine communicator_;

  /// @brief The chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  mundy::mesh::BulkData::AutomaticAuraOption aura_option_;

  /// @brief Pointer to an existing field data manager.
  stk::mesh::FieldDataManager *field_data_manager_ptr_;

  /// @brief Upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned bucket_capacity_;

  /// @brief A flag specifying if upward connectivity will be enabled or not.
  bool upward_connectivity_flag_;

  /// \brief If the spacial dimension is set or not.
  bool spatial_dimension_is_set_ = false;

  /// \brief If the names of each rank are set or not.
  bool entity_rank_names_is_set_ = false;

  /// \brief If the MPI communicator is set or not.
  bool comm_is_set_ = false;

  /// \brief If the aura option is set or not.
  bool aura_option_is_set_ = false;

  /// \brief If the field manager is set or not.
  bool field_data_manager_ptr_is_set_ = false;

  /// \brief If the bucket capacity is set or not.
  bool bucket_capacity_is_set_ = false;

  /// \brief If the upward connectivity flag is set or not.
  bool enable_upward_connectivity_is_set_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> part_ranked_field_maps_{
      stk::topology::NUM_RANKS};

  /// \brief A map from part name to the part params for that part.
  std::map<std::string, std::shared_ptr<PartRequirements>> mesh_part_map_;

  /// \brief Any attributes associated with this mesh.
  stk::CSet mesh_attributes_;
};  // MeshRequirements

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_MESHREQUIREMENTS_HPP_

# The goal of this script is to generate all of our aggregate and augment classes without having to manually write them all out.

class DataItemInfo:
    def __init__(self, name_upper, name_camel, name_lower, rank, data_type, documentation):
        """
        Initialize a DataItemInfo object.

        :param name_upper: The name of the data item in uppercase (e.g., EXAMPLEDATA1)
        :param name_camel: The name of the data item in camel case (e.g., ExampleData1)
        :param name_lower: The name of the data item in lowercase (e.g., example_data1)
        :param rank: The expected rank of the data item (e.g., NODE_RANK, EDGE_RANK, etc.)
        :param data_type: The type of the data item if shared (default is None)
        :param documentation: The doxygen documentation for the data item (default is an empty string)
        """
        self.name_upper = name_upper
        self.name_camel = name_camel
        self.name_lower = name_lower
        self.rank = rank
        self.shared_data_type = data_type
        self.documentation = documentation

    def __repr__(self):
        return (f"DataItemInfo(name_upper={self.name_upper}, name_camel={self.name_camel}, "
                f"name_lower={self.name_lower}, rank={self.rank}, data_type={self.data_type}, "
                f"documentation={self.documentation})")
    
class AggregateInfo:
    def __init__(self, name_upper, name_camel, name_lower, valid_topologies, documentation, data_items):
        """
        Initialize an AggregateInfo object.

        :param name_upper: The name of the class in uppercase (e.g., EXAMPLENAME)
        :param name_camel: The name of the class in camel case (e.g., ExampleName)
        :param name_lower: The name of the class in lowercase (e.g., example_name)
        :param valid_topologies: The list of valid topologies for the class
        :param documentation: The doxygen documentation for the class
        :param data_items: A list of DataItemInfo objects
        """
        self.name_upper = name_upper
        self.name_camel = name_camel
        self.name_lower = name_lower
        self.valid_topologies = valid_topologies
        self.documentation = documentation
        self.data_items = data_items

    def __repr__(self):
        return (f"AggregateInfo(name_upper={self.name_upper}, name_camel={self.name_camel}, "
                f"name_lower={self.name_lower}, valid_topologies={self.valid_topologies}, "
                f"documentation={self.documentation}, data_items={self.data_items})")

class EntityGetterInfo:
    def __init__(self, name, entity_access, entity_index_access, requirements=None):
        """
        Initialize an EntityGetter object.

        Example use:
            EntityGetterInfo(name='ellipsoid', entity_access='Base::entity()', entity_index_access='Base::entity_index()')

        :param name: The name of the entity getter
        :param entity_access: The access pattern for the entity getter
        :param entity_index_access: The access pattern for the entity index getter
        :param requirements: The requirements for the entity getter (default is None)
        """
        self.name = name
        self.entity_access = entity_access
        self.entity_index_access = entity_index_access
        self.requirements = requirements

    def __repr__(self):
        return f"EntityGetter(name={self.name}, entity_access={self.entity_access}, entity_index_access={self.entity_index_access})"

class DataGetterInfo:
    def __init__(self, name, generate_access_pattern, generate_ngp_access_pattern):
        """
        Initialize a DataGetter object.

        Example use:
            DataGetter(name='radius', 
                generate_access_pattern=
                    lambda data, object_used_for_access: f'stk::mesh::field_data({data}, {object_used_for_access})[0]'),
                generate_ngp_access_pattern=
                    lambda data, object_used_for_access: f'{data}({object_used_for_access}, 0)')

        :param name: The name of the data getter
        :param access_pattern: The access pattern for the data getter
        """
        self.name = name
        self.generate_access_pattern = generate_access_pattern
        self.generate_ngp_access_pattern = generate_ngp_access_pattern

    def __repr__(self):
        return (f"DataGetter(name={self.name}, generate_access_pattern={self.generate_access_pattern}), "
                f"generate_ngp_access_pattern={self.generate_ngp_access_pattern})")

def tagged_data_access(tag_name, data, access_pattern, object_used_for_access):
    access = ""
    access += f"if constexpr (std::is_base_of_v<tag_type::FIELD, {tag_name}>) {{\n"
    access += f"  return {access_pattern(data, object_used_for_access)};\n"
    access += f"}} else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, {tag_name}>) {{\n"
    access += f"  const auto part_ptrs = data_.parts();\n"
    access += f"  unsigned num_parts = part_ptrs.size();\n"
    access += f"  for (unsigned i = 0; i < num_parts; ++i) {{\n"
    access += f"    if (data_.bulk_data().bucket({object_used_for_access}).member(*part_ptr[i])) {{\n"

    data_subset = f"{data}()[i]"

    access += f"      return {access_pattern(data_subset, object_used_for_access)};\n"
    access += f"    }}\n"
    access += f"  }}\n"
    access += f"}} else if constexpr (std::is_base_of_v<tag_type::SHARED, {tag_name}>) {{\n"
    access += f"  return {data}();\n"
    access += f"}} else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, {tag_name}>) {{\n"
    access += f"  const auto part_ptrs = data_.parts();\n"
    access += f"  unsigned num_parts = part_ptrs.size();\n"
    access += f"  for (unsigned i = 0; i < num_parts; ++i) {{\n"
    access += f"    if (data_.bulk_data().bucket({object_used_for_access}).member(*part_ptr[i])) {{\n"

    data_subset = f"{data}()[i]"

    access += f"      return {access_pattern(data_subset, object_used_for_access)};\n"
    access += f"    }}\n"
    access += f"  }}\n"
    access += f"}} else {{\n"
    access += f"  MUNDY_THROW_ASSERT(false, std::invalid_argument, \"Invalid tag type. This should be unreachable.\");\n"
    access += f"}}\n"

    return access

def scalar_access_pattern(data, object_used_for_access):
    return f'stk::mesh::field_data({data}, {object_used_for_access})[0]'

def ngp_scalar_access_pattern(data, object_used_for_access):
    return f'{data}({object_used_for_access}, 0)'

def vector3_access_pattern(data, object_used_for_access):
    return f'mundy::mesh::vector3_field_data({data}, {object_used_for_access})'

def ngp_vector3_access_pattern(data, object_used_for_access):
    return vector3_access_pattern(data, object_used_for_access)

def quaternion_access_pattern(data, object_used_for_access):
    return f'mundy::mesh::quaternion_field_data({data}, {object_used_for_access})'

def ngp_quaternion_access_pattern(data, object_used_for_access):
    return quaternion_access_pattern(data, object_used_for_access)

class EntityViewInfo:
    def __init__(self, aggregate_info, entity_getters, generate_data_getters):
        """
        Initialize an EntityViewInfo object.

        :param aggregate_info: An AggregateInfo object we are generating an entity view for
        :param entity_getters: A list of EntityGetter objects
        :param generate_data_getters: A list of GenerateDataGetter objects
        """
        self.aggregate_info = aggregate_info
        self.entity_getters = entity_getters
        self.generate_data_getters = generate_data_getters

    def __repr__(self):
        return (f"EntityViewInfo(aggregate_info={self.aggregate_info}, "
                f"entity_getters={self.entity_getters}, generate_data_getters={self.generate_data_getters})")

class TagInfo:
    def __init__(self, name_upper, name_lower, parent_tag, value):
        """
        Initialize a TagInfo object.

        :param name_upper: The name of the tag in uppercase (e.g., LINE)
        :param name_lower: The name of the tag in lowercase (e.g., line)
        :param patent_tag: The parent tag of the tag (e.g., tag_type::FIELD)
        :param value: The value of the tag
        """
        self.name_upper = name_upper
        self.name_lower = name_lower
        self.parent_tag = parent_tag
        self.value = value

    def __eq__(self, other):
        return self.name_upper == other.name_upper

    def __hash__(self):
        return hash(self.name_upper)

    def __repr__(self):
        return f"TagInfo(name_upper={self.name_upper}, name_lower={self.name_lower}, parent_tag={self.parent_tag}, value={self.value})"

class PlaceholderReplacement:
    def __init__(self, placeholder, replacement):
        """
        Initialize a PlaceholderReplacement object.

        :param placeholder: The placeholder string to be replaced
        :param replacement: The replacement string
        """
        self.placeholder = placeholder
        self.replacement = replacement

    def __repr__(self):
        return f"PlaceholderReplacement(placeholder={self.placeholder}, replacement={self.replacement})"
    
def read_template_file(file_path):
    """
    Read the content of the template file.

    :param file_path: The path to the template file
    :return: The content of the file as a string
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def perform_replacements(content, replacements):
    """
    Perform the replacements in the content.

    :param content: The original content
    :param replacements: A list of PlaceholderReplacement objects
    :return: The updated content
    """
    for replacement in replacements:
        content = content.replace(replacement.placeholder, replacement.replacement)
    return content

def save_updated_content(file_path, content):
    """
    Save the updated content back to the file.

    :param file_path: The path to the file
    :param content: The updated content
    """
    with open(file_path, 'w') as file:
        file.write(content)

def generate_topological_entity_view_replacements(entity_info):
    """
    Generate a list of PlaceholderReplacement objects based on the given AggregateInfo.

    These replacements are to create a TopologicalEntityView class.

    :param entity_info: An EbtityInfo object
    :return: A list of PlaceholderReplacement objects
    """
    aggregate_info = entity_info.aggregate_info
    replacements = []

    # Replace the names
    replacements.append(PlaceholderReplacement('EXAMPLENAME', aggregate_info.name_upper))
    replacements.append(PlaceholderReplacement('ExampleName', aggregate_info.name_camel))
    replacements.append(PlaceholderReplacement('example_name', aggregate_info.name_lower))

    # Replace the entity getters
    #   - entity_getters_placeholder
    #     '''
    #       stk::mesh::Entity ellipsoid_entity() const {
    #         return Base::entity();
    #       }
    #    
    #       stk::mesh::Entity center_node_entity() const {
    #         return Base::connected_node(0);
    #       }
    #     '''
    #
    # generate_entity_getters is a python function that takes in the topology returns [the name of the entity, a string for how the entity is accessed]
    entity_getters = ""
    for item in entity_info.entity_getters:
        if item.requirements:
            entity_getters += f"stk::mesh::Entity {item.name}_entity() const \n"
            entity_getters += f"requires({item.requirements}) {{\n"
            entity_getters += f"  return {item.entity_access};\n"
            entity_getters += f"}}\n\n"
        else:
            entity_getters += f"stk::mesh::Entity {item.name}_entity() const {{\n"
            entity_getters += f"  return {item.entity_access};\n"
            entity_getters += f"}}\n\n"
    
    entity_index_getters = ""
    for item in entity_info.entity_getters:
        if item.requirements:
            entity_index_getters += f"unsigned {item.name}_entity_index() const \n"
            entity_index_getters += f"requires({item.requirements}) {{\n"
            entity_index_getters += f"  return {item.entity_index_access};\n"
            entity_index_getters += f"}}\n\n"
        else:
            entity_index_getters += f"unsigned {item.name}_entity_index() const {{\n"
            entity_index_getters += f"  return {item.entity_index_access};\n"
            entity_index_getters += f"}}\n\n"

    replacements.append(PlaceholderReplacement('entity_getters_placeholder', entity_getters))
    replacements.append(PlaceholderReplacement('entity_index_getters_placeholder', entity_index_getters))

    # Replace the data getters
    #   - data_getters_placeholder
    #     '''
    #       decltype(auto) center() {
    #         return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_entity());
    #       }
    #    
    #       decltype(auto) center() const {
    #         return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_entity());
    #       }
    #    
    #       decltype(auto) orientation() {
    #         return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_entity());
    #       }
    #    
    #       decltype(auto) orientation() const {
    #         return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_entity());
    #       }
    #     '''

    # generate_access_pattern is a python function that takes in a string representing the data and the object used for access and returns a string
    # generate_object_used_for_access is a python function that takes in the topology and returns a string

    data_getters = ""
    for constness in ['const', '']:
        for item in entity_info.generate_data_getters:
            data_getters += f"decltype(auto) {item.name}() {constness} {{\n"

            for valid_topology in aggregate_info.valid_topologies:
                data_getters += f"  if constexpr (topology_t == stk::topology::{valid_topology}) {{\n"

                object_used_for_access = item.generate_object_used_for_access(valid_topology)
                access_pattern = item.generate_access_pattern(f"data_.{item.name}_data()", object_used_for_access)

                data_getters += f"    return {access_pattern};\n"
                data_getters += f"  }} else "

            data_getters += "{\n"
            data_getters += f"    MUNDY_THROW_ASSERT(false, std::invalid_argument, \"Invalid topology. This should be unreachable.\");\n"
            data_getters += f"}}\n\n"
    
    ngp_data_getters = ""
    for constness in ['const', '']:
        for item in entity_info.generate_ngp_data_getters:
            ngp_data_getters += f"decltype(auto) {item.name}() {constness} {{\n"

            for valid_topology in aggregate_info.valid_topologies:
                ngp_data_getters += f"  if constexpr (topology_t == stk::topology::{valid_topology}) {{\n"

                object_used_for_access = item.generate_ngp_object_used_for_access(valid_topology)
                access_pattern = item.generate_ngp_access_pattern(f"data_.{item.name}_data()", object_used_for_access)

                ngp_data_getters += f"    return {access_pattern};\n"
                ngp_data_getters += f"  }} else "

            ngp_data_getters += "{\n"
            ngp_data_getters += f"    MUNDY_THROW_ASSERT(false, std::invalid_argument, \"Invalid topology. This should be unreachable.\");\n"
            ngp_data_getters += f"}}\n\n"

    replacements.append(PlaceholderReplacement('getters_for_data_placeholder', data_getters))
    replacements.append(PlaceholderReplacement('getters_for_ngp_data_placeholder', ngp_data_getters))



def generate_topological_aggregate_replacements(aggregate_info):
    """
    Generate a list of PlaceholderReplacement objects based on the given AggregateInfo.

    Note: 
      Things being replaced must not be subsets of other things being replaced.
      This means that we sometimes place ngp awkwardly in the middle of a placeholder when 
      placing ngp_ at the beginning would be more readable.

    :param aggregate_info: An AggregateInfo object
    :return: A list of PlaceholderReplacement objects
    """
    replacements = []

    # Replace the names
    replacements.append(PlaceholderReplacement('EXAMPLENAME', aggregate_info.name_upper))
    replacements.append(PlaceholderReplacement('ExampleName', aggregate_info.name_camel))
    replacements.append(PlaceholderReplacement('example_name', aggregate_info.name_lower))

    # Replace the documentation
    replacements.append(PlaceholderReplacement('example_discussion_placeholder', aggregate_info.documentation))

    # Replace the template tags (with and without defaults)
    #   - list_of_templated_tags_with_defaults_placeholder becomes:
    #     '''
    #       typename ExampleData1Tag = EXAMPLE_DATA1_IS_FIELD,  //
    #       typename ExampleData2Tag = EXAMPLE_DATA2_IS_FIELD
    #     '''
    #   - list_of_templated_tags_placeholder
    #     '''
    #       typename ExampleData1Tag, typename ExampleData2Tag
    #     '''
    #   - list_of_tags_placeholder
    #     '''
    #       ExampleData1Tag, ExampleData2Tag
    #     '''
    templated_tags_with_defaults = ", //\n".join(
        [f"typename {item.name_camel}DataTag = {item.name_upper}_DATA_IS_FIELD" for item in aggregate_info.data_items]
    )
    templated_tags = ", ".join(
        [f"typename {item.name_camel}DataTag" for item in aggregate_info.data_items]
    )
    tag_list = ", ".join(
        [f"{item.name_camel}DataTag" for item in aggregate_info.data_items]
    )
    replacements.append(PlaceholderReplacement('list_of_templated_tags_with_defaults_placeholder', templated_tags_with_defaults))
    replacements.append(PlaceholderReplacement('list_of_templated_tags_placeholder', templated_tags))
    replacements.append(PlaceholderReplacement('list_of_tags_placeholder', tag_list))

    # Replace the static assert for valid topologies
    #   - static_assert_valid_topology_placeholder;
    #     '''
    #       static_assert(OurTopology::value == valid_topology[0] || OurTopology::value == valid_topology[1],
    #                     "The topology of an ellipsoid must be either valid_topology[0] or valid_topology[1]");
    #     '''
    is_valid_topology = " || ".join([f"OurTopology::value == {topology}" for topology in aggregate_info.valid_topologies])
    topology_or = " or ".join([f"{topology}" for topology in aggregate_info.valid_topologies])
    static_assert_string = f"The topology of an {aggregate_info.name_lower} must be either {topology_or}"
    static_assert_valid_topology = f"static_assert({is_valid_topology},\n\"{static_assert_string}\");"
    replacements.append(PlaceholderReplacement('static_assert_valid_topology_placeholder;', static_assert_valid_topology))

    # Replace the data type aliases (both ngp and non-ngp)
    data_type_aliases = "\n".join(
        [f"using {item.name_lower}_data_t = map_tag_to_data_type_t</* Tag */ {item.name_camel}DataTag, //\n"
         f"                                           /* Scalar */ scalar_t, //\n"
         f"                                           /* Shared type */ {item.shared_data_type}>;" for item in aggregate_info.data_items]
    )
    ngp_data_type_aliases = "\n".join(
        [f"using {item.name_lower}_data_t = map_tag_to_ngp_data_type_t</* Tag */ {item.name_camel}DataTag, //\n"
         f"                                           /* Scalar */ scalar_t, //\n"
         f"                                           /* Shared type */ {item.shared_data_type}>;" for item in aggregate_info.data_items]
    )
    replacements.append(PlaceholderReplacement('data_type_aliases_placeholder;', data_type_aliases))
    replacements.append(PlaceholderReplacement('data_type_ngp_aliases_placeholder;', ngp_data_type_aliases))

    # Replace the input data placeholders and their initializations
    #   - list_of_input_data_placeholder
    #     '''
    #       example_data1, example_data2
    #     '''
    #   - list_of_input_data_initialization_placeholder
    #     '''
    #       example_data1_(example_data1_), example_data2_(example_data2_)
    #     '''
    list_of_input_data = ", ".join([item.name_lower for item in aggregate_info.data_items])
    list_of_input_data_initialization = ", ".join([f"{item.name_lower}_data_({item.name_lower}_data_)" for item in aggregate_info.data_items])

    replacements.append(PlaceholderReplacement('list_of_input_data_placeholder', list_of_input_data))
    replacements.append(PlaceholderReplacement('list_of_input_data_initialization_placeholder', list_of_input_data_initialization))

    # Replace the input data doxygen documentation   
    #   - /// params_documentation_placeholder
    #     '''
    #       /// \param example_data1 The example_data1 data documentation (shared type: ExampleData1SharedType)
    #       /// \param example_data2 The example_data2 data documentation (shared type: ExampleData2SharedType)
    #     '''
    params_documentation = "\n".join(
        [f"/// \\param {item.name_lower}_data {item.documentation} (shared type: {item.shared_data_type})" 
            for item in aggregate_info.data_items]
    )
    replacements.append(PlaceholderReplacement('/// params_documentation_placeholder', params_documentation))

    # Replace the assert correct rank for input data (both ngp and non-ngp)
    #   There are two options here: 
    #    If the rank is SAME_RANK_AS_TOPOLOGY, we must assert that the the rank of the data item is rank_t
    #    Otherwise, the rank of the data item must equal the given rank
    rank_assertions = []
    for item in aggregate_info.data_items:
        if item.rank == 'SAME_RANK_AS_TOPOLOGY':
            rank_assertions.append(
                f"if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::FIELD>) {{\n"
                f"  MUNDY_THROW_ASSERT({item.name_lower}_data_->entity_rank() == rank_t, std::invalid_argument,\n"
                f"                     \"The {item.name_lower}_data data must be a field of rank_t\");\n"
                f"}} else if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::VECTOR_OF_FIELDS>) {{\n"
                f"  for (const auto {item.name_lower}_data_field_ptr_ : {item.name_lower}_data_) {{\n"
                f"    MUNDY_THROW_ASSERT({item.name_lower}_data_field_ptr_->entity_rank() == rank_t, std::invalid_argument,\n"
                f"                       \"The {item.name_lower}_data data must be a vector of fields of rank_t\");\n"
                f"  }}\n"
                f"}}"
            )
        else:
            rank_assertions.append(
                f"if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::FIELD>) {{\n"
                f"  MUNDY_THROW_ASSERT({item.name_lower}_data_->entity_rank() == {item.rank}, std::invalid_argument,\n"
                f"                     \"The {item.name_lower}_data data must be a field of {item.rank}\");\n"
                f"}} else if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::VECTOR_OF_FIELDS>) {{\n"
                f"  for (const auto {item.name_lower}_data_field_ptr_ : {item.name_lower}_data_) {{\n"
                f"    MUNDY_THROW_ASSERT({item.name_lower}_data_field_ptr_->entity_rank() == {item.rank}, std::invalid_argument,\n"
                f"                       \"The {item.name_lower}_data data must be a vector of fields of {item.rank}\");\n"
                f"  }}\n"
                f"}}"
            )
    assert_correct_rank_for_input_data = "\n\n".join(rank_assertions)

    ngp_rank_assertions = []
    for item in aggregate_info.data_items:
        if item.rank == 'SAME_RANK_AS_TOPOLOGY':
            ngp_rank_assertions.append(
                f"if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::FIELD>) {{\n"
                f"  MUNDY_THROW_ASSERT({item.name_lower}_data_->get_rank() == rank_t, std::invalid_argument,\n"
                f"                     \"The {item.name_lower}_data data must be a field of rank_t\");\n"
                f"}} else if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::VECTOR_OF_FIELDS>) {{\n"
                f"  for (const auto {item.name_lower}_data_field_ptr_ : {item.name_lower}_data_) {{\n"
                f"    MUNDY_THROW_ASSERT({item.name_lower}_data_field_ptr_->get_rank() == rank_t, std::invalid_argument,\n"
                f"                       \"The {item.name_lower}_data data must be a vector of fields of rank_t\");\n"
                f"  }}\n"
                f"}}"
            )
        else:
            ngp_rank_assertions.append(
                f"if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::FIELD>) {{\n"
                f"  MUNDY_THROW_ASSERT({item.name_lower}_data_->get_rank() == {item.rank}, std::invalid_argument,\n"
                f"                     \"The {item.name_lower}_data data must be a field of {item.rank}\");\n"
                f"}} else if constexpr (std::is_same_v<{item.name_camel}DataTag, tag_type::VECTOR_OF_FIELDS>) {{\n"
                f"  for (const auto {item.name_lower}_data_field_ptr_ : {item.name_lower}_data_) {{\n"
                f"    MUNDY_THROW_ASSERT({item.name_lower}_data_field_ptr_->get_rank() == {item.rank}, std::invalid_argument,\n"
                f"                       \"The {item.name_lower}_data data must be a vector of fields of {item.rank}\");\n"
                f"  }}\n"
                f"}}"
            )
    assert_correct_rank_for_input_ngp_data = "\n\n".join(ngp_rank_assertions)
    
    replacements.append(PlaceholderReplacement('assert_correct_rank_for_input_data_placeholder;', assert_correct_rank_for_input_data))
    replacements.append(PlaceholderReplacement('assert_correct_rank_for_input_ngp_data_placeholder;', assert_correct_rank_for_input_ngp_data))

    # Replace the getters for tags and data (both ngp and non-ngp)
    # NGP is the same as the non-ngp but with KOKKOS_INLINE_FUNCTION
    #   - getters_for_tags_placeholder;  (be sure to replace the ";". It's only there to allow the templates to be auto-formatted)
    #     '''
    #       /// \brief Get the tag for the example_data1
    #       static constexpr ExampleData1Tag get_example_data1_tag() {
    #         return ExampleData1Tag{};
    #       }
    #       
    #       /// \brief Get the tag for the example_data2
    #       static constexpr ExampleData2Tag get_example_data2_tag() {
    #         return ExampleData2Tag{};
    #       }
    #     '''
    #
    #   - getters_for_data_placeholder;
    #     '''
    #       /// \brief Get the example_data1
    #       const example_data1_t& example_data1() const {
    #         return example_data1_;
    #       }
    #       
    #       /// \brief Get the example_data2
    #       const example_data2_t& example_data2() const {
    #         return example_data2_;
    #       }
    #     '''

    getters_for_tags = "\n".join(
        [f"/// \\brief Get the tag for the {item.name_lower}_data\n"
         f"static constexpr {item.name_camel}DataTag get_{item.name_lower}_data_tag() {{\n"
         f"  return {item.name_camel}DataTag{{}};\n"
         f"}}\n" for item in aggregate_info.data_items]
    )
    getters_for_data = "\n".join(
        [f"/// \\brief Get the {item.name_lower}_data\n"
         f"const {item.name_lower}_data_t& {item.name_lower}_data() const {{\n"
         f"  return {item.name_lower}_data_;\n"
         f"}}\n" for item in aggregate_info.data_items]
    )

    ngp_getters_for_tags = "\n".join(
        [f"/// \\brief Get the tag for the {item.name_lower}_data\n"
         f"KOKKOS_INLINE_FUNCTION\n"
         f"static constexpr {item.name_camel}DataTag get_{item.name_lower}_data_tag() {{\n"
         f"  return {item.name_camel}DataTag{{}};\n"
         f"}}\n" for item in aggregate_info.data_items]
    )
    ngp_getters_for_data = "\n".join(
        [f"/// \\brief Get the {item.name_lower}_data\n"
         f"KOKKOS_INLINE_FUNCTION\n"
         f"const {item.name_lower}_data_t& {item.name_lower}_data() const {{\n"
         f"  return {item.name_lower}_data_;\n"
         f"}}\n" for item in aggregate_info.data_items]
    )

    replacements.append(PlaceholderReplacement('getters_for_tags_placeholder;', getters_for_tags))
    replacements.append(PlaceholderReplacement('getters_for_data_placeholder;', getters_for_data))
    replacements.append(PlaceholderReplacement('getters_for_ngp_tags_placeholder;', ngp_getters_for_tags))
    replacements.append(PlaceholderReplacement('getters_for_ngp_data_placeholder;', ngp_getters_for_data))

    # Replace the list of internal data
    #  - list_of_internal_data_placeholder;
    #    '''
    #      example_data1_t example_data1_;  ///< example_data1 description
    #      example_data2_t example_data2_;  ///< example_data2 description
    #    '''
    internal_data = "\n".join(
        [f"{item.name_lower}_data_t {item.name_lower}_data_;  ///< {item.documentation}" for item in aggregate_info.data_items]
    )

    replacements.append(PlaceholderReplacement('list_of_internal_data_placeholder;', internal_data))
    
    # Replace the type to tag deduction placeholders (both ngp and non-ngp)
    #  - type_to_tag_deduction_placeholder;
    #    '''
    #      using ExampleData1Tag = map_data_type_to_tag_t<ExampleData1Type>;
    #      using ExampleData2Tag = map_data_type_to_tag_t<ExampleData2Type>;
    #    '''
    #
    #  - type_to_ngp_tag_deduction_placeholder;
    #    '''
    #      using ExampleData1Tag = map_ngp_data_type_to_tag_t<ExampleData1Type>;
    #      using ExampleData2Tag = map_ngp_data_type_to_tag_t<ExampleData2Type>;
    #    '''

    type_to_tag_deduction = "\n".join(
        [f"using {item.name_camel}DataTag = map_data_type_to_tag_t<{item.name_camel}DataType>;" for item in aggregate_info.data_items]
    )
    ngp_type_to_tag_deduction = "\n".join(
        [f"using {item.name_camel}DataTag = map_ngp_data_type_to_tag_t<{item.name_camel}DataType>;" for item in aggregate_info.data_items]
    )

    replacements.append(PlaceholderReplacement('type_to_tag_deduction_placeholder;', type_to_tag_deduction))
    replacements.append(PlaceholderReplacement('type_to_ngp_tag_deduction_placeholder;', ngp_type_to_tag_deduction))

    # Replace the list of data
    #  - list_of_data_placeholder
    #   '''
    #     example_data1, example_data2
    #   '''
    #  - list_of_type_deduced_data_placeholder
    #   '''
    #     const ExampleData1Type &example_data1, const ExampleData2Type &example_data2
    #   '''
    #  - list_of_templated_data_types_placeholder
    #   '''
    #     ExampleData1Type, ExampleData2Type
    #   '''
    data_list = ", ".join([item.name_lower for item in aggregate_info.data_items])
    type_deduced_data_list = ", ".join([f"const {item.name_camel}DataType &{item.name_lower}_data" for item in aggregate_info.data_items])
    templated_data_types = ", ".join([f"{item.name_camel}DataType" for item in aggregate_info.data_items])
    replacements.append(PlaceholderReplacement('list_of_data_placeholder', data_list))
    replacements.append(PlaceholderReplacement('list_of_type_deduced_data_placeholder', type_deduced_data_list))
    replacements.append(PlaceholderReplacement('list_of_templated_data_types_placeholder', templated_data_types))

    return replacements

def create_topological_aggregate(template_path, aggregate_file_path, aggregate_info):
    """
    Use the TopologicalAggTemplate.hpp file to generate a new aggregate class.

    :param file_path: The path to the TopologicalAggTemplate.hpp file
    :param aggregate_file_path: The path to the output file
    :param aggregate_info: An AggregateInfo object for which we want to generate the class
    """
    content = read_template_file(template_path)
    replacements = generate_topological_replacements(aggregate_info)
    updated_content = perform_replacements(content, replacements)
    save_updated_content(aggregate_file_path, updated_content)

def random_unsigned():
    """
    Generate a random unsigned value between 0 and 4,294,967,295.

    :return: A random unsigned value
    """
    import random
    return random.randint(0, 2**32 - 1)

def create_tags(aggregate_info):
    """
    Create a set of tags for the given aggregate class.

    Each aggregate has its own tag with an all uppercase name. Such as LINE for the LineData aggregate. Each 
    data item has 4 tags consisting of its name followed by a qualifier about its type: *_IS_FIELD, 
    *_IS_VECTOR_OF_FIELDS, *_IS_SHARED, *_IS_VECTOR_OF_SHARED. These tags all inherit from centralized types 
    to make them easier to tell apart such as tag_type::FIELD or tag_type::AGGREGATE. We then associate with 
    each tag a random unsigned value between 0 and 4,294,967,295. All of this is done to ensure that the set
    of tags for an aggregate uniquely identifies it.

    :param aggregate_info: An AggregateInfo object
    """

    tags = []
    tags.append(TagInfo(name_upper=aggregate_info.name_upper, 
                        name_lower=aggregate_info.name_lower,
                        parent_tag='tag_type::AGGREGATE', 
                        value=random_unsigned()))
    
    qualifiers = []
    for item in aggregate_info.data_items:
        qualifiers.append(TagInfo(name_upper=f"{item.name_upper}_DATA_IS_FIELD", 
                            name_lower=f"{item.name_lower}_data_is_field",
                            parent_tag='tag_type::FIELD', 
                            value=random_unsigned()))
        qualifiers.append(TagInfo(name_upper=f"{item.name_upper}_DATA_IS_VECTOR_OF_FIELDS", 
                            name_lower=f"{item.name_lower}_data_is_vector_of_fields",
                            parent_tag='tag_type::VECTOR_OF_FIELDS', 
                            value=random_unsigned())) 
        qualifiers.append(TagInfo(name_upper=f"{item.name_upper}_DATA_IS_SHARED", 
                            name_lower=f"{item.name_lower}_data_is_shared",
                            parent_tag='tag_type::SHARED', value=random_unsigned()))    
        qualifiers.append(TagInfo(name_upper=f"{item.name_upper}_DATA_IS_VECTOR_OF_SHARED", 
                            name_lower=f"{item.name_lower}_data_is_vector_of_shared",
                            parent_tag='tag_type::VECTOR_OF_SHARED', 
                            value=random_unsigned()))

    return tags, qualifiers

def write_tags_to_file(template_tag_file_path, tag_file_path, aggregate_infos):
    """
    Create a file that contains the tags for a set of aggregates.

    Each tag has the following format:
        /// @brief The Tag identifying our data type
        struct NAME : public PARENT_NAME {
          static constexpr unsigned value = VALUE;
        };
        //
        constexpr line_tag_v = LINE::value;

        /// @brief The inverse map from tag value to tag type 
        template <>
        struct value_to_tag_type<3727727993> {
          using type = LINE;
        };

    These tags are placed in the template by replacing the all_tags_placeholder with the tags.

    :param template_tag_file_path: The path to the template file
    :param tag_file_path: The path to the output file
    :param tags: A list of TagInfo objects
    """
    tags = []
    qualifiers = []
    for aggregate_info in aggregate_infos:
        aggregate_tags, aggregate_qualifiers = create_tags(aggregate_info)
        tags.extend(aggregate_tags)
        qualifiers.extend(aggregate_qualifiers)

    # Sort and unique the tags based on their name_upper
    tags = sorted(list(set(tags)), key=lambda tag: tag.name_upper)
    qualifiers = sorted(list(set(qualifiers)), key=lambda tag: tag.name_upper)

    # Place the qualifiers at the very end for organization and readability
    tags.extend(qualifiers)

    tag_content = "\n\n".join(
        [f"/// @brief The Tag identifying our data type\n"
        f"struct {tag.name_upper} : public {tag.parent_tag} {{\n"
        f"  static constexpr unsigned value = {tag.value};\n"
        f"}};\n"
        f"//\n"
        f"constexpr auto {tag.name_lower}_v = {tag.name_upper}::value;\n"
        f"\n"
        f"/// @brief The inverse map from tag value to tag type\n"
        f"template <>\n"
        f"struct value_to_tag_type<{tag.value}> {{\n"
        f"  using type = {tag.name_upper};\n"
        f"}};\n" for tag in tags]
    )

    tag_template_content = read_template_file(template_tag_file_path)
    replacements = [PlaceholderReplacement('all_tags_placeholder', tag_content)]
    updated_tag_content = perform_replacements(tag_template_content, replacements)
    save_updated_content(tag_file_path, updated_tag_content)

if __name__ == '__main__':
    topological_template_path = 'TemplateTopologicalAgg.hpp'
    tag_template_path = 'TemplateTags.hpp'

    # Lets create our real aggregates
    #   - EllipsoidData 
    #       Data: center (NODE_RANK), orientation (SAME_RANK_AS_TOPOLOGY), radii (SAME_RANK_AS_TOPOLOGY)
    #       Valid topologies: NODE, PARTICLE
    #   - LineData
    #       Data: center (NODE_RANK), direction (SAME_RANK_AS_TOPOLOGY)
    #       Valid topologies: NODE, PARTICLE
    #   - PointData 
    #       Data: center (NODE_RANK)
    #       Valid topologies: NODE, PARTICLE
    #   - SphereData 
    #       Data: center (NODE_RANK), radius (SAME_RANK_AS_TOPOLOGY)
    #       Valid topologies: NODE, PARTICLE
    #   - SpherocylinderData 
    #       Data: center (NODE_RANK), orientation (SAME_RANK_AS_TOPOLOGY), radius (SAME_RANK_AS_TOPOLOGY), length (SAME_RANK_AS_TOPOLOGY)
    #       Valid topologies: NODE, PARTICLE
    #   - SpherocylinderSegmentData 
    #       Data: node_coords (NODE_RANK)
    #       Valid topologies: LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, SPRING_3
    #   - VSegmentData 
    #       Data: node_coords (NODE_RANK)
    #       Valid topologies: SPRING_3, TRI_3, SHELL_TRI_3

    ellipsoid_data = AggregateInfo(
        name_upper='ELLIPSOID',
        name_camel='Ellipsoid',
        name_lower='ellipsoid',
        valid_topologies=['NODE', 'PARTICLE'],
        documentation="The topology of an ellipsoid directly effects the access pattern for the underlying data:\n"
                      "///   - NODE: All data is stored on a single node\n"
                      "///   - PARTICLE: The center is stored on a node, whereas the orientation and radii are stored on the\n"
                      "///   element-rank particle",
        data_items=[
            DataItemInfo(name_upper='CENTER', 
                         name_camel='Center', 
                         name_lower='center',
                         rank='NODE_RANK', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Center of each ellipsoid (NODE_RANK).'),
            DataItemInfo(name_upper='ORIENTATION', 
                         name_camel='Orientation', 
                         name_lower='orientation',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='mundy::math::Quaternion<scalar_t>', 
                         documentation='Orientation of each ellipsoid (SAME_RANK_AS_TOPOLOGY).'),
            DataItemInfo(name_upper='RADII', 
                         name_camel='Radii', 
                         name_lower='radii',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Radii of each ellipsoid (SAME_RANK_AS_TOPOLOGY).')
        ])

    ellipsoid_entity_view = EntityViewInfo(
        aggregate_info=ellipsoid_data,
        entity_getters=[
            EntityGetterInfo(name='ellipsoid', 
                             entity_access='Base::entity()', 
                             entity_index_access='Base::entity_index()')
            EntityGetterInfo(name='center_node',
                             entity_access='Base::connected_node(0)',
                             entity_index_access='Base::connected_node_index(0)',
                             requirements='topology_t == stk::topology::PARTICLE')],
        data_getters=[
            DataGetterInfo(name='center', generate_access_pattern=vector3_access_pattern, 
                            generate_ngp_access_pattern=ngp_vector3_access_pattern),
            DataGetterInfo(name='orientation', generate_access_pattern=quaternion_access_pattern,
                            generate_ngp_access_pattern=ngp_quaternion_access_pattern),
            DataGetterInfo(name='radii', generate_access_pattern=vector3_access_pattern,
                            generate_ngp_access_pattern=ngp_vector3_access_pattern)
        ])
    




    line_data = AggregateInfo(
        name_upper='LINE',
        name_camel='Line',
        name_lower='line',
        valid_topologies=['NODE', 'PARTICLE'],
        documentation="",
        data_items=[
            DataItemInfo(name_upper='CENTER', 
                         name_camel='Center', 
                         name_lower='center',
                         rank='NODE_RANK', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Center of each line (NODE_RANK).'),
            DataItemInfo(name_upper='DIRECTION', 
                         name_camel='Direction', 
                         name_lower='direction',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Direction of each line (SAME_RANK_AS_TOPOLOGY).')
        ])
    
    point_data = AggregateInfo(
        name_upper='POINT',
        name_camel='Point',
        name_lower='point',
        valid_topologies=['NODE', 'PARTICLE'],
        documentation="",
        data_items=[
            DataItemInfo(name_upper='CENTER', 
                         name_camel='Center', 
                         name_lower='center',
                         rank='NODE_RANK', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Center of each point (NODE_RANK).')
        ])

    sphere_data = AggregateInfo(
        name_upper='SPHERE',
        name_camel='Sphere',
        name_lower='sphere',
        valid_topologies=['NODE', 'PARTICLE'],
        documentation="",
        data_items=[
            DataItemInfo(name_upper='CENTER', 
                         name_camel='Center', 
                         name_lower='center',
                         rank='NODE_RANK', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Center of each sphere (NODE_RANK).'),
            DataItemInfo(name_upper='RADIUS', 
                         name_camel='Radius', 
                         name_lower='radius',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='scalar_t', 
                         documentation='Radius of each sphere (SAME_RANK_AS_TOPOLOGY).')
        ])
    
    spherocylinder_data = AggregateInfo(
        name_upper='SPHEROCYLINDER',
        name_camel='Spherocylinder',
        name_lower='spherocylinder',
        valid_topologies=['NODE', 'PARTICLE'],
        documentation="",
        data_items=[
            DataItemInfo(name_upper='CENTER', 
                         name_camel='Center', 
                         name_lower='center',
                         rank='NODE_RANK', 
                         data_type='mundy::math::Vector3<scalar_t>', 
                         documentation='Center of each spherocylinder (NODE_RANK).'),
            DataItemInfo(name_upper='ORIENTATION', 
                         name_camel='Orientation', 
                         name_lower='orientation',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='mundy::math::Quaternion<scalar_t>', 
                         documentation='Orientation of each spherocylinder (SAME_RANK_AS_TOPOLOGY).'),
            DataItemInfo(name_upper='RADIUS', 
                         name_camel='Radius', 
                         name_lower='radius',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='scalar_t', 
                         documentation='Radius of each spherocylinder (SAME_RANK_AS_TOPOLOGY).'),
            DataItemInfo(name_upper='LENGTH', 
                         name_camel='Length', 
                         name_lower='length',
                         rank='SAME_RANK_AS_TOPOLOGY', 
                         data_type='scalar_t', 
                         documentation='Length of each spherocylinder (SAME_RANK_AS_TOPOLOGY).')
        ])
    
    spherocylinder_segment_data = AggregateInfo(
        name_upper='SPHEROCYLINDER_SEGMENT',
        name_camel='SpherocylinderSegment',
        name_lower='spherocylinder_segment',
        valid_topologies=['LINE_2', 'LINE_3', 'BEAM_2', 'BEAM_3', 'SPRING_2', 'SPRING_3'],
        documentation="",
        data_items=[
            DataItemInfo(name_upper='NODE_COORDS', 
                         name_camel='NodeCoords', 
                         name_lower='node_coords',
                         rank='NODE_RANK', 
                         data_type='std::vector<mundy::math::Vector3<scalar_t>>', 
                         documentation='Coordinates of the nodes of each segment (NODE_RANK).')
        ])
    
    v_segment_data = AggregateInfo(
        name_upper='V_SEGMENT',
        name_camel='VSegment',
        name_lower='v_segment',
        valid_topologies=['SPRING_3', 'TRI_3', 'SHELL_TRI_3'],
        documentation="",
        data_items=[
            DataItemInfo(name_upper='NODE_COORDS', 
                         name_camel='NodeCoords', 
                         name_lower='node_coords',
                         rank='NODE_RANK', 
                         data_type='std::vector<mundy::math::Vector3<scalar_t>>', 
                         documentation='Coordinates of the nodes of each V segment (NODE_RANK).')
        ])

    aggs = [ellipsoid_data, line_data, point_data, sphere_data, spherocylinder_data, spherocylinder_segment_data, v_segment_data]
    for agg in aggs:
        create_topological_aggregate(topological_template_path, f'{agg.name_camel}Data.hpp', agg)
    
    write_tags_to_file(tag_template_path, 'Tags.hpp', aggs)

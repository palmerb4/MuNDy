// C++ core
#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace stk {

namespace mesh {

template <typename T>
class Field {
 public:
  Field(const std::string& name) : name_(name) {
  }

  std::string name() const {
    return name_;
  }

 private:
  std::string name_;
};  // Field

class Part {
 public:
  Part(const std::string& name) : name_(name) {
  }

  std::string name() const {
    return name_;
  }

 private:
  std::string name_;
};  // Part

class Attribute {
 public:
  Attribute(const std::string& name) : name_(name) {
  }

  std::string name() const {
    return name_;
  }

 private:
  std::string name_;
};  // Attribute

}  // namespace mesh

}  // namespace stk

/// \def MUNDY_THROW_ASSERT
/// \brief Throw an exception if the given assertion is false.
///
/// This macro is a a revised version of Teuchos' \c TEUCHOS_TEST_FOR_EXCEPTION
/// macro with improved logic. Unlike
/// \c TEUCHOS_TEST_FOR_EXCEPTION, this macro will throw an exception if the
/// assertion is false. If the assertion is true, nothing happens. This macro is
/// intended to be used in place of \c TEUCHOS_TEST_FOR_EXCEPTION in order to
/// improve code readability.
///
/// \param assertion_to_test The assertion to test
/// \param exception_to_throw The exception to throw if the assertion is false
/// \param message_to_print The message to print if the assertion is false
#define MUNDY_THROW_ASSERT(assertion_to_test, exception_to_throw, message_to_print)                  \
  do {                                                                                               \
    const bool assertion_failed = !(assertion_to_test);                                              \
    if (assertion_failed) {                                                                          \
      std::ostringstream omsg;                                                                       \
      omsg << "Assertion failed in " << __func__ << "\nFile: " << __FILE__ << "\nLine: " << __LINE__ \
           << "\nMessage: " << message_to_print << std::endl;                                        \
      const std::string& omsgstr = omsg.str();                                                       \
      throw exception_to_throw(omsgstr);                                                             \
    }                                                                                                \
  } while (0)

namespace mundy {

namespace meta {

class MetaMethod {
 public:
  virtual std::type_index get_type_index() const = 0;
  virtual std::string get_name() const = 0;
  virtual std::string get_namespace() const = 0;
  std::string get_full_name() const {
    return get_namespace() + "::" + get_name();
  }
  virtual void run() = 0;
};  // MetaMethod


/// \class MetaMethodHierarchy (Think of this as a collection of drop down menus)
/// Given a string name at runtime, construct the corresponding MetaMethod
///
/// Methods are organized into hierarchies. There is one GlobalMetaMethodHierarchy that contains all MetaMethods registered with their full name.
///
/// Otherwise, each MetaMethod may be declared as a subset of a certain group (identified by a string name) and given a string name that is unique within that group.
/// The explicit requirements for the uniqueness of the string names for groups and MetaMethods are the same as namespaces and classes in C++.
///  A subgroup can have one, and only one, parent group.
///  A subgroup cannot have the same name as one of its parent group.
///  A MetaMethod within a subgroup cannot have the same name as a method in any of its parent groups.
/// These requirements cannot be enforced at static time, but they are enforced at runtime.
///
/// Convention-wise, group names follow namespace snake_case conventions and MetaMethod names follow class CamelCase conventions.
class MetaMethodHierarchy {

  /* 
  
  The full names of all methods are stored in a single vector.
  The full name of a method may be used to fetch its create_new_instance function.
  Groups are simply a tree-structure of strings. Each Group has a name and an optional parent group name.
  Each method in a group is associated with a custom string name. Within the group hierarchy, all that is stored about this method is its 
  group-specific name and full name.
  
  
  */


  //! \name Internal hierarchy of method names.
  //@{

  /// \brief A helper class for generating a hierarchy of string names.
  class Group {
   public:
    /// \brief Construct a new Group object wit
    /// \param group_name [in] The name of this group.
    /// \param parent_group_name [in] The name of our parent group.
    Group(const std::string& group_name, const std::string& parent_group_name = "");

    /// \brief Add a subgroup to the current group.
    void add_subgroup(std::shared_ptr<Group> subgroup);

    /// \brief Add a method to the current group.
    void add_method(const std::string& local_method_name, const std::string& full_method_name);

    /// \brief Get our name.
    std::string name() const;

    /// \brief Get our parent's name.
    std::string parent_name() const;

    /// \brief Get one of our subgroups by name.
    std::shared_ptr<Group> get_subgroup(const std::string& name) const;

    /// \brief Get all of our subgroups.
    std::vector<std::shared_ptr<Group>> get_subgroups() const;

    /// \brief Get one of our methods by its group-specific name.
    std::string get_method(const std::string& local_method_name) const;

    /// \brief Get the local names of all of our methods.
    std::vector<std::string> get_methods() const;

   private:
    const std::string group_name_;
    const std::string parent_group_name_;
    std::vector<std::shared_ptr<Group>> subgroups_;
    std::unordered_map<std::string, std::string> methods_local_to_full_;
  };  // Group

  /// \brief A non-member builder class for \c Group's that accounts for the static initialization order.
  class GroupHierarchyManager {
   public:
    static std::shared_ptr<Group> create_group(const std::string& group_name, const std::string& parent_group_name);

    /// \brief Print the hierarchy.
    static void print_hierarchy(std::ostream& os = std::cout);

   private:
    static void print_hierarchy(std::shared_ptr<Group> group, std::ostream& os = std::cout, int depth = 0);

    static inline std::map<std::string, std::shared_ptr<Group>> root_group_map_;
    static inline std::map<std::string, std::shared_ptr<Group>> orphaned_group_map_;
  };  // GroupHierarchyManager




  /// \brief A non-member builder class for \c Group's that accounts for the static initialization order.
  class StringTreeManager {
   public:
    static std::shared_ptr<Group> create_node(const unsigned& id, const std::string& name,
                                                       const std::vector<std::string>& parent_names);

    /// \brief Print the tree.
    static void print_tree(std::ostream& os = std::cout);

   private:
    /// \brief Print the tree.
    /// \param node The node in the tree to print.
    /// \param depth The depth of the node in the tree.
    static void print_tree(std::shared_ptr<Group> node, std::ostream& os = std::cout, int depth = 0);

    static inline std::map<std::string, std::shared_ptr<Group>> root_node_map_;
    static inline std::map<std::string, std::shared_ptr<Group>> node_map_;
  };  // StringTreeManager

  //@}

};


/*
Each MetaMethod is limited in what it can intake. We want to support MetaMethods
that accept Fields, Parts, other MetaMethods, and Attributes that are either
const (a promise that they aren't modified) or non-const (may be modified). The
MetaMethodFunctionFactory contains the set of all 6 types of inputs that a
certain MetaMethod accepts and the ability to call any of them using their
string name.

Aka. If ExampleMetaMethod only accepts a const node coordinates field, then
MetaMethodFunctionFactory<ExampleMetaMethod> will store that ExampleMetaMethod
accepts a const version of this field and allow you to access it with a string
via
MetaMethodFunctionFactory<ExampleMetaMethod>::get(an_example_meta_method_instance,
"const_node_coordinates_field"). This will return a pointer to the const node
coordinates field, accessed via calling
an_example_meta_method_instance.get_const_node_coordinates_field(). The same
will work for the set method.
*/

/// \class MetaMethodFunctionFactory
/// Given an instance of a meta method, call one of its set/get functions based
/// on the string name of the function and the inputs to the function.
class MetaMethodFunctionFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A getter function type that takes in a MetaMethod instance and
  /// returns a pointer to a type.
  template <typename T>
  using Getter = std::function<T*(const MetaMethod* const)>;

  /// \brief A setter function type that takes in a MetaMethodType instance
  /// and a pointer to a type.
  template <typename T>
  using Setter = std::function<void(MetaMethod* const, T* const)>;
  //@}

  //! \name Our Getters
  //@{

  /// \brief Get the number of functions this method recognizes.
  static int num_registered_functions(const MetaMethod* const meta_method_ptr) {
    auto method_t_index = meta_method_ptr->get_type_index();
    return static_cast<int>(get_internal_function_strings()[method_t_index].size());
  }

  /// \brief Get the set of all registered functions for the provided
  /// MetaMethod.
  static std::vector<std::string> get_function_strings(const MetaMethod* const meta_method_ptr) {
    auto method_t_index = meta_method_ptr->get_type_index();
    return get_internal_function_strings()[method_t_index];
  }

  /// \brief Get the set of all registered function as a single string.
  static std::string get_functions_as_string(const MetaMethod* const meta_method_ptr) {
    auto method_t_index = meta_method_ptr->get_type_index();
    std::string all_functions_as_string;
    for (int i = 0; i < num_registered_functions(meta_method_ptr); ++i) {
      all_functions_as_string += get_internal_function_strings()[method_t_index][i];
      if (i != num_registered_functions(meta_method_ptr) - 1) {
        all_functions_as_string += ", ";
      }
    }
    return all_functions_as_string;
  }

  /// \brief Get if the provided function string is valid or not for the
  /// provided MetaMethod.
  /// \param function_string [in] A function string that may or may not
  /// correspond to a registered function.
  static bool is_valid_function(const MetaMethod* const meta_method_ptr, const std::string& function_string) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return std::count(get_internal_function_strings()[method_t_index].begin(),
                      get_internal_function_strings()[method_t_index].end(), function_string) != 0;
  }
  //@}

  //! \name MetaMethod function interface
  //@{

  /*
  We have 8 getters and 8 setters that call the appropriate getter/setter
  function based on the string name. Getters return a pointer to the desired
  object. Setters are overloaded. They either take a pointer to the desired
  object or they accept the sufficient parameters to fetch the object and set
  it from the BulkData.

  get_const_field<field_value_type>
  get_field<field_value_type>
  get_const_part
  get_part
  get_const_meta_method<OtherMetaMethodType>
  get_meta_method<OtherMetaMethodType>
  get_const_attribute<attribute_type>
  get_attribute<attribute_type>

  set_const_field<field_value_type>
  set_field<field_value_type>
  set_const_part
  set_part
  set_const_meta_method<OtherMetaMethodType>
  set_meta_method<OtherMetaMethodType>
  set_const_attribute<attribute_type>
  set_attribute<attribute_type>
  */

  /// \brief Get a pointer to a const field of the provided type.
  template <typename FieldValueType>
  static stk::mesh::Field<FieldValueType>* get_const_field(const MetaMethod* const meta_method_ptr,
                                                           const std::string& field_fuction_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return std::any_cast<stk::mesh::Field<FieldValueType>*>(
        get_const_field_getters()[method_t_index][field_fuction_name](meta_method_ptr));
  }

  /// \brief Get a pointer to a field of the provided type.
  template <typename FieldValueType>
  static stk::mesh::Field<FieldValueType>* get_field(const MetaMethod* const meta_method_ptr,
                                                     const std::string& field_fuction_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return std::any_cast<stk::mesh::Field<FieldValueType>*>(
        get_field_getters()[method_t_index][field_fuction_name](meta_method_ptr));
  }

  /// \brief Get a pointer to a const part.
  static stk::mesh::Part* get_const_part(const MetaMethod* const meta_method_ptr,
                                         const std::string& part_function_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return get_const_part_getters()[method_t_index][part_function_name](meta_method_ptr);
  }

  /// \brief Get a pointer to a part.
  static stk::mesh::Part* get_part(const MetaMethod* const meta_method_ptr, const std::string& part_function_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return get_part_getters()[method_t_index][part_function_name](meta_method_ptr);
  }

  /// \brief Get a pointer to a const meta method of the provided type.
  template <typename OtherMetaMethodType>
  static OtherMetaMethodType* get_const_meta_method(const MetaMethod* const meta_method_ptr,
                                                    const std::string& meta_method_function_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return std::any_cast<OtherMetaMethodType*>(
        get_const_meta_method_getters()[method_t_index][meta_method_function_name](meta_method_ptr));
  }

  /// \brief Get a pointer to a meta method of the provided type.
  template <typename OtherMetaMethodType>
  static OtherMetaMethodType* get_meta_method(const MetaMethod* const meta_method_ptr,
                                              const std::string& meta_method_function_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return std::any_cast<OtherMetaMethodType*>(
        get_meta_method_getters()[method_t_index][meta_method_function_name](meta_method_ptr));
  }

  /// \brief Get a pointer to a const attribute.
  static stk::mesh::Attribute* get_const_attribute(const MetaMethod* const meta_method_ptr,
                                                   const std::string& attribute_function_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return get_const_attribute_getters()[method_t_index][attribute_function_name](meta_method_ptr);
  }

  /// \brief Get a pointer to an attribute.
  static stk::mesh::Attribute* get_attribute(const MetaMethod* const meta_method_ptr,
                                             const std::string& attribute_function_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    return get_attribute_getters()[method_t_index][attribute_function_name](meta_method_ptr);
  }

  /// \brief Set a const field of the provided type using a pointer to the
  /// field.
  template <typename FieldValueType>
  static void set_const_field(MetaMethod* const meta_method_ptr, const std::string& field_function_name,
                              stk::mesh::Field<FieldValueType>* const field_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_field_setters()[method_t_index][field_function_name](meta_method_ptr, std::any(field_ptr));
  }

  /// \brief Set a field of the provided type using a pointer to the field.
  template <typename FieldValueType>
  static void set_field(MetaMethod* const meta_method_ptr, const std::string& field_function_name,
                        stk::mesh::Field<FieldValueType>* const field_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_field_setters()[method_t_index][field_function_name](meta_method_ptr, std::any(field_ptr));
  }

  /// \brief Set a const part using a pointer to the part.
  static void set_const_part(MetaMethod* const meta_method_ptr, const std::string& part_function_name,
                             stk::mesh::Part* const part_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_part_setters()[method_t_index][part_function_name](meta_method_ptr, part_ptr);
  }

  /// \brief Set a part using a pointer to the part.
  static void set_part(MetaMethod* const meta_method_ptr, const std::string& part_function_name,
                       stk::mesh::Part* const part_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_part_setters()[method_t_index][part_function_name](meta_method_ptr, part_ptr);
  }

  /// \brief Set a const meta method of the provided type using a pointer to
  /// the meta method.
  template <typename OtherMetaMethodType>
  static void set_const_meta_method(MetaMethod* const meta_method_ptr, const std::string& meta_method_function_name,
                                    OtherMetaMethodType* const other_meta_method_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_meta_method_setters()[method_t_index][meta_method_function_name](meta_method_ptr,
                                                                               std::any(other_meta_method_ptr));
  }

  /// \brief Set a meta method of the provided type using a pointer to the
  /// meta method.
  template <typename OtherMetaMethodType>
  static void set_meta_method(MetaMethod* const meta_method_ptr, const std::string& meta_method_function_name,
                              OtherMetaMethodType* const other_meta_method_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_meta_method_setters()[method_t_index][meta_method_function_name](meta_method_ptr,
                                                                         std::any(other_meta_method_ptr));
  }

  /// \brief Set a const attribute using a pointer to the attribute.
  static void set_const_attribute(MetaMethod* const meta_method_ptr, const std::string& attribute_function_name,
                                  stk::mesh::Attribute* const attribute_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_attribute_setters()[method_t_index][attribute_function_name](meta_method_ptr, attribute_ptr);
  }

  /// \brief Set an attribute using a pointer to the attribute.
  static void set_attribute(MetaMethod* const meta_method_ptr, const std::string& attribute_function_name,
                            stk::mesh::Attribute* const attribute_ptr) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_attribute_setters()[method_t_index][attribute_function_name](meta_method_ptr, attribute_ptr);
  }

  /// \brief Set a const field of the provided type using the name of the
  /// field.
  static void set_const_field(MetaMethod* const meta_method_ptr, const std::string& field_function_name,
                              const std::string& field_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_field_setters_by_name()[method_t_index][field_function_name](meta_method_ptr, field_name);
  }

  /// \brief Set a field of the provided type using the name of the field.
  static void set_field(MetaMethod* const meta_method_ptr, const std::string& field_function_name,
                        const std::string& field_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_field_setters_by_name()[method_t_index][field_function_name](meta_method_ptr, field_name);
  }

  /// \brief Set a const part using the name of the part.
  static void set_const_part(MetaMethod* const meta_method_ptr, const std::string& part_function_name,
                             const std::string& part_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_part_setters_by_name()[method_t_index][part_function_name](meta_method_ptr, part_name);
  }

  /// \brief Set a part using the name of the part.
  static void set_part(MetaMethod* const meta_method_ptr, const std::string& part_function_name,
                       const std::string& part_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_part_setters_by_name()[method_t_index][part_function_name](meta_method_ptr, part_name);
  }

  /// \brief Set a const meta method of the provided type using the name of
  /// the meta method.
  static void set_const_meta_method(MetaMethod* const meta_method_ptr, const std::string& meta_method_function_name,
                                    const std::string& meta_method_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_meta_method_setters_by_name()[method_t_index][meta_method_function_name](meta_method_ptr,
                                                                                       meta_method_name);
  }

  /// \brief Set a meta method of the provided type using the name of the meta
  /// method.
  static void set_meta_method(MetaMethod* const meta_method_ptr, const std::string& meta_method_function_name,
                              const std::string& meta_method_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_meta_method_setters_by_name()[method_t_index][meta_method_function_name](meta_method_ptr, meta_method_name);
  }

  /// \brief Set a const attribute using the name of the attribute.
  static void set_const_attribute(MetaMethod* const meta_method_ptr, const std::string& attribute_function_name,
                                  const std::string& attribute_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_const_attribute_setters_by_name()[method_t_index][attribute_function_name](meta_method_ptr, attribute_name);
  }

  /// \brief Set an attribute using the name of the attribute.
  static void set_attribute(MetaMethod* const meta_method_ptr, const std::string& attribute_function_name,
                            const std::string& attribute_name) {
    const auto method_t_index = meta_method_ptr->get_type_index();
    get_attribute_setters_by_name()[method_t_index][attribute_function_name](meta_method_ptr, attribute_name);
  }
  //@}

  //! \name Registration
  //@{

  /// \brief Reset the factory to its initial state.
  ///
  /// This function removes all registered functions and clears all internal
  /// data structures.
  static void reset() {
    get_internal_function_strings().clear();
    get_const_field_getters().clear();
    get_field_getters().clear();
    get_const_part_getters().clear();
    get_part_getters().clear();
    get_const_meta_method_getters().clear();
    get_meta_method_getters().clear();
    get_const_attribute_getters().clear();
    get_attribute_getters().clear();

    get_const_field_setters().clear();
    get_field_setters().clear();
    get_const_part_setters().clear();
    get_part_setters().clear();
    get_const_meta_method_setters().clear();
    get_meta_method_setters().clear();
    get_const_attribute_setters().clear();
    get_attribute_setters().clear();

    get_const_field_setters_by_name().clear();
    get_field_setters_by_name().clear();
    get_const_part_setters_by_name().clear();
    get_part_setters_by_name().clear();
    get_const_meta_method_setters_by_name().clear();
    get_meta_method_setters_by_name().clear();
    get_const_attribute_setters_by_name().clear();
    get_attribute_setters_by_name().clear();
  }

  /// \brief Register a new const field getter.
  template <typename MetaMethodType, typename FieldValueType>
  static bool register_const_field_getter(const std::string& field_function_name,
                                          Getter<stk::mesh::Field<FieldValueType>> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, field_function_name);
    get_internal_function_strings()[method_t_index].push_back(field_function_name);
    get_const_field_getters()[method_t_index].insert(std::make_pair(field_function_name, make_any_getter(getter)));
    return true;
  }

  /// \brief Register a new field getter.
  template <typename MetaMethodType, typename FieldValueType>
  static bool register_field_getter(const std::string& field_function_name,
                                    Getter<stk::mesh::Field<FieldValueType>> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, field_function_name);
    get_internal_function_strings()[method_t_index].push_back(field_function_name);
    get_field_getters()[method_t_index].insert(std::make_pair(field_function_name, make_any_getter(getter)));
    return true;
  }

  /// \brief Register a new const part getter.
  template <typename MetaMethodType>
  static bool register_const_part_getter(const std::string& part_function_name, Getter<stk::mesh::Part> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, part_function_name);
    get_internal_function_strings()[method_t_index].push_back(part_function_name);
    get_const_part_getters()[method_t_index].insert(std::make_pair(part_function_name, getter));
    return true;
  }

  /// \brief Register a new part getter.
  template <typename MetaMethodType>
  static bool register_part_getter(const std::string& part_function_name, Getter<stk::mesh::Part> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, part_function_name);
    get_internal_function_strings()[method_t_index].push_back(part_function_name);
    get_part_getters()[method_t_index].insert(std::make_pair(part_function_name, getter));
    return true;
  }

  /// \brief Register a new const meta method getter.
  template <typename MetaMethodType, typename OtherMetaMethodType>
  static bool register_const_meta_method_getter(const std::string& meta_method_function_name,
                                                Getter<OtherMetaMethodType> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, meta_method_function_name);
    get_internal_function_strings()[method_t_index].push_back(meta_method_function_name);
    get_const_meta_method_getters()[method_t_index].insert(
        std::make_pair(meta_method_function_name, make_any_getter(getter)));
    return true;
  }

  /// \brief Register a new meta method getter.
  template <typename MetaMethodType, typename OtherMetaMethodType>
  static bool register_meta_method_getter(const std::string& meta_method_function_name,
                                          Getter<OtherMetaMethodType> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, meta_method_function_name);
    get_internal_function_strings()[method_t_index].push_back(meta_method_function_name);
    get_meta_method_getters()[method_t_index].insert(
        std::make_pair(meta_method_function_name, make_any_getter(getter)));
    return true;
  }

  /// \brief Register a new const attribute getter.
  template <typename MetaMethodType>
  static bool register_const_attribute_getter(const std::string& attribute_function_name,
                                              Getter<stk::mesh::Attribute> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, attribute_function_name);
    get_internal_function_strings()[method_t_index].push_back(attribute_function_name);
    get_const_attribute_getters()[method_t_index].insert(std::make_pair(attribute_function_name, getter));
    return true;
  }

  /// \brief Register a new attribute getter.
  template <typename MetaMethodType>
  static bool register_attribute_getter(const std::string& attribute_function_name,
                                        Getter<stk::mesh::Attribute> getter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, attribute_function_name);
    get_internal_function_strings()[method_t_index].push_back(attribute_function_name);
    get_attribute_getters()[method_t_index].insert(std::make_pair(attribute_function_name, getter));
    return true;
  }

  /// \brief Register a new const field setter.
  template <typename MetaMethodType, typename FieldValueType>
  static bool register_const_field_setter(const std::string& field_function_name,
                                          Setter<stk::mesh::Field<FieldValueType>> setter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, field_function_name);
    get_internal_function_strings()[method_t_index].push_back(field_function_name);
    get_const_field_setters()[method_t_index].insert(std::make_pair(field_function_name, make_any_setter(setter)));
    return true;
  }

  /// \brief Register a new field setter.
  template <typename MetaMethodType, typename FieldValueType>
  static bool register_field_setter(const std::string& field_function_name,
                                    Setter<stk::mesh::Field<FieldValueType>> setter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, field_function_name);
    get_internal_function_strings()[method_t_index].push_back(field_function_name);
    get_field_setters()[method_t_index].insert(std::make_pair(field_function_name, make_any_setter(setter)));
    return true;
  }

  /// \brief Register a new const part setter.
  template <typename MetaMethodType>
  static bool register_const_part_setter(const std::string& part_function_name, Setter<stk::mesh::Part> setter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, part_function_name);
    get_internal_function_strings()[method_t_index].push_back(part_function_name);
    get_const_part_setters()[method_t_index].insert(std::make_pair(part_function_name, setter));
    return true;
  }

  /// \brief Register a new part setter.
  template <typename MetaMethodType>
  static bool register_part_setter(const std::string& part_function_name, Setter<stk::mesh::Part> setter) {
    std::type_index method_t_index = typeid(MetaMethodType);
    assert_function_string_is_unique(method_t_index, part_function_name);
    get_internal_function_strings()[method_t_index].push_back(part_function_name);
    get_part_setters()[method_t_index].insert(std::make_pair(part_function_name, setter));
    return true;
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief An std::any getter function type that takes in a MetaMethodType
  /// instance and returns an std::any. This is used for fields and meta
  /// methods to overcome the temporary loss of type information when storing
  /// them in a map.
  using AnyGetter = std::function<std::any(const MetaMethod* const)>;

  /// \brief An std::any setter function type that takes in a MetaMethodType
  /// instance and an std::any.
  using AnySetter = std::function<void(MetaMethod* const, const std::any&)>;

  /// \brief A field setter function type that takes in a MetaMethodType
  /// instance and a field name.
  using FieldSetterByName = std::function<void(MetaMethod* const, const std::string&)>;

  /// \brief A part setter function type that in a MetaMethodType instance and
  /// a part name.
  using PartSetterByName = std::function<void(MetaMethod* const, const std::string&)>;

  /// \brief A meta method setter function type that in a MetaMethodType
  /// instance and a meta method name.
  using MetaMethodSetterByName = std::function<void(MetaMethod* const, const std::string&)>;

  /// \brief An attribute setter function type that in a MetaMethodType
  /// instance and an attribute name.
  using AttributeSetterByName = std::function<void(MetaMethod* const, const std::string&)>;
  //@}

  //! \name Internal helpers
  //@{

  /// \brief Make an AnyGetter from a Getter.
  template <typename T>
  static AnyGetter make_any_getter(Getter<T> getter) {
    return [getter](const MetaMethod* const meta_method_ptr) -> std::any { return std::any(getter(meta_method_ptr)); };
  }

  /// \brief Make an AnySetter from a Setter.
  template <typename T>
  static AnySetter make_any_setter(Setter<T> setter) {
    return [setter](MetaMethod* const meta_method_ptr, const std::any& value) {
      setter(meta_method_ptr, std::any_cast<T*>(value));
    };
  }

  /// \brief Assert that the function string is not already registered.
  static void assert_function_string_is_unique(const MetaMethod* const meta_method_ptr,
                                               const std::string& function_string) {
    MUNDY_THROW_ASSERT(
        !is_valid_function(meta_method_ptr, function_string), std::invalid_argument,
        "MetaMethodFunctionFactory: The provided function "
            << function_string << " already exists for the provided MetaMethod(" << meta_method_ptr->get_full_name()
            << "). The valid functions for this method are are: " << get_functions_as_string(meta_method_ptr));
  }

  static void assert_function_string_is_unique(const std::type_index& method_t_index,
                                               const std::string& function_string) {
    const double func_is_unique =
        std::count(get_internal_function_strings()[method_t_index].begin(),
                   get_internal_function_strings()[method_t_index].end(), function_string) == 0;
    MUNDY_THROW_ASSERT(func_is_unique, std::invalid_argument,
                       "MetaMethodFunctionFactory: The provided function "
                           << function_string << " already exists for the provided MetaMethod(" << method_t_index.name()
                           << ").");
  }
  //@}

  //! \name Internal getters
  //@{

  static std::unordered_map<std::type_index, std::vector<std::string>>& get_internal_function_strings() {
    static std::unordered_map<std::type_index, std::vector<std::string>> internal_function_strings;
    return internal_function_strings;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>>& get_const_field_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>> const_field_getters;
    return const_field_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>>& get_field_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>> field_getters;
    return field_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Part>>>&
  get_const_part_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Part>>>
        const_part_getters;
    return const_part_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Part>>>&
  get_part_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Part>>> part_getters;
    return part_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>>&
  get_const_meta_method_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>> const_meta_method_getters;
    return const_meta_method_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>>& get_meta_method_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnyGetter>> meta_method_getters;
    return meta_method_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Attribute>>>&
  get_const_attribute_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Attribute>>>
        const_attribute_getters;
    return const_attribute_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Attribute>>>&
  get_attribute_getters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Getter<stk::mesh::Attribute>>>
        attribute_getters;
    return attribute_getters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>>& get_const_field_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>> const_field_setters;
    return const_field_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>>& get_field_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>> field_setters;
    return field_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Part>>>&
  get_const_part_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Part>>>
        const_part_setters;
    return const_part_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Part>>>&
  get_part_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Part>>> part_setters;
    return part_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>>&
  get_const_meta_method_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>> const_meta_method_setters;
    return const_meta_method_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>>& get_meta_method_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AnySetter>> meta_method_setters;
    return meta_method_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Attribute>>>&
  get_const_attribute_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Attribute>>>
        const_attribute_setters;
    return const_attribute_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Attribute>>>&
  get_attribute_setters() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, Setter<stk::mesh::Attribute>>>
        attribute_setters;
    return attribute_setters;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, FieldSetterByName>>&
  get_const_field_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, FieldSetterByName>>
        const_field_setters_by_name;
    return const_field_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, FieldSetterByName>>&
  get_field_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, FieldSetterByName>>
        field_setters_by_name;
    return field_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, PartSetterByName>>&
  get_const_part_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, PartSetterByName>>
        const_part_setters_by_name;
    return const_part_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, PartSetterByName>>&
  get_part_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, PartSetterByName>> part_setters_by_name;
    return part_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, MetaMethodSetterByName>>&
  get_const_meta_method_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, MetaMethodSetterByName>>
        const_meta_method_setters_by_name;
    return const_meta_method_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, MetaMethodSetterByName>>&
  get_meta_method_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, MetaMethodSetterByName>>
        meta_method_setters_by_name;
    return meta_method_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AttributeSetterByName>>&
  get_const_attribute_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AttributeSetterByName>>
        const_attribute_setters_by_name;
    return const_attribute_setters_by_name;
  }

  static std::unordered_map<std::type_index, std::unordered_map<std::string, AttributeSetterByName>>&
  get_attribute_setters_by_name() {
    static std::unordered_map<std::type_index, std::unordered_map<std::string, AttributeSetterByName>>
        attribute_setters_by_name;
    return attribute_setters_by_name;
  }
  //@}
};  // MetaFactory

}  // namespace meta

}  // namespace mundy

/// \brief Macro to declare a setter, getter, and private data member for a
/// field
/// \param owning_class_type The class that owns the field
/// \param field_value_type The type of the field
/// \param field_name The name of the field
#define MUNDY_DECLARE_CONST_FIELD(owning_class_type, field_value_type, field_name)                       \
 public:                                                                                                 \
  /**                                                                                                    \
   * \brief Set the field_rank-rank field_name field.                                                    \
   *                                                                                                     \
   * \param const_##field_name##_field_ptr Pointer to field_value_type                                   \
   * to set.                                                                                             \
   * \return A reference to the owning_class_type instance for chaining.                                 \
   */                                                                                                    \
  owning_class_type& set_const_##field_name##_field(                                                     \
      stk::mesh::Field<field_value_type>* const const_##field_name##_field_ptr) {                        \
    const_##field_name##_field_ptr_ = const_##field_name##_field_ptr;                                    \
    return *this;                                                                                        \
  }                                                                                                      \
                                                                                                         \
  /**                                                                                                    \
   * \brief Get the field_rank-rank field_name field.                                                    \
   *                                                                                                     \
   * \return A pointer to the field_name field.                                                          \
   */                                                                                                    \
  stk::mesh::Field<field_value_type>* get_const_##field_name##_field() const {                           \
    return const_##field_name##_field_ptr_;                                                              \
  }                                                                                                      \
                                                                                                         \
 private:                                                                                                \
  stk::mesh::Field<field_value_type>* const_##field_name##_field_ptr_;                                   \
  static inline bool const_##field_name##_field_is_registered_ = []() {                                  \
    MetaMethodFunctionFactory::register_const_field_getter<owning_class_type, field_value_type>(         \
        std::string("get_const_") + #field_name + "_field",                                              \
        [](const MetaMethod* const obj_ptr) -> stk::mesh::Field<field_value_type>* {                     \
          return static_cast<const owning_class_type* const>(obj_ptr)->get_const_##field_name##_field(); \
        });                                                                                              \
    MetaMethodFunctionFactory::register_const_field_setter<owning_class_type, field_value_type>(         \
        std::string("set_const_") + #field_name + "_field",                                              \
        [](MetaMethod* const obj_ptr,                                                                    \
           stk::mesh::Field<field_value_type>* const const_##field_name##_field_ptr) -> void {           \
          static_cast<owning_class_type* const>(obj_ptr)->set_const_##field_name##_field(                \
              const_##field_name##_field_ptr);                                                           \
        });                                                                                              \
    return true;                                                                                         \
  }();

namespace mundy {

namespace meta {

class MyMetaMethod : public MetaMethod {
 public:
  std::type_index get_type_index() const override {
    return typeid(MyMetaMethod);
  }

  std::string get_name() const override {
    return std::string("MyMetaMethod");
  }

  std::string get_namespace() const override {
    return std::string("mundy::meta");
  }

  void run() override {
    std::cout << "Running MyMetaMethod" << std::endl;
    std::cout << "a field name: " << const_a_field_ptr_->name() << std::endl;
    std::cout << "b field name: " << const_b_field_ptr_->name() << std::endl;
  }

  MUNDY_DECLARE_CONST_FIELD(MyMetaMethod, int, a);
  MUNDY_DECLARE_CONST_FIELD(MyMetaMethod, double, b);
};

}  // namespace meta

}  // namespace mundy

int main() {
  // Create a fake a and b field
  stk::mesh::Field<int> a("a");
  stk::mesh::Field<double> b("b");

  // Create a MyMetaMethod instance and run it
  mundy::meta::MyMetaMethod().set_const_a_field(&a).set_const_b_field(&b).run();

  // Demonstrate that MetaMethodFunctionFactory works even without knowing MyMetaMethod's explicit type
  mundy::meta::MyMetaMethod my_meta_method;
  mundy::meta::MetaMethod * base_ptr = &my_meta_method;

  using OurFunctionFactory =
  mundy::meta::MetaMethodFunctionFactory;
  OurFunctionFactory::set_const_field<int>(base_ptr,
  "set_const_a_field", &a);
  OurFunctionFactory::set_const_field<double>(base_ptr,
  "set_const_b_field", &b); 
  std::cout << "a field name2: "
            << OurFunctionFactory::get_const_field<int>(base_ptr,
            "get_const_a_field")->name() << std::endl;
  std::cout << "b field name2: "
            << OurFunctionFactory::get_const_field<double>(base_ptr,
            "get_const_b_field")->name() << std::endl;
  base_ptr->run();
  return 0;
}

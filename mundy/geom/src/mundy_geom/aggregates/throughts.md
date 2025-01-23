# Aggregates should be augmentable! 

The add_augment function should check if the specified augment is compatible with the current class.
Compatibility likely comes down to checking topology/rank at compile time

#include <iostream>
#include <vector>
#include <array>

// Base Sphere class
class Sphere {
private:
    std::vector<double> center_data_; // Center coordinates
    double radius_;

public:
    Sphere(const std::vector<double>& center_data, double radius)
        : center_data_(center_data), radius_(radius) {}

    const std::vector<double>& center_data() const { return center_data_; }
    double radius() const { return radius_; }

    void print() const {
        std::cout << "Sphere: Center = (";
        for (size_t i = 0; i < center_data_.size(); ++i) {
            std::cout << center_data_[i];
            if (i + 1 < center_data_.size()) std::cout << ", ";
        }
        std::cout << "), Radius = " << radius_ << '\n';
    }

    // Chaining to add the next augment (supports additional template parameters)
    template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<Sphere, AugmentTemplates...>(*this, std::forward<Args>(args)...);
    }
};

// Mobility augment
template <typename Base>
class MobilityAugment : public Base {
private:
    std::vector<double> velocity_data_; // Velocity for each dimension

public:
    MobilityAugment(const Base& base, const std::vector<double>& velocity_data)
        : Base(base), velocity_data_(velocity_data) {}

    const std::vector<double>& velocity_data() const { return velocity_data_; }

    void update_position() {
        auto& center = this->center_data();
        std::cout << "Updating position with velocity: ";
        for (size_t i = 0; i < velocity_data_.size(); ++i) {
            std::cout << velocity_data_[i] << " ";
        }
        std::cout << '\n';
    }

    void print() const {
        Base::print();
        std::cout << "Velocity = (";
        for (size_t i = 0; i < velocity_data_.size(); ++i) {
            std::cout << velocity_data_[i];
            if (i + 1 < velocity_data_.size()) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    // Chaining to add the next augment
    template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<MobilityAugment, AugmentTemplates...>(*this, std::forward<Args>(args)...);
    }
};

// Color augment
template <typename Base, typename ColorType>
class ColorAugment : public Base {
private:
    ColorType color_data_; // RGB color

public:
    ColorAugment(const Base& base, const ColorType& color_data)
        : Base(base), color_data_(color_data) {}

    const ColorType& color_data() const { return color_data_; }

    void paint() {
        std::cout << "Painting object with color: (";
        for (size_t i = 0; i < 3; ++i) {
            std::cout << color_data_[i];
            if (i + 1 < 3) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    void print() const {
        Base::print();
        std::cout << "Color = (" << color_data_[0] << ", " << color_data_[1]
                  << ", " << color_data_[2] << ")\n";
    }

    // Chaining to add the next augment
    template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<ColorAugment, AugmentTemplates...>(*this, std::forward<Args>(args)...);
    }
};

// Example usage
int main() {
    std::vector<double> center = {1.0, 2.0, 3.0};
    double radius = 5.0;
    std::vector<double> velocity = {0.1, 0.2, 0.3};
    std::array<double, 3> color = {0.5, 0.7, 0.9};

    // Build the object with chaining
    auto augmented_sphere = Sphere(center, radius)
        .add_augment<MobilityAugment>(velocity)
        .add_augment<ColorAugment, std::array<double, 3>>(color);

    // Use the augmented object
    augmented_sphere.print();
    augmented_sphere.update_position();
    augmented_sphere.paint();

    return 0;
}



# The current design
In my original vision, aggregates weren't something that was declared up front. I saw them as something that was potentially created right before a method. This has the advantage of allowing aggregates to be seen as loose collections of fields/parts/shared data that are meant to be used to streamline functions. It's easy to store fields and parts and then use them to construct an aggregate before calling a function, thereby making it clear what is being acted upon. 

In this regard, aggregates are pre-baked collections of entities defined by their PartVector and the fields/shared objects that compose them.We design our methods to act on these aggregates, often independent of certain aspects of their data access. We have aggregates that represent abstract collections of objects such as ellipsoids, colored spheres, motile rods, inertialess fibers. These aggregates are defined by their unique set of Tags (we use set in the mathematical sense, as tags are independent of order). For example, colored spheres would be identified by the Tags (COLORED and SPHERES). We try to use nouns for the core aggregates and adjectives for their augments.

Aggregates are registered with their keys by type specializing aggregate_type, their entity view is registered by specializing entity_view_type. Because we often want to be able to ~lose~ aggregate type information and then recover it using their tags, you can register any aggregate object with our AggregateFactory via AggregateFactory::register<AggregateType, Set, of, keys>("string_name", aggregate). For us, we choose to perform the stash within the constructor, by calling AggregateFactory::register<AggregateType, Set, of, keys>(our_name, *this*). Note, aggregates are assumed to be copyable views. By this we mean that they do not contain any data that isn't free to be copied around. Aggregates can be accessed using their set of keys (independent of their order) via AggregateFactory::get<Set, of, keys>("string_name") or via AggregateFactory::get_all<Set, of, keys>(). Under the surface, register will stash away a copy of the given aggregate and will store a function for accessing said copy based on the given string name. The keys passed into the get-all function will then be used to fetch all copies that match the given keys or the name could be used to fetch a specific instance at the cost of string comparisons. Currently, we use a singleton factory design pattern, which does assume that there is only one BulkData. In the future, we could relax this by having BulkData "own" an instance of the AggregateFactory, requiring only small modifications to achieve.

Unlike part or fields, Aggregates may be created AFTER calling commit without incurring overhead cost; though, aggregates should be declared outside of repeated for loops to avoid incurring multiple string lookups. For example, if an aggregate has 4 keys, we will perform 4 map queries to see if the given aggregate exists.

We have a set of key types and a string name as our keys and a function whose type is determined by the set of keys. Unfortunately, due to our use CRTP for the augments, where an augment inherits from the object it extends, I don't think we can write for_each<HAS_AABB>, since the HAS_AABB augment is not uniquely identified by its tag alone. For some Tags, this makes sense. For example, MOTILE can't be accessed without its core object. HAS_AABB, HAS_COLOR, HAS_MASS, HAS_DENSITY could so long as we know their rank, so for_each<ELEM_RANK, HAS_AABB> would work. This underscores the core difference between aggregates and augments. Augments cannot be accessed without the aggregates they extend. This is reasonable since we aren't using type-based disbatch, so we won't write something like for_each<HAS_AABB>(compute_aabbs). We will write for_each<SPHERE, HAS_AABB>(compute_aabb_sphere). 

I'd say that we want users to call wrapper functions using objects that are non-ngp, so that we can fetch their NGP data, sync the fields to the correct space, call some function, and then mark any modified fields as modified. Users won't really call for_each much and those that do will write wrapper functions that mirror our wrapper functions. The nice thing about Tags is that users can create multiple species of Spheres and Spherocylinder, each with their own fields/shared values for radius/length and node coordinates and then call 

mundy::geom::compute_aabb_spheres(bulk_data, PartSubsetSelector, buffer_distance);  // Acts on all spheres that have aabb that are in the given selector.
mundy::geom::compute_aabb_sherocylinders(bulk_data, PartSubsetSelector, buffer_distance);

This is as opposed to 
mundy::geom::compute_aabb_spheres(sphere_data1, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_spheres(sphere_data2, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_spheres(sphere_data3, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_spheres(sphere_data4, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_spheres(sphere_data5, PartSubsetSelector, buffer_distance);

mundy::geom::compute_aabb_sherocylinders(sherocylinder_data1, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_sherocylinders(sherocylinder_data2, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_sherocylinders(sherocylinder_data3, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_sherocylinders(sherocylinder_data4, PartSubsetSelector, buffer_distance);
mundy::geom::compute_aabb_sherocylinders(sherocylinder_data5, PartSubsetSelector, buffer_distance);

This also means that the functions for acting on our augments are python compatible since they simply take in the bulk_data and don't care about the explicit type. The thing that would need to be made python compatible is the actual declaration of the augmented aggregates, which simply requires explicit template instantiation and is within the ability of super users.

Ok, back to our AggregateFactory. We need our AggregateFactory to interface with an AggregateRepository. The goal is to provide access to aggregate copy functions through a set of keys. Importantly, these are actual sets so duplicates are ignored and their order doesn't matter. Now, type specialization maintains order, so aggregate_type<Tags...> will return a type that matches that of the provided order and casting an aggregate of one order to another is invalid. This makes our life a bit difficult. The set of Tags should always contain one and only one aggregate in any position. It may contain any number of augments and qualifiers. mundy::core::is_subset_of_v<SetA, SetB> will state if SetB is a subset of SetA, these sets are simply mundy::core::type_set<Tags...>. How then can we go from the set of tags provided to AggregateFactory::register to an efficient way to access the constructor given a valid subset of the tags. I don't want us to store the function for generating the aggregate for all permutations of tags. Can we sort and unique a type_list to get a type_set? Yes. That prevents us from needing to store permutations, but doesn't help with our tree issue. If we have 4 tags, we currently need to store one generation function for each of the following:

Tag1, Tag2, Tag3, Tag4
Tag1, Tag2, Tag3
Tag1, Tag3, Tag4
Tag2, Tag3, Tag4
Tag1, Tag2
Tag1, Tag3
Tag1, Tag4
Tag2, Tag3
Tag2, Tag4
Tag3, Tag4
Tag1
Tag2
Tag3
Tag4

element_rank colored inertial spheres

element_rank: no predicates
colored:      requires ranked
inertial:     requires one of our shapes
spheres:      requires a rank and a topology

The problem with using type specialization of an aggregate_type is that type specialization cannot handle any arbitrary sorting. It also fails to account for implicit dependencies, such as sphere_tag actually requiring a rank and a topology. There's something weird here in that things like inertial spheres just requires ~A~ sphere data, it doesn't care if that data has a rank or a topology, so some aggregates shouldn't be required to know all of the necessary tags to construct them. We could concatenate our tags with Base's tags. In this regard, we could have a required set of tags. All of this provided far too high of a burden on creating new aggregates.

Sorry, I got lost there. If register is called during construction, then then we will perform one registration per construction of an augment. This does have some flaws. The biggest of which is that users might overlook dependencies. If I want to access all element rank objects that have color, it's likely that the colored sphere will be missed. 

That's a wee bit much! Now, we do have a logical sorting to our tags, so we can break this into trees with the highest level being that with the smallest value but only if we knew which Tags were aggregate tags, which tags had dependencies, and which tags were qualifiers. Part of the problem is that 

Flaws:
  1. Assumes no conditional augments.
  2. Assumes that each object is accessible with only one of the tags. Not true for dependent tags such as Motility.
  3. Treats qualifying tags the same as aggregate tags. has_shared_radius_tag qualifies the sphere_tag.

AggregateTag, type_list<QualifierTags>, type_list<AugmentTags>


In STK, they are able to perform part selection because the declaration of part subsets is explicit. Our aggregates have violated that. 
They declare all of their parts and then declare part subsets. The tree becomes clear. Our tree is lost because we are marking things dependent that aren't dependent because it's cleaner that way because we want to create overdamped colored textured spheres and sometimes access their color and texture. The simple type-based method where we wrote compute_aabb_spheres(sphere_data, subset_selector) isn't that bad. It does have the issue that it cannot naturally support user-defined ValidSphereData, whereas in the Tag-based approach, we can completely drop the need for the sphere_data object and just pass around Tags. compute_aabb_spheres(bulk_data, subset_selector)





Why is it not just

std::tuple<Tags1...> Tags for registered agg 1
std::tuple<Tags2...> Tags for registered agg 2
std::tuple<Tags3...> Tags for registered agg 3
std::tuple<Tags4...> Tags for registered agg 4
std::tuple<Tags5...> Tags for registered agg 5

Given the fact that we want to act on COLORED SPHERES, will COLORED SPHERES even result in a valid aggregate? Yes. Ok, then which of the given 
aggs has COLORED and SPHERES within their tags. This can be done at runtime using the vector of tag values.

The only reason that this won't work is that we cannot use aggregate_type specialization since we need the given tags to be sorted. Next, we need implicit assumptions to be explicit such that SPHERES have the ELEMENT_RANK or NODE_RANK tag. Even in this form, we still can't fetch all objects that have color without knowing what their rank is since augments cannot live on their own. That isn't true if the ranked object and topological object Base types include a rank. That would allow you to loop over all objects within a given set that have mass and double their mass. We would need to support AABBData and AABBDataAugment. The AABBDataAugment would inherit from the AABBData. This way, we can have augments live on their own if they so choose. Motility less so. Motility depends on the underlying data and only makes sense as an augment to shape data.

The Tags hard code type dependence. The free functions do not.......... This cannot be emphasized enough. compute_aabb_spheres(any_class_that_satisfies_the_constraint) is far better. Can we simply offer the AggregateRegistry as a means of losing and then recovering aggregate types? We would need such a thing for the python interface. 

Let's play looser: We create a set of tags. Aggregates need not be associated with them and they don't know anything about aggregate or entity view types. The tags are just identifiers that are registered with the AggregateFactory which can be used to stash and then retrieve an aggregate using some subset of the existing tags. 












































































## How to check for subsets:
#include <type_traits>
#include <iostream>

// 1. Wrap types into a parameter-pack container (a type_list)
template <typename... Ts>
struct type_list {};

// 2. contains<T, Pack...> checks if T is one of the Pack types
template <typename T, typename... Pack>
struct contains : std::bool_constant<(std::is_same_v<T, Pack> || ...)> {};

// 3. is_subset_of<type_list<A...>, type_list<B...>> checks if all A... are in B...
template <typename, typename>
struct is_subset_of;

template <typename... As, typename... Bs>
struct is_subset_of<type_list<As...>, type_list<Bs...>>
  : std::bool_constant<(contains<As, Bs...>::value && ...)> {};

// Helper variable templates for clarity
template <typename T, typename... Pack>
inline constexpr bool contains_v = contains<T, Pack...>::value;

template <typename AList, typename BList>
inline constexpr bool is_subset_of_v = is_subset_of<AList, BList>::value;

// Demo
int main()
{
    using SetA = type_list<int, double>;
    using SetB = type_list<int, float, double, char>;

    static_assert( is_subset_of_v<SetA, SetB>, "SetA should be a subset of SetB" );
    std::cout << "is_subset_of<SetA, SetB> = " 
              << std::boolalpha 
              << is_subset_of_v<SetA, SetB>
              << std::endl;  // prints true

    using SetC = type_list<int, bool>;
    static_assert( !is_subset_of_v<SetC, SetB>, "SetC should NOT be a subset of SetB" );
    std::cout << "is_subset_of<SetC, SetB> = "
              << std::boolalpha
              << is_subset_of_v<SetC, SetB>
              << std::endl;  // prints false
}

For the sake of having the ability to store and sort tags at compile-time, I have elected to store a static constexpr value on each tag. To avoid conflicts, we have elected to store the value as an unsigned int (which ranges from 0 to 4,294,967,295) and to choose the values randomly without a set seed or counter. Comically, we simply ask Google "random number between 0 to 4,294,967,295" any time we want to make a new tag! The statistical chance of a conflict is low and users are free to add new tags without worrying about the set of used or not used tags.

struct sphere_tag {
    static constexpr unsigned value = 3166131987;
};
static constexpr sphere_tag_v = sphere_tag::value;

and we offer an inverse map accordingly so that we can map from constexpr value to type.

template <unsigned Value>
struct value_to_tag_type {
    using type = void;
};

template <unsigned Value>
using value_to_tag_type_t = value_to_tag_type<Value>::type;

template<>
struct value_to_tag_type<3166131987> {
    using type = sphere_tag;
};

## How to sort by value
#include <type_traits>
#include <iostream>

// -----------------------------------------------------------------------------
// 1) A container for a list of types
// -----------------------------------------------------------------------------
template <typename... Ts>
struct type_list {};

// -----------------------------------------------------------------------------
// 2) A helper to concatenate two type_list types
// -----------------------------------------------------------------------------
template <typename, typename>
struct concat;

template <typename... Ts, typename... Us>
struct concat<type_list<Ts...>, type_list<Us...>> {
    using type = type_list<Ts..., Us...>;
};

template <typename L1, typename L2>
using concat_t = typename concat<L1, L2>::type;

// -----------------------------------------------------------------------------
// 3) A helper to split a type_list into two parts at index N
// -----------------------------------------------------------------------------
template <typename List, std::size_t N>
struct split;

template <std::size_t N>
struct split<type_list<>, N> {
    using first = type_list<>;
    using second = type_list<>;
};

template <typename T, typename... Ts>
struct split<type_list<T, Ts...>, 0> {
    using first = type_list<>;
    using second = type_list<T, Ts...>;
};

template <typename T, typename... Ts, std::size_t N>
struct split<type_list<T, Ts...>, N> {
private:
    // Recursively split the tail
    using splitted_tail = split<type_list<Ts...>, N - 1>;

public:
    // Put T into the `first` part and keep splitted_tail's second
    using first  = concat_t<type_list<T>, typename splitted_tail::first>;
    using second = typename splitted_tail::second;
};

// -----------------------------------------------------------------------------
// 4) A comparison trait: sorts by sizeof(T) < sizeof(U)
//    (You could replace this with any custom comparison you like)
// -----------------------------------------------------------------------------
template <typename T, typename U>
using smaller_than = std::bool_constant<(T::value < U::value)>;

// -----------------------------------------------------------------------------
// 5) merge<Compare, List1, List2> merges two sorted type_lists
// -----------------------------------------------------------------------------
template <template<typename, typename> class Compare, typename List1, typename List2>
struct merge;

template <template<typename, typename> class Compare>
struct merge<Compare, type_list<>, type_list<>> {
    using type = type_list<>;
};

template <template<typename, typename> class Compare, typename... Ts>
struct merge<Compare, type_list<>, type_list<Ts...>> {
    using type = type_list<Ts...>;
};

template <template<typename, typename> class Compare, typename... Ts>
struct merge<Compare, type_list<Ts...>, type_list<>> {
    using type = type_list<Ts...>;
};

template <
    template<typename, typename> class Compare,
    typename T, typename... Ts,
    typename U, typename... Us
>
struct merge<Compare, type_list<T, Ts...>, type_list<U, Us...>> {
    using type = std::conditional_t<
        Compare<T,U>::value,
        // If T < U, T goes first; then merge the rest
        concat_t< type_list<T>,
                  typename merge<Compare, type_list<Ts...>, type_list<U, Us...>>::type >,
        // Otherwise U goes first; then merge the rest
        concat_t< type_list<U>,
                  typename merge<Compare, type_list<T, Ts...>, type_list<Us...>>::type >
    >;
};

// -----------------------------------------------------------------------------
// 6) mergesort<Compare, List> - does a compile-time merge sort
// -----------------------------------------------------------------------------
template <template<typename, typename> class Compare, typename List>
struct mergesort;

template <template<typename, typename> class Compare>
struct mergesort<Compare, type_list<>> {
    using type = type_list<>;
};

template <template<typename, typename> class Compare, typename T>
struct mergesort<Compare, type_list<T>> {
    using type = type_list<T>;
};

template <template<typename, typename> class Compare, typename T, typename U, typename... Ts>
struct mergesort<Compare, type_list<T, U, Ts...>> {
private:
    static constexpr std::size_t count = 2 + sizeof...(Ts);
    static constexpr std::size_t half  = count / 2;

    using splitted      = split<type_list<T, U, Ts...>, half>;
    using left_sorted  = typename mergesort<Compare, typename splitted::first>::type;
    using right_sorted = typename mergesort<Compare, typename splitted::second>::type;

public:
    using type = typename merge<Compare, left_sorted, right_sorted>::type;
};

// -----------------------------------------------------------------------------
// 7) unique<List> - remove consecutive duplicates from a sorted list
// -----------------------------------------------------------------------------
template <typename List>
struct unique;

template <>
struct unique<type_list<>> {
    using type = type_list<>;
};

template <typename T>
struct unique<type_list<T>> {
    using type = type_list<T>;
};

template <typename T, typename U, typename... Ts>
struct unique<type_list<T, U, Ts...>> {
private:
    // Unique tail after removing consecutive dups in (U, Ts...)
    using unique_tail = typename unique<type_list<U, Ts...>>::type;

    // If T == U, skip T
public:
    using type = std::conditional_t<
        std::is_same_v<T, U>,
        unique_tail,
        concat_t< type_list<T>, unique_tail >
    >;
};

// -----------------------------------------------------------------------------
// 8) Convenience alias: sort_and_unique_t<List, Compare>
//    sorts the list under Compare<T,U> and removes duplicates
// -----------------------------------------------------------------------------
template <typename List,
          template<typename, typename> class CompareValue = smaller_than>
struct sort_and_unique_value {
private:
    using sorted = typename mergesort<CompareValue, List>::type;

public:
    using type = typename unique<sorted>::type;
};

template <typename List,
          template<typename, typename> class CompareValue = smaller_than>
using sort_and_unique_value_t = typename sort_and_unique_value<List, CompareValue>::type;


struct TagName1 {
    static constexpr size_t value = 0xA1B2C3D4; // Randomly chosen unique value
};

struct TagName2 {
    static constexpr size_t value = 0x1A2B3C4D; // Another random unique value
};

struct TagName3 {
    static constexpr size_t value = 0xABCDEF12; // Another unique random value
};

struct TagName4 {
    static constexpr size_t value = 0x12345678; // Another unique random value
};

// -----------------------------------------------------------------------------
// Demo
// -----------------------------------------------------------------------------
int main() {
    std::cout << "TagName1 value: " << TagName1::value << "\n";
    std::cout << "TagName2 value: " << TagName2::value << "\n";
    std::cout << "TagName3 value: " << TagName3::value << "\n";
    std::cout << "TagName4 value: " << TagName4::value << "\n";


    using my_list = type_list<TagName1, TagName1, TagName3, TagName2, TagName1, TagName4, TagName2>;
    using my_set = sort_and_unique_value_t<my_list>;

    // Let's verify at compile time:
    constexpr bool ok = std::is_same_v<
        my_set,
        type_list<TagName4, TagName2, TagName1, TagName3>
    >;

    // Compare tags (compile-time or runtime)
    std::cout << "TagName1 < TagName2? " << std::boolalpha 
              << smaller_than<TagName1, TagName2>::value << "\n";

    return 0;
}

An important topic remains unaddressed: How to lose aggregate scalar type? I think we should move to instead templating our entity_view. That said, without knowing our scalar type, I don't see how we can store our shared values. We would need to use something like an std::any or a void*. (I just ran some tests to see if std::any or void* could be compiler optimized in the way we want where operations on reused data are precomputed before the loop and the answer was that it did not work on std::any but it did on void*). I feel like MundyMath should offer a ScalarBase, VectorBase, MatrixBase, etc. that all follow type erasure. This is only necessary for non-ngp entity views. The NGP data is fetched using its scalar type.






Having multiple aggregates per Part is risky if the user isn't careful, as this could lead to double counting. We should (in debug mode), assert that all aggregates with overlapping keys have non-intersecting sets.

The infrastructure for declaring aggregates and performing the for_each should go into MundyMesh. This will allow us to place our AggregateFactory within the BulkData or MetaData.



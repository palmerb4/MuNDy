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
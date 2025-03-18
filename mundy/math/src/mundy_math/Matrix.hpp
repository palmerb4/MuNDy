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

#ifndef MUNDY_MATH_MATRIX_HPP_
#define MUNDY_MATH_MATRIX_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>  // for std::decay_t
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>    // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>        // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>           // for mundy::math::Array
#include <mundy_math/MaskedView.hpp>      // for mundy::math::MaskedView
#include <mundy_math/ShiftedView.hpp>     // for mundy::math::ShiftedView
#include <mundy_math/StridedView.hpp>     // for mundy::math::StridedView
#include <mundy_math/Tolerance.hpp>       // for mundy::math::get_zero_tolerance
#include <mundy_math/TransposedView.hpp>  // for mundy::math::TransposedView
#include <mundy_math/Vector.hpp>          // for mundy::math::Vector
#include <mundy_math/impl/MatrixImpl.hpp> 

namespace mundy {

namespace math {

/// \brief (Implementation) Type trait to determine if a type is a Matrix
template <typename TypeToCheck>
struct is_matrix_impl : std::false_type {};
//
template <typename T, size_t N, size_t M, typename Accessor, typename OwnershipType>
struct is_matrix_impl<Matrix<T, N, M, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a Matrix
template <typename T>
struct is_matrix : public is_matrix_impl<std::decay_t<T>> {};
//
template <typename TypeToCheck>
constexpr bool is_matrix_v = is_matrix<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid Matrix type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our quaternions.
template <typename MatrixType>
concept ValidMatrixType =
    is_matrix_v<std::decay_t<MatrixType>> &&
    requires(std::decay_t<MatrixType> matrix, const std::decay_t<MatrixType> const_matrix, size_t i) {
      typename std::decay_t<MatrixType>::scalar_t;
      { matrix[i] } -> std::convertible_to<typename std::decay_t<MatrixType>::scalar_t>;
      { matrix(i) } -> std::convertible_to<typename std::decay_t<MatrixType>::scalar_t>;
      { matrix(i, i) } -> std::convertible_to<typename std::decay_t<MatrixType>::scalar_t>;
      { const_matrix[i] } -> std::convertible_to<const typename std::decay_t<MatrixType>::scalar_t>;
      { const_matrix(i) } -> std::convertible_to<const typename std::decay_t<MatrixType>::scalar_t>;
      { const_matrix(i, i) } -> std::convertible_to<const typename std::decay_t<MatrixType>::scalar_t>;
    };  // ValidMatrixType

/// \brief Class for an NxM (num rows x num columns) matrix with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
///
/// This class is designed to be used with Kokkos. It is a simple NxM matrix with arithmetic entries implemented without
/// for loops (to provide compile-time optimization for small matrix sizes). It is templated on the type of the entries,
/// Accessor type, and the number of rows and columns. See Accessor.hpp for more details on the Accessor type
/// requirements.
///
/// The goal of Matrix is to be a lightweight class that can be used with Kokkos to perform mathematical operations on
/// matrices in RNxM. It does not own or manage the underlying data, but rather it is templated on an Accessor type that
/// provides access to the underlying data. This allows us to use Matrix with Kokkos Views, raw pointers, or any other
/// type that meets the ValidAccessor requirements without copying the data. This is especially important for
/// GPU-compatable code.
///
/// Matrixs can be constructed by passing an accessor to the constructor. However, if the accessor has a N*M-argument
/// constructor, then the Matrix can also be constructed by passing the elements directly to the constructor (in
/// row-major order). Similarly, if the accessor has an initializer list constructor, then the Matrix can be constructed
/// by passing an initializer list to the constructor. This is a convenience feature which makes working with the
/// default accessor (Array<T, N*M>) easier. For example, the following are all valid ways to construct a Matrix:
///
/// \code{.cpp}
///   // Constructs a Matrix with the default accessor (Array<int, 9>)
///   Matrix<int, 3, 3> mat1({1, 2, 3, 4, 5, 6, 7, 8, 9});
///   Matrix<int, 3, 3> mat2(1, 2, 3, 4, 5, 6, 7, 8, 9);
///   Matrix<int, 3, 3> mat3(Array<int, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9});
///   Matrix<int, 3, 3> mat4;
///   mat4.set(1, 2, 3, 4, 5, 6, 7, 8, 9);
///
///   // Construct a Matrix from a double array
///   double data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
///   MatrixView<double, 3, 3, double*> mat5(data);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the Matrix class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the Matrix that use it.
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
  requires std::is_arithmetic_v<T>
class Matrix<T, N, M, Accessor, Ownership::Views> {
 public:
  //! \name Internal data
  //@{

  /// \brief A reference or a pointer to an external data accessor.
  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using scalar_t = T;

  /// \brief The non-const type of the entries
  using non_const_scalar_t = std::remove_const_t<T>;

  /// \brief Our ownership type
  using ownership_t = Ownership::Views;

  /// \brief The number of rows
  static constexpr size_t num_rows = N;

  /// \brief The number of columns
  static constexpr size_t num_cols = M;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor since we don't own the data.
  KOKKOS_INLINE_FUNCTION Matrix() = delete;

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr Matrix(Accessor& data)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(data) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr Matrix(Accessor data)
    requires std::is_pointer_v<Accessor>
      : accessor_(data) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Matrix() = default;

  /// \brief Shallow copy constructor. Stores a reference to the accessor in the other matrix.
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const Matrix<T, N, M, Accessor, Ownership::Views>& other)
      : accessor_(other.data()) {
  }

  /// \brief Shallow move constructor. Stores and moves the reference to the accessor from the other matrix.
  KOKKOS_INLINE_FUNCTION constexpr Matrix(Matrix<T, N, M, Accessor, Ownership::Views>&& other)
      : accessor_(std::move(other.data())) {
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
  template <typename OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(
      const Matrix<T, N, M, OtherAccessor, OtherOwnershipType>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  template <typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(
      const Matrix<T, N, M, Accessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N * M>{}, *this, value);
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(
      Matrix<T, N, M, OtherAccessor, Ownership::Owns>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(
      Matrix<T, N, M, Accessor, Ownership::Owns>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(
      Matrix<T, N, M, OtherAccessor, Ownership::Views>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator=(
      Matrix<T, N, M, Accessor, Ownership::Views>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via flat index
  /// \param[in] row The row index.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator[](size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via flat index
  /// \param[in] row The row index.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator[](size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via flat index
  /// \param[in] index The flat index.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via flat index
  /// \param[in] index The flat index.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via row and column indices
  /// \note This operator is preferred over using m[row][col]
  /// \param[in] row The row index.
  /// \param[in] col The column index.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t row, size_t col) {
    return accessor_[row * N + col];
  }

  /// \brief Const element access operators
  /// \note This operator is preferred over using m[row][col]
  /// \param[in] row The row index.
  /// \param[in] col The column index.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t row, size_t col) const {
    return accessor_[row * N + col];
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> data() {
    return accessor_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> data() const {
    return accessor_;
  }

  /// \brief Get a copy of a certain column of the matrix
  /// \param[in] col The column index.
  KOKKOS_INLINE_FUNCTION
  constexpr Vector<non_const_scalar_t, N> copy_column(size_t col) const {
    return impl::copy_column_impl(std::make_index_sequence<N>{}, *this, col);
  }

  /// \brief Get a copy of a certain row of the matrix
  /// \param[in] row The row index.
  KOKKOS_INLINE_FUNCTION
  constexpr Vector<non_const_scalar_t, M> copy_row(size_t row) const {
    return impl::copy_row_impl(std::make_index_sequence<M>{}, *this, row);
  }

  /// \brief Get a view into a certain column of the matrix
  /// \tparam[in] col The column index.
  template <size_t col>
  KOKKOS_INLINE_FUNCTION constexpr auto view_column() {
    // To explain, because the data is stored in row-major order, we need to stride by N to access the contents of a
    // column and then shift by the column index to access the contents of the desired column.
    constexpr size_t shift = col;
    constexpr size_t stride = M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);
    auto strided_shifted_data_accessor = get_owning_strided_accessor<T, stride>(shifted_data_accessor);
    return get_owning_vector<T, N>(std::move(strided_shifted_data_accessor));
  }

  /// \brief Get a view into a certain column of the matrix
  /// \tparam[in] col The column index.
  template <size_t col>
  KOKKOS_INLINE_FUNCTION constexpr auto view_column() const {
    // To explain, because the data is stored in row-major order, we need to stride by N to access the contents of a
    // column and then shift by the column index to access the contents of the desired column.
    constexpr size_t shift = col;
    constexpr size_t stride = M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);
    auto strided_shifted_data_accessor = get_owning_strided_accessor<T, stride>(shifted_data_accessor);
    return get_owning_vector<T, N>(std::move(strided_shifted_data_accessor));
  }

  // #pragma todo Simply because you own the accessor doesn't mean you own the data.

  /// \brief Get a view into a certain row of the matrix
  /// \tparam[in] row The row index.
  template <size_t row>
  KOKKOS_INLINE_FUNCTION constexpr auto view_row() {
    // To explain, because the data is stored in row-major order, we need to shift by row * N to get the correct row.
    // Once shifted, we can then get a view of the row.
    constexpr size_t shift = row * M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);

    // Pass ownership of the local shifted view to the vector
    return get_owning_vector<T, M>(std::move(shifted_data_accessor));
  }

  /// \brief Get a view into a certain row of the matrix
  /// \tparam[in] row The row index.
  template <size_t row>
  KOKKOS_INLINE_FUNCTION constexpr auto view_row() const {
    // To explain, because the data is stored in row-major order, we need to shift by row * N to get the correct row.
    // Once shifted, we can then get a view of the row.
    constexpr size_t shift = row * M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);

    // Pass ownership of the local shifted view to the vector
    return get_owning_vector<T, M>(std::move(shifted_data_accessor));
  }

  /// \brief Get a view into the diagonal of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_diagonal() {
    // To explain, because the data is stored in row-major order, we need to stride by N+1 to access the contents of the
    // diagonal.
    constexpr size_t stride = N + 1;
    auto strided_data_accessor = get_strided_view<T, stride>(accessor_);
    return get_owning_vector<T, Kokkos::min(N, M)>(std::move(strided_data_accessor));
  }

  /// \brief Get a view into the diagonal of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_diagonal() const {
    // To explain, because the data is stored in row-major order, we need to stride by N+1 to access the contents of the
    // diagonal.
    constexpr size_t stride = N + 1;
    auto strided_data_accessor = get_strided_view<T, stride>(accessor_);
    return get_owning_vector<T, Kokkos::min(N, M)>(std::move(strided_data_accessor));
  }

  /// \brief Get a view into the transpose of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_transpose() {
    // Isn't this neat? We can get a transposed view of the matrix without copying the data and then use any of our
    // existing function/operations on it!
    auto transposed_data_accessor = get_transposed_view<T, N, M>(accessor_);
    return get_owning_matrix<T, M, N>(std::move(transposed_data_accessor));
  }

  /// \brief Get a view into the transpose of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_transpose() const {
    // Isn't this neat? We can get a transposed view of the matrix without copying the data and then use any of our
    // existing function/operations on it!
    auto transposed_data_accessor = get_transposed_view<T, N, M>(accessor_);
    return get_owning_matrix<T, M, N>(std::move(transposed_data_accessor));
  }

  /// \brief Get a view into the matrix excluding a certain row and column
  /// This is known as the minor of the element at that row/column.
  /// \tparam[in] row The row index to drop.
  /// \tparam[in] col The column index to drop
  template <size_t row_to_exclude, size_t col_to_exclude>
  KOKKOS_INLINE_FUNCTION constexpr auto view_minor() {
    // To explain, we use a compile-time mask to exclude the given row and column from the submatrix.
    constexpr size_t newN = N - 1;
    constexpr size_t newM = M - 1;
    constexpr Kokkos::Array<bool, N * M> mask = impl::create_row_and_col_mask<N, M, row_to_exclude, col_to_exclude>();
    auto masked_data_accessor = get_masked_view<T, N * M, mask>(accessor_);
    return get_owning_matrix<T, newN, newM>(std::move(masked_data_accessor));
  }

  /// \brief Get a view into the matrix excluding a certain row and column
  /// This is known as the minor of the element at that row/column.
  /// \tparam[in] row The row index to drop.
  /// \tparam[in] col The column index to drop
  template <size_t row_to_exclude, size_t col_to_exclude>
  KOKKOS_INLINE_FUNCTION constexpr auto view_minor() const {
    // To explain, we use a compile-time mask to exclude the given row and column from the submatrix.
    constexpr size_t newN = N - 1;
    constexpr size_t newM = M - 1;
    constexpr Kokkos::Array<bool, N * M> mask = impl::create_row_and_col_mask<N, M, row_to_exclude, col_to_exclude>();
    auto masked_data_accessor = get_masked_view<T, N * M, mask>(accessor_);
    return get_owning_matrix<T, newN, newM>(std::move(masked_data_accessor));
  }

  /// \brief Cast (and copy) the matrix to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return impl::cast_impl<U>(std::make_index_sequence<N * M>{}, *this);
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the matrix
  template <typename... Args>
    requires(sizeof...(Args) == N * M) &&
            (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set(Args&&... args) {
    impl::set_impl(std::make_index_sequence<N * M>{}, *this, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set all elements of the matrix using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Matrix is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_impl(std::make_index_sequence<N * M>{}, *this, accessor);
  }

  /// \brief Set a certain row of the matrix
  /// \param[in] i The row index.
  template <typename... Args>
    requires(sizeof...(Args) == M) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set_row(const size_t& i, Args&&... args) {
    impl::set_row_impl(std::make_index_sequence<M>{}, *this, i, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set a certain row of the matrix
  /// \param[in] i The row index.
  /// \param[in] row The row vector.
  template <ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr void set_row(const size_t& i,
                                                const Vector<T, M, OtherAccessor, OtherOwnershipType>& row)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_row_impl(std::make_index_sequence<M>{}, *this, i, row);
  }

  /// \brief Set a certain column of the matrix
  /// \param[in] j The column index.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set_column(const size_t& j, Args&&... args) {
    impl::set_column_impl(std::make_index_sequence<N>{}, *this, j, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set a certain column of the matrix
  /// \param[in] j The column index.
  /// \param[in] col The column vector.
  template <ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr void set_column(const size_t& j,
                                                   const Vector<T, N, OtherAccessor, OtherOwnershipType>& col)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_column_impl(std::make_index_sequence<N>{}, *this, j, col);
  }

  /// \brief Fill all elements of the matrix with a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION
  constexpr void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N * M>{}, *this, value);
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr Matrix<T, N, M> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr Matrix<T, N, M> operator-() const {
    return impl::unary_minus_impl(std::make_index_sequence<N * M>{}, *this);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Matrix-matrix addition
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_matrix_addition_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Self-matrix addition
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor>& operator+=(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_matrix_addition_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Matrix-matrix subtraction
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_matrix_subtraction_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Self-matrix subtraction
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator-=(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_matrix_subtraction_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Matrix-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
    requires std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar) const {
    return impl::matrix_scalar_addition_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator+=(const U& scalar) {
    impl::self_scalar_addition_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }

  /// \brief Matrix-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
    requires std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar) const {
    return impl::matrix_scalar_subtraction_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator-=(const U& scalar) {
    impl::self_scalar_subtraction_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Matrix-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(
      const Matrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_matrix_multiplication_impl(std::make_index_sequence<N * OtherM>{}, *this, other);
  }

  /// \brief Self-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator*=(
      const Matrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    constexpr bool all_sizes_match = (N == OtherM) && (M == OtherN) && (N == M);
    static_assert(all_sizes_match,
                  "Self-matrix multiplication is not supported for non-square matrices of different sizes.");
    impl::self_matrix_multiplication_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Matrix-vector multiplication
  /// \param[in] other The other vector.
  template <typename U, typename OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Vector<U, M, OtherAccessor, OtherOwnershipType>& other) const {
    // Pass in index sequence for the vector size
    return impl::matrix_vector_multiplication_impl(std::make_index_sequence<M>{}, *this, other);
  }

  /// \brief Matrix-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
    requires std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar) const {
    return impl::matrix_scalar_multiplication_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator*=(const U& scalar) {
    impl::self_scalar_multiplication_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }

  /// \brief Matrix-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
    requires std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar) const {
    return impl::matrix_scalar_division_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Views>& operator/=(const U& scalar) {
    impl::self_scalar_division_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream& operator<<(std::ostream& os,
                                  const Matrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& mat);

  // We are friends with all Matrixs  regardless of their Accessor or type
  template <typename U, size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_arithmetic_v<U>
  friend class Matrix;
  //@}
};  // class Matrix (non-owning)

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
  requires std::is_arithmetic_v<T>
class Matrix {
 public:
  //! \name Internal data
  //@{

  /// \brief Our data accessor. Owning
  Accessor accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using scalar_t = T;

  /// \brief The non-const type of the entries
  using non_const_scalar_t = std::remove_const_t<T>;

  /// \brief Our ownership type
  using ownership_t = OwnershipType;

  /// \brief The number of rows
  static constexpr size_t num_rows = N;

  /// \brief The number of columns
  static constexpr size_t num_cols = M;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Assume elements are uninitialized.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr Matrix()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] accessor The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr Matrix(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Constructor to initialize all elements explicitly.
  /// Requires the number of arguments to be N and the type of each to be T.
  /// Only enabled if the Accessor has a N-argument constructor.
  template <typename... Args>
    requires(sizeof...(Args) == N * M) &&
            (std::is_convertible_v<Args, T> && ...) && HasNArgConstructor<Accessor, T, N * M>
  KOKKOS_INLINE_FUNCTION explicit constexpr Matrix(Args&&... args)
      : accessor_{static_cast<T>(std::forward<Args>(args))...} {
  }

  /// \brief Constructor to initialize all elements via initializer list
  /// \param[in] list The initializer list.
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const std::initializer_list<T>& list)
    requires HasInitializerListConstructor<Accessor, T>
      : accessor_(list) {
    MUNDY_THROW_ASSERT(list.size() == N * M, std::invalid_argument, "Matrix: Initializer list must have 3 elements.");
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Matrix() = default;

  /// \brief Deep copy constructor
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const Matrix<T, N, M, Accessor, Ownership::Owns>& other)
    requires HasCopyConstructor<Accessor>
      : accessor_(other.accessor_) {
  }

  /// \brief Deep copy constructor
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const Matrix<T, N, M, Accessor, Ownership::Views>& other)
    requires HasCopyConstructor<Accessor>
      : accessor_(other.accessor_) {
  }

  /// \brief Deep copy constructor
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const Matrix<T, N, M, Accessor, Ownership::Owns>& other)
    requires(!HasCopyConstructor<Accessor>) && HasNonConstAccessOperator<Accessor, T>
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Deep copy constructor
  template <typename OtherAccessor>
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const Matrix<T, N, M, OtherAccessor, Ownership::Owns>& other) : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Deep copy constructor
  template <typename OtherAccessor>
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr Matrix(const Matrix<T, N, M, OtherAccessor, Ownership::Views>& other) : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Deep move constructor
  KOKKOS_INLINE_FUNCTION constexpr Matrix(Matrix<T, N, M, Accessor, Ownership::Owns>&& other)
    requires(HasCopyConstructor<Accessor> || HasMoveConstructor<Accessor>)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherAccessor>
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr Matrix(Matrix<T, N, M, OtherAccessor, Ownership::Owns>&& other) : accessor_() {
    // Other owns its accessor but that doesn't mean that it owns the data the accessor accesses.
    // Since the accessor neither has a copy constructor nor a move constructor, we must deep copy.
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
  }
  /// \brief Deep move constructor
  template <typename OtherAccessor>
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr Matrix(Matrix<T, N, M, OtherAccessor, Ownership::Views>&& other) : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      const Matrix<T, N, M, OtherAccessor, Ownership::Owns>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      const Matrix<T, N, M, Accessor, Ownership::Owns>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      const Matrix<T, N, M, OtherAccessor, Ownership::Views>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      const Matrix<T, N, M, Accessor, Ownership::Views>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N * M>{}, *this, value);
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      Matrix<T, N, M, OtherAccessor, Ownership::Owns>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      Matrix<T, N, M, Accessor, Ownership::Owns>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      Matrix<T, N, M, OtherAccessor, Ownership::Views>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator=(
      Matrix<T, N, M, Accessor, Ownership::Views>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N * M>{}, *this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via flat index
  /// \param[in] row The row index.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator[](size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via flat index
  /// \param[in] row The row index.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator[](size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via flat index
  /// \param[in] index The flat index.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via flat index
  /// \param[in] index The flat index.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via row and column indices
  /// \note This operator is preferred over using m[row][col]
  /// \param[in] row The row index.
  /// \param[in] col The column index.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t row, size_t col) {
    return accessor_[row * N + col];
  }

  /// \brief Const element access operators
  /// \note This operator is preferred over using m[row][col]
  /// \param[in] row The row index.
  /// \param[in] col The column index.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t row, size_t col) const {
    return accessor_[row * N + col];
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr Accessor& data() {
    return accessor_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr const Accessor& data() const {
    return accessor_;
  }

  /// \brief Get a copy of a certain column of the matrix
  /// \param[in] col The column index.
  KOKKOS_INLINE_FUNCTION
  constexpr Vector<non_const_scalar_t, N> copy_column(size_t col) const {
    return impl::copy_column_impl(std::make_index_sequence<N>{}, *this, col);
  }

  /// \brief Get a copy of a certain row of the matrix
  /// \param[in] row The row index.
  KOKKOS_INLINE_FUNCTION
  constexpr Vector<non_const_scalar_t, M> copy_row(size_t row) const {
    return impl::copy_row_impl(std::make_index_sequence<M>{}, *this, row);
  }

  /// \brief Get a view into a certain column of the matrix
  /// \tparam[in] col The column index.
  template <size_t col>
  KOKKOS_INLINE_FUNCTION constexpr auto view_column() {
    // To explain, because the data is stored in row-major order, we need to stride by N to access the contents of a
    // column and then shift by the column index to access the contents of the desired column.
    constexpr size_t shift = col;
    constexpr size_t stride = M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);
    auto strided_shifted_data_accessor = get_owning_strided_accessor<T, stride>(shifted_data_accessor);
    return get_owning_vector<T, N>(std::move(strided_shifted_data_accessor));
  }

  /// \brief Get a view into a certain column of the matrix
  /// \tparam[in] col The column index.
  template <size_t col>
  KOKKOS_INLINE_FUNCTION constexpr auto view_column() const {
    // To explain, because the data is stored in row-major order, we need to stride by N to access the contents of a
    // column and then shift by the column index to access the contents of the desired column.
    constexpr size_t shift = col;
    constexpr size_t stride = M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);
    auto strided_shifted_data_accessor = get_owning_strided_accessor<T, stride>(shifted_data_accessor);
    return get_owning_vector<T, N>(std::move(strided_shifted_data_accessor));
  }

  /// \brief Get a view into a certain row of the matrix
  /// \tparam[in] row The row index.
  template <size_t row>
  KOKKOS_INLINE_FUNCTION constexpr auto view_row() {
    // To explain, because the data is stored in row-major order, we need to shift by row * N to get the correct row.
    // Once shifted, we can then get a view of the row.
    constexpr size_t shift = row * M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);
    return get_owning_vector<T, M>(std::move(shifted_data_accessor));
  }

  /// \brief Get a view into a certain row of the matrix
  /// \tparam[in] row The row index.
  template <size_t row>
  KOKKOS_INLINE_FUNCTION constexpr auto view_row() const {
    // To explain, because the data is stored in row-major order, we need to shift by row * N to get the correct row.
    // Once shifted, we can then get a view of the row.
    constexpr size_t shift = row * M;
    auto shifted_data_accessor = get_shifted_view<T, shift>(accessor_);
    return get_owning_vector<T, M>(std::move(shifted_data_accessor));
  }

  /// \brief Get a view into the diagonal of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_diagonal() {
    // To explain, because the data is stored in row-major order, we need to stride by N+1 to access the contents of the
    // diagonal.
    constexpr size_t stride = N + 1;
    auto strided_data_accessor = get_strided_view<T, stride>(accessor_);
    return get_owning_vector<T, Kokkos::min(N, M)>(std::move(strided_data_accessor));
  }

  /// \brief Get a view into the diagonal of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_diagonal() const {
    // To explain, because the data is stored in row-major order, we need to stride by N+1 to access the contents of the
    // diagonal.
    constexpr size_t stride = N + 1;
    auto strided_data_accessor = get_strided_view<T, stride>(accessor_);
    return get_owning_vector<T, Kokkos::min(N, M)>(std::move(strided_data_accessor));
  }

  /// \brief Get a view into the transpose of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_transpose() {
    // Isn't this neat? We can get a transposed view of the matrix without copying the data and then use any of our
    // existing function/operations on it!
    auto transposed_data_accessor = get_transposed_view<T, N, M>(accessor_);
    return get_owning_matrix<T, M, N>(std::move(transposed_data_accessor));
  }

  /// \brief Get a view into the transpose of the matrix
  KOKKOS_INLINE_FUNCTION
  constexpr auto view_transpose() const {
    // Isn't this neat? We can get a transposed view of the matrix without copying the data and then use any of our
    // existing function/operations on it!
    auto transposed_data_accessor = get_transposed_view<T, N, M>(accessor_);
    return get_owning_matrix<T, M, N>(std::move(transposed_data_accessor));
  }

  /// \brief Get a view into the matrix excluding a certain row and column
  /// This is known as the minor of the element at that row/column.
  /// \tparam[in] row The row index to drop.
  /// \tparam[in] col The column index to drop
  template <size_t row_to_exclude, size_t col_to_exclude>
  KOKKOS_INLINE_FUNCTION constexpr auto view_minor() {
    // To explain, we use a compile-time mask to exclude the given row and column from the submatrix.
    constexpr size_t newN = N - 1;
    constexpr size_t newM = M - 1;
    constexpr Kokkos::Array<bool, N * M> mask = impl::create_row_and_col_mask<N, M, row_to_exclude, col_to_exclude>();
    auto masked_data_accessor = get_masked_view<T, N * M, mask>(accessor_);
    return get_owning_matrix<T, newN, newM>(std::move(masked_data_accessor));
  }

  /// \brief Get a view into the matrix excluding a certain row and column
  /// This is known as the minor of the element at that row/column.
  /// \tparam[in] row The row index to drop.
  /// \tparam[in] col The column index to drop
  template <size_t row_to_exclude, size_t col_to_exclude>
  KOKKOS_INLINE_FUNCTION auto constexpr view_minor() const {
    // To explain, we use a compile-time mask to exclude the given row and column from the submatrix.
    constexpr size_t newN = N - 1;
    constexpr size_t newM = M - 1;
    constexpr Kokkos::Array<bool, N * M> mask = impl::create_row_and_col_mask<N, M, row_to_exclude, col_to_exclude>();
    auto masked_data_accessor = get_masked_view<T, N * M, mask>(accessor_);
    return get_owning_matrix<T, newN, newM>(std::move(masked_data_accessor));
  }

  /// \brief Cast (and copy) the matrix to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return impl::cast_impl<U>(std::make_index_sequence<N * M>{}, *this);
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the matrix
  template <typename... Args>
    requires(sizeof...(Args) == N * M) &&
            (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set(Args&&... args) {
    impl::set_impl(std::make_index_sequence<N * M>{}, *this, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set all elements of the matrix using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Matrix is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_impl(std::make_index_sequence<N * M>{}, *this, accessor);
  }

  /// \brief Set a certain row of the matrix
  /// \param[in] i The row index.
  template <typename... Args>
    requires(sizeof...(Args) == M) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set_row(const size_t& i, Args&&... args) {
    impl::set_row_impl(std::make_index_sequence<M>{}, *this, i, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set a certain row of the matrix
  /// \param[in] i The row index.
  /// \param[in] row The row vector.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set_row(const size_t& i, const Vector<T, M, OtherAccessor>& row)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_row_impl(std::make_index_sequence<M>{}, *this, i, row);
  }

  /// \brief Set a certain column of the matrix
  /// \param[in] j The column index.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set_column(const size_t& j, Args&&... args) {
    impl::set_column_impl(std::make_index_sequence<N>{}, *this, j, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set a certain column of the matrix
  /// \param[in] j The column index.
  /// \param[in] col The column vector.
  template <typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set_column(const size_t& j, const Vector<T, N, OtherAccessor>& col)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_column_impl(std::make_index_sequence<N>{}, *this, j, col);
  }

  /// \brief Fill all elements of the matrix with a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION
  constexpr void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N * M>{}, *this, value);
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr Matrix<T, N, M> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr Matrix<T, N, M> operator-() const {
    return impl::unary_minus_impl(std::make_index_sequence<N * M>{}, *this);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Matrix-matrix addition
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_matrix_addition_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Self-matrix addition
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor>& operator+=(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_matrix_addition_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Matrix-matrix subtraction
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_matrix_subtraction_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Self-matrix subtraction
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor>& operator-=(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_matrix_subtraction_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }

  /// \brief Matrix-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar) const {
    return impl::matrix_scalar_addition_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_addition_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }

  /// \brief Matrix-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar) const {
    return impl::matrix_scalar_subtraction_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_subtraction_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Matrix-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(
      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_matrix_multiplication_impl(std::make_index_sequence<N * M>{}, *this, other);
  }

  /// \brief Self-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor, typename OtherOwnershipType, size_t OtherN, size_t OtherM>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator*=(
      const Matrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    constexpr bool all_sizes_match = (N == OtherM) && (M == OtherN) && (N == M);
    static_assert(all_sizes_match,
                  "Self-matrix multiplication is not supported for non-square matrices of different sizes.");
    impl::self_matrix_multiplication_impl(std::make_index_sequence<N * M>{}, *this, other);
    return *this;
  }
  /// \brief Matrix-vector multiplication
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Vector<U, M, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::matrix_vector_multiplication_impl(std::make_index_sequence<M>{}, *this, other);
  }

  /// \brief Matrix-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar) const {
    return impl::matrix_scalar_multiplication_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator*=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_multiplication_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }

  /// \brief Matrix-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar) const {
    return impl::matrix_scalar_division_impl(std::make_index_sequence<N * M>{}, *this, scalar);
  }

  /// \brief Self-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr Matrix<T, N, M, Accessor, Ownership::Owns>& operator/=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_division_impl(std::make_index_sequence<N * M>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get the identity matrix
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> identity() {
    constexpr size_t min_dim = M < N ? M : N;
    return identity_impl(std::make_index_sequence<min_dim>{});
  }

  /// \brief Get the ones matrix
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> ones() {
    return ones_impl(std::make_index_sequence<N * M>{});
  }

  /// \brief Get the zero matrix
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> zeros() {
    return zeros_impl(std::make_index_sequence<N * M>{});
  }

  /// \brief Get a diagonal matrix from a vector
  /// \param[in] vec The vector.
  template <typename U, size_t OtherN, typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> diagonal(const Vector<U, OtherN, OtherAccessor>& vec) {
    constexpr size_t min_dim = M < N ? M : N;
    static_assert(OtherN == min_dim, "Matrix: Diagonal vector must have the same size as the smallest dimension.");
    return diagonal_impl(std::make_index_sequence<min_dim>{}, vec);
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream& operator<<(std::ostream& os,
                                  const Matrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& mat);

  // We are friends with all Matrixs  regardless of their Accessor or type
  template <typename U, size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_arithmetic_v<U>
  friend class Matrix;
  //@}

 private:
  //! \name Private helper functions
  //@{

  /// \brief Get the identity matrix
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> identity_impl(std::index_sequence<Is...>) {
    // Is should be of length min(N, M)
    Matrix<std::remove_const_t<T>, N, M> result = zeros();
    ((result(Is, Is) = static_cast<T>(1)), ...);
    return result;
  }

  /// \brief Get the ones matrix
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> ones_impl(std::index_sequence<Is...>) {
    // Is should be of size M * N
    Matrix<std::remove_const_t<T>, N, M> result;
    ((result[Is] = static_cast<T>(1)), ...);
    return result;
  }

  /// \brief Get a matrix of zeros
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> zeros_impl(std::index_sequence<Is...>) {
    // Is should be of size M * N
    Matrix<std::remove_const_t<T>, N, M> result;
    ((result[Is] = static_cast<T>(0)), ...);
    return result;
  }

  /// \brief Get a diagonal matrix from a vector
  template <size_t... Is, typename U, size_t OtherN, typename OtherAccessor>
  KOKKOS_INLINE_FUNCTION static constexpr Matrix<T, N, M> diagonal_impl(std::index_sequence<Is...>,
                                                                        const Vector<U, OtherN, OtherAccessor>& vec) {
    // Is should be of length min(N, M). As should the vec.
    constexpr size_t min_dim = M < N ? M : N;
    static_assert(OtherN == min_dim,
                  "The vector must have the same number of elements as the minimum dimension of the "
                  "matrix.");
    Matrix<std::remove_const_t<T>, N, M> result;
    ((result(Is, Is) = static_cast<T>(vec[Is])), ...);
    return result;
  }
  //@}
};  // class Matrix (owning)

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor = Array<T, N * M>>
  requires std::is_arithmetic_v<T>
using MatrixView = Matrix<T, N, M, Accessor, Ownership::Views>;

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor = Array<T, N * M>>
  requires std::is_arithmetic_v<T>
using OwningMatrix = Matrix<T, N, M, Accessor, Ownership::Owns>;

static_assert(is_matrix_v<Matrix<int, 3, 4>>, "Odd, default matrix is not a matrix.");
static_assert(is_matrix_v<Matrix<int, 3, 4, Array<int, 12>>>,
              "Odd, default matrix with Array accessor is not a matrix.");
static_assert(is_matrix_v<MatrixView<int, 3, 4>>, "Odd, MatrixView is not a matrix.");
static_assert(is_matrix_v<OwningMatrix<int, 3, 4>>, "Odd, OwningMatrix is not a matrix.");

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the matrix to an output stream
/// \param[in] os The output stream.
/// \param[in] mat The matrix.
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
std::ostream& operator<<(std::ostream& os, const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  for (size_t i = 0; i < N; ++i) {
    os << "(";
    for (size_t j = 0; j < M; ++j) {
      os << mat(i, j);
      if (j < M - 1) {
        os << ", ";
      }
    }
    os << ")";
    if (i < N - 1) {
      os << std::endl;
    }
  }
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// \brief Matrix-matrix equality (element-wise within a tolerance)
/// \param[in] mat1 The first matrix.
/// \param[in] mat2 The second matrix.
/// \param[in] tol The tolerance (default is determined by the given type).
template <size_t N, size_t M, typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1,
          ValidAccessor<T> Accessor2, typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr bool is_close(
    const Matrix<U, N, M, Accessor1, OwnershipType1>& mat1, const Matrix<T, N, M, Accessor2, OwnershipType2>& mat2,
    const std::common_type_t<T, U>& tol = get_zero_tolerance<std::common_type_t<T, U>>()) {
  return impl::is_close_impl(std::make_index_sequence<N * M>{}, mat1, mat2, tol);
}

/// \brief Matrix-matrix equality (element-wise within a relaxed tolerance)
/// \param[in] mat1 The first matrix.
/// \param[in] mat2 The second matrix.
/// \param[in] tol The tolerance (default is determined by the given type).
template <size_t N, size_t M, typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1,
          ValidAccessor<T> Accessor2, typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr bool is_approx_close(
    const Matrix<U, N, M, Accessor1, OwnershipType1>& mat1, const Matrix<T, N, M, Accessor2, OwnershipType2>& mat2,
    const std::common_type_t<T, U>& tol = get_relaxed_zero_tolerance<std::common_type_t<T, U>>()) {
  return is_close(mat1, mat2, tol);
}
//@}

//! \name Non-member addition and subtraction operators
//@{

/// \brief Scalar-matrix addition
/// \param[in] scalar The scalar.
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar, const Matrix<T, N, M, Accessor, OwnershipType>& mat)
    -> Matrix<std::common_type_t<T, U>, N, M> {
  return mat + scalar;
}

/// \brief Scalar-matrix subtraction
/// \param[in] scalar The scalar.
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar, const Matrix<T, N, M, Accessor, OwnershipType>& mat)
    -> Matrix<std::common_type_t<T, U>, N, M> {
  return -mat + scalar;
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-matrix multiplication
/// \param[in] scalar The scalar.
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar, const Matrix<T, N, M, Accessor, OwnershipType>& mat)
    -> Matrix<std::common_type_t<T, U>, N, M> {
  return mat * scalar;
}

/// \brief Vector matrix multiplication (v^T M)
/// \param[in] vec The vector.
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename U, typename T, ValidAccessor<T> Accessor1, typename OwnershipType1,
          ValidAccessor<U> Accessor2, typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const Vector<U, N, Accessor1, OwnershipType1>& vec,
                                                const Matrix<T, N, M, Accessor2, OwnershipType2>& mat)
    -> Vector<std::common_type_t<T, U>, M> {
  // Use view symmantics to avoid copying the matrix during the transpose.
  return mat.view_transpose() * vec;
}
//@}

//! \name Basic arithmetic reduction operations
//@{

/// \brief Matrix determinant
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto determinant(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  static_assert(N == M, "The determinant is only defined for square matrices.");
  return impl::determinant_impl(std::make_index_sequence<N>{}, mat);
}

/// \brief Matrix trace
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto trace(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return sum(mat.view_diagonal());
}

/// \brief Sum of all elements
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto sum(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::sum_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Product of all elements
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto product(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::product_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Minimum element of the matrix
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto min(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::min_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Maximum element of the matrix
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto max(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::max_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Mean of all elements (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType mean(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return static_cast<OutputType>(sum(mat)) / OutputType(N * M);
}

/// \brief Mean of all elements (returns a float if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType mean_f(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return mean(mat);
}

/// \brief Variance of all elements (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType variance(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::variance_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Variance of all elements (returns a float if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType variance_f(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return variance(mat);
}

/// \brief Standard deviation of all elements (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType stddev(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::standard_deviation_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Standard deviation of all elements (returns a float if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType stddev_f(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return stddev(mat);
}
//@}

//! \name Special matrix operations
//@{

/// \brief Matrix transpose
/// \param[in] mat The matrix.
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr Matrix<T, M, N> transpose(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::transpose_impl(std::make_index_sequence<N * M>{}, mat);
}

/// \brief Matrix cofactors
/// \param[in] mat The matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION Matrix<T, N, N> constexpr cofactors(const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  return impl::cofactors_impl(std::make_index_sequence<N * N>{}, mat);
}

/// \brief Matrix adjugate
/// \param[in] mat The matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION Matrix<T, N, N> constexpr adjugate(const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  return transpose(cofactors(mat));
}

/// \briuf Matrix inverse (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION Matrix<OutputType, N, N> constexpr inverse(const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  const auto det = determinant(mat);
  MUNDY_THROW_ASSERT(det != T(0), std::runtime_error, "Matrix<T>: matrix is singular.");
  return adjugate(mat).template cast<OutputType>() / det;
}

/// \brief Matrix inverse (returns a float if T is an integral type, otherwise returns T)
/// \tparam T The input matrix element type.
/// \tparam Accessor The accessor for the Matrix, assuming this is part of your implementation.
/// \tparam OutputElementType The output matrix element type, defaults T if T is an integral type (e.g., float or
/// double) and float otherwise.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputElementType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr auto inverse_f(const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  return inverse(mat);
}

/// \brief Matrix Frobenius inner product
/// \param[in] a The left matrix.
/// \param[in] b The right matrix.
template <size_t N, size_t M, typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1,
          ValidAccessor<T> Accessor2, typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr auto frobenius_inner_product(const Matrix<U, N, M, Accessor1, OwnershipType1>& a,
                                                              const Matrix<T, N, M, Accessor2, OwnershipType2>& b) {
  return impl::frobenius_inner_product_impl(std::make_index_sequence<N * M>{}, a, b);
}
//@}

//! \name Special vector operations with matrices
//@{

/// \brief Outer product of two vectors
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, size_t M, typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1,
          ValidAccessor<T> Accessor2, typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr auto outer_product(const Vector<U, N, Accessor1, OwnershipType1>& a,
                                                    const Vector<T, M, Accessor2, OwnershipType2>& b)
    -> Matrix<std::common_type_t<T, U>, N, M> {
  return impl::outer_product_impl(std::make_index_sequence<N * M>{}, a, b);
}
//@}

//! \name Matrix norms
//@{

/// \brief Matrix Frobenius norm
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto frobenius_norm(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return Kokkos::sqrt(frobenius_inner_product(mat, mat));
}

/// \brief Matrix infinity norm
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto infinity_norm(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::infinity_norm_impl(std::make_index_sequence<N>{}, mat);
}

/// \brief Matrix 1-norm
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto one_norm(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return impl::one_norm_impl(std::make_index_sequence<N>{}, mat);
}

/// \brief Matrix 2-norm
template <size_t N, size_t M, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto two_norm(const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return Kokkos::sqrt(frobenius_inner_product(mat, mat));
}
//@}

//! \name Matrix<T, Accessor> views
//@{

/// \brief A helper function to create a Matrix<T, Accessor> based on a given (valid) accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a Matrix<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   Matrix<T, Accessor> mat(data);
/// \endcode
/// you can write
/// \code
///   auto mat = get_matrix3_view<T>(data);
/// \endcode
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_matrix_view(Accessor& data) {
  return MatrixView<T, N, M, Accessor>(data);
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_matrix_view(Accessor&& data) {
  return MatrixView<T, N, M, Accessor>(std::forward<Accessor>(data));
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_matrix(Accessor& data) {
  return OwningMatrix<T, N, M, Accessor>(data);
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_matrix(Accessor&& data) {
  return OwningMatrix<T, N, M, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_MATRIX_HPP_

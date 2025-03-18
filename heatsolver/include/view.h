#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>

namespace heatsolver {

// Index for SpaceTime Domain
template <size_t Dim>
using NDIndex_t = std::array<size_t, Dim>;

// Continuous Index for SpaceTime Domain
template <size_t Dim>
using NDCIndex_t = std::array<double, Dim>;

template <size_t Dim>
size_t prod_components(const NDIndex_t<Dim>& index) {
  size_t prod = 1;
  for (size_t i = 0; i < Dim; ++i) {
    prod *= index[i];
  }
  return prod;
}

template <size_t Dim>
NDIndex_t<Dim> floor_index(const NDCIndex_t<Dim>& index) {
  NDIndex_t<Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    result[i] = static_cast<size_t>(std::floor(index[i]));
  }
  return result;
}

template <size_t Dim>
NDIndex_t<Dim> ceil_index(const NDCIndex_t<Dim>& index) {
  NDIndex_t<Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    result[i] = static_cast<size_t>(std::ceil(index[i]));
  }
  return result;
}

template <size_t Dim>
NDCIndex_t<Dim> to_continious_index(const NDIndex_t<Dim>& index) {
  NDCIndex_t<Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    result[i] = static_cast<double>(index[i]);
  }
  return result;
}

/**
 * @brief Interface for N dimensional array
 */
template <typename T, size_t Dim>
class NDArray {
 public:
  static constexpr size_t kDim = Dim;
  using value_type = T;
  using index_type = NDIndex_t<Dim>;
  using cindex_type = NDCIndex_t<Dim>;

  template <bool IsConst>
  class FlatIterator {
    size_t m_index{0};
    using array_ptr_t =
        typename std::conditional_t<IsConst, NDArray<T, Dim> const*,
                                    NDArray<T, Dim>*>;
    array_ptr_t m_array;

   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = int64_t;
    using reference = typename std::conditional_t<IsConst, T const&, T&>;
    using pointer = typename std::conditional_t<IsConst, T const*, T*>;

    explicit FlatIterator(array_ptr_t array, size_t index = 0)
        : m_array(array), m_index(index) {}
    FlatIterator& operator++() {
      ++m_index;
      return *this;
    }
    FlatIterator operator++(int) {
      FlatIterator retval = *this;
      ++(*this);
      return retval;
    }
    FlatIterator& operator--() {
      --m_index;
      return *this;
    }
    FlatIterator operator--(int) {
      FlatIterator retval = *this;
      --(*this);
      return retval;
    }
    FlatIterator& operator+=(difference_type rhs) {
      m_index += rhs;
      return *this;
    }
    FlatIterator& operator-=(difference_type rhs) {
      m_index -= rhs;
      return *this;
    }

    difference_type operator-(const FlatIterator& rhs) const {
      return m_index - rhs.m_index;
    }
    FlatIterator operator+(difference_type rhs) const {
      return FlatIterator(m_array, m_index + rhs);
    }
    FlatIterator operator-(difference_type rhs) const {
      return FlatIterator(m_array, m_index - rhs);
    }
    friend FlatIterator operator+(difference_type lhs,
                                  const FlatIterator& rhs) {
      return FlatIterator(rhs.m_array, lhs + rhs.m_index);
    }
    friend FlatIterator operator-(difference_type lhs,
                                  const FlatIterator& rhs) {
      return FlatIterator(rhs.m_array, lhs - rhs.m_index);
    }

    bool operator>(const FlatIterator& rhs) const {
      return m_index > rhs.m_index;
    }
    bool operator<(const FlatIterator& rhs) const {
      return m_index < rhs.m_index;
    }
    bool operator>=(const FlatIterator& rhs) const {
      return m_index >= rhs.m_index;
    }
    bool operator<=(const FlatIterator& rhs) const {
      return m_index <= rhs.m_index;
    }
    bool operator==(FlatIterator other) const {
      return m_index == other.m_index && m_array == other.m_array;
    }
    bool operator!=(FlatIterator other) const { return !(*this == other); }
    reference operator*() const {
      return m_array->value(m_array->unflat_index(m_index));
    }
  };

  virtual ~NDArray() = default;

  virtual FlatIterator<false> begin() { return FlatIterator<false>(this, 0); }
  virtual FlatIterator<false> end() {
    return FlatIterator<false>(this, prod_components(shape()) - 1);
  }

  virtual FlatIterator<true> begin() const {
    return FlatIterator<true>(this, 0);
  }
  virtual FlatIterator<true> end() const {
    return FlatIterator<true>(this, prod_components(shape()) - 1);
  }

  virtual const index_type& shape() const noexcept = 0;

  virtual index_type unflat_index(size_t index) const noexcept = 0;
  virtual size_t flat_index(const index_type& index) const noexcept = 0;

  virtual T& value(const index_type& index) = 0;
  virtual const T& value(const index_type& index) const = 0;

  /**
   * @brief Return interpolated data.
   */
  virtual T value(const cindex_type& index) const = 0;
};

template <size_t Dim>
class SpaceTimeMesh {
 public:
  static constexpr size_t kDim = Dim;

  using index_type = NDIndex_t<Dim>;
  using cindex_type = NDCIndex_t<Dim>;

  virtual ~SpaceTimeMesh() = default;

  /**
   * @brief Get dx
   */
  virtual const cindex_type& dx() const noexcept = 0;

  /**
   * @brief Get size
   */
  virtual const cindex_type& size() const noexcept = 0;

  /**
   * @brief Set size and update dx
   */
  virtual void size(cindex_type size) noexcept = 0;

  /**
   * @brief Get point with lowest space coordinates i.e. origin
   */
  virtual const cindex_type& origin() const noexcept = 0;

  /**
   * @brief Set point with lowest space coordinates i.e. origin
   */
  virtual void origin(cindex_type size) noexcept = 0;

  /**
   * @brief Get shape
   */
  virtual const index_type& shape() const noexcept = 0;

  /**
   * @brief Set shape and update dx
   */
  virtual void shape(index_type shape) noexcept = 0;

  /**
   * @brief Get position from index
   */
  virtual cindex_type index_to_position(
      const index_type& index) const noexcept = 0;

  /**
   * @brief Get position from continious index
   */
  virtual cindex_type index_to_position(
      const cindex_type& index) const noexcept = 0;

  /**
   * @brief Get continious index from position
   */
  virtual cindex_type position_to_index(
      const cindex_type& pos) const noexcept = 0;

  /**
   * @brief Return space boundary axis  if index lay on space boundary else
   * return nullopt
   */
  std::optional<size_t> check_boundary(const index_type& index) const noexcept;
};

template <size_t Dim>
std::optional<size_t> SpaceTimeMesh<Dim>::check_boundary(
    const index_type& index) const noexcept {
  for (size_t i = 0; i < Dim - 1; ++i) {
    if (index[i] == 0 || index[i] == this->shape()[i] - 1) {
      return i;
    }
  }
  return std::nullopt;
}

}  // namespace heatsolver

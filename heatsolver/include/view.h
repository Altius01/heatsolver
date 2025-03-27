#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>

namespace heatsolver {

/**
 * @brief Index for SpaceTime Domain
 */
template <size_t Dim>
using NDIndex_t = std::array<size_t, Dim>;

/**
 * @brief Continuous Index for SpaceTime Domain
 */
template <size_t Dim>
using NDCIndex_t = std::array<double, Dim>;

/**
 * @brief Compute index's product of components
 */
template <size_t Dim>
size_t prod_components(const NDIndex_t<Dim>& index) {
  return std::accumulate(index.begin(), index.end(), 1, std::multiplies<>());
}

/**
 * @brief Return floor index
 */
template <size_t Dim>
NDIndex_t<Dim> floor_index(const NDCIndex_t<Dim>& index) {
  NDIndex_t<Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    result[i] = static_cast<size_t>(std::floor(index[i]));
  }
  return result;
}

/**
 * @brief Return ceil index
 */
template <size_t Dim>
NDIndex_t<Dim> ceil_index(const NDCIndex_t<Dim>& index) {
  NDIndex_t<Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    result[i] = static_cast<size_t>(std::ceil(index[i]));
  }
  return result;
}

/**
 * @brief Convert NDIndex_t to NDCIndex_t
 */
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
template <typename T, size_t Dim, typename Tref = T&,
          typename Tconstref = T const&>
class NDArray {
 public:
  static constexpr size_t kDim = Dim;
  using value_type = T;
  using value_reference = Tref;
  using value_const_reference = Tconstref;
  using index_type = NDIndex_t<Dim>;
  using cindex_type = NDCIndex_t<Dim>;

  /**
   * @brief Random acess 1D iterator for NDArray
   */
  template <bool IsConst>
  class FlatIterator {
   protected:
    size_t m_index{0};
    using array_ptr_t =
        typename std::conditional_t<IsConst,
                                    NDArray<T, Dim, Tref, Tconstref> const*,
                                    NDArray<T, Dim, Tref, Tconstref>*>;
    array_ptr_t m_array;

   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = int64_t;
    using reference =
        typename std::conditional_t<IsConst, value_const_reference,
                                    value_reference>;
    using pointer = typename std::conditional_t<IsConst, T const*, T*>;

    explicit FlatIterator(array_ptr_t array, size_t index = 0)
        : m_array(array), m_index(index) {}
    virtual ~FlatIterator() = default;
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
    return FlatIterator<false>(this, prod_components(shape()));
  }

  virtual FlatIterator<true> begin() const {
    return FlatIterator<true>(this, 0);
  }
  virtual FlatIterator<true> end() const {
    return FlatIterator<true>(this, prod_components(shape()));
  }

  /**
   * @brief Get shape
   */
  virtual const index_type& shape() const noexcept = 0;

  /**
   * @brief Get NDIndex from flat index
   */
  virtual index_type unflat_index(size_t index) const noexcept = 0;

  /**
   * @brief Get flat index from NDIndex
   */
  virtual size_t flat_index(const index_type& index) const noexcept = 0;

  /**
   * @brief Set value
   */
  virtual value_reference value(const index_type& index) = 0;

  /**
   * @brief Get value
   */
  virtual value_const_reference value(const index_type& index) const = 0;

  /**
   * @brief Return interpolated data.
   */
  virtual value_type value(const cindex_type& index) const = 0;
};

/**
 * @brief Return l2 norm of NDArray
 *
 * @param a
 */
template <size_t Dim, typename ValueType, typename ATref, typename ATconstref>
ValueType l2_norm(NDArray<ValueType, Dim, ATref, ATconstref>& a) {
  ValueType norm = 0;
  for (const auto& a_value : a) {
    norm += (a_value * a_value);
  }
  return std::sqrt(norm);
}

/**
 * @brief Return difference's l2 norm of two NDArrays. If array's shapes are
 * inequal to each other diff value of -1 is returned.
 */
template <size_t Dim, typename ValueType, typename ATref, typename ATconstref,
          typename BTref, typename BTconstref>
ValueType l2_diff_norm(const NDArray<ValueType, Dim, ATref, ATconstref>& a,
                       const NDArray<ValueType, Dim, BTref, BTconstref>& b) {
  if (a.shape() != b.shape()) {
    return -1.0;
  }
  ValueType error = 0;
  auto b_it = b.begin();
  for (const auto& a_value : a) {
    error += (a_value - *b_it) * (a_value - *b_it);
    ++b_it;
  }
  return std::sqrt(error);
}

/**
 * @brief SpaceTime Mesh interface
 */
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

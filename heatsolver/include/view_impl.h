#pragma once

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "view.h"

namespace heatsolver {

// Implementation for 2D spatial case
constexpr size_t kDim = 3;
constexpr size_t kSpaceDim = 2;
using Index_t = NDIndex_t<kDim>;
using CIndex_t = NDCIndex_t<kDim>;

class ProblemSpaceTimeFunction : public NDArray<double, kDim> {
 public:
  explicit ProblemSpaceTimeFunction(
      std::function<cindex_type(const cindex_type&)> index_to_position,
      std::function<value_type(const cindex_type&)> foo,
      const index_type& shape = {});

  ~ProblemSpaceTimeFunction() override;

  ProblemSpaceTimeFunction(const ProblemSpaceTimeFunction& other) {
    m_foo = other.m_foo;
    m_index_to_position = other.m_index_to_position;
  }

  ProblemSpaceTimeFunction& operator=(const ProblemSpaceTimeFunction& other) {
    if (this == &other) {
      return *this;
    }
    m_foo = other.m_foo;
    m_index_to_position = other.m_index_to_position;
    return *this;
  }

  ProblemSpaceTimeFunction(ProblemSpaceTimeFunction&& other) noexcept {
    m_foo = std::move(other.m_foo);
    m_index_to_position = std::move(other.m_index_to_position);
  }

  ProblemSpaceTimeFunction& operator=(
      ProblemSpaceTimeFunction&& other) noexcept {
    m_foo = std::move(other.m_foo);
    m_index_to_position = std::move(other.m_index_to_position);
    return *this;
  }

  const index_type& shape() const noexcept override { return m_shape; };

  index_type& shape(index_type index) noexcept { return m_shape = index; };
  index_type& shape(const index_type& index) noexcept {
    return m_shape = index;
  };
  index_type& shape(index_type&& index) noexcept {
    return m_shape = std::move(index);
  };

  index_type unflat_index(size_t index) const noexcept override;

  size_t flat_index(const index_type& index) const noexcept override;

  value_reference value(const index_type& index) override;
  value_const_reference value(const index_type& index) const override;
  /**
   * @brief Return linear interpolated data.
   */
  value_type value(const cindex_type& index) const override {
    return m_foo(m_index_to_position(index));
  }

 protected:
  index_type m_shape{};
  std::function<value_type(const cindex_type&)> m_foo;
  std::function<cindex_type(const cindex_type&)> m_index_to_position;

  static std::string to_string(const index_type& index) {
    std::string result;
    for (auto i : index) {
      result += std::to_string(i) + " ";
    }
    return result;
  }
};

/**
 * @brief Class for storing discrete scalar spatial 2D and time field: field(x,
 * y, t)
 */
class ProblemSpaceTimeFunctionArray : public NDArray<double, kDim> {
 public:
  explicit ProblemSpaceTimeFunctionArray(const index_type& shape)
      : m_shape(shape), m_data(prod_components(shape)) {}

  ProblemSpaceTimeFunctionArray(const index_type& shape,
                                std::vector<value_type> data)
      : m_shape(shape), m_data(std::move(data)) {}

  ProblemSpaceTimeFunctionArray(const ProblemSpaceTimeFunctionArray& other) {
    m_shape = other.m_shape;
    m_data = other.m_data;
  }

  ProblemSpaceTimeFunctionArray& operator=(
      const ProblemSpaceTimeFunctionArray& other) {
    if (this == &other) {
      return *this;
    }

    m_shape = other.m_shape;
    m_data = other.m_data;
    return *this;
  }

  ProblemSpaceTimeFunctionArray(
      ProblemSpaceTimeFunctionArray&& other) noexcept {
    m_shape = other.m_shape;
    m_data = std::move(other.m_data);
  }

  ProblemSpaceTimeFunctionArray& operator=(
      ProblemSpaceTimeFunctionArray&& other) noexcept {
    m_shape = other.m_shape;
    m_data = std::move(other.m_data);
    return *this;
  }

  ~ProblemSpaceTimeFunctionArray() override = default;

  void fill(value_type value) {
    std::fill(m_data.begin(), m_data.end(), value);
  }

  const index_type& shape() const noexcept override { return m_shape; }

  index_type unflat_index(size_t index) const noexcept override;
  size_t flat_index(const index_type& index) const noexcept override;

  value_type& value(const index_type& index) override;
  const value_type& value(const index_type& index) const override;

  /**
   * @brief Return linear interpolated data.
   */
  value_type value(const cindex_type& index) const override;

 protected:
  value_type interpolate_value(const cindex_type& index) const;
  bool check_index(const index_type& index) const noexcept;

  index_type m_shape;
  std::vector<value_type> m_data;
};

class ProblemSpaceFunctionArray : public ProblemSpaceTimeFunctionArray {
 public:
  explicit ProblemSpaceFunctionArray(const index_type& shape)
      : ProblemSpaceTimeFunctionArray(strip_time_dim(shape)) {}

  ProblemSpaceFunctionArray(const index_type& shape,
                            std::vector<value_type> data)
      : ProblemSpaceTimeFunctionArray(strip_time_dim(shape), std::move(data)) {}

  ProblemSpaceFunctionArray(const ProblemSpaceFunctionArray& other) = default;
  ProblemSpaceFunctionArray& operator=(const ProblemSpaceFunctionArray& other) =
      default;
  ProblemSpaceFunctionArray(ProblemSpaceFunctionArray&& other) noexcept =
      default;
  ProblemSpaceFunctionArray& operator=(
      ProblemSpaceFunctionArray&& other) noexcept = default;

  value_type& value(const index_type& index) override {
    return ProblemSpaceTimeFunctionArray::value(strip_time_index(index));
  }
  const value_type& value(const index_type& index) const override {
    return ProblemSpaceTimeFunctionArray::value(strip_time_index(index));
  }

  /**
   * @brief Return interpolated data.
   */
  value_type value(const cindex_type& index) const override {
    return ProblemSpaceTimeFunctionArray::value(strip_time_index(index));
  }

 private:
  static index_type strip_time_dim(const index_type& index) {
    auto result = index;
    result.back() = 1;
    return result;
  }
  static index_type strip_time_index(const index_type& index) {
    auto result = index;
    result.back() = 0;
    return result;
  }
  static cindex_type strip_time_index(const cindex_type& index) {
    auto result = index;
    result.back() = 0;
    return result;
  }
};

class ProblemMesh final : public SpaceTimeMesh<kDim> {
 public:
  ProblemMesh() = default;

  /**
   * @brief Get dx
   */
  const cindex_type& dx() const noexcept override { return m_dx; }

  /**
   * @brief Get size
   */
  const cindex_type& size() const noexcept override { return m_size; }

  /**
   * @brief Set size and update dx
   */
  void size(cindex_type size) noexcept override {
    m_size = size;
    update_dx();
  }

  /**
   * @brief Get point with lowest space coordinates i.e. origin
   */
  const cindex_type& origin() const noexcept override { return m_origin; }

  /**
   * @brief Set point with lowest space coordinates i.e. origin
   */
  void origin(cindex_type size) noexcept override { m_origin = size; }

  /**
   * @brief Get shape
   */
  const index_type& shape() const noexcept override { return m_shape; }

  /**
   * @brief Set shape and update dx
   */
  void shape(index_type shape) noexcept override {
    m_shape = shape;
    update_dx();
  }

  /**
   * @brief Get position from index
   */
  cindex_type index_to_position(
      const index_type& index) const noexcept override {
    cindex_type pos{};
    for (size_t i = 0; i < kDim; ++i) {
      pos[i] = ((index[i] + 0.5) * m_dx[i]) + m_origin[i];
    }
    return pos;
  }

  /**
   * @brief Get position from continious index
   */
  cindex_type index_to_position(
      const cindex_type& index) const noexcept override {
    cindex_type pos{};
    for (size_t i = 0; i < kDim; ++i) {
      pos[i] = ((index[i] + 0.5) * m_dx[i]) + m_origin[i];
    }
    return pos;
  }

  /**
   * @brief Get continious index from position
   */
  cindex_type position_to_index(
      const cindex_type& pos) const noexcept override {
    cindex_type index{};
    for (size_t i = 0; i < kDim; ++i) {
      index[i] = (pos[i] - m_origin[i]) / m_dx[i] - 0.5;
    }
    return index;
  }

 private:
  void update_dx() {
    for (size_t i = 0; i < kDim; ++i) {
      m_dx[i] =
          (m_shape[i] > 0
               ? m_size[i] / m_shape[i]
               : std::numeric_limits<std::decay_t<decltype(m_size[i])>>::max());
    }
  }

  index_type m_shape{};
  cindex_type m_dx{};
  cindex_type m_size{};
  cindex_type m_origin{};
};

}  // namespace heatsolver

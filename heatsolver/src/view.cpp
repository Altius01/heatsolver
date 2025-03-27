#include "view.h"

#include "view_impl.h"

namespace heatsolver {

namespace {
std::unordered_map<
    const ProblemSpaceTimeFunction*,
    std::unordered_map<std::string, ProblemSpaceTimeFunction::value_type>>
    problem_space_time_function_cache;
}

ProblemSpaceTimeFunction::ProblemSpaceTimeFunction(
    std::function<cindex_type(const cindex_type&)> index_to_position,
    std::function<value_type(const cindex_type&)> foo, const index_type& shape)
    : m_index_to_position(std::move(index_to_position)),
      m_foo(std::move(foo)),
      m_shape(shape) {
  problem_space_time_function_cache.emplace(
      this, std::unordered_map<std::string, value_type>());
}

ProblemSpaceTimeFunction::~ProblemSpaceTimeFunction() {
  problem_space_time_function_cache.erase(this);
};

ProblemSpaceTimeFunction::value_reference ProblemSpaceTimeFunction::value(
    const index_type& index) {
  auto str_idx = to_string(index);
  if (problem_space_time_function_cache.at(this).find(str_idx) ==
      problem_space_time_function_cache.at(this).end()) {
    problem_space_time_function_cache.at(this).emplace(
        str_idx,
        m_foo(m_index_to_position(to_continious_index(std::move(index)))));
  }
  return problem_space_time_function_cache.at(this).at(str_idx);
}

ProblemSpaceTimeFunction::value_const_reference ProblemSpaceTimeFunction::value(
    const index_type& index) const {
  auto str_idx = to_string(index);
  if (problem_space_time_function_cache.at(this).find(str_idx) ==
      problem_space_time_function_cache.at(this).end()) {
    problem_space_time_function_cache.at(this).emplace(
        str_idx,
        m_foo(m_index_to_position(to_continious_index(std::move(index)))));
  }
  return problem_space_time_function_cache.at(this).at(str_idx);
}

ProblemSpaceTimeFunction::index_type ProblemSpaceTimeFunction::unflat_index(
    size_t index) const noexcept {
  Index_t idx{};

  size_t tmp_flat_shape = prod_components(m_shape);
  for (size_t i = 0; i < kDim; ++i) {
    tmp_flat_shape /= m_shape[i];
    idx[i] = index / tmp_flat_shape;
    index %= tmp_flat_shape;
  }
  return idx;
}

size_t ProblemSpaceTimeFunction::flat_index(
    const ProblemSpaceTimeFunction::index_type& index) const noexcept {
  size_t idx = 0;

  for (size_t i = 0; i < kDim; ++i) {
    idx = idx * m_shape[i] + index[i];
  }
  return idx;
}

ProblemSpaceTimeFunctionArray::index_type
ProblemSpaceTimeFunctionArray::unflat_index(size_t index) const noexcept {
  Index_t idx{};

  size_t tmp_flat_shape = m_data.size();
  for (size_t i = 0; i < kDim; ++i) {
    tmp_flat_shape /= m_shape[i];
    idx[i] = index / tmp_flat_shape;
    index %= tmp_flat_shape;
  }
  return idx;
}

size_t ProblemSpaceTimeFunctionArray::flat_index(
    const ProblemSpaceTimeFunctionArray::index_type& index) const noexcept {
  size_t idx = 0;

  for (size_t i = 0; i < kDim; ++i) {
    idx = idx * m_shape[i] + index[i];
  }
  return idx;
}

double& ProblemSpaceTimeFunctionArray::value(
    const ProblemSpaceTimeFunctionArray::index_type& index) {
  return m_data.at(flat_index(index));
}

const double& ProblemSpaceTimeFunctionArray::value(
    const ProblemSpaceTimeFunctionArray::index_type& index) const {
  return m_data.at(flat_index(index));
}

double ProblemSpaceTimeFunctionArray::value(
    const ProblemSpaceTimeFunctionArray::cindex_type& index) const {
  return interpolate_value(index);
}

double ProblemSpaceTimeFunctionArray::interpolate_value(
    const ProblemSpaceTimeFunctionArray::cindex_type& index) const {
  auto floor_idx = floor_index(index);
  auto xd = index[0] - floor_idx[0];
  auto yd = index[1] - floor_idx[1];
  auto zd = index[2] - floor_idx[2];

  double weights[2][2][2] = {
      {{(1 - xd) * (1 - yd) * (1 - zd), (1 - xd) * (1 - yd) * zd},
       {(1 - xd) * yd * (1 - zd), (1 - xd) * yd * zd}},
      {{xd * (1 - yd) * (1 - zd), xd * (1 - yd) * zd},
       {xd * yd * (1 - zd), xd * yd * zd}}};

  double result = 0.0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        if (weights[i][j][k] > std::numeric_limits<double>::epsilon())
          result += m_data.at(flat_index(
                        {std::min(floor_idx[0] + i, m_shape[0] - 1),
                         std::min(floor_idx[1] + j, m_shape[1] - 1),
                         std::min(floor_idx[2] + k, m_shape[2] - 1)})) *
                    weights[i][j][k];
      }
    }
  }

  return result;
}

}  // namespace heatsolver

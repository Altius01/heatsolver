#include "view.h"

#include "view_impl.h"

namespace heatsolver {

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

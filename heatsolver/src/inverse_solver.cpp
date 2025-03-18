#include "inverse_solver.h"

#include <osqp++.h>

#include <iostream>

#include "Eigen/src/IterativeLinearSolvers/LeastSquareConjugateGradient.h"
#include "Eigen/src/SparseQR/SparseQR.h"
#include "view.h"
#include "view_impl.h"

namespace heatsolver {

Index_t InverseSolver::unflat_spatial_index(size_t index) const noexcept {
  Index_t idx{};

  const auto& shape = m_data.mesh->shape();
  size_t tmp_flat_shape = 1;
  for (size_t i = 0; i < kSpaceDim; ++i) {
    tmp_flat_shape *= shape[i];
  }

  for (size_t i = 0; i < kSpaceDim; ++i) {
    tmp_flat_shape /= shape[i];
    idx[i] = index / tmp_flat_shape;
    index %= tmp_flat_shape;
  }
  idx[kSpaceDim] = 0;
  return idx;
}

size_t InverseSolver::flat_spatial_index(const Index_t& index) const noexcept {
  size_t idx = 0;

  const auto& shape = m_data.mesh->shape();
  for (size_t i = 0; i < kSpaceDim; ++i) {
    idx = idx * shape[i] + index[i];
  }
  return idx;
}

size_t InverseSolver::add_time_to_flat_index(size_t flat_index,
                                             size_t time_idx) {
  return (flat_index * m_data.mesh->shape()[kSpaceDim]) + time_idx;
}

InverseSolverData::Function_t::value_type InverseSolver::rhs(Index_t index) {
  auto result = m_data.temperature->value(index);
  index[kSpaceDim] -= 1;
  result -= m_data.temperature->value(index);
  result /= m_data.mesh->dx()[kSpaceDim];
  index[kSpaceDim] += 1;
  return result - m_data.heat_source->value(index);
}

void InverseSolver::assemble_rhs_k(size_t time_idx) {
  const auto& shape = m_data.mesh->shape();
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      Index_t index = {i, j, time_idx};

      auto flat_index = flat_spatial_index(index);

      m_rhs.coeffRef(add_time_to_flat_index(flat_index, time_idx)) = rhs(index);
    }
  }
}

void InverseSolver::assemble_system_k(size_t time_idx) {
  const auto& shape = m_data.mesh->shape();
  const auto& dx = m_data.mesh->dx();
  auto inv_dx = dx;
  for (auto& elem : inv_dx) {
    elem = 1.0 / elem;
  }
  auto inv_dx_squared = inv_dx;
  for (auto& elem : inv_dx) {
    elem *= elem;
  }

  size_t spatial_components = 1;

  for (size_t i = 0; i < kSpaceDim; ++i) {
    spatial_components *= shape[i];
  }

  std::vector<Triplet> triplet_list;
  triplet_list.reserve(5 * spatial_components);

  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      Index_t index = {i, j, time_idx};
      Index_t index_right = {i + 1, j, time_idx};
      Index_t index_left = {i - 1, j, time_idx};
      Index_t index_up = {i, j + 1, time_idx};
      Index_t index_down = {i, j - 1, time_idx};

      auto flat_index = flat_spatial_index(index);
      auto flat_index_right = flat_spatial_index(index_right);
      auto flat_index_left = flat_spatial_index(index_left);
      auto flat_index_up = flat_spatial_index(index_up);
      auto flat_index_down = flat_spatial_index(index_down);

      decltype(m_system)::value_type coef_center = 0.0;
      decltype(m_system)::value_type coef_right = 0.0;
      decltype(m_system)::value_type coef_left = 0.0;
      decltype(m_system)::value_type coef_up = 0.0;
      decltype(m_system)::value_type coef_down = 0.0;

      decltype(m_system)::value_type temperature_center =
          m_data.temperature->value(index);
      decltype(m_system)::value_type temperature_right = 0.0;
      decltype(m_system)::value_type temperature_left = 0.0;
      decltype(m_system)::value_type temperature_up = 0.0;
      decltype(m_system)::value_type temperature_down = 0.0;

      // Коэффициенты по x
      if (i == 0) {
        // Левая граница: разность вперёд
        temperature_right = m_data.temperature->value(index_right);
        coef_right =
            (temperature_right - temperature_center) * 0.5 * inv_dx_squared[0];
        coef_center -= coef_right;  // Поток через правую грань
      } else if (i == shape[0] - 1) {
        // Правая граница: разность назад
        temperature_left = m_data.temperature->value(index_left);
        coef_left =
            (temperature_center - temperature_left) * 0.5 * inv_dx_squared[0];
        coef_center += coef_left;  // Поток через левую грань (входящий)
      } else {
        // Внутренние точки: центральная разность
        temperature_right = m_data.temperature->value(index_right);
        temperature_left = m_data.temperature->value(index_left);
        coef_right =
            (temperature_right - temperature_center) * 0.5 * inv_dx_squared[0];
        coef_left =
            (temperature_center - temperature_left) * 0.5 * inv_dx_squared[0];
        coef_center -= (coef_right + coef_left);
      }

      // Коэффициенты по y
      if (j == 0) {
        // Нижняя граница: разность вперёд
        temperature_up = m_data.temperature->value(index_up);
        coef_up =
            (temperature_up - temperature_center) * 0.5 * inv_dx_squared[1];
        coef_center -= coef_up;
      } else if (j == shape[1] - 1) {
        // Верхняя граница: разность назад
        temperature_down = m_data.temperature->value(index_down);
        coef_down =
            (temperature_down - temperature_center) * 0.5 * inv_dx_squared[1];
        coef_center += coef_down;
      } else {
        // Внутренние точки: центральная разность
        temperature_up = m_data.temperature->value(index_up);
        temperature_down = m_data.temperature->value(index_down);
        coef_up =
            (temperature_up - temperature_center) * 0.5 * inv_dx_squared[1];
        coef_down =
            (temperature_down - temperature_center) * 0.5 * inv_dx_squared[1];
        coef_center -= (coef_up + coef_down);
      }

      // Заполняем triplet-список
      triplet_list.emplace_back(add_time_to_flat_index(flat_index, time_idx),
                                flat_index, coef_center);
      if (i < shape[0] - 1)
        triplet_list.emplace_back(add_time_to_flat_index(flat_index, time_idx),
                                  flat_index_right, coef_right);
      if (i > 0)
        triplet_list.emplace_back(add_time_to_flat_index(flat_index, time_idx),
                                  flat_index_left, coef_left);
      if (j < shape[1] - 1)
        triplet_list.emplace_back(add_time_to_flat_index(flat_index, time_idx),
                                  flat_index_up, coef_up);
      if (j > 0)
        triplet_list.emplace_back(add_time_to_flat_index(flat_index, time_idx),
                                  flat_index_down, coef_down);
    }
  }

  m_system.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

InverseSolver::SpMat InverseSolver::assemble_regularisation() {
  const auto& shape = m_data.mesh->shape();
  const auto& dx = m_data.mesh->dx();
  auto inv_dx = dx;
  for (auto& elem : inv_dx) {
    elem = 1.0 / elem;
  }
  auto inv_dx_squared = inv_dx;
  for (auto& elem : inv_dx) {
    elem *= elem;
  }

  size_t spatial_components = 1;

  for (size_t i = 0; i < kSpaceDim; ++i) {
    spatial_components *= shape[i];
  }

  std::vector<Triplet> triplet_list;
  triplet_list.reserve(5 * spatial_components);

  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      Index_t index = {i, j, 0};
      Index_t index_right = {i + 1, j, 0};
      Index_t index_left = {i - 1, j, 0};
      Index_t index_up = {i, j + 1, 0};
      Index_t index_down = {i, j - 1, 0};

      auto flat_index = flat_spatial_index(index);
      auto flat_index_right = flat_spatial_index(index_right);
      auto flat_index_left = flat_spatial_index(index_left);
      auto flat_index_up = flat_spatial_index(index_up);
      auto flat_index_down = flat_spatial_index(index_down);

      double coef_center = 0.0;
      double coef_right = 0.0;
      double coef_left = 0.0;
      double coef_up = 0.0;
      double coef_down = 0.0;

      // По x
      if (i == 0) {
        coef_right = inv_dx_squared[0];
        coef_center -= coef_right;
      } else if (i == shape[0] - 1) {
        coef_left = inv_dx_squared[0];
        coef_center -= coef_left;
      } else {
        coef_right = inv_dx_squared[0];
        coef_left = inv_dx_squared[0];
        coef_center -= (coef_right + coef_left);
      }

      // По y
      if (j == 0) {
        coef_up = inv_dx_squared[1];
        coef_center -= coef_up;
      } else if (j == shape[1] - 1) {
        coef_down = inv_dx_squared[1];
        coef_center -= coef_down;
      } else {
        coef_up = inv_dx_squared[1];
        coef_down = inv_dx_squared[1];
        coef_center -= (coef_up + coef_down);
      }

      // Заполняем triplet-список
      triplet_list.emplace_back(flat_index, flat_index, coef_center);
      if (i < shape[0] - 1)
        triplet_list.emplace_back(flat_index, flat_index_right, coef_right);
      if (i > 0)
        triplet_list.emplace_back(flat_index, flat_index_left, coef_left);
      if (j < shape[1] - 1)
        triplet_list.emplace_back(flat_index, flat_index_up, coef_up);
      if (j > 0)
        triplet_list.emplace_back(flat_index, flat_index_down, coef_down);
    }
  }

  SpMat reg(spatial_components * shape[kSpaceDim], spatial_components);
  reg.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return reg;
}

void InverseSolver::assemble_system_new(double lambda) {
  const auto& shape = m_data.mesh->shape();
  size_t spatial_components = 1;
  for (size_t i = 0; i < kSpaceDim; ++i) {
    spatial_components *= shape[i];
  }

  m_system.resize(spatial_components * shape[kSpaceDim], spatial_components);
  m_rhs.resize(spatial_components * shape[kSpaceDim]);

  for (int k = 1; k < shape[kSpaceDim]; ++k) {
    assemble_system_k(k);
    assemble_rhs_k(k);
  }
  auto reg = assemble_regularisation();
  m_system += lambda * reg;

  m_rhs = m_system.transpose() * m_rhs;
  m_system = m_system.transpose() * m_system;
}

namespace {
void print_spmat(InverseSolver::SpMat& mat) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      printf("%.2f ", mat.coeff(i, j));
    }
    std::cout << "\n";
  }
}
}  // namespace

Eigen::VectorXd InverseSolver::solve(double lambda) {
  assemble_system_new(lambda);

  auto cols = m_system.cols();
  auto rows = m_system.cols();

  Eigen::VectorXd x(cols);

  const double k_infinity = std::numeric_limits<double>::infinity();

  SpMat constraint(rows, cols);
  constraint.setIdentity();

  Eigen::VectorXd lower_bounds = Eigen::VectorXd::Zero(cols);
  Eigen::VectorXd upper_bounds = Eigen::VectorXd::Constant(cols, k_infinity);
  osqp::OsqpInstance instance;
  instance.objective_matrix = m_system;
  instance.objective_vector = -m_rhs;
  instance.constraint_matrix = constraint;
  instance.lower_bounds = lower_bounds;
  instance.upper_bounds = upper_bounds;

  osqp::OsqpSolver solver;
  osqp::OsqpSettings settings;
  // Edit settings if appropriate.
  auto status = solver.Init(instance, settings);
  // Assuming status.ok().
  osqp::OsqpExitCode exit_code = solver.Solve();
  // Assuming exit_code == OsqpExitCode::kOptimal.
  double optimal_objective = solver.objective_value();
  Eigen::VectorXd optimal_solution = solver.primal_solution();

  return optimal_solution;
}
}  // namespace heatsolver

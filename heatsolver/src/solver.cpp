#include "solver.h"

namespace heatsolver {

void HeatSolver::solve() {
  for (size_t i = 0; i < m_data->mesh->shape()[kSpaceDim] - 1; ++i) {
    time_step(i);
  }
}

void HeatSolver::time_step(size_t time_index) {
  auto time_step = m_data->mesh->dx()[kSpaceDim];
  // X sweep

  const auto &shape = m_data->mesh->shape();
#pragma omp parallel for
  for (size_t j = 0; j < shape[1]; ++j) {
    Index_t index = {1, j, time_index};
    assemble_and_solve_system(0, *m_data->temperature, *m_temporary_solution,
                              index, time_step);
  }

// Y sweep
#pragma omp parallel for
  for (size_t i = 0; i < shape[0]; ++i) {
    Index_t index = {i, 1, time_index + 1};
    index[0] = i;
    assemble_and_solve_system(1, *m_temporary_solution, *m_data->temperature,
                              index, time_step);
  }
}

void HeatSolver::assemble_and_solve_system(
    size_t dim, const HeatProblemData::Function_t &prev_solution,
    HeatProblemData::Function_t &next_solution, Index_t index,
    double time_step) {
  if (dim > kSpaceDim) {
    return;
  }

  const auto &mesh = m_data->mesh;
  TriDiagMatrixSolver m_solver(mesh->shape()[dim]);

  auto dx_dim = mesh->dx()[dim];
  auto shape = mesh->shape();
  auto shape_dim = shape[dim];

  Index_t tmp_index = index;

  CIndex_t tmp_cindex = to_continious_index(index);
  tmp_cindex.back() += dim * m_cindex_time_step;

  auto inv_h_squared = 1 / (dx_dim * dx_dim);
  auto tau_inv_h_squared = time_step * inv_h_squared;

  (m_solver.m_upper)[0] = 0.0;
  (m_solver.m_lower)[0] = 0.0;
  (m_solver.m_diag)[0] = 1.0;

  tmp_cindex[dim] = 0;
  (m_solver.m_rhs)[0] = m_data->dirichlet_boundary[dim]->value(tmp_cindex);

  (m_solver.m_upper)[shape_dim - 1] = 0.0;
  (m_solver.m_lower)[shape_dim - 1] = 0.0;
  (m_solver.m_diag)[shape_dim - 1] = 1.0;

  tmp_cindex[dim] = 1;
  (m_solver.m_rhs)[shape_dim - 1] =
      m_data->dirichlet_boundary[dim]->value(tmp_cindex);

#pragma omp parallel for
  for (size_t index_i = 1; index_i < shape_dim - 1; ++index_i) {
    auto tmp_index_copy = tmp_index;
    auto tmp_cindex_copy = tmp_cindex;
    tmp_index_copy[dim] = index_i;
    auto u_prev = prev_solution.value(tmp_index_copy);

    tmp_cindex_copy[dim] = index_i - 0.5;
    auto k_im = m_data->coefficient->value(tmp_cindex_copy);

    tmp_cindex_copy[dim] = index_i + 0.5;
    auto k_ip = m_data->coefficient->value(tmp_cindex_copy);
    tmp_cindex_copy[dim] = index_i;

    auto upper_value = tau_inv_h_squared * k_ip;
    (m_solver.m_upper)[index_i] = -upper_value;
    auto lower_value = tau_inv_h_squared * k_im;
    (m_solver.m_lower)[index_i] = -lower_value;

    (m_solver.m_diag)[index_i] = 1.0 + upper_value + lower_value;

    auto boundary_idx = mesh->check_boundary(index);
    if (boundary_idx) {
      auto b_idx = boundary_idx.value();
      tmp_cindex_copy[b_idx] = index[b_idx] > 0 ? 1 : 0;
      (m_solver.m_rhs)[index_i] =
          m_data->dirichlet_boundary[b_idx]->value(tmp_cindex_copy);
    } else {
      (m_solver.m_rhs)[index_i] =
          (time_step * m_data->heat_source->value(tmp_cindex_copy)) + (u_prev);
    }
  }

  tmp_index = index;
  m_solver.solve([&](size_t index_i, double value) {
    tmp_index[dim] = index_i;
    next_solution.value(tmp_index) = value;
  });
}

}  // namespace heatsolver

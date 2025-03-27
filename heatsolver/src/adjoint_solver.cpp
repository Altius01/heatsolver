#include "adjoint_solver.h"

#include <algorithm>
#include <memory>
#include <type_traits>

#include "solver.h"
#include "view.h"
#include "view_impl.h"

namespace heatsolver {

void HeatAdjointSolver::assemble_and_solve_system(
    size_t dim, const HeatProblemData::Function_t& prev_solution,
    HeatProblemData::Function_t& next_solution, Index_t index,
    double time_step) {
  if (dim > kSpaceDim) {
    return;
  }

  const auto& mesh = m_data->mesh;
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

  (m_solver.m_rhs)[0] = 0;

  (m_solver.m_upper)[shape_dim - 1] = 0.0;
  (m_solver.m_lower)[shape_dim - 1] = 0.0;
  (m_solver.m_diag)[shape_dim - 1] = 1.0;

  (m_solver.m_rhs)[shape_dim - 1] = 0;

  for (size_t index_i = 1; index_i < shape_dim - 1; ++index_i) {
    auto tmp_index_copy = tmp_index;
    auto tmp_cindex_copy = tmp_cindex;
    tmp_index_copy[dim] = index_i;
    auto u_prev = prev_solution.value(tmp_index);

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
      (m_solver.m_rhs)[index_i] = 0;
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

void HeatAdjointSolver::time_step(size_t time_index) {
  HeatSolver::time_step(time_index);
}

AdjointCoefficientSolver::AdjointCoefficientSolver(
    std::shared_ptr<HeatProblemData::Mesh_t> mesh,
    std::shared_ptr<HeatProblemData::Function_t> target_temperature,
    std::shared_ptr<HeatProblemData::Function_t> heat_source,
    std::shared_ptr<HeatProblemData::Function_t> coefficient)
    : m_target_temperature(std::move(target_temperature)) {
  m_target_temperature_norm = l2_norm(*m_target_temperature);

  m_coefficient = std::move(coefficient);

  m_mean_coefficient_gradient =
      std::make_shared<ProblemSpaceFunctionArray>(mesh->shape());
  std::reinterpret_pointer_cast<ProblemSpaceFunctionArray>(
      m_mean_coefficient_gradient)
      ->fill(0);

  m_mean_squared_coefficient_gradient =
      std::make_shared<ProblemSpaceFunctionArray>(mesh->shape());
  std::reinterpret_pointer_cast<ProblemSpaceFunctionArray>(
      m_mean_squared_coefficient_gradient)
      ->fill(0);

  m_data = std::make_shared<HeatProblemData>();
  m_data->mesh = std::move(mesh);
  m_data->temperature =
      std::make_shared<ProblemSpaceTimeFunctionArray>(m_data->mesh->shape());
  std::copy(m_target_temperature->begin(), m_target_temperature->end(),
            m_data->temperature->begin());
  m_data->coefficient = m_coefficient;
  m_data->heat_source = std::move(heat_source);
  m_data->dirichlet_boundary = {
      std::make_shared<ProblemSpaceTimeFunction>(
          [&](const ProblemSpaceTimeFunction::cindex_type& index) {
            return index;
          },
          [&](const ProblemSpaceTimeFunction::cindex_type& index) {
            auto index_copy = index;
            if (index[0] > 0)
              index_copy[0] = m_target_temperature->shape()[0] - 1;
            return m_target_temperature->value(index_copy);
          }),
      std::make_shared<ProblemSpaceTimeFunction>(
          [&](const ProblemSpaceTimeFunction::cindex_type& index) {
            return index;
          },
          [&](const ProblemSpaceTimeFunction::cindex_type& index) {
            auto index_copy = index;
            if (index[1] > 0)
              index_copy[1] = m_target_temperature->shape()[1] - 1;

            return m_target_temperature->value(index_copy);
          }),
  };

  m_adjoint_data = std::make_shared<HeatProblemData>();
  m_adjoint_data->mesh = m_data->mesh;
  m_adjoint_data->temperature =
      std::make_shared<ProblemSpaceTimeFunctionArray>(m_data->mesh->shape());
  m_adjoint_data->coefficient = m_coefficient;
  m_adjoint_data->heat_source = std::make_shared<ProblemSpaceTimeFunction>(
      [&](const ProblemSpaceTimeFunction::cindex_type& index) { return index; },
      [&](const ProblemSpaceTimeFunction::cindex_type& index) {
        return m_target_temperature->value(index) -
               m_data->temperature->value(index);
      },
      m_data->mesh->shape());

  m_data->dirichlet_boundary = {
      std::make_shared<ProblemSpaceTimeFunction>(
          [&](const ProblemSpaceTimeFunction::cindex_type& index) {
            return index;
          },
          [&](const ProblemSpaceTimeFunction::cindex_type& /*index*/) {
            return 0;
          }),
      std::make_shared<ProblemSpaceTimeFunction>(
          [&](const ProblemSpaceTimeFunction::cindex_type& index) {
            return index;
          },
          [&](const ProblemSpaceTimeFunction::cindex_type& /*index*/) {
            return 0;
          })};

  m_solver = std::make_shared<HeatSolver>(m_data);
  m_adjoint_solver = std::make_shared<HeatAdjointSolver>(m_adjoint_data);
}

void AdjointCoefficientSolver::calculate_coefficient_gradient() {
  auto dt = m_data->mesh->dx()[kSpaceDim];

  auto inv_h_squared = m_data->mesh->dx();
  std::for_each(inv_h_squared.begin(), inv_h_squared.end(),
                [](auto& x) { x = 1.0 / (x * x); });

#pragma omp parallel for
  for (size_t i = 0; i < m_data->mesh->shape()[0]; ++i) {
    for (size_t j = 0; j < m_data->mesh->shape()[1]; ++j) {
      std::remove_pointer_t<decltype(m_mean_coefficient_gradient)>::
          element_type::value_type value = 0;
      std::remove_pointer_t<decltype(m_mean_coefficient_gradient)>::
          element_type::value_type grad_regularization = 0;

      decltype(value) mult_x = 1;
      decltype(value) mult_y = 1;

      std::array<int, 2> i_shift{1, -1};
      std::array<int, 2> j_shift{1, -1};

      if (i == 0) {
        i_shift[1] = 0;
      } else if (i == m_data->mesh->shape()[0] - 1) {
        i_shift[0] = 0;
      } else {
        mult_x = 0.5;
      }

      if (j == 0) {
        j_shift[1] = 0;
      } else if (j == m_data->mesh->shape()[1] - 1) {
        j_shift[0] = 0;
      } else {
        mult_y = 0.5;
      }

      for (size_t time_idx = 0; time_idx < m_data->mesh->shape()[kSpaceDim];
           ++time_idx) {
        decltype(value) dt_dx = 1;
        decltype(value) dt_dy = 1;

        decltype(value) dadj_dx = 1;
        decltype(value) dadj_dy = 1;

        dt_dx = (m_data->temperature->value({i + i_shift[0], j, time_idx}) -
                 m_data->temperature->value({i + i_shift[1], j, time_idx})) *
                mult_x;
        dadj_dx =
            (m_adjoint_data->temperature->value({i + i_shift[0], j, time_idx}) -
             m_adjoint_data->temperature->value(
                 {i + i_shift[1], j, time_idx})) *
            mult_x;

        dt_dy = (m_data->temperature->value({i, j + j_shift[0], time_idx}) -
                 m_data->temperature->value({i, j + j_shift[1], time_idx})) *
                mult_y;

        dadj_dy =
            (m_adjoint_data->temperature->value({i, j + j_shift[0], time_idx}) -
             m_adjoint_data->temperature->value(
                 {i, j + j_shift[1], time_idx})) *
            mult_y;

        auto diag_dt_pdx_pdy =
            (m_data->temperature->value(
                 {i + i_shift[0], j + j_shift[0], time_idx}) -
             m_data->temperature->value(
                 {i + i_shift[1], j + j_shift[1], time_idx})) *
            mult_x * mult_y;

        auto diag_dadj_pdx_pdy =
            (m_adjoint_data->temperature->value(
                 {i + i_shift[0], j + j_shift[0], time_idx}) -
             m_adjoint_data->temperature->value(
                 {i + i_shift[1], j + j_shift[1], time_idx})) *
            mult_x * mult_y;

        auto diag_dt_pdx_mdy =
            (m_data->temperature->value(
                 {i + i_shift[0], j + j_shift[1], time_idx}) -
             m_data->temperature->value(
                 {i + i_shift[1], j + j_shift[0], time_idx})) *
            mult_x * mult_y;

        auto diag_dadj_pdx_mdy =
            (m_adjoint_data->temperature->value(
                 {i + i_shift[0], j + j_shift[1], time_idx}) -
             m_adjoint_data->temperature->value(
                 {i + i_shift[1], j + j_shift[0], time_idx})) *
            mult_x * mult_y;

        dt_dx = 0.5 * dt_dx + 0.25 * (diag_dt_pdx_mdy + diag_dt_pdx_pdy);
        dt_dy = 0.5 * dt_dy + 0.25 * (diag_dt_pdx_pdy - diag_dt_pdx_mdy);

        dadj_dx =
            0.5 * dadj_dx + 0.25 * (diag_dadj_pdx_mdy + diag_dadj_pdx_pdy);
        dadj_dy =
            0.5 * dadj_dy + 0.25 * (diag_dadj_pdx_pdy - diag_dadj_pdx_mdy);

        value += (dt_dx * dadj_dx) * inv_h_squared[0] +
                 (dt_dy * dadj_dy) * inv_h_squared[1];
      }
      value *= dt;

      if (i == 0) {
        grad_regularization += (m_coefficient->value({i + 2, j, 0}) -
                                2 * m_coefficient->value({i + 1, j, 0}) +
                                m_coefficient->value({i, j, 0})) *
                               inv_h_squared[0];
      } else if (i == m_data->mesh->shape()[0] - 1) {
        grad_regularization += (m_coefficient->value({i, j, 0}) -
                                2 * m_coefficient->value({i - 1, j, 0}) +
                                m_coefficient->value({i - 2, j, 0})) *
                               inv_h_squared[0];
      } else {
        grad_regularization += (m_coefficient->value({i + 1, j, 0}) -
                                2 * m_coefficient->value({i, j, 0}) +
                                m_coefficient->value({i - 1, j, 0})) *
                               inv_h_squared[0];
      }

      if (j == 0) {
        grad_regularization += (m_coefficient->value({i, j + 2, 0}) -
                                2 * m_coefficient->value({i, j + 1, 0}) +
                                m_coefficient->value({i, j, 0})) *
                               inv_h_squared[1];
      } else if (j == m_data->mesh->shape()[1] - 1) {
        grad_regularization += (m_coefficient->value({i, j, 0}) -
                                2 * m_coefficient->value({i, j - 1, 0}) +
                                m_coefficient->value({i, j - 2, 0})) *
                               inv_h_squared[1];
      } else {
        grad_regularization += (m_coefficient->value({i, j + 1, 0}) -
                                2 * m_coefficient->value({i, j, 0}) +
                                m_coefficient->value({i, j - 1, 0})) *
                               inv_h_squared[1];
      }

      auto value_updated =
          value - (m_grad_reg_multiplier * grad_regularization) +
          (m_norm_reg_multiplier * m_coefficient->value({i, j, 0}));

      m_mean_coefficient_gradient->value({i, j, 0}) =
          m_momentum_decay_multiplier *
              m_mean_coefficient_gradient->value({i, j, 0}) +
          (1 - m_momentum_decay_multiplier) * value_updated;

      m_mean_squared_coefficient_gradient->value({i, j, 0}) =
          m_squared_momentum_decay_multiplier *
              m_mean_squared_coefficient_gradient->value({i, j, 0}) +
          (1 - m_squared_momentum_decay_multiplier) * value_updated *
              value_updated;
    }
  }
}

void AdjointCoefficientSolver::solve_step(double beta_1, double beta_2) {
  m_solver->solve();
  m_adjoint_solver->solve();
  calculate_coefficient_gradient();

#pragma omp parallel for
  for (size_t i = 0; i < m_data->mesh->shape()[0]; ++i) {
    for (size_t j = 0; j < m_data->mesh->shape()[1]; ++j) {
      auto momentum = m_mean_coefficient_gradient->value({i, j, 0}) * beta_1;
      auto second_momentum =
          m_mean_squared_coefficient_gradient->value({i, j, 0}) * beta_2;
      m_coefficient->value(Index_t({i, j, 0})) =
          std::max(0.0, m_coefficient->value(Index_t({i, j, 0})) -
                            (m_learning_rate * momentum /
                             (std::sqrt(second_momentum) + 1e-8)));
    }
  }
}

HeatProblemData::Function_t::value_type AdjointCoefficientSolver::getError()
    const {
  HeatProblemData::Function_t::value_type error =
      l2_diff_norm(*m_target_temperature, *m_data->temperature);
  return error / m_target_temperature_norm;
}

std::shared_ptr<HeatProblemData::Function_t> AdjointCoefficientSolver::solve(
    size_t iterations, size_t check_stop_conditions_per_steps,
    double ans_rtol_stop_condition) {
  for (size_t i = 0; i < iterations; ++i) {
    if (i > 0 && check_stop_conditions_per_steps > 0 &&
        i % check_stop_conditions_per_steps == 0) {
      auto relative_error = getError();

      printf("Step: %zu | rel error: %lf \n", i, relative_error);
      if (relative_error <= ans_rtol_stop_condition)
        return std::move(m_coefficient);
    }

    auto beta_1 = 1 / (1 - std::pow(m_momentum_decay_multiplier, i + 1));
    auto beta_2 =
        1 / (1 - std::pow(m_squared_momentum_decay_multiplier, i + 1));

    solve_step(beta_1, beta_2);
  }

  return std::move(m_coefficient);
}
}  // namespace heatsolver

#pragma once

#include <cstddef>

#include "solver.h"
#include "view_impl.h"
namespace heatsolver {

/**
 * @brief Holder struct for all necessary data
 */
class HeatAdjointSolver : public HeatSolver {
 public:
  explicit HeatAdjointSolver(std::shared_ptr<HeatProblemData> data)
      : HeatSolver(std::move(data)) {}

 protected:
  void time_step(size_t time_index) override;
  void assemble_and_solve_system(
      size_t dim, const HeatProblemData::Function_t& prev_solution,
      HeatProblemData::Function_t& next_solution, Index_t index,
      double time_step) override;
};

/**
 * @brief Finds heat conduction coefficient field using gradient descent method
 * (Adam) with analytical Error functional's gradient estimation.
 */
class AdjointCoefficientSolver {
 public:
  explicit AdjointCoefficientSolver(
      std::shared_ptr<HeatProblemData::Mesh_t> mesh,
      std::shared_ptr<HeatProblemData::Function_t> target_temperature,
      std::shared_ptr<HeatProblemData::Function_t> heat_source,
      std::shared_ptr<HeatProblemData::Function_t> coefficient);

  /**
   * @brief Solve inverse 2D heat problem of finding heat conduction coefficient
   * field
   *
   * @param max_iterations Maximum number of iterations
   * @param check_stop_conditions_per_steps Defines frequency of checking stop
   * condition
   * @param ans_rtol_stop_condition Relative accuracy upon reaching which the
   * calculations will be stopped
   */
  std::shared_ptr<HeatProblemData::Function_t> solve(
      size_t max_iterations, size_t check_stop_conditions_per_steps = 0,
      double ans_rtol_stop_condition = 1e-6);

  // Made these fields public as they are essentially configs

  // Gradient descend step size
  HeatProblemData::Function_t::value_type m_learning_rate{1e-3};
  // "Smoothing" regularisation weight
  HeatProblemData::Function_t::value_type m_grad_reg_multiplier{1e-8};
  // "Norm" regularisation weight
  HeatProblemData::Function_t::value_type m_norm_reg_multiplier{1e-8};
  // Adam's decay weight for first momentum of gradient
  HeatProblemData::Function_t::value_type m_momentum_decay_multiplier{0.999};
  // Adam's decay weight for second momentum of gradient
  HeatProblemData::Function_t::value_type m_squared_momentum_decay_multiplier{
      0.9};

 private:
  void solve_step(double beta_1, double beta_2);
  void calculate_coefficient_gradient();
  void addGradRegularisation(
      std::shared_ptr<HeatProblemData::Function_t>& gradient);
  HeatProblemData::Function_t::value_type getError() const;

  HeatProblemData::Function_t::value_type m_target_temperature_norm{0};

  std::shared_ptr<HeatProblemData> m_data;
  std::shared_ptr<HeatProblemData> m_adjoint_data;
  std::shared_ptr<HeatProblemData::Function_t> m_target_temperature;
  std::shared_ptr<HeatProblemData::Function_t> m_coefficient;
  std::shared_ptr<HeatProblemData::Function_t> m_mean_coefficient_gradient;
  std::shared_ptr<HeatProblemData::Function_t>
      m_mean_squared_coefficient_gradient;

  std::shared_ptr<HeatSolver> m_solver;
  std::shared_ptr<HeatAdjointSolver> m_adjoint_solver;
};

}  // namespace heatsolver

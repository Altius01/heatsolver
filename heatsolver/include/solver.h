#pragma once

#include <cstddef>
#include <functional>
#include <memory>

// #include "problem.h"

#include "view_impl.h"
namespace heatsolver {

/**
 * @brief Holder struct for all necessary data
 */
struct HeatProblemData {
  using Mesh_t = SpaceTimeMesh<kDim>;
  using Function_t = NDArray<double, kDim>;
  std::shared_ptr<const Mesh_t> mesh;
  std::shared_ptr<Function_t> temperature;
  std::shared_ptr<const Function_t> heat_source;
  std::shared_ptr<const Function_t> coefficient;
  std::array<std::shared_ptr<const Function_t>, kSpaceDim> dirichlet_boundary;
};

/**
 * @brief Simple sequential realisation of Thomas algorith
 */
struct TriDiagMatrixSolver {
  TriDiagMatrixSolver() = default;
  explicit TriDiagMatrixSolver(size_t size)
      : m_upper(size),
        m_diag(size),
        m_lower(size),
        m_rhs(size),
        m_p(size),
        m_q(size) {};

  void solve(
      const std::function<void(size_t, double)> &solution_setter) noexcept;

  void print_system() const noexcept;

  std::vector<double> m_upper;
  std::vector<double> m_diag;
  std::vector<double> m_lower;
  std::vector<double> m_rhs;
  std::vector<double> m_p;
  std::vector<double> m_q;
};

/**
 * @brief Solver for forward (find temperature) heat equation in 2D
 */
class HeatSolver {
 public:
  explicit HeatSolver(std::shared_ptr<HeatProblemData> data)
      : m_data(std::move(data)) {
    m_temporary_solution =
        std::make_shared<ProblemSpaceFunctionArray>(m_data->mesh->shape());
  };

  /**
   * @brief Solves heat equation
   */
  virtual void solve();

 protected:
  virtual void time_step(size_t time_index);
  virtual void assemble_and_solve_system(
      size_t dim, const HeatProblemData::Function_t &prev_solution,
      HeatProblemData::Function_t &next_solution, Index_t index,
      double time_step);

  friend class TriDiagMatrixSolver;

  std::shared_ptr<HeatProblemData> m_data;

  std::shared_ptr<ProblemSpaceFunctionArray> m_temporary_solution;
  double m_cindex_time_step{0.5};
};

}  // namespace heatsolver

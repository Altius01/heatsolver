#pragma once

#include <cstddef>
#include <functional>
#include <optional>

// #include "problem.h"

#include "view_impl.h"
namespace heatsolver {

// inline static std::optional<size_t> get_boundary_idx(size_t dim, Index_t
// index,
//                                                      Index_t shape) {
//   for (size_t i = 0; i < index.size(); ++i) {
//     if (dim == i) continue;
//
//     if (index[i] == 0 || index[i] == shape[i] - 1) {
//       return i;
//     }
//   }
//   return std::nullopt;
// }

struct HeatProblemData {
  using Mesh_t = SpaceTimeMesh<kDim>;
  using Function_t = NDArray<double, kDim>;

  std::shared_ptr<const Mesh_t> mesh;
  std::shared_ptr<Function_t> temperature;
  std::shared_ptr<const Function_t> heat_source;
  std::shared_ptr<const Function_t> coefficient;
  std::array<std::shared_ptr<const Function_t>, kSpaceDim> dirichlet_boundary;
};

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

class HeatSolver {
 public:
  explicit HeatSolver(HeatProblemData data) : m_data(std::move(data)) {
    m_temporary_solution =
        std::make_shared<ProblemSpaceFunctionArray>(m_data.mesh->shape());
  };

  void solve();

 private:
  void time_step(size_t time_index);
  void assemble_and_solve_system(
      size_t dim, const HeatProblemData::Function_t &prev_solution,
      HeatProblemData::Function_t &next_solution, Index_t index,
      double time_step);

  friend class TriDiagMatrixSolver;

  HeatProblemData m_data;

  std::shared_ptr<ProblemSpaceFunctionArray> m_temporary_solution;
  double m_cindex_time_step{0.5};
};

}  // namespace heatsolver

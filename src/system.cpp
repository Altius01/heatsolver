#include <iostream>

#include "solver.h"

namespace heatsolver {

void TriDiagMatrixSolver::print_system() const noexcept {
  int n = m_diag.size();

  for (int i = 0; i < n; ++i) {
    if (i > 0 && i < n - 1) {
      for (int j = 0; j < i - 1; ++j) {
        printf("X ");
      }
      printf("%.2f %.2f %.2f ", m_lower[i], m_diag[i], m_upper[i]);
      for (int j = 0; j < n - (i + 2); ++j) {
        printf("X ");
      }
    }
    if (i == n - 1) {
      for (int j = 0; j < i - 1; ++j) {
        printf("X ");
      }
      printf("   %.2f %.2f", m_lower[i], m_diag[i]);
    }

    if (i == 0) {
      printf("%.2f %.2f    ", m_diag[i], m_upper[i]);
      for (int j = 0; j < n - (i + 2); ++j) {
        printf("X ");
      }
    }
    printf("\n");
  }
  printf("\n\n");
}

void TriDiagMatrixSolver::solve(
    const std::function<void(size_t, double)>& solution_setter) noexcept {
  int n = m_diag.size();

  m_p[0] = m_upper[0] / m_diag[0];
  m_q[0] = m_rhs[0] / m_diag[0];
  for (int i = 1; i < n; i++) {
    auto denom = m_diag[i] - (m_lower[i] * m_p[i - 1]);
    m_p[i] = (i < n - 1) ? m_upper[i] / denom : 0.0;
    m_q[i] = (m_rhs[i] - m_lower[i] * m_q[i - 1]) / denom;
  }
  auto prev_solution = m_q[n - 1];
  solution_setter(n - 1, prev_solution);
  for (int i = n - 2; i >= 0; i--) {
    prev_solution = m_q[i] - m_p[i] * prev_solution;
    solution_setter(i, prev_solution);
  }
  // print_system();
}
}  // namespace heatsolver

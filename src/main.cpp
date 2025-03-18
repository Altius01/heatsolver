#include <algorithm>
#include <cmath>
#include <iostream>

#include "inverse_solver.h"
#include "io.h"
#include "solver.h"
#include "view_impl.h"

constexpr size_t kDim = 2;

using namespace heatsolver;  // NOLINT

namespace {
double heat_source_foo(heatsolver::CIndex_t /*index*/) { return 0; }

double coefficient_foo(heatsolver::CIndex_t /* index */) { return 1; }

double g_boundary(size_t dim, heatsolver::CIndex_t /* index */) {
  if (dim == 0) return 0;
  return 0;
}

double initial_temperature(heatsolver::CIndex_t index) {
  return sin(M_PI * index[0]) * sin(M_PI * index[1]);
}

void print_solution(const std::vector<std::vector<double>>& u) {
  for (const auto& i : u) {
    for (double j : i) {
      printf("%.2f ", j);
    }
    std::cout << "\n";
  }
  std::cout << "\n\n";
}

void print_solution(const NDArray<double, 3>& u, size_t time_index) {
  const auto shape = u.shape();
  Index_t idx = {0, 0, time_index};
  for (size_t x_it = 0; x_it < shape[0]; ++x_it) {
    idx[0] = x_it;
    for (size_t y_it = 0; y_it < shape[1]; ++y_it) {
      idx[1] = y_it;
      printf("%.2f ", u.value(idx));
    }
    std::cout << "\n";
  }
  std::cout << "\n\n";
}

void set_true_solution(const ProblemMesh& mesh,
                       std::vector<std::vector<double>>& u, double time) {
  const auto& shape = mesh.shape();
  for (size_t x_it = 0; x_it < shape[0]; ++x_it) {
    for (size_t y_it = 0; y_it < shape[1]; ++y_it) {
      auto pos = mesh.index_to_position(Index_t({x_it, y_it, 0}));
      auto x = pos[0];
      auto y = pos[1];

      u[x_it][y_it] =
          sin(M_PI * x) * sin(M_PI * y) * exp(-2 * M_PI * M_PI * time);
    }
  }
}

double l2_error(const std::vector<std::vector<double>>& set_true_solution,
                const NDArray<double, 3>& u) {
  const auto& shape = u.shape();
  double error = 0;
  Index_t idx = {0, 0, shape[2] - 1};
  for (size_t x_it = 0; x_it < shape[0]; ++x_it) {
    idx[0] = x_it;
    for (size_t y_it = 0; y_it < shape[1]; ++y_it) {
      idx[1] = y_it;
      error += (u.value(idx) - set_true_solution[x_it][y_it]) *
               (u.value(idx) - set_true_solution[x_it][y_it]);
    }
  }
  return sqrt(error) / (shape[0] * shape[1]);
}

}  // namespace

using namespace heatsolver;  // NOLINT

int main(int /*argc*/, char* /*argv*/[]) {
  auto mesh = std::make_shared<ProblemMesh>();  // Mesh2D();
  size_t shape_x = 11;
  size_t shape_y = 5;
  size_t shape_time = 100;

  double size_x = (static_cast<double>(shape_x) / (shape_x - 1)) * 1.0;
  double size_y = (static_cast<double>(shape_y) / (shape_y - 1)) * 1.0;
  double time = (static_cast<double>(shape_time) / (shape_time - 1)) * 0.05;

  mesh->shape({shape_x, shape_y, shape_time});
  mesh->size({size_x, size_y, time});

  const auto dx = mesh->dx();
  const auto shape = mesh->shape();

  mesh->origin({-dx[0] * 0.5, -dx[1] * 0.5, -dx[2] * 0.5});
  const auto origin = mesh->origin();
  std::cout << "Shape: x:" << shape_x << ", y:" << shape_y
            << ", t:" << shape_time << "\n";
  std::cout << "Size: x:" << size_x << ", y:" << size_y << ", t:" << time
            << "\n";
  std::cout << "dx: " << dx[0] << ", dy: " << dx[0] << ", dt: " << dx[2]
            << "\n";
  std::cout << "Origin: x:" << origin[0] << ", y:" << origin[1]
            << ", t: " << origin[2] << "\n";

  auto temperature = std::make_shared<ProblemSpaceTimeFunctionArray>(shape);
  auto heat_source = std::make_shared<ProblemSpaceTimeFunctionArray>(shape);
  auto coefficient = std::make_shared<ProblemSpaceFunctionArray>(shape);

  Index_t y_boundary_shape = {shape_x, 2, shape_time};
  Index_t x_boundary_shape = {2, shape_y, shape_time};
  auto y_boundary =
      std::make_shared<ProblemSpaceTimeFunctionArray>(y_boundary_shape);
  auto x_boundary =
      std::make_shared<ProblemSpaceTimeFunctionArray>(x_boundary_shape);

  Index_t idx = {0, 0, 0};
  CIndex_t pos{};
  for (size_t i = 0; i < shape_x; ++i) {
    idx[0] = i;
    for (size_t j = 0; j < shape_y; ++j) {
      idx[1] = j;
      idx[2] = 0;

      pos = mesh->index_to_position(idx);

      temperature->value(idx) = initial_temperature(pos);
      coefficient->value(idx) = coefficient_foo(pos);
      for (size_t k = 0; k < shape_time; ++k) {
        idx[2] = k;
        pos = mesh->index_to_position(idx);
        heat_source->value(idx) = heat_source_foo(pos);
      }
    }
  }

  Index_t bond_idx{};
  for (size_t i = 0; i < shape_x; ++i) {
    idx[0] = i;
    bond_idx[0] = i;
    for (size_t k = 0; k < shape_time; ++k) {
      idx[2] = k;
      idx[1] = 0;
      bond_idx[1] = 0;
      pos = mesh->index_to_position(idx);
      y_boundary->value(bond_idx) = g_boundary(1, pos);
      idx[1] = shape_y - 1;
      bond_idx[1] = 1;
      pos = mesh->index_to_position(idx);
      y_boundary->value(bond_idx) = g_boundary(1, pos);
    }
  }

  for (size_t i = 0; i < shape_y; ++i) {
    idx[1] = i;
    bond_idx[1] = i;
    for (size_t k = 0; k < shape_time; ++k) {
      idx[2] = k;

      idx[0] = 0;
      bond_idx[0] = 0;
      pos = mesh->index_to_position(idx);
      x_boundary->value(bond_idx) = g_boundary(0, pos);
      idx[0] = shape_x - 1;
      bond_idx[0] = 1;
      pos = mesh->index_to_position(idx);
      x_boundary->value(bond_idx) = g_boundary(0, pos);
    }
  }

  HeatProblemData problem{
      mesh, temperature, heat_source, coefficient, {x_boundary, y_boundary}};

  HeatSolver solver(problem);

  std::cout << "Initial Solution:\n";
  print_solution(*problem.temperature, 0);

  solver.solve();

  std::cout << "Solution:\n";
  print_solution(*problem.temperature, shape_time - 1);

  std::vector<std::vector<double>> u_true(shape_x,
                                          std::vector<double>(shape_y));
  set_true_solution(*mesh, u_true, time);

  std::cout << "True solution:\n";
  print_solution(u_true);

  std::cout << "Error: " << l2_error(u_true, *(problem.temperature)) << "\n";

  writeSpaceTimeFunction(*(problem.temperature), "temperature", "solution.h5");

  InverseSolverData inverse_data{mesh, problem.temperature, problem.heat_source,
                                 coefficient};

  InverseSolver inverse_solver(inverse_data);
  auto coef_sol = inverse_solver.solve(1e-1);

  // std::cout << "Coefficient solution:\n";
  for (size_t i = 0; i < shape_x; i++) {
    for (size_t j = 0; j < shape_y; j++) {
      // std::cout << coef_sol((i * shape_y) + j) << " ";
      coefficient->value(Index_t({i, j, 0})) = coef_sol((i * shape_y) + j);
    }
    // std::cout << "\n";
  }

  // std::cout << "Initial Solution:\n";
  // print_solution(*problem.temperature, 0);

  solver.solve();

  // std::cout << "Solution:\n";
  // print_solution(*problem.temperature, shape_time - 1);

  std::cout << "L2 Error: " << l2_error(u_true, *(problem.temperature)) << "\n";
  return 0;
}

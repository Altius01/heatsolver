#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#include "adjoint_solver.h"
#include "io.h"
#include "solver.h"
#include "view.h"
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

}  // namespace

using namespace heatsolver;  // NOLINT

int main(int /*argc*/, char* /*argv*/[]) {
  auto mesh = std::make_shared<ProblemMesh>();  // Mesh2D();
  // Задаем сетку задачи
  size_t shape_x = 11;
  size_t shape_y = 11;
  size_t shape_time = 10;

  // Задаем физический размер сетки (учитываем, что значения храняться в центрах
  // вокселей в пространстве-времени)
  double size_x_m = 1.0;
  double size_y_m = 1.0;
  double time_sec = 0.05;
  double size_x = (static_cast<double>(shape_x) / (shape_x - 1)) * size_x_m;
  double size_y = (static_cast<double>(shape_y) / (shape_y - 1)) * size_y_m;
  double time = (static_cast<double>(shape_time) / (shape_time - 1)) * time_sec;

  mesh->shape({shape_x, shape_y, shape_time});
  mesh->size({size_x, size_y, time});

  const auto dx = mesh->dx();
  const auto shape = mesh->shape();

  // Сдвигаем поцизию точки с индексом [0, 0, 0] в 0
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

  // Устанавливаем начальные условия
#pragma omp parallel for
  for (size_t i = 0; i < shape_x; ++i) {
    CIndex_t pos{};
    Index_t idx = {0, 0, 0};
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

  // Граница параллельная оси x храниться в NDArray с shape = (shape_x, 2,
  // shape_time), где второй по счету индекс отвечает за то - левая граница или
  // правая
#pragma omp parallel for
  for (size_t i = 0; i < shape_x; ++i) {
    Index_t bond_idx{};
    CIndex_t pos{};
    Index_t idx = {0, 0, 0};
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

#pragma omp parallel for
  for (size_t i = 0; i < shape_y; ++i) {
    Index_t bond_idx{};
    CIndex_t pos{};
    Index_t idx = {0, 0, 0};
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

  auto problem = std::make_shared<HeatProblemData>();
  problem->mesh = mesh;
  problem->temperature = temperature;
  problem->heat_source = heat_source;
  problem->coefficient = coefficient;
  problem->dirichlet_boundary =
      std::array<std::shared_ptr<const HeatProblemData::Function_t>, kSpaceDim>(
          {x_boundary, y_boundary});

  HeatSolver solver(problem);

  solver.solve();

  // Точное решение в случае равномерного коэффициента теплопроводности равного
  // 1
  auto u_true_foo = std::make_shared<ProblemSpaceTimeFunction>(
      [&](const ProblemSpaceTimeFunction::cindex_type& index) {
        return mesh->index_to_position(index);
      },
      [&](const ProblemSpaceTimeFunction::cindex_type& index) {
        auto x = index[0];
        auto y = index[1];
        auto t = index[2];
        return sin(M_PI * x) * sin(M_PI * y) * exp(-2 * M_PI * M_PI * t);
      },
      shape);

  // Записываем точное решение на диск
  application::io::writeSpaceTimeFunction(*u_true_foo, "true_temperature",
                                          "true_solution.h5");

  auto mean_l2_error_forward_problem =
      l2_diff_norm(*u_true_foo, *(problem->temperature));
  std::cout << "Forward problem solution Error: "
            << mean_l2_error_forward_problem << "\n";

  // Записываем решение на диск
  application::io::writeSpaceTimeFunction(*(problem->temperature),
                                          "temperature", "solution.h5");

  // InverseSolverData inverse_data{mesh, problem->temperature,
  //                                problem->heat_source, coefficient};
  //
  // InverseSolver inverse_solver(inverse_data);
  // inverse_solver.solve(0);

  auto copy_forward_solution = std::make_shared<ProblemSpaceTimeFunctionArray>(
      problem->temperature->shape());
  std::copy(problem->temperature->begin(), problem->temperature->end(),
            copy_forward_solution->begin());

  // Задаем начальное приближение для коэффициента теплопроводности
  auto adjoint_start_coefficient =
      std::make_shared<ProblemSpaceFunctionArray>(shape);
  adjoint_start_coefficient->fill(0);

  AdjointCoefficientSolver adjoint_coefficient_solver(
      mesh, copy_forward_solution, heat_source, adjoint_start_coefficient);

  // Конфигурируем солвер для обратной задачи
  adjoint_coefficient_solver.m_momentum_decay_multiplier = 0.9;
  adjoint_coefficient_solver.m_squared_momentum_decay_multiplier = 0.8;
  adjoint_coefficient_solver.m_learning_rate = 1e-1;
  adjoint_coefficient_solver.m_grad_reg_multiplier = 1e-8;
  adjoint_coefficient_solver.m_norm_reg_multiplier = 1e-5;

  auto adjoint_coefficient = adjoint_coefficient_solver.solve(10000, 100, 1e-6);

  problem->coefficient = adjoint_coefficient;
  solver.solve();

  // Записываем решение обратной задачи на диск
  application::io::writeSpaceTimeFunction(
      *adjoint_coefficient, "adjoint_coefficient", "adjoint_coefficient.h5");

  // Записываем реальный коэффициент тпелопроводности
  application::io::writeSpaceTimeFunction(*coefficient, "coefficient",
                                          "coefficient.h5");

  // Записываем температуру полученную решением прямой задачи с учетом
  // найденного коэффициента теплопроводности
  application::io::writeSpaceTimeFunction(
      *problem->temperature, "adjoint_solution", "adjoint_solution.h5");

  auto true_sol_norm = l2_norm(*copy_forward_solution);

  std::cout << "L2 Error: "
            << l2_diff_norm(*copy_forward_solution, *(problem->temperature)) /
                   true_sol_norm
            << "\n";
  return 0;
}

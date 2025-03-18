#pragma once

#include <Eigen/Sparse>
#include <memory>

#include "Eigen/src/Core/Matrix.h"
#include "view_impl.h"

namespace heatsolver {

struct InverseSolverData {
  using Mesh_t = SpaceTimeMesh<kDim>;
  using Function_t = NDArray<double, kDim>;
  std::shared_ptr<const Mesh_t> mesh;
  std::shared_ptr<const Function_t> temperature;
  std::shared_ptr<const Function_t> heat_source;
  std::shared_ptr<Function_t> coefficient;
};

class InverseSolver {
 public:
  using SpMat = Eigen::SparseMatrix<ProblemSpaceTimeFunctionArray::value_type>;
  using SpVec = Eigen::SparseVector<ProblemSpaceTimeFunctionArray::value_type>;
  using Vec = Eigen::VectorXd;
  using Triplet = Eigen::Triplet<ProblemSpaceTimeFunctionArray::value_type>;

  explicit InverseSolver(InverseSolverData data) : m_data(std::move(data)) {};

  Eigen::VectorXd solve(double lambda);

 private:
  void assemble_system();
  void assemble_system_new(double lambda);
  void assemble_system_k(size_t time_idx);
  void assemble_rhs_k(size_t time_idx);
  SpMat assemble_regularisation();

  InverseSolverData::Function_t::value_type rhs(Index_t index);

  Index_t unflat_spatial_index(size_t index) const noexcept;
  size_t flat_spatial_index(const Index_t& index) const noexcept;
  size_t add_time_to_flat_index(size_t flat_index, size_t time_idx);

  InverseSolverData m_data;

  Vec m_rhs;
  SpMat m_system;
};
}  // namespace heatsolver

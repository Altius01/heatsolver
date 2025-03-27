#pragma once

#include <memory>

#include "view_impl.h"

namespace heatsolver {

/**
 * @brief Holder struct for all necessary data
 */
struct InverseSolverData {
  using Mesh_t = SpaceTimeMesh<kDim>;
  using Function_t = NDArray<double, kDim>;
  std::shared_ptr<const Mesh_t> mesh;
  std::shared_ptr<const Function_t> temperature;
  std::shared_ptr<const Function_t> heat_source;
  std::shared_ptr<Function_t> coefficient;
};

class InverseSolverPrivateData;

/**
 * @brief Finds heat conduction coefficient field by solving sparce linear
 * equations system. The overdetermined system of equations is constructed from
 * the temperature field measurements at each time step
 */
class InverseSolver {
 public:
  using value_type = ProblemSpaceTimeFunctionArray::value_type;

  explicit InverseSolver(InverseSolverData data);

  /**
   * @brief Solve inverse 2D heat problem of finding heat conduction coefficient
   * field
   *
   * @param lambda "Smoothing" regularisation weight
   */
  void solve(double lambda);

 private:
  void assemble_system();
  void assemble_system_new(double lambda);
  void assemble_system_k(size_t time_idx);
  void assemble_rhs_k(size_t time_idx);
  void assemble_regularisation(void* reg);

  InverseSolverData::Function_t::value_type rhs(Index_t index);

  Index_t unflat_spatial_index(size_t index) const noexcept;
  size_t flat_spatial_index(const Index_t& index) const noexcept;
  size_t add_time_to_flat_index(size_t flat_index, size_t time_idx);

  InverseSolverData m_data;

  std::shared_ptr<InverseSolverPrivateData> m_private_data;
};
}  // namespace heatsolver

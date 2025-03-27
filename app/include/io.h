#pragma once

#include <highfive/highfive.hpp>

#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "view.h"
#include "view_impl.h"

namespace application::io {

template <typename T, size_t Dim>
void writeSpaceTimeFunction(const heatsolver::NDArray<T, Dim>& array,
                            const std::string& dataset_name,
                            const std::string& path) {
  HighFive::File file(path, HighFive::File::Truncate);

  const auto& shape = array.shape();

  std::vector<T> data(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));

  heatsolver::NDIndex_t<Dim> idx{};
  std::copy(array.begin(), array.end(), data.begin());

  HighFive::DataSet dataset =
      file.createDataSet<T>(dataset_name, HighFive::DataSpace(shape));
  dataset.write_raw(data.data());
}

inline std::shared_ptr<heatsolver::ProblemSpaceTimeFunctionArray>
readSpaceTimeFunction(const std::string& dataset_name, const std::string& path);
}  // namespace application::io

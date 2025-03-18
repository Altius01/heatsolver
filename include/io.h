#pragma once

#include <highfive/highfive.hpp>
#include <type_traits>

#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "view.h"
#include "view_impl.h"

namespace heatsolver {

template <typename T, size_t Dim>
void writeSpaceTimeFunction(const NDArray<T, Dim>& array,
                            const std::string& dataset_name,
                            const std::string& path) {
  HighFive::File file(path, HighFive::File::Truncate);

  const auto& shape = array.shape();

  // copy data to std::vector

  boost::multi_array<T, Dim> data(boost::extents[shape[0]][shape[1]][shape[2]]);

  NDIndex_t<Dim> idx{};
  for (size_t i = 0; i < shape[0]; ++i) {
    idx[0] = i;
    for (size_t j = 0; j < shape[1]; ++j) {
      idx[1] = j;
      for (size_t k = 0; k < shape[2]; ++k) {
        idx[2] = k;
        data[i][j][k] = array.value(idx);
      }
    }
  }

  HighFive::DataSet dataset = file.createDataSet(dataset_name, data);
}

inline std::shared_ptr<ProblemSpaceTimeFunctionArray> readSpaceTimeFunction(
    const std::string& dataset_name, const std::string& path) {
  HighFive::File file(path, HighFive::File::ReadOnly);

  HighFive::DataSet dataset = file.getDataSet(dataset_name);

  boost::multi_array<ProblemSpaceTimeFunctionArray::value_type,
                     ProblemSpaceTimeFunctionArray::kDim>
      data;

  dataset.read(data);

  auto dataset_shape = dataset.getSpace().getDimensions();
  NDIndex_t<ProblemSpaceTimeFunctionArray::kDim> shape{
      dataset_shape[0], dataset_shape[1], dataset_shape[2]};

  auto result = std::make_shared<ProblemSpaceTimeFunctionArray>(shape);
  NDIndex_t<ProblemSpaceTimeFunctionArray::kDim> idx{};
  for (size_t i = 0; i < shape[0]; ++i) {
    idx[0] = i;
    for (size_t j = 0; j < shape[1]; ++j) {
      idx[1] = j;
      for (size_t k = 0; k < shape[2]; ++k) {
        idx[2] = k;
        result->value(idx) = data[i][j][k];
      }
    }
  }
  return result;
}

}  // namespace heatsolver

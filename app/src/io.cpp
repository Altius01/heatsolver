#include "io.h"

#include <functional>
#include <numeric>

#include "view.h"

namespace application::io {
inline std::shared_ptr<heatsolver::ProblemSpaceTimeFunctionArray>
readSpaceTimeFunction(const std::string& dataset_name,
                      const std::string& path) {
  HighFive::File file(path, HighFive::File::ReadOnly);

  HighFive::DataSet dataset = file.getDataSet(dataset_name);
  auto dataset_shape = dataset.getSpace().getDimensions();

  std::vector<heatsolver::ProblemSpaceTimeFunctionArray::value_type> data(
      std::accumulate(dataset_shape.begin(), dataset_shape.end(), 1,
                      std::multiplies<>()));

  dataset.read_raw<heatsolver::ProblemSpaceTimeFunctionArray::value_type>(
      data.data());

  heatsolver::NDIndex_t<heatsolver::ProblemSpaceTimeFunctionArray::kDim> shape{
      dataset_shape[0], dataset_shape[1], dataset_shape[2]};

  auto result =
      std::make_shared<heatsolver::ProblemSpaceTimeFunctionArray>(shape);
  heatsolver::NDIndex_t<heatsolver::ProblemSpaceTimeFunctionArray::kDim> idx{};
  std::copy(data.begin(), data.end(), result->begin());
  return result;
}

}  // namespace application::io

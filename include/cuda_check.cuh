#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cacheline_atomicity {

inline void check_cuda(cudaError_t error,
                       const char* expression,
                       const char* file,
                       int line) {
  if (error == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << "CUDA call failed: " << expression << " at " << file << ":" << line
          << " (" << cudaGetErrorName(error) << ": "
          << cudaGetErrorString(error) << ")";
  throw std::runtime_error(message.str());
}

}  // namespace cacheline_atomicity

#define CUDA_CHECK(expression) \
  ::cacheline_atomicity::check_cuda((expression), #expression, __FILE__, __LINE__)

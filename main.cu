#include "cacheline_atomicity_test.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using cacheline_atomicity::ExperimentConfig;
using cacheline_atomicity::ExperimentResult;

std::optional<std::uint64_t> parse_uint64(std::string_view text) {
  try {
    std::size_t consumed = 0;
    const auto value = std::stoull(std::string(text), &consumed, 10);
    if (consumed != text.size()) {
      return std::nullopt;
    }
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

std::pair<std::string_view, std::string_view> split_argument(std::string_view argument) {
  const std::size_t separator = argument.find('=');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 >= argument.size()) {
    throw std::invalid_argument(
        "arguments must use key=value syntax, for example cache_line_count=1024");
  }
  return {argument.substr(0, separator), argument.substr(separator + 1)};
}

void print_usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [cache_line_count=<count>] [timeout_ms=<ms>]\n"
      << '\n'
      << "Arguments:\n"
      << "  cache_line_count   Number of 128-byte cache lines to allocate\n"
      << "                     Default: " << ExperimentConfig{}.cache_line_count << '\n'
      << "  timeout_ms         How long the persistent kernel should run\n"
      << "                     Default: " << ExperimentConfig{}.timeout_ms << '\n'
      << '\n'
      << "Example:\n"
      << "  " << argv0 << " cache_line_count=1024 timeout_ms=5000\n";
}

ExperimentConfig parse_arguments(int argc, char** argv) {
  ExperimentConfig config;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }

    const auto [key, value_text] = split_argument(argument);
    const auto value = parse_uint64(value_text);
    if (!value.has_value()) {
      std::ostringstream message;
      message << "invalid integer value for " << key;
      throw std::invalid_argument(message.str());
    }

    if (key == "cache_line_count") {
      if (*value == 0 || *value > static_cast<std::uint64_t>(UINT32_MAX)) {
        throw std::invalid_argument(
            "cache_line_count must be in the range [1, 4294967295]");
      }
      config.cache_line_count = static_cast<std::uint32_t>(*value);
      continue;
    }

    if (key == "timeout_ms") {
      if (*value == 0) {
        throw std::invalid_argument("timeout_ms must be a positive integer");
      }
      config.timeout_ms = *value;
      continue;
    }

    std::ostringstream message;
    message << "unknown argument key: " << key;
    throw std::invalid_argument(message.str());
  }

  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ExperimentConfig config = parse_arguments(argc, argv);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
      throw std::runtime_error("no CUDA devices are available");
    }
    constexpr int device_ordinal = 0;
    CUDA_CHECK(cudaSetDevice(device_ordinal));
    config.print();

    std::cout << "Test with tile_size=32" << std::endl;
    const ExperimentResult result32 = cacheline_atomicity::run_experiment<32>(config);
    result32.print();
    std::cout << "Test with tile_size=16" << std::endl;
    const ExperimentResult result16 = cacheline_atomicity::run_experiment<16>(config);
    result16.print();
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}

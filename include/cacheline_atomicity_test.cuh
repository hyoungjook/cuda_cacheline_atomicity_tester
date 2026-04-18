#pragma once

#include "cuda_check.cuh"
#include "experiment_config.cuh"
#include "cacheline_access.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <stdexcept>

namespace cacheline_atomicity {

inline constexpr int kCacheLineBytes = 128;
inline constexpr int kCacheLineWords = kCacheLineBytes / sizeof(std::uint32_t);
inline constexpr int kTileSize = 32;
inline constexpr int kThreadsPerBlock = 128;

struct DeviceCounters {
  std::uint64_t total_loops = 0;
  std::uint64_t clean_weak_reads = 0;
  std::uint64_t clean_atomic_reads = 0;
  __device__ void atomic_add(const DeviceCounters& other) {
    #define do_atomic_add(member) \
    cuda::atomic_ref member##_ref(member); \
    member##_ref.fetch_add(other.member)
    do_atomic_add(total_loops);
    do_atomic_add(clean_weak_reads);
    do_atomic_add(clean_atomic_reads);
    #undef do_atomic_add
  }
};

__device__ std::uint32_t* line_at(std::uint32_t* cache_lines, std::uint32_t line_index) {
  return cache_lines + (static_cast<std::size_t>(line_index) * kCacheLineWords);
}
template <typename elem_type, typename tile_type>
__device__ bool check_clean_line(elem_type per_lane_elem, const tile_type& tile) {
  auto first_lane_elem = tile.shfl(per_lane_elem, 0) & ~latch_mask<elem_type>;
  auto mismatch = tile.ballot(per_lane_elem != first_lane_elem) & ~static_cast<elem_type>(1u);
  return mismatch == 0 ;
}

__global__ __launch_bounds__(kThreadsPerBlock) void tester_kernel(
    std::uint32_t* cache_lines,
    std::uint32_t cache_line_count,
    volatile int* stop_flag,
    DeviceCounters* counters) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<kTileSize>(block);
  const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState rand_state;
  curand_init(1234ull, global_thread_id, 0, &rand_state);
  DeviceCounters local_counters;
  tile.sync();

  while (true) {
    int stop_requested = 0;
    if (tile.thread_rank() == 0) {
      stop_requested = *stop_flag;
    }
    stop_requested = tile.shfl(stop_requested, 0);
    if (stop_requested != 0) {
      break;
    }

    const std::uint32_t rand_value = curand(&rand_state);
    const auto write_line_index = tile.shfl(rand_value, 0) % cache_line_count;
    const std::uint32_t write_elem = tile.shfl(rand_value, 1);
    const auto read_weak_line_index = tile.shfl(rand_value, 2) % cache_line_count;
    const auto read_atomic_line_index = tile.shfl(rand_value, 3) % cache_line_count;

    // write
    const auto write_line = line_at(cache_lines, write_line_index);
    cacheline_acquire(write_line, tile);
    cacheline_store_release(write_line, write_elem, tile);

    // read_weak
    const auto read_weak_elem = cacheline_load<false>(line_at(cache_lines, read_weak_line_index), tile);
    if (check_clean_line(read_weak_elem, tile)) {
      local_counters.clean_weak_reads++;
    }

    // read_atomic
    const auto read_atomic_elem = cacheline_load<true>(line_at(cache_lines, read_atomic_line_index), tile);
    if (check_clean_line(read_atomic_elem, tile)) {
      local_counters.clean_atomic_reads++;
    }

    local_counters.total_loops++;
  }

  if (tile.thread_rank() == 0) {
    counters->atomic_add(local_counters);
  }
}

inline ExperimentResult run_experiment(const ExperimentConfig& config) {
  if (config.cache_line_count == 0) {
    throw std::invalid_argument("cache line count must be positive");
  }
  if (config.timeout_ms == 0) {
    throw std::invalid_argument("timeout must be positive");
  }

  const std::size_t allocation_bytes =
      static_cast<std::size_t>(config.cache_line_count) * kCacheLineBytes;

  std::uint32_t* cache_lines = nullptr;
  DeviceCounters* counters = nullptr;
  int* stop_flag = nullptr;
  cudaStream_t kernel_stream = nullptr;
  ExperimentResult result{};

  try {
    CUDA_CHECK(cudaMalloc(&cache_lines, allocation_bytes));
    CUDA_CHECK(cudaMallocHost(&counters, sizeof(DeviceCounters)));
    CUDA_CHECK(cudaMallocHost(&stop_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(cache_lines, 0, allocation_bytes));
    *counters = DeviceCounters{};
    *reinterpret_cast<volatile int*>(stop_flag) = 0;
    CUDA_CHECK(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking));
    if (reinterpret_cast<uintptr_t>(cache_lines) % kCacheLineBytes != 0) {
      throw std::runtime_error("cudaMalloc buffer is not cacheline aligned");
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, tester_kernel, kThreadsPerBlock, 0));
    int num_blocks = blocks_per_sm * properties.multiProcessorCount;
    if (num_blocks <= 0) {
      throw std::runtime_error("failed to compute a valid launch configuration");
    }

    const auto start_time = std::chrono::steady_clock::now();
    tester_kernel<<<num_blocks, kThreadsPerBlock, 0, kernel_stream>>>(
      cache_lines, config.cache_line_count, stop_flag, counters);

    CUDA_CHECK(cudaGetLastError());
    std::this_thread::sleep_for(std::chrono::milliseconds(config.timeout_ms));
    *reinterpret_cast<volatile int*>(stop_flag) = 1;
    CUDA_CHECK(cudaStreamSynchronize(kernel_stream));
    const auto stop_time = std::chrono::steady_clock::now();

    result.total_writes = counters->total_loops;
    result.total_weak_reads = counters->total_loops;
    result.clean_weak_reads = counters->clean_weak_reads;
    result.torn_weak_reads = counters->total_loops - counters->clean_weak_reads;
    result.total_atomic_reads = counters->total_loops;
    result.clean_atomic_reads = counters->clean_atomic_reads;
    result.torn_atomic_reads = counters->total_loops - counters->clean_atomic_reads;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(stop_time - start_time).count();
  } catch (...) {
    if (kernel_stream != nullptr) {
      cudaStreamDestroy(kernel_stream);
    }
    if (stop_flag != nullptr) {
      cudaFreeHost(stop_flag);
    }
    if (counters != nullptr) {
      cudaFreeHost(counters);
    }
    if (cache_lines != nullptr) {
      cudaFree(cache_lines);
    }
    throw;
  }

  CUDA_CHECK(cudaStreamDestroy(kernel_stream));
  CUDA_CHECK(cudaFreeHost(stop_flag));
  CUDA_CHECK(cudaFreeHost(counters));
  CUDA_CHECK(cudaFree(cache_lines));

  return result;
}

}  // namespace cacheline_atomicity

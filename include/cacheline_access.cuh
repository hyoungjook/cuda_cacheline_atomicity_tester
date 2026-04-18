#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>

namespace cacheline_atomicity {

template <bool atomic, typename elem_type, typename tile_type>
__device__ elem_type cacheline_load(elem_type* line, const tile_type& tile) {
  elem_type elem;
  if constexpr (atomic) {
    cuda::atomic_ref<elem_type, cuda::thread_scope_device> ref(line[tile.thread_rank()]);
    tile.sync();
    elem = ref.load(cuda::memory_order_acquire);
    tile.sync();
  }
  else {
    elem = line[tile.thread_rank()];
  }
  return elem;
}

template <typename elem_type>
inline constexpr elem_type latch_mask = static_cast<elem_type>(1) << (sizeof(elem_type) * 8 - 1);

template <typename elem_type, typename tile_type>
__device__ void cacheline_acquire(elem_type* line, const tile_type& tile) {
  while (true) {
    elem_type old;
    if (tile.thread_rank() == 0) {
      cuda::atomic_ref<elem_type, cuda::thread_scope_device> ref(line[0]);
      old = ref.fetch_or(latch_mask<elem_type>, cuda::memory_order_relaxed);
    }
    old = tile.shfl(old, 0);
    if ((old & latch_mask<elem_type>) == 0) {
      break;
    }
  }
  //tile.sync();
  //cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
}

template <typename elem_type, typename tile_type>
__device__ void cacheline_store_release(elem_type* line, elem_type elem, const tile_type& tile) {
  elem = elem & ~latch_mask<elem_type>;
  cuda::atomic_ref<elem_type, cuda::thread_scope_device> ref(line[tile.thread_rank()]);
  tile.sync();
  ref.store(elem, cuda::memory_order_release);
  tile.sync();
}

}  // namespace cacheline_atomicity

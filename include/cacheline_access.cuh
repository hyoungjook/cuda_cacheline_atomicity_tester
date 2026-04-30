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
    tile.sync();
    elem = line[tile.thread_rank()];
    tile.sync();
  }
  return elem;
}

template <typename elem_type>
inline constexpr elem_type latch_mask = static_cast<elem_type>(1) << (sizeof(elem_type) * 8 - 1);

template <typename elem_type, typename tile_type>
__device__ void cacheline_acquire(elem_type* line, const tile_type& tile) {
  if (tile.thread_rank() == 0) {
    while (true) {
      cuda::atomic_ref<elem_type, cuda::thread_scope_device> ref(line[0]);
      auto old = ref.fetch_or(latch_mask<elem_type>, cuda::memory_order_relaxed);
      if ((old & latch_mask<elem_type>) == 0) {
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
        break;
      }
    }
  }
  tile.sync();
}

template <typename elem_type, typename tile_type>
__device__ void cacheline_store_release(elem_type* line, elem_type elem, const tile_type& tile) {
  //  Here, we (1) first weak-store elements and then (2) unlatch, to ensure single-writer.
  //  After cacheline atomicity is verified, in real use case, the programmer can instead
  //  fuse unlatch with element store into one coalesced store, exploiting cacheline atomicity.
  elem = elem & ~latch_mask<elem_type>;
  if (tile.thread_rank() == 0) {
    elem = elem | latch_mask<elem_type>;
  }
  tile.sync();
  line[tile.thread_rank()] = elem;
  tile.sync();
  if (tile.thread_rank() == 0) {
    cuda::atomic_ref<elem_type, cuda::thread_scope_device> ref(line[0]);
    ref.store(elem & ~latch_mask<elem_type>, cuda::memory_order_release);
  }
}

}  // namespace cacheline_atomicity

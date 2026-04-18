#pragma once

#include <cstdint>
#include <iostream>

namespace cacheline_atomicity {

struct ExperimentConfig {
  std::uint32_t cache_line_count = 1000000;
  std::uint64_t timeout_ms = 60000;
  void print() const {
    std::cout << "Configs:" << std::endl;
    std::cout << "  cache_line_count: " << cache_line_count << std::endl;
    std::cout << "  timeout_ms: " << timeout_ms << std::endl;
    std::cout << std::endl;
  }
};

struct ExperimentResult {
  std::uint64_t total_writes = 0;
  std::uint64_t total_weak_reads = 0;
  std::uint64_t clean_weak_reads = 0;
  std::uint64_t torn_weak_reads = 0;
  std::uint64_t total_atomic_reads = 0;
  std::uint64_t clean_atomic_reads = 0;
  std::uint64_t torn_atomic_reads = 0;
  double elapsed_ms = 0;
  void print() const {
    std::cout << "Results:" << std::endl;
    std::cout << "  Elapsed: " << elapsed_ms << " ms" << std::endl;
    std::cout << "  Writes: " << total_writes << '\n';
    std::cout << "  Reads(weak): " << total_weak_reads << '\n';
    std::cout << "    Clean weak reads: " << clean_weak_reads << '\n';
    std::cout << "    Torn weak reads: " << torn_weak_reads << '\n';
    std::cout << "  Reads(atomic): " << total_atomic_reads << '\n';
    std::cout << "    Clean atomic reads: " << clean_atomic_reads << '\n';
    std::cout << "    Torn atomic reads: " << torn_atomic_reads << '\n';
    if (torn_weak_reads == 0 && torn_atomic_reads == 0) {
      std::cout << "No torn read was observed.\n";
    }
  }
};

}  // namespace cacheline_atomicity

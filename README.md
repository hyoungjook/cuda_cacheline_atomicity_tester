# cuda_cacheline_atomicity_tester

Small CUDA experiment for checking whether an NVIDIA GPU appears to provide
128-byte cache-line atomicity under latched coalesced writes and concurrent coalesced reads.

## Build

```bash
./compile.sh
```

## Run

```bash
./build/cuda_cacheline_atomicity_tester
./build/cuda_cacheline_atomicity_tester cache_line_count=1 timeout_ms=1000
```

Arguments:
- `cache_line_count`: number of 128-byte cache lines to allocate
  Default: `1000000`
- `timeout_ms`: how long the kernel should run before the host stops it
  Default: `60000`

## What to expect

For any input arguments, the program should observe **NO TORN READS AT ALL**.

Sample expected output looks like:

```
Results:
  Elapsed: 60007 ms
  Writes: 104902882
  Reads(weak): 104902882
    Clean weak reads: 104902882
    Torn weak reads: 0
  Reads(atomic): 104902882
    Clean atomic reads: 104902882
    Torn atomic reads: 0
No torn read was observed.
```

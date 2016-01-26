# Simple parallel sort benchmark

This repo contains a simple benchmark of parallel sort, to be run on a
system with a multi-core CPU and an NVIDIA GPU.

## Requirements
 * NVIDIA CUDA Toolkit version 7.5 or higher
 * OpenMP
 * Bash
 * GNU coreutils (make etc.)

## How to compile
Simply execute
```
make -j
```

## How to run
Run the script named `benchmark.sh`. It will produce a log file named
`log.txt` that looks like this:
```
Loading 184m (184000000) integers...
Parallel sort on GPU: 892.14 ms
Parallel sort on CPU: 3654.55 ms

SPEED-UP ON GPU = 4.10
  THROUGHPUT ON GPU = 0.79 MB/s
  THROUGHPUT ON CPU = 0.19 MB/s

Loading 92m (92000000) integers...
Parallel sort on GPU: 442.63 ms
Parallel sort on CPU: 1669.03 ms

SPEED-UP ON GPU = 3.77
  THROUGHPUT ON GPU = 0.79 MB/s
  THROUGHPUT ON CPU = 0.21 MB/s
...
```

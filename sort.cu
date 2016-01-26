#include <cstdio>
#include <algorithm>
#include <chrono>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

#include "cuda_safety_wrapper.h"
#include "parse_size_t.h"

double sort_cpu(unsigned int *array, size_t nelem);
double sort_gpu(unsigned int *array, size_t nelem);

void load_from_file(const char *filename, unsigned int *array,
    size_t nelem)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error: could not open \'%s\' for reading.\n", filename);
        exit(1);
    }

    size_t nelem_read = fread(array, sizeof(*array), nelem, f);
    if (nelem_read < nelem) {
        printf("Error: input file contains fewer than %zu elements.\n",
            nelem);
        fclose(f);
        exit(1);
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s [input file] [number of integers]\n"
               "reads a list of integers from a binary file and sorts"
               "it in two ways.\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    size_t nelem;

    if ( (nelem = parse_size_t(argv[2])) == 0) {
        printf("Error: number of integers is not correctly formatted.\n"
               "       Accepted format: plain numbers OR"
               "numbers with common suffixes (K for thousands, M for "
               "millions, B for billions)\n"
               "       Examples: 1300, 100K, 20M, 1B\n");
        return 1;
    }

    printf("Loading %s (%zu) integers...\n", argv[2], nelem);

    unsigned int *array = (unsigned int *)malloc(nelem * sizeof(*array));
    if (!array) {
        printf("Error: could not allocate an array of %zu elements.\n",
            nelem);
        return 1;
    }

    load_from_file(filename, array, nelem);
    double gpu_time = sort_gpu(array, nelem);

    load_from_file(filename, array, nelem);
    double cpu_time = sort_cpu(array, nelem);

    printf("\nSPEED-UP ON GPU = %.2lf\n", cpu_time/gpu_time);
    printf("  THROUGHPUT ON GPU = %.2lf MB/s\n",
        nelem*sizeof(*array)/1048576.0 / gpu_time);
    printf("  THROUGHPUT ON CPU = %.2lf MB/s\n",
        nelem*sizeof(*array)/1048576.0 / cpu_time);

    free(array);
}

double sort_cpu(unsigned int *array, size_t nelem)
{
    auto start = std::chrono::steady_clock::now();

    thrust::sort(thrust::omp::par, array, array+nelem);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end-start;
    printf("Parallel sort on CPU: %.2lf ms\n", diff.count());

    return diff.count();
}

double sort_gpu(unsigned int *array, size_t nelem)
{
    unsigned int *array_d;

    CUDA_CHECK(cudaMalloc(&array_d, nelem*sizeof(*array)));
    CUDA_CHECK(cudaMemcpy(array_d, array, nelem*sizeof(*array),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();

    thrust::sort(thrust::cuda::par, array_d, array_d+nelem);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end-start;
    printf("Parallel sort on GPU: %.2lf ms\n", diff.count());

    CUDA_CHECK(cudaMemcpy(array, array_d, nelem*sizeof(*array),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(array_d));

    return diff.count();
}

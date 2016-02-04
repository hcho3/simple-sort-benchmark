#include <cstdio>
#include <algorithm>
#include <chrono>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

#include "cuda_safety_wrapper.h"
#include "parse_size_t.h"
#include "Buffer.h"

double sort_cpu(unsigned int *array, size_t nelem);
double sort_gpu(unsigned int *array, size_t nelem, size_t segment_len,
    size_t num_segments);

void load_from_file(const char *filename, unsigned int *array,
    size_t nelem);
bool valid_segment_length(size_t segment_len, size_t nelem);
size_t compute_least_segment_length(size_t nelem);

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s [input file] [number of integers] "
               "[segment size]\n"
               "reads a list of integers from a binary file and sorts"
               "it in two ways.\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    size_t nelem;
    size_t segment_len;

    if ( (nelem = parse_size_t(argv[2])) == 0) {
        printf("Error: number of integers is not correctly formatted.\n"
               "       Accepted format: plain numbers OR"
               "numbers with common suffixes (K for thousands, M for "
               "millions, B for billions)\n"
               "       Examples: 1300, 100K, 20M, 1B\n");
        return 1;
    }

    if ( (segment_len = parse_size_t(argv[3])) == 0) {
        printf("Error: segment size is not correctly formatted.\n"
               "       Accepted format: plain numbers OR"
               "numbers with common suffixes (K for thousands, M for "
               "millions, B for billions)\n"
               "       Examples: 1300, 100K, 20M, 1B\n");
        return 1;
    }
    if (!valid_segment_length(segment_len, nelem)) {
        printf("Segment size is too small. Try a size of at least %zu.\n",
            compute_least_segment_length(nelem));
        return 1;
    }

    size_t num_segments = (nelem+segment_len-1)/segment_len;

    printf("Loading %s (%zu) integers...\n", argv[2], nelem);
    printf("segment_len = %s (%zu)\n", argv[3], segment_len);
    printf("num_segments = %zu\n", num_segments);

    unsigned int *array = (unsigned int *)malloc(nelem * sizeof(*array));
    if (!array) {
        printf("Error: could not allocate an array of %zu elements.\n",
            nelem);
        return 1;
    }

    load_from_file(filename, array, nelem);
    double gpu_time = sort_gpu(array, nelem, segment_len, num_segments);

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

double sort_gpu(unsigned int *array, size_t nelem, size_t segment_len,
    size_t num_segments)
{
    size_t M = segment_len;
    size_t N = nelem;

    Buffer<unsigned int> disk(N, array, 0);
    Buffer<unsigned int> *disk_seg
        = new Buffer<unsigned int>[num_segments];

    unsigned int *mem_buf;
    cudaMalloc(&mem_buf, sizeof(unsigned int)*M);
    unsigned int *mem2_buf;
    cudaMalloc(&mem2_buf, sizeof(unsigned int)*M);

    Buffer<unsigned int> mem[2];
    mem[0] = Buffer<unsigned int>(M, mem_buf, 0);
    mem[1] = Buffer<unsigned int>(M, mem2_buf, 0);

    thrust::device_ptr<unsigned int> dev_ptr[2];
    dev_ptr[0] = thrust::device_ptr<unsigned int>(mem[0].buf);
    dev_ptr[1] = thrust::device_ptr<unsigned int>(mem[1].buf);
    
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    auto start = std::chrono::steady_clock::now();

    size_t seg_len = fetch_into_gpu_async(mem[0], disk, M, stream[0]);
    size_t seg_len2 = 0;
    if (num_segments >= 2) {
        disk.cursor += seg_len;
        seg_len2 = fetch_into_gpu_async(mem[1], disk, M, stream[1]);
        disk.cursor -= seg_len;
    }

    for (size_t k = 0; k < num_segments; k += 2) {
        thrust::sort(thrust::cuda::par.on(stream[0]), dev_ptr[0],
            dev_ptr[0]+seg_len);
        disk_seg[k] = fetch_from_gpu_async_and_slice(disk, mem[0],
            seg_len, stream[0]);

        if (k+2 < num_segments) {
            disk.cursor += seg_len + seg_len2;
            seg_len = fetch_into_gpu_async(mem[0], disk, M, stream[0]);
            disk.cursor -= seg_len2;
        } else {
            disk.cursor += seg_len;
        }

        if (k+1 < num_segments) {
            thrust::sort(thrust::cuda::par.on(stream[1]), dev_ptr[1],
                dev_ptr[1]+seg_len2);
            disk_seg[k+1] = fetch_from_gpu_async_and_slice(disk, mem[1],
                seg_len2, stream[1]);
        }

        if (k+3 < num_segments) {
            disk.cursor += seg_len2 + seg_len;
            seg_len2 = fetch_into_gpu_async(mem[1], disk, M, stream[1]);
            disk.cursor -= seg_len;
        } else {
            disk.cursor += seg_len2;
        }
    }

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end-start;
    printf("Parallel sort on GPU: %.2lf ms\n", diff.count());

    cudaFree(mem[0].buf);
    cudaFree(mem[1].buf);

    delete [] disk_seg;

    return diff.count();
}

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

bool valid_segment_length(size_t seg_len, size_t nelem)
{
    if (seg_len <= 0)
        return false;

    size_t num_segments = (nelem+seg_len-1)/seg_len;
    return seg_len >= num_segments;
}

size_t compute_least_segment_length(size_t nelem)
{
    size_t left, right, mid;
    bool (*const valid)(size_t, size_t) = &valid_segment_length;

    left = 0;
    right = nelem;
    while (left <= right) {
        mid = (left+right)/2;
        if (valid(mid-1, nelem) && valid(mid, nelem))
            right = mid-2;
        else if ( !valid(mid-1, nelem) && !valid(mid, nelem))
            left = mid+1;
        else
            break;
    }

    while (true) {
        if (valid(mid-1, nelem) && valid(mid, nelem))
            mid--;
        else if ( !valid(mid-1, nelem) && !valid(mid, nelem))
            mid++;
        else
            break;
    }

    return mid;
}


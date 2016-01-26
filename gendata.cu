#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <chrono>

#include <curand.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda_safety_wrapper.h"
#include "parse_size_t.h"

void generate_random_int(unsigned int *array, size_t nelem);

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) {
        printf("Usage: %s [output file] [number of integers]\n"
               "  OR   %s [number of integers]\n"
               "This utility randomly generates non-negative integers "
               "and produces a binary dump.\n"
               "If output file is not specified, random integers are "
               "generated but not written to a file.\n",
               argv[0], argv[0]);
        return 1;
    }
    bool file_output;
    const char *filename;
    const char *nelem_str;

    if (argc == 2) { // no output file
        file_output = false;
        nelem_str = argv[1];
    } else {
        file_output = true;
        filename = argv[1];
        nelem_str = argv[2];
    }

    size_t nelem;

    if ( (nelem = parse_size_t(nelem_str)) == 0) {
        printf("Error: number of integers is not correctly formatted.\n"
               "       Accepted format: plain numbers OR"
               "numbers with common suffixes (K for thousands, M for "
               "millions, B for billions)\n"
               "       Examples: 1300, 100K, 20M, 1B\n");
        return 1;
    }

    printf("Generating %s (%zu) integers...%s\n", nelem_str, nelem,
        (file_output) ? "" : "(DRY RUN)");

    FILE *f;
    if (file_output) {
        f = fopen(filename, "w");
        if (!f) {
            printf("Error: could not open \'%s\' for writing.\n", filename);
            return 1;
        }
    }

    unsigned int *array = (unsigned int *)malloc(nelem * sizeof(*array));
    if (!array) {
        printf("Error: could not allocate an array of %zu elements.\n",
            nelem);

        if (file_output)
            fclose(f);
        return 1;
    }

    generate_random_int(array, nelem);

    if (file_output) {
        size_t nelem_written = fwrite(array, sizeof(*array), nelem, f);
        if (nelem_written < nelem) {
            printf("Error: output file is incomplete.\n");
            fclose(f);
            return 1;
        }
        fclose(f);
    }

    return 0;
}

void generate_random_int(unsigned int *array, size_t nelem)
{
    srand(time(NULL));
    int _seed = rand();
    curandGenerator_t gen;

    unsigned int *array_d;

    CUDA_CHECK(cudaMalloc(&array_d, nelem*sizeof(*array)));
    
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, _seed));
    CURAND_CHECK(curandGenerate(gen, array_d, nelem));
        // generate the random numbers
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaMemcpy(array, array_d, nelem*sizeof(*array),
               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(array_d));
}


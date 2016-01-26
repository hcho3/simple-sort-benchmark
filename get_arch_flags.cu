#include <cstdio>

int main(void)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("-gencode arch=compute_%d%d,code=sm_%d%d\n",
        prop.major, prop.minor, prop.major, prop.minor);

    return 0;
}

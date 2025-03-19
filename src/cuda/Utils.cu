#include "Utils.h"
unsigned int prng(unsigned int x){
    x = (x + 0x7ED55D16) + (x << 12);
    x = (x ^ 0xC761C23C) ^ (x >> 19);
    x = (x + 0x165667B1) + (x << 5);
    x = (x + 0xD3A2646C) ^ (x << 9);
    x = (x + 0xFD7046C5) + (x << 3);
    x = (x ^ 0xB55A4F09) ^ (x >> 16);
    return x;
}

void check_cuda(cudaError_t err){
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
__device__ void map_executor(int n, MapOperator map_operator, void* context)
{
    for(int id = threadIdx.x; id < n; id += blockDim.x){
        map_operator(id, context);
    }
}

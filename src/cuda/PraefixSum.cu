#include "PraefixSum.h"

__global__ void praefix_sum_kernel1(float * Y, int classes, int n, int k){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id * k + k - 1 < n){
        for(int c = 0; c < classes; c++){
            Y[(id * k + k - 1)*classes+c] += Y[(id * k + k - 1 - k/2)*classes+c];
        }
    }
}

__global__ void praefix_sum_kernel2(float * Y, int classes, int n, int k){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id * k + k + k/2 - 1 < n){
        for(int c = 0; c < classes; c++){
            Y[(id * k + k + k/2 - 1)*classes+c] += Y[(id * k + k - 1)*classes+c];
        }
    }
}

void praefix_sum_small_data(float * Y, int classes, int n){
    int blocks = (n + PRAEFIX_THREADS - 1) / PRAEFIX_THREADS;
    int k;
    for(k = 2; k < 2*n; k*=2){
        int current_blocks = (blocks + k - 1)/k;
        praefix_sum_kernel1<<<current_blocks, PRAEFIX_THREADS>>>(Y, classes, n, k);
    }
    for(k=k/2; k >= 2; k/=2){
        int current_blocks = (blocks + k - 1)/k;
        praefix_sum_kernel2<<<current_blocks, PRAEFIX_THREADS>>>(Y, classes, n, k);
    }
}


__host__ void praefix_sum_big_data(std::vector<float>::iterator Y_begin, int classes, long n){
    
    constexpr long CAPACITY = 100000; //randomly choosen
    long part_size = CAPACITY/classes;

    //we work with part_size that is a potenz of 2 if possible
    float* Y_device;
    check_cuda(cudaMalloc((void**)&Y_device, part_size * classes * sizeof(float)));

    for (long i = 0; i < n; i += part_size - 1) {
        long current_size = min(part_size, n - i);
        auto Y_it = Y_begin + i * classes;
        check_cuda(cudaMemcpy(Y_device, &(*Y_it), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice));
        praefix_sum_small_data(Y_device, classes, current_size);
        check_cuda(cudaMemcpy(&(*Y_it), Y_device, current_size * sizeof(float) * classes, cudaMemcpyDeviceToHost));
    }

    check_cuda(cudaFree(Y_device));

}
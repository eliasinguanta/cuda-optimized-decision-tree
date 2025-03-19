#include <iostream>
#include <vector>

#ifndef UTILS_H
#define UTILS_H
#define MAX_CLASSES 30

unsigned int prng(unsigned int x);

template<typename T>
void printArray(const T* arr, int n){
    std::cout << "Array contents: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
};
template <typename T>
void printVector(typename std::vector<T>::iterator arr, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << *arr << " ";
        ++arr;
    }
    std::cout << std::endl;
}


void check_cuda(cudaError_t err);

template<typename T>
__device__ void load(T* from, T* to, int n){
    for(int id = threadIdx.x; id < n; id += blockDim.x){
        to[id] = from[id];
    }
}
template<typename T>
__device__ void load(T* to, T value, int n){
    for(int id = threadIdx.x; id < n; id += blockDim.x){
        to[id] = value;
    }
}

using MapOperator = void (*)(int, void*);

__device__ void map_executor(int n, MapOperator map_operator, void* context);

struct SplitPoint{
    long index;
    int feature;
    float evaluation; //0 is best, 2 is worst
};
#endif
#include "Reduce.h"

//************************************************//
//                reduce min index                //
//************************************************//

__device__ int minimum(int l, int r, void* array){
    float* arr = (float*)array;
    if(arr[l] < arr[r] || (arr[l] == arr[r] && l > r)) return l;
    return r;
}
//as an example we reduce to max
__device__ int reduce_index_of_min(int n, float* context){
    if(n <= 1) return 0;
    constexpr int m = 1024; //number of threads would be optimal
    __shared__ int shared_memory[m];

    int best_id = 0;//threadIdx.x;
    for(int id = threadIdx.x; id < n; id += blockDim.x){
        best_id = minimum(best_id, id, context);
    }

    shared_memory[threadIdx.x] = best_id;

    __syncthreads();
    for(int size = (n < blockDim.x)?n :blockDim.x, half = (size+1)/2; size > 1; size = half, half = (size+1)/2){
        if(threadIdx.x + half < size){
            shared_memory[threadIdx.x] = minimum(shared_memory[threadIdx.x], shared_memory[threadIdx.x + half], context);
        }
        __syncthreads();
    }

    return shared_memory[0];
}

__global__ void reduce_index_of_min_blockwise(int n, float* array, int* best_indicies){
    int inner_block_size = coersing_factor * blockDim.x;
    if(inner_block_size * blockIdx.x < n){
        int current_size = (inner_block_size < n - inner_block_size * blockIdx.x)? inner_block_size: n - inner_block_size * blockIdx.x;
        best_indicies[blockIdx.x] = reduce_index_of_min(current_size, array + inner_block_size * blockIdx.x) + inner_block_size * blockIdx.x;
    }else best_indicies[blockIdx.x] = 0; //(we dont check boundaries of the indicies so we have to make sure they are correct)
}

__global__ void reduce_index_of_min_block_results(int blocks, float* array, int* best_indicies){
    if(blocks <= 1) return;
    constexpr int m = 1024;
    __shared__ int shared_memory[m];

    for(int id = threadIdx.x + blockDim.x; id < blocks; id += blockDim.x){
        best_indicies[threadIdx.x] = minimum(best_indicies[threadIdx.x], best_indicies[id], array);
    }

    if(threadIdx.x < blocks) shared_memory[threadIdx.x] = best_indicies[threadIdx.x];

    __syncthreads();
    for(int size = (blocks < blockDim.x)?blocks :blockDim.x, half = (size+1)/2; size > 1; size = half, half = (size+1)/2){
        if(threadIdx.x + half < size){
            shared_memory[threadIdx.x] = minimum(shared_memory[threadIdx.x], shared_memory[threadIdx.x + half], array);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) best_indicies[0] = shared_memory[0];
}

__host__ long reduce_index_of_min_big_data(std::vector<float> data){
    //bound for array-index and therefor for array
    //we assume float array

    constexpr int part_size = 1<<10;
    constexpr int threads = 8;
    int blocks = (part_size + (threads*coersing_factor) - 1) / (threads*coersing_factor);

    float* part_device;
    int* best_indicies_device;
    long best_index_global = 0;
    cudaMalloc((void**)&part_device, sizeof(float)*part_size);
    cudaMalloc((void**)&best_indicies_device, sizeof(int)*blocks);
    
    int* best_indicies_host = (int*) malloc(sizeof(int)*blocks);

    for(long i = 0; i<data.size(); i+= (long)(part_size)){
        long current_size = min((long)(part_size), data.size() - i);
        cudaMemcpy(part_device, data.data() + i, current_size * sizeof(float), cudaMemcpyHostToDevice);

        reduce_index_of_min_blockwise<<<blocks, threads>>>(current_size, part_device, best_indicies_device);
        cudaMemcpy(best_indicies_host, best_indicies_device, sizeof(int)*blocks, cudaMemcpyDeviceToHost);

        reduce_index_of_min_block_results<<<1, 1024>>>(blocks, part_device, best_indicies_device);

        cudaMemcpy(best_indicies_host, best_indicies_device, sizeof(int)*blocks, cudaMemcpyDeviceToHost);
        if(data[((long)(best_indicies_host[0]))+i]< data[best_index_global]) best_index_global = ((long)(best_indicies_host[0]))+i;
    }

    cudaFree(part_device);
    cudaFree(best_indicies_device);
    free(best_indicies_host);
    return best_index_global;
}

//************************************************//
//              reduce sum of error               //
//************************************************//

__device__ float reduce_error(float* P, float* Y, float* D, int classes, int n){
    if(n <= 0) return 0;
    constexpr int m = 1024; //number of threads would be optimal
    __shared__ float shared_memory[m];

    float sum = 0.0f;
    for(int id = threadIdx.x; id < n; id += blockDim.x){
        int max_pr_index = 0;
        int max_tr_index = 0;
        float max_pr = -__FLT_MAX__;
        float max_tr = -__FLT_MAX__;
        for(int c = 0; c<classes; c++){
            float current_pr = P[id * classes + c];
            float current_tr = Y[id * classes + c];
            if(current_pr >= max_pr){
                max_pr_index = c;
                max_pr = current_pr;
            } 
            if(current_tr >= max_tr){
                max_tr_index = c;
                max_tr = current_tr;
            } 
        }
        if(max_pr_index != max_tr_index) sum += D[id];
    }

    shared_memory[threadIdx.x] = sum;

    __syncthreads();
    for(int size = (n < blockDim.x)?n :blockDim.x, half = (size+1)/2; size > 1; size = half, half = (size+1)/2){
        if(threadIdx.x + half < size){
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + half];
        }
        __syncthreads();
    }

    return shared_memory[0];
}
__global__ void reduce_sum_of_error_blockwise(float* P, float* Y, float* D, float* error_device, int classes, int n){
    int inner_block_size = coersing_factor * blockDim.x;
    int offset = inner_block_size * blockIdx.x;
    int current_size = (inner_block_size < n - offset)? inner_block_size: n - offset;
    float error = reduce_error(P + offset*classes, Y + offset*classes, D + offset, classes, current_size);
    if(threadIdx.x == 0) error_device[blockIdx.x] = error;
}

__global__ void reduce_sum_of_error(int n, float* error_device){
    if(n <= 1) return;
    constexpr int m = 1024;
    __shared__ float shared_memory[m];

    shared_memory[threadIdx.x] = 0.0f;

    for(int id = threadIdx.x; id < n; id += blockDim.x){
        shared_memory[threadIdx.x] += error_device[id];
    }

    __syncthreads();
    
    for(int size = min(n, blockDim.x), half = (size+1)/2; size > 1; size = half, half = (size+1)/2){
        if(threadIdx.x < half && threadIdx.x + half < size){
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + half];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) error_device[0] = shared_memory[0];
}

//************************************************//
//                  reduce sum                    //
//************************************************//

__device__ float reduce_sum(float* array, int n){
    if(n <= 0) return 0;
    constexpr int m = 1024; //number of threads would be optimal
    __shared__ float shared_memory[m];

    float sum = 0.0f;
    for(int id = threadIdx.x; id < n; id += blockDim.x){
        sum += array[id];
    }

    shared_memory[threadIdx.x] = sum;

    __syncthreads();

    for(int size = min(n, blockDim.x), half = (size+1)/2; size > 1; size = half, half = (size+1)/2){
        if(threadIdx.x < half && threadIdx.x + half < size){
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + half];
        }
        __syncthreads();
    }

    return shared_memory[0];
}

__global__ void reduce_sum_blockwise(float * array, int n)
{
    int inner_block_size = coersing_factor * blockDim.x;
    int offset = inner_block_size * blockIdx.x;
    int current_size = (inner_block_size < n - offset)? inner_block_size: n - offset;
    float sum = reduce_sum(array + offset, current_size);
    if(threadIdx.x == 0) array[blockIdx.x] = sum;
}

__global__ void reduce_sum(int n, float * array)
{
    if(n <= 1) return;
    constexpr int m = 1024;
    __shared__ float shared_memory[m];

    shared_memory[threadIdx.x] = 0.0f;

    for(int id = threadIdx.x; id < n; id += blockDim.x){
        shared_memory[threadIdx.x] += array[id];
    }

    __syncthreads();
    
    for(int size = min(n, blockDim.x), half = (size+1)/2; size > 1; size = half, half = (size+1)/2){
        if(threadIdx.x < half && threadIdx.x + half < size){
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + half];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) array[0] = shared_memory[0];
}

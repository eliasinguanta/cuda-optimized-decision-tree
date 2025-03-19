#include "BitonicSort.h"

#define DEBUG 0
#define SORT_THREADS_LOG 5 //min 1 max 10
#define SHARED_MEM_OPT 1 //seems to have no really diff

inline __device__ void compare_and_swap(int s, int b, float* X_row, int* I, bool dir){
    if(dir == (X_row[s] > X_row[b])){
        //swap the value
        float tmp = X_row[s];
        X_row[s] = X_row[b];
        X_row[b] = tmp;

        //keep the index
        int i = I[s];
        I[s] = I[b];
        I[b] = i;
    }
}

/*
* we sort an inner block entirely in shared memory
*/
__global__ void sort_k_2048_block_entirely(int k, int i, int n, float* X_row, int* I){

    k = 1<<k;
    i = 1<<i;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int m = 1<<(SORT_THREADS_LOG+1); //should be twice as large as number of threads
    
    __shared__ float shared_values[m]; 
    __shared__ int shared_indicies[m];

    //index where the innerblock is that we sort currently
    int offset = (id/(m/2)) * m;
    
    //the position of the innerblock in the outer block decides in which direction we sort
    bool dir = (id % i < i/2)? true: false;

    //load inner block in shared memory (or multiple inner blocks)
    for(int j = threadIdx.x; j < m && j + offset < n; j+=blockDim.x){
        shared_values[j] = X_row[j + offset];
        shared_indicies[j] = I[j + offset];
    }
    __syncthreads();

    //sort inner block in shared memory
    for(int l = k; l>1; l/=2){
        int index = (threadIdx.x/(l/2)) * l + threadIdx.x % (l/2);
        compare_and_swap(index, index + l/2, shared_values, shared_indicies, dir);
        __syncthreads();
    }

    //load inner block in global memory (or multiple inner blocks)
    for(int j = threadIdx.x; j < m && j + offset < n; j+=blockDim.x){
        X_row[j + offset] = shared_values[j];
        I[j + offset] = shared_indicies[j];
    }
}

/*
* we sort an outer block entirely in shared memory
*/
__global__ void sort_i_2048_block_entirely(int i, int n, float* X_row, int* I){

    i = 1<<i;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int m = 1<<(SORT_THREADS_LOG+1);
    __shared__ float shared_values[m]; 
    __shared__ int shared_indicies[m];

    //index where the outer block is that we sort currently
    int offset = (id/(m/2)) * m;

    //load outer block in shared memory (or multiple inner blocks)
    for(int j = threadIdx.x; j < m && j + offset < n; j+=blockDim.x){
        shared_values[j] = X_row[j + offset];
        shared_indicies[j] = I[j + offset];
    }
    __syncthreads();

    //sort outer block in shared memory
    for(int l = 2; l<=i; l*=2){
        for(int k = l; k>1; k/=2){
            int index = (threadIdx.x/(k/2)) * k + threadIdx.x % (k/2);
            bool dir = (index % l < l/2)? true: false;
            compare_and_swap(index, index + k/2, shared_values, shared_indicies, dir);    
            __syncthreads();
        }
    }

    //load outer block in global memory (or multiple inner blocks)
    for(int j = threadIdx.x; j < m && j + offset < n; j+=blockDim.x){
        X_row[j + offset] = shared_values[j];
        I[j + offset] = shared_indicies[j];
    }
}

/*
* one parallel step of bitonic sort
*/
__global__ void sort_k_block(int k, int i, int n, float* X_row, int* I){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < (n >> 1)){
        bool dir = (id & ((1<<i)-1)) < (1<<(i-1))? true: false;
        int index = ((id >> (k - 1)) << k) + (id & ((1 << (k - 1)) - 1));
        compare_and_swap(index, index + (1<<k)/2, X_row, I, dir);
    }
}


__host__ void sort_indicies(float* X_row, int* I, int l){
    int blocks = ((1<<l) + (1<<SORT_THREADS_LOG) - 1) / (1<<SORT_THREADS_LOG);
    
#if SHARED_MEM_OPT
    int block_size = min(SORT_THREADS_LOG + 1, l);
    sort_i_2048_block_entirely<<<blocks, (1<<SORT_THREADS_LOG)>>>(block_size, 1<<l, X_row, I);
    for(int i = block_size; i<=l; i++){
        int k = i;
        for(; k>SORT_THREADS_LOG+1; k--){
            sort_k_block<<<blocks,(1<<SORT_THREADS_LOG)>>>(k, i, 1<<l, X_row, I);
        }
        sort_k_2048_block_entirely<<<blocks, (1<<SORT_THREADS_LOG)>>>(k, i, 1<<l, X_row, I);
    }
#else
    for(int i = 1; i<=l; i++){
        for(int k = i; k>0; k--){
            sort_k_block<<<blocks,(1<<SORT_THREADS_LOG)>>>(k, i, 1<<l, X_row, I);
        }
    }
#endif
}

//////
__host__ void merge_sort_merge(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, int selected_feature, int part_size) {
    for (long merge_size = 2 * part_size; merge_size < 2 * n; merge_size *= 2) {
        for (long offset = 0; offset < n; offset += merge_size) {
            long left_end = min(offset + merge_size / 2, n);
            long right_end = min(offset + merge_size, n);

            long left = offset;
            long right = left_end;

            while (left < right && right < right_end) {
                if (*(X_begin + left * features + selected_feature) <= *(X_begin + right * features + selected_feature)) {
                    left++;
                } else {
                    // shift vector from right to left position
                    for (long i = right; i > left; i--) {
                        for (int j = 0; j < features; j++) {
                            std::swap(*(X_begin + i * features + j), *(X_begin + (i - 1) * features + j));
                        }
                        for (int j = 0; j < classes; j++) {
                            std::swap(*(Y_begin + i * classes + j), *(Y_begin + (i - 1) * classes + j));
                        }
                    }
                    left++;
                    right++;
                }
            }
        }
    }
}
__global__ void init_host(float* X, float* X_row, int* I, int features, int n, int l, int selected_feature){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id<n) {
        X_row[id] = X[id*features+selected_feature];
        I[id] = id;
    }else if(id < (1<<l)) {
        X_row[id] = FLT_MAX;
    }
}
__global__ void init_host2(float* X, float* X_row, int* I, int features, int n, int l, SplitPoint* split_point){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id<n) {
        X_row[id] = X[id*features+ split_point->feature];
        I[id] = id;
    }else if(id < (1<<l)) {
        X_row[id] = FLT_MAX;
    }
}

__global__ void apply_index_host(float* X_old, float* Y_old, float* X_new, float* Y_new, int* I, int features, int classes, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        for(int f = 0; f<features; f++){
            X_new[id*features+f] = X_old[I[id]*features+f];
        }
        for(int c = 0; c<classes; c++){
            Y_new[id*classes+c] = Y_old[I[id]*classes+c];
        }
    }
}


__host__ void sort_big_data(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, int selected_feature) {

    constexpr long CAPACITY = (long)(1<<20); //in bytes
    long part_size; for(part_size = 1; part_size<(CAPACITY/(classes+features))/2; part_size*=2);

    //we work with part_size that is a potenz of 2 if possible
    float* X_old_device;
    float* Y_old_device;
    int* I_device;
    float* X_row;
    float* X_new_device;
    float* Y_new_device;


    cudaMalloc((void**)&X_old_device, part_size * features * sizeof(float));
    cudaMalloc((void**)&Y_old_device, part_size * classes * sizeof(float));
    cudaMalloc((void**)&I_device, part_size * sizeof(int));
    cudaMalloc((void**)&X_row, part_size * sizeof(float));
    cudaMalloc((void**)&X_new_device, part_size * features * sizeof(float));
    cudaMalloc((void**)&Y_new_device, part_size * classes * sizeof(float));


    for (long i = 0; i < n; i += part_size) {

        long current_size = min(part_size, n - i);

        auto X_it = X_begin + i * features;
        auto Y_it = Y_begin + i * classes;

        cudaMemcpy(X_old_device, &(*X_it), current_size * sizeof(float) * features, cudaMemcpyHostToDevice);
        cudaMemcpy(Y_old_device, &(*Y_it), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice);

        int l; for(l = 1; (1<<l)<current_size; l++);
        float* X_row_host = (float*) malloc(sizeof(float)*(1<<l)); 

        constexpr int threads = 32;
        int blocks = ((1<<l) + threads - 1) / threads;
        init_host<<<blocks, threads>>>(X_old_device, X_row, I_device, features, n, l, selected_feature);

            //check_cuda(cudaMemcpy(X_row_host, X_row, sizeof(float)*(1<<l), cudaMemcpyDeviceToHost));
            //printArray<float>(X_row_host, 1<<l);
        sort_indicies(X_row, I_device, l);
            //check_cuda(cudaMemcpy(X_row_host, X_row, sizeof(float)*(1<<l), cudaMemcpyDeviceToHost));
            //printArray<float>(X_row_host, 1<<l);
        apply_index_host<<<blocks, threads>>>(X_old_device, Y_old_device, X_new_device, Y_new_device, I_device, features, classes, n);


        cudaMemcpy(&(*X_it), X_new_device, current_size * sizeof(float) * features, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(*Y_it), Y_new_device, current_size * sizeof(float) * classes, cudaMemcpyDeviceToHost);
    }

    cudaFree(X_old_device);
    cudaFree(Y_old_device);
    cudaFree(X_new_device);
    cudaFree(Y_new_device);
    cudaFree(I_device);
    cudaFree(X_row);

    merge_sort_merge(X_begin, Y_begin, features, classes, n, selected_feature, part_size);
}
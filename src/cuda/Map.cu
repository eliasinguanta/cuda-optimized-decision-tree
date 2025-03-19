#include "Map.h"
#define MAX_CLASSES 100

__device__ float gini(float* Y, float* L, int classes){

    float sum = 0.0;
    float squared_sum = 0.0;
    for(int c = 0; c<classes; c++)sum += (L == NULL)?Y[c]:L[c] - Y[c];
    for(int c = 0; c<classes; c++){
        float p = ((L == NULL)?Y[c]:L[c] - Y[c])/sum;
        squared_sum += p*p;
    }
    return 1.0 - squared_sum;
}

__device__ float multi_log_loss(float* Y, float* L, int classes){

    float max_pr = -__FLT_MAX__;
    int max_pr_index = 0;
    float sum = 0;

    for(int c = 0; c<classes; c++){
        float current = (L == NULL)?Y[c]:L[c] - Y[c];
        sum += current;
        if(current >= max_pr){
            max_pr = current;
            max_pr_index = c;
        }
    }
    float prediction = (L == NULL)?Y[max_pr_index]:L[max_pr_index] - Y[max_pr_index];
    prediction /= sum;
    return - log(prediction);
}

__device__ void evaluate_distribution(float* Y, float* G, float* L, int classes, int n, int offset, int real_n){

    constexpr int m = 8; //we use copies of the last distribution to have less race condition
    __shared__ float L_copies[m][MAX_CLASSES];
    if(threadIdx.x == 0){
        for(int c = 0; c<classes; c++) L_copies[0][c]=L[c];
    }
    __syncthreads();

    for(int i = 1; i < m; i *= 2){
        if(threadIdx.x + i < m){
            for(int c = 0; c < classes; c++){
                L_copies[threadIdx.x + i][c] = L_copies[threadIdx.x][c];
            }
        }
        __syncthreads();
    }

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){

        float gini_coef_left = multi_log_loss(Y + id*classes, NULL, classes);
        float gini_coef_right = 0.0;

        if(id < real_n - 1){
            gini_coef_right = multi_log_loss(Y + id*classes, L_copies[threadIdx.x % m], classes);
        }

        //result
        G[id]= (gini_coef_left*(id+1+offset)+gini_coef_right*(real_n - (id+1+offset)))/real_n;

    }
    
}

__global__ void map_gini_kernel(float* Y, float* G, float* L, int classes, int n, int offset, int real_n){
    evaluate_distribution(Y, G, L, classes, n, offset, real_n);
}

void evaluate_distributions_small_data(float* Y, float* G, float* L, int classes, int n, int offset, int real_n){
    constexpr int threads = 32;
    int blocks = (n + threads - 1) / threads;
    map_gini_kernel<<<blocks, threads>>>(Y, G, L, classes, n, offset, real_n);
}


__host__ void evaluate_distributions_big_data(std::vector<float>::iterator Y_begin, std::vector<float>::iterator G_begin, int classes, long n){
    constexpr long CAPACITY = 100000;//(long)(INT_MAX); //in bytes
    long part_size = CAPACITY/classes;

    //we work with part_size that is a potenz of 2 if possible
    float* Y_device;
    float* G_device;
    float* Y_device_last;
    cudaMalloc((void**)&Y_device, part_size * classes * sizeof(float));
    cudaMalloc((void**)&G_device, part_size * sizeof(float));
    cudaMalloc((void**)&Y_device_last, classes * sizeof(float));

    cudaMemcpy(Y_device_last, &(*(Y_begin + (n-1) * classes)), sizeof(float) * classes, cudaMemcpyHostToDevice);

    for (long i = 0; i < n; i += part_size) {
        long current_size = min(part_size, n - i);

        auto Y_it = Y_begin + i * classes;
        auto G_it = G_begin + i;

        cudaMemcpy(Y_device, &(*Y_it), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice);
        evaluate_distributions_small_data(Y_device, G_device, Y_device_last, classes, current_size, i, n);
        cudaMemcpy(&(*G_it), G_device, current_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(Y_device);
    cudaFree(G_device);
}

/*
 * the same value in X_row can map to different evaluation because of the way
 * we compute it so we take the last one and disqualify all other duplicates
*/
__global__ void fix_duplicates(float* G, float* X_row, int n){
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < n - 1){
        if(X_row[id + 1] == X_row[id]){
            G[id]= __FLT_MAX__;
        }
    }
}

__host__ void handle_duplicates(float* G, float* X_row, int n){
    constexpr int threads = 32;
    int blocks = (n + threads - 1)/threads;
    fix_duplicates<<<blocks, threads>>>(G, X_row, n);
}

__global__ void divide_by_sum(float* array, int n, float sum){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < n) array[id] /= sum;
}

__global__ void compute_data_weights(float* P, float* Y, float* D, float alpha, int classes, int n){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id < n){
        int max_tr = 0;
        int max_pr = 0;
        for(int c = 1; c<classes; c++){
            if(P[id*classes+max_pr]<=P[id*classes+c])max_pr=c;
            if(Y[id*classes+max_tr]<=Y[id*classes+c])max_tr=c;
        }
        D[id] *= ((max_tr != max_pr)?exp(-alpha):exp(alpha));
    }
}
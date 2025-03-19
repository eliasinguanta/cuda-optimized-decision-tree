#include "CudaAdaBoost.h"


inline bool isClose(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

void changed_weights(std::vector<float>::iterator P_begin, std::vector<float>::iterator Y_begin, std::vector<float>::iterator D_begin, float* alpha, int classes, unsigned long n){
    int part_size = 1<<20/classes; //more or less randomly selected
    
    //input buffer
    float* Y_device;
    float* P_device;
    float* D_device;

    //output buffer
    float* error_device;

    //standart config 
    constexpr int threads = 32;
    int rblocks  = (part_size + (threads*coersing_factor) - 1) / (threads*coersing_factor);

    check_cuda(cudaMalloc((void**)&Y_device, part_size * classes * sizeof(float)));
    check_cuda(cudaMalloc((void**)&P_device, part_size * classes * sizeof(float)));
    check_cuda(cudaMalloc((void**)&D_device, part_size * sizeof(float)));
    check_cuda(cudaMalloc((void**)&error_device, rblocks * sizeof(float)));
    

    //we sum up the errors

    float* error_host  = (float*) malloc(sizeof(float)*rblocks);
    float error = 0;
    for(int offset = 0; offset<n; offset+=part_size){

        int current_size = (int)min(n - ((long)offset), (long)part_size);
        rblocks  = (current_size + (threads*coersing_factor) - 1) / (threads*coersing_factor);

        check_cuda(cudaMemcpy(Y_device, &(*(Y_begin + offset * classes)), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(P_device, &(*(P_begin + offset * classes)), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(D_device, &(*(D_begin + offset)), current_size * sizeof(float), cudaMemcpyHostToDevice));
 
        reduce_sum_of_error_blockwise<<<rblocks, threads>>>(P_device, Y_device, D_device, error_device, classes, current_size);
        reduce_sum_of_error<<<1, 1024>>>(rblocks, error_device);

        check_cuda(cudaMemcpy(error_host, error_device, rblocks * sizeof(float), cudaMemcpyDeviceToHost));
        error += error_host[0];
    }

    //set tree weight

    if(isClose(error, 0.0f)){
        *alpha = 1.0f;
        return;
    }
    if(isClose(error, 1.0f)){
        *alpha = 0.0f;
        return;
    }
    *alpha = 0.5f * log(((1.0f-(error))/error) + 1);

    //compute the new data weights and the sum

    float weight_sum_buffer = 0.0f;
    float weight_sum = 0.0f;

    for(int offset = 0; offset<n; offset+=part_size){

        int current_size = (int)min(n - ((long)offset), (long)part_size);
        rblocks  = (current_size + (threads*coersing_factor) - 1) / (threads*coersing_factor);

        cudaMemcpy(Y_device, &(*(Y_begin + offset * classes)), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice);
        cudaMemcpy(P_device, &(*(P_begin + offset * classes)), current_size * sizeof(float) * classes, cudaMemcpyHostToDevice);
        cudaMemcpy(D_device, &(*(D_begin + offset)), current_size * sizeof(float), cudaMemcpyHostToDevice);

        compute_data_weights<<<1,1024>>>(Y_device, P_device, D_device, *alpha, classes, current_size);
        cudaMemcpy(&(*(D_begin + offset)), D_device, current_size * sizeof(float), cudaMemcpyDeviceToHost);
        reduce_sum_blockwise<<<rblocks, threads>>>(D_device, current_size);
        reduce_sum<<<1, 1024>>>(rblocks, D_device);
        cudaMemcpy(&weight_sum_buffer, D_device, sizeof(float), cudaMemcpyDeviceToHost);
        weight_sum += weight_sum_buffer;
    }

    //use the computed sum to norm the data weights
    for(int offset = 0; offset<n; offset+=part_size){
        int current_size = (int)min(n - ((long)offset), (long)part_size);

        cudaMemcpy(D_device, &(*(D_begin + offset)), current_size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int mthreads = 32;
        int mblocks  = (current_size + mthreads - 1) / mthreads;
        divide_by_sum<<<mblocks, mthreads>>>(D_device, current_size, weight_sum);
        cudaMemcpy(&(*(D_begin + offset)), D_device, current_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(Y_device);
    cudaFree(P_device);
    cudaFree(D_device);
    cudaFree(error_device);
}


void CudaAdaBoost::fit(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n)
{
    int part_size = n/number_trees;
    std::vector<float> data_weights(n);
    std::vector<float> P(n*classes);

    std::fill(data_weights.begin(), data_weights.end(), 1.0f / n);
    
    for(int i = 0; i<number_trees; i++){
        int current_size = (int)min(n - (long)i, (long)part_size);
        trees[i].fit(X_begin+(i*part_size)*features, Y_begin+(i*part_size)*classes,current_size);
    }

    for(int i = 0; i<number_trees; i++){
        trees[i].predict(X_begin, P.begin(), n);
        changed_weights(P.begin(), Y_begin, data_weights.begin(), &tree_weights[i], classes, n);
    }
}

void CudaAdaBoost::predict(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n)
{
    std::vector<float> current(n*classes);
    
    for(int i = 0; i<number_trees; i++){
        trees[i].predict(X_begin, current.begin(), n);
        for(int j = 0; j<n*classes; j++){
            *(P_begin + j) += current[j] * tree_weights[i] / number_trees;
        }
    }

}

float CudaAdaBoost::accuracy(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n)
{
    std::vector<float> P(n*classes);
    predict(X_begin, P.begin(), n);
    unsigned long correct = 0;
    for(int i = 0; i<n; i++){
       // if(i%10==0)std::cout<<i<<std::endl;
        int max_tr = 0; 
        int max_pr = 0;
        for(int c = 1; c<classes; c++){
            if(*(P.begin() + i*classes+max_pr)<=*(P.begin() + i*classes+c))max_pr=c;
            if(*(Y_begin + i*classes+max_tr)<=*(Y_begin + i*classes+c))max_tr=c;
        }
        if(max_pr == max_tr) correct += 1;
    }
    return (float)correct / (float)n;
}

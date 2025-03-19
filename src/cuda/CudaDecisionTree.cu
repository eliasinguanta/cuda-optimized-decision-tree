
#include "CudaDecisionTree.h"

#define MAX_LEAF_NUM_CONSTANT 512
#define MAX_INNERNODE_NUM_CONSTANT 512 - 1
#define MAX_CLASSES_CONSTANT 6

__constant__ int decisionDim_constant[MAX_INNERNODE_NUM_CONSTANT];
__constant__ float decisionValue_constant[MAX_INNERNODE_NUM_CONSTANT];
__constant__ float distributions_constant[MAX_LEAF_NUM_CONSTANT * MAX_CLASSES_CONSTANT];

__global__ void prediction_no_constant(float* X, float* P, int* decisionDim, float* decisionValue, float* distributions, int features, int classes, int n, int max_innernode_num){
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < n){
        int index;
        for(index = 0; index < max_innernode_num; index = (X[id * features + decisionDim[index]] <= decisionValue[index])?(2*index+1):(2*index+2));
        for(unsigned int k = 0; k < classes; k++){
            P[id*classes + k] = distributions[(index - max_innernode_num)*classes+k];
        }
    }
}

__global__ void predict_constant(float* X, float* P, int features, int classes, int n, int max_innernode_num){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < n){

        //find correct leaf
        int index;
        for(index = 0; index < max_innernode_num; index = (X[id * features + decisionDim_constant[index]] <= decisionValue_constant[index])?(2*index+1):(2*index+2));
        
        //load leaf distribution
        for(unsigned int k = 0; k < classes; k++){
            P[id*classes + k] = distributions_constant[(index - max_innernode_num)*classes+k];
        }
    }
}
void CudaDecisionTree::predict_with_constant_mem(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n){

    check_cuda(cudaMemcpyToSymbol(decisionDim_constant, decision_features.data(), max_innernode_num * sizeof(int)));
    check_cuda(cudaMemcpyToSymbol(decisionValue_constant, decision_values.data(), max_innernode_num * sizeof(float)));
    check_cuda(cudaMemcpyToSymbol(distributions_constant, distributions.data(), max_leaf_num * classes * sizeof(float)));

    constexpr int threads = 32;
    int part_size = (1<<20)/(features + classes);
    int blocks = (part_size + threads - 1)/threads;

    float* X_device;
    float* P_device;
    check_cuda(cudaMalloc((void**)&X_device, part_size * features * sizeof(float)));
    check_cuda(cudaMalloc((void**)&P_device, part_size * classes * sizeof(float)));

    for(int offset = 0; offset<n; offset+=part_size){

        int current_size = (int)min(n - ((long)offset), (long)part_size);
        check_cuda(cudaMemcpy(X_device, &(*(X_begin + offset * features)), current_size * sizeof(float) * features, cudaMemcpyHostToDevice));
        predict_constant<<<blocks, threads>>>(X_device, P_device, features, classes, current_size, max_innernode_num);
        check_cuda(cudaMemcpy(&(*(P_begin + offset * classes)), P_device, current_size * sizeof(float) * classes, cudaMemcpyDeviceToHost));
    }
    check_cuda(cudaFree(X_device));
    check_cuda(cudaFree(P_device));
}

void CudaDecisionTree::predict_with_out_constant_mem(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n){
    constexpr int threads = 32;
    int part_size = (1<<20)/(features + classes);
    int blocks = (part_size + threads - 1)/threads;

    int* decisionDim_device;
    float* decisionValue_device; 
    float* distributions_device;
    float* X_device;
    float* P_device;

    check_cuda(cudaMalloc((void**)&decisionDim_device, max_innernode_num * sizeof(int)));
    check_cuda(cudaMalloc((void**)&decisionValue_device, max_innernode_num * sizeof(float)));
    check_cuda(cudaMalloc((void**)&distributions_device, max_leaf_num * classes * sizeof(float)));
    check_cuda(cudaMalloc((void**)&X_device, part_size * features * sizeof(float)));
    check_cuda(cudaMalloc((void**)&P_device, part_size * classes * sizeof(float)));

    check_cuda(cudaMemcpy(decisionDim_device, decision_features.data(),  max_innernode_num * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(decisionValue_device, decision_values.data(), max_innernode_num * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(distributions_device, distributions.data(), max_leaf_num * classes * sizeof(float), cudaMemcpyHostToDevice));

    for(int offset = 0; offset<n; offset+=part_size){
        int current_size = (int)min(n - ((long)offset), (long)part_size);
        check_cuda(cudaMemcpy(X_device, &(*(X_begin + offset * features)), current_size * sizeof(float) * features, cudaMemcpyHostToDevice));
        prediction_no_constant<<<blocks,threads>>>(X_device, P_device, decisionDim_device, decisionValue_device, distributions_device, features, classes, current_size, max_innernode_num);
        check_cuda(cudaMemcpy(&(*(P_begin + offset * classes)), P_device, current_size * sizeof(float) * classes, cudaMemcpyDeviceToHost));
    }

    check_cuda(cudaFree(decisionDim_device));
    check_cuda(cudaFree(decisionValue_device));
    check_cuda(cudaFree(distributions_device));
    check_cuda(cudaFree(X_device));
    check_cuda(cudaFree(P_device));
}

void CudaDecisionTree::predict(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n)
{
    if(max_leaf_num <= MAX_LEAF_NUM_CONSTANT && max_innernode_num <= MAX_INNERNODE_NUM_CONSTANT && classes <= MAX_CLASSES_CONSTANT){
        predict_with_constant_mem(X_begin, P_begin, n);
    }else{
        predict_with_out_constant_mem(X_begin, P_begin, n);
    }
}

void CudaDecisionTree::fit_rek(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n, unsigned int index)
{
    if(index < max_innernode_num){
        if(n < 2){
            decision_features[index]=0;
            decision_values[index]=__FLT_MAX__;
            fit_rek(X_begin, Y_begin, n, 2*index+1);
            fit_rek(X_begin, Y_begin, 0, 2*index+2);
        }else{
            SplitPoint split_point;
            split_node_small_data(X_begin, Y_begin, features, classes, n, &split_point);
            decision_features[index]=split_point.feature;
            if(split_point.index == n - 1 )decision_values[index] = __FLT_MAX__;
            else decision_values[index] = *(X_begin + split_point.index*features+split_point.feature);
            fit_rek(X_begin, Y_begin, split_point.index+1, (index+1)*2-1);
            fit_rek(X_begin + (split_point.index+1)*features, Y_begin + (split_point.index+1)*classes, n - (split_point.index+1), (index+1)*2);            
        }
    }else{
        if(n==0){
            for(unsigned int i = 0; i<classes; i++){
                distributions[(index - max_innernode_num)*classes + i] = 1.0f/classes;
            }
        }else{
            for(unsigned int j = 0; j<classes; j++){
                distributions[(index - max_innernode_num)*classes+j] = 0.0f;
            }
            for(unsigned int i = 0; i<n; i++){
                for(unsigned int j = 0; j<classes; j++){
                    distributions[(index - max_innernode_num)*classes+j] += *(Y_begin + i*classes + j);
                }
            }
            for(unsigned int j = 0; j<classes; j++){
                distributions[(index - max_innernode_num)*classes+j] /= n;
            }
        }
    }
}

void CudaDecisionTree::fit(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n)
{
    fit_rek(X_begin, Y_begin, n, 0);
}

float CudaDecisionTree::accuracy(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n)
{
    std::vector<float> P(n*classes);//buffer
    predict(X_begin, P.begin(), n);

    unsigned int correct = 0;

    for(unsigned int i = 0; i < n; i++){
        unsigned int max_gt = 0;
        unsigned int max_pr = 0;
        auto Y_dist = Y_begin + i*classes;
        auto P_dist = P.begin() + i*classes;
        for(unsigned int j = 0; j<classes; j++){
            if(*(Y_dist + j)>= *(Y_dist + max_gt))max_gt=j;
            if(*(P_dist + j)>= *(P_dist + max_pr))max_pr=j;
        }
        if(max_gt==max_pr) correct++;
    }

    return (float)correct/n;
}

void CudaDecisionTree::printTreeParameter()
{
    std::cout<<"Nodes: ";
    for(int i = 0; i<CudaDecisionTree::max_innernode_num; i++){
        std::cout<<" ("<<i<<"  "<<decision_values[i]<<"  "<<decision_features[i]<<")";
    }
    std::cout<<std::endl;

    std::cout<<"Leafs: ";
    for(int i = 0; i<CudaDecisionTree::max_leaf_num; i++){
        int max = 0;
        for(int j = 1; j<CudaDecisionTree::classes; j++){
            if(CudaDecisionTree::distributions[i*classes+j]>CudaDecisionTree::distributions[i*classes+max])max = j;
        }
        std::cout<<" "<<max;
    }
    std::cout<<std::endl;
}

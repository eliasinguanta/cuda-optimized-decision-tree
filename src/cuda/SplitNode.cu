#include "SplitNode.h"

#define DEBUG 0


__global__ void init_split_point(SplitPoint* split_point){
    if(threadIdx.x + blockDim.x * blockIdx.x == 0){
        split_point->evaluation = FLT_MAX;
        split_point->index = 0;
        split_point->feature = 0;    
    }
}
__global__ void update_split_point(SplitPoint* split_point, int* best_indicies_device, float* G_buffer, int selected_feature){
    if(threadIdx.x + blockDim.x * blockIdx.x == 0){
        if(split_point->evaluation > G_buffer[best_indicies_device[0]]){
            split_point->evaluation = G_buffer[best_indicies_device[0]];
            split_point->index = best_indicies_device[0];
            split_point->feature = selected_feature;
        }
    }
}
__host__ void split_node_host(float * X_old, float * Y_old, float * X_new, float * Y_new, float* X_row, int* I, float * G_buffer, int* best_indicies_device, int features, int classes, long n, SplitPoint* split_point){

    int l; for(l = 1; (1<<l)<n; l++); //l is the smallest potenz of 2 bigger or equal to n
    
    //standart config for musing all sm if the code logic allows it
    constexpr int threads = 32;
    int blocks = ((1<<l)+threads-1) / threads;
    
    //reduce config respects the coersing_factor
    int rblocks  = (n + (threads*coersing_factor) - 1) / (threads*coersing_factor);

    init_split_point<<<1, 1>>>(split_point);
    for(int selected_feature = 0; selected_feature < features; selected_feature++){

        //sort the data set with respect to the selected feature
        init_host<<<blocks, threads>>>(X_old, X_row, I, features, n, l, selected_feature);
        sort_indicies(X_row, I, l);
        apply_index_host<<<blocks, threads>>>(X_old, Y_old, X_new, Y_new, I, features, classes, n);
        check_cuda(cudaMemcpy(X_old, X_new, n*features*sizeof(float), cudaMemcpyDeviceToDevice));
        check_cuda(cudaMemcpy(Y_old, Y_new, n*classes*sizeof(float), cudaMemcpyDeviceToDevice));

        //computes the absolut class distribution (Y_new is one hot codet)
        praefix_sum_small_data(Y_new, classes, n);

        //compute evaluation
        float* L = Y_new + (n-1) * classes;
        evaluate_distributions_small_data(Y_new, G_buffer, L, classes, n, 0,  n);
        handle_duplicates(G_buffer, X_row, n); //one hard to find edge case

        //find index of the minimum evaluation
        reduce_index_of_min_blockwise<<<rblocks, threads>>>(n, G_buffer, best_indicies_device);
        reduce_index_of_min_block_results<<<1, 1024>>>(rblocks, G_buffer, best_indicies_device);
        
        update_split_point<<<1,1>>>(split_point, best_indicies_device, G_buffer, selected_feature);

    }

    //apply best split point
    init_host2<<<blocks, threads>>>(X_old, X_row, I, features, n, l, split_point);
    sort_indicies(X_row, I, l);
    apply_index_host<<<blocks, threads>>>(X_old, Y_old, X_new, Y_new, I, features, classes, n);
    check_cuda(cudaMemcpy(X_old, X_new, n*features*sizeof(float), cudaMemcpyDeviceToDevice));
    check_cuda(cudaMemcpy(Y_old, Y_new, n*classes*sizeof(float), cudaMemcpyDeviceToDevice));
}

__host__ void split_node_small_data(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, SplitPoint* split_point_host){
    int l; for(l = 1; (1<<l)<n; l++);
    constexpr int threads = 32;
    int blocks = ((1<<l)+threads-1) / threads;

    float* X_old_device;
    float* Y_old_device;
    float* X_new_device;
    float* Y_new_device;
    float* X_row;
    int* I_device;
    float* G_buffer;
    int* best_index_device;
    SplitPoint* split_point_device;

    check_cuda(cudaMalloc((void**)&X_old_device, n * features * sizeof(float)));
    check_cuda(cudaMalloc((void**)&Y_old_device, n * classes * sizeof(float)));
    check_cuda(cudaMalloc((void**)&X_new_device, n * features * sizeof(float)));
    check_cuda(cudaMalloc((void**)&Y_new_device, n * classes * sizeof(float)));
    check_cuda(cudaMalloc((void**)&X_row, (1<<l) * sizeof(float)));
    check_cuda(cudaMalloc((void**)&I_device, (1<<l) * sizeof(int)));
    check_cuda(cudaMalloc((void**)&G_buffer, n * sizeof(float)));
    check_cuda(cudaMalloc((void**)&best_index_device, blocks * sizeof(int)));

    check_cuda(cudaMalloc((void**)&split_point_device, sizeof(SplitPoint)));

    check_cuda(cudaMemcpy(X_old_device, &(*X_begin), n * sizeof(float) * features, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(Y_old_device, &(*Y_begin), n * sizeof(float) * classes, cudaMemcpyHostToDevice));
    
    split_node_host(X_old_device, Y_old_device, X_new_device, Y_new_device, X_row, I_device, G_buffer, best_index_device, features, classes, n, split_point_device);

    check_cuda(cudaMemcpy(split_point_host, split_point_device, sizeof(SplitPoint), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&(*X_begin), X_old_device, n * sizeof(float) * features, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&(*Y_begin), Y_old_device, n * sizeof(float) * classes, cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(X_old_device));
    check_cuda(cudaFree(Y_old_device));
    check_cuda(cudaFree(X_new_device));
    check_cuda(cudaFree(Y_new_device));
    check_cuda(cudaFree(X_row));
    check_cuda(cudaFree(I_device));
    check_cuda(cudaFree(G_buffer));
    check_cuda(cudaFree(split_point_device));

}

__host__ void split_node_big_data(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, SplitPoint* split_point)
{

    split_point->feature = 0;
    split_point->index = 0;
    split_point->evaluation = FLT_MAX;

    std::vector<float> Y_buffer(n*classes);
    std::vector<float> G(n);


    for(int selected_feature = 0; selected_feature < features; selected_feature++){
        sort_big_data(X_begin, Y_begin, features, classes, n, selected_feature);
        std::copy(Y_begin, Y_begin + n*classes, Y_buffer.begin());
        praefix_sum_big_data(Y_buffer.begin(), classes, n);
        evaluate_distributions_big_data(Y_buffer.begin(), G.begin(), classes, n);

        long index = reduce_index_of_min_big_data(G);
       // std::cout<<"G[index]: " <<G[index]<<", index: "<<index<<std::endl;
        if(G[index]<split_point->evaluation){
            split_point->feature = selected_feature;
            split_point->index = index;
            split_point->evaluation = G[index];
        }
    }
    sort_big_data(X_begin, Y_begin, features, classes, n, split_point->feature);
}

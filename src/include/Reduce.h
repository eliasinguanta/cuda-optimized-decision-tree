#ifndef REDUCE_H
#define REDUCE_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cfloat>
#include <cassert>


#define coersing_factor 3

//************************************************//
//               Split Node functions             //
//************************************************//

/**
 * @brief find the (highest) index of the value in data that is minimal
 * 
 * @param n size of data
 * @param data 
 * @return index
 */
__device__ int reduce_index_of_min(int n, float* data);

/**
 * @brief same as reduce_index_of_min but with tiling and on host
 * 
 * @param data 
 * @return index
 */
long reduce_index_of_min_big_data(std::vector<float> data);

/**
 * @brief computes the minimum of different parts with a block each
 * 
 * @param n size of array
 * @param array 
 * @param best_indicies buffer for the result
 */
__global__ void reduce_index_of_min_blockwise(int n, float* array, int* best_indicies);

/**
 * @brief has to run with only one block.
 * Used to find the minium of the results of reduce_index_of_min_blockwise
 * 
 * @param blocks that were used in reduce_index_of_min_blockwise (or in other words size of best_indicies)
 * @param array contains the values ​​referenced in best indices 
 * @param best_indicies contains the results of reduce_index_of_min_blockwise
 */
__global__ void reduce_index_of_min_block_results(int blocks, float* array, int* best_indicies);

//************************************************//
//               AdaBoost functions               //
//************************************************//

/**
 * @brief sums up errors in different parts or error_device with a block each
 * 
 * @param P predictions
 * @param Y true labels
 * @param D data weights
 * @param error_device result buffer
 * @param classes 
 * @param n 
 */
__global__ void reduce_sum_of_error_blockwise(float* P, float* Y, float* D, float* error_device, int classes, int n);

/**
 * @brief sums up the results of reduce_sum_of_error_blockwise
 * 
 * @param blocks used in reduce_sum_of_error_blockwise (size of error_device)
 * @param error_device results of the blocks
 */
__global__ void reduce_sum_of_error(int blocks, float* error_device);

/**
 * @brief sums up different parts of an array
 * 
 * @param array 
 * @param n 
 */
__global__ void reduce_sum_blockwise(float* array, int n);

/**
 * @brief sums up the results of reduce_sum_blockwise
 * 
 * @param blocks used in reduce_sum_blockwise
 * @param array 
 */
__global__ void reduce_sum(int blocks, float* array);

#endif
#ifndef SORT_H
#define SORT_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cfloat>
#include "Utils.h"
#include <cassert>
//************************************************//
//               Split Node functions             //
//************************************************//

/*
 * sorts X_row but keeps track of the indicies in I. in this way we can sort a our dataset without
 * swapping the whole data point in every step and do it instead once at the end
 * sorts with bitonic sort in n*log(n)^2 steps with n/2 threads at the same time (if hardware allows it so for n<= 2048)
 */
__host__ void sort_indicies(float* X_row, int* I, int l);

/*
 * It takes a row from our dataset, where each data point is represented by a column.
 * Fills an array with indicies 0 -> number of data points
 */
__global__ void init_host(float* X, float* X_row, int* I, int features, int n, int l, int selected_feature);

/*
 * does the same as init_host but takes the selected_feature from split_point
 */
__global__ void init_host2(float* X, float* X_row, int* I, int features, int n, int l, SplitPoint* selected_feature);

/*
 * the array I contains the new order of our data set now we apply the new order to the data set (that has not been touched until now)
 */
__global__ void apply_index_host(float* X_old, float* Y_old, float* X_new, float* Y_new, int* I, int features, int classes, int n);


/**
* This function is currently only used for testing.
* Made for data that is to big to fit on the gpu it is sortet partially on gpu and merged on cpu with merge sort
*/
//
__host__ void sort_big_data(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, int selected_feature);
#endif
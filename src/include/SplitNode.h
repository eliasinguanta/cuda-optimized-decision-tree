#ifndef SPLIT_NODE_H
#define SPLIT_NODE_H


#include "BitonicSort.h"
#include "Map.h"
#include "PraefixSum.h"
#include "Reduce.h"

/**
 * @brief we load the data once one the gpu, try every split point for every feature, split the data set,
 * evaluate and apply the best one (according to gini or logloss) and load it back once
 * 
 * @param X_begin 
 * @param Y_begin 
 * @param features 
 * @param classes 
 * @param n data points in the given data set
 * @param split_point buffer for the result
 */
void split_node_small_data(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, SplitPoint* split_point);

/**
 * @brief basicly like split_node_small_data but with tiling so more memory exchange between CPU and GPU
 * 
 * @param X_begin 
 * @param Y_begin 
 * @param features 
 * @param classes 
 * @param n data points in the given data set
 * @param split_point buffer for the result
 */
void split_node_big_data(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, int features, int classes, long n, SplitPoint* split_point);

#endif
#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <iostream>
#include <cmath>
#include <stdlib.h>
#include "vector"
#include "SplitNode.h"
#include <cassert>


class CudaDecisionTree{
private:

    unsigned int max_tree_size;// (2 << depth) - 1;
    unsigned int max_leaf_num;// (1 << depth);
    unsigned int max_innernode_num;// (1 << depth) - 1;

    std::vector<int> decision_features; //contains the features of the nodes
    std::vector<float> decision_values; //contains the split-values of the nodes
    std::vector<float> distributions; //contains the leaf distributions

    /**
     * @brief predicts test data on the gpu but has some size conditions because it uses constant memory
     * 
     * @param X_begin test data
     * @param P_begin result buffer
     * @param n size of test data
     */
    void predict_with_constant_mem(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n);
    
    /**
     * @brief predicts test data on the gpu
     * 
     * @param X_begin test data
     * @param P_begin result buffer
     * @param n number of data points
     */
    void predict_with_out_constant_mem(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n);

    /**
     * @brief the fit function of our model. Builds the decision tree so fills
     * decision_features, decision_values and distributions
     * 
     * @param X_begin train data
     * @param Y_begin train label
     * @param n number of data points
     * @param index 
     */
    void fit_rek(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n, unsigned int index);

public:
    unsigned int features;
    unsigned int classes;
    unsigned int depth;
    
    /**
     * @brief wrapper for the real fit function of our model. 
     * 
     * @param X_begin train data
     * @param Y_begin train label
     * @param n number of data points
     */
    void fit(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n);

    /**
     * @brief predicts test data
     * 
     * @param X_begin test data
     * @param P_begin result buffer
     * @param n data points in test data
     */
    void predict(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n);

    CudaDecisionTree(unsigned int dimensions, unsigned int classes, unsigned int depth):features(dimensions),classes(classes),depth(depth),max_tree_size((2 << depth) - 1),max_leaf_num(1<<depth),max_innernode_num((1<<depth)-1){
        decision_features.resize(max_innernode_num);
        decision_values.resize(max_innernode_num);
        distributions.resize(max_leaf_num * classes);
    };

    /**
     * @brief computes (correct predictions) / (all predictions)
     * 
     * @param X_begin test data
     * @param Y_begin test labels
     * @param n number of data points
     * @return accuracy
     */
    float accuracy(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n);
    
    /**
     * @brief for debugging has format (node index, decision_values, decision_features) for nodes and prediction class for leafs
     * 
     */
    void printTreeParameter();

};

#endif
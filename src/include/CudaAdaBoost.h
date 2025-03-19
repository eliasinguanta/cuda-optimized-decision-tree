#include "CudaDecisionTree.h"

#ifndef ADA_BOOST_H
#define ADA_BOOST_H

class CudaAdaBoost{
public:
    unsigned int features;
    unsigned int classes;
    unsigned int depth;
    unsigned int number_trees;
    
    std::vector<CudaDecisionTree> trees;
    std::vector<float> tree_weights;

    /**
     * @brief trains the classifer
     * 
     * @param X_begin train data
     * @param Y_begin train label
     * @param n number of data points in train data set
     */
    void fit(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n);
    
    /**
     * @brief predicts labels
     * 
     * @param X_begin test data
     * @param P_begin buffer for results
     * @param n number of data points in test data set
     */
    void predict(std::vector<float>::iterator X_begin, std::vector<float>::iterator P_begin, unsigned long n);

    CudaAdaBoost(unsigned int dimensions, unsigned int classes, unsigned int depth, unsigned int number_of_base_classifier):features(dimensions),classes(classes),depth(depth),number_trees(number_of_base_classifier){
        trees.reserve(number_of_base_classifier);
        for (unsigned int i = 0; i < number_of_base_classifier; ++i) {
            trees.emplace_back(dimensions, classes, depth);
        }
    
        tree_weights.resize(number_of_base_classifier, 1.0f);
    };


    /**
     * @brief computes (correct predictions) / (all predictions)
     * 
     * @param X_begin test data
     * @param Y_begin test label
     * @param n number of data points in test data
     * @return accuracy 
     */
    float accuracy(std::vector<float>::iterator X_begin, std::vector<float>::iterator Y_begin, unsigned long n);


};

#endif

#ifndef MAP_H
#define MAP_H

#include <vector>

//************************************************//
//               Split Node functions             //
//************************************************//

/**
 * @brief Currently only used for testing
 * 
 * @param Y_begin absolut class distribution of left site
 * @param G_begin buffer for results
 * @param classes 
 * @param n 
 */
void evaluate_distributions_big_data(std::vector<float>::iterator Y_begin, std::vector<float>::iterator G_begin, int classes, long n);

/**
 * @brief recieves absolut class distributions and computes a metrix like gini or multi-logloss
 * absolut_distributions is a flat array that contains the absolut class distribution of the data set 
 * that is on the left site (placed after sorting)
 * 
 * @param absolut_distributions absolut class distribution of left site
 * @param result_buffer
 * @param L last distribution, with absolut_distributions we can compute the absolut class distribution of right site.
 * @param classes 
 * @param n 
 * @param offset global position of the current absolut_distributions-part if tiling is used
 * @param real_n global size of absolut_distributions if tiling is used
 */
void evaluate_distributions_small_data(float* absolut_distributions, float* result_buffer, float* L, int classes, int n, int offset, int real_n);

/**
 * @brief computes gini or logloss of one distribution
 * 
 * @param absolut_distributions is a distributen (so #absolut_distributions == classes)
 * @param result_buffer (just one float)
 * @param L is the last distribution
 * @param classes 
 * @param part_size size of absolut_distributions-part (if we dont use tiling its equal to n)
 * @param offset global position of the current absolut_distributions-part if tiling is used
 * @param n global size of absolut_distributions if tiling is used
 * @return gini or logloss but writes it in result_buffer 
 */
__device__ void evaluate_distribution(float* absolut_distribution, float* result_buffer, float* last_absolut_distribution, int classes, int part_size, int offset, int n);

/**
 * @brief if we have duplicates in X_row we can get errors in our evaluations. that ist corrected here
 * 
 * @param evaluations of class distributions
 * @param X_row row of our data set where each data point is a column
 * @param n size of evaluations
 */
void handle_duplicates(float* evaluations, float* X_row, int n);


//************************************************//
//               AdaBoost functions               //
//************************************************//

/**
 * @brief we just apply x = x/sum
 * 
 * @param array
 * @param n size of array
 * @param sum 
 */
__global__ void divide_by_sum(float* array, int n, float sum);

/**
 * @brief we update the data weights in the AdaBoost fit function. 
 * data_weight *= (wrong predicted) ? exp(-alpha) : exp(alpha);
 * 
 * @param P predictions
 * @param Y true labels
 * @param D data weights
 * @param alpha weight of the decision tree that is used for the prediction
 * @param classes 
 * @param n number of data points
 */
__global__ void compute_data_weights(float* P, float* Y, float* D, float alpha, int classes, int n);
#endif
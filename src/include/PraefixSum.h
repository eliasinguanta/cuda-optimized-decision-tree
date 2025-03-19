#ifndef PRAEFIX_H
#define PRAEFIX_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include "Utils.h"

#define PRAEFIX_THREADS 1024
#define PRAEFIX_MAX_N 1<<24

//************************************************//
//               Split Node functions             //
//************************************************//

/**
 * @brief computes the präfix sum of each class so for each row in the data set Y where a
 * one hot encoded distribution is a column
 * 
 * @param Y labels (one hot encoded) of the (train-) data set
 * @param classes 
 * @param n number of labels in Y
 */
void praefix_sum_small_data(float * Y, int classes, int n);

/**
 * @brief same as praefix_sum_small_data just with tiling. was used for testing
 * 
 * @param Y_begin labels (one hot encoded) of the (train-) data set
 * @param classes 
 * @param n number of labels in Y
 */
void praefix_sum_big_data(std::vector<float>::iterator Y_begin, int classes, long n);

#endif
# CART Decision Tree and AdaBoost with CUDA

## **Overview**
This project implements a CART Decision Tree in C++, utilizing CUDA for split-point calculations. AdaBoost applies the Decision Tree.

## **Functionality**
- **CART Decision Tree:**
  - Identifies the best split point for each feature by:
    - Sorting data based on the selected feature (Bitonic Sort)
    - Computes class distributions using präfix sum.
    - Evaluates splits using log-loss.
    - Finds the best split with a reduce operation.

  - Repeats the process for all features.
  - Designed as a simple CART implementation without optimizations or regularization.

- **AdaBoost:**
  - Splits data into equally sized subsets.
  - Trains a Decision Tree for each subset.
  - Computing the tree weights and data weights is also Cuda optimized
  
## **Execution**
### **Requirements:**
- Nothing. I hardcoded the small data-sets to make it really easy to run the tests.
- I implemented and tested it in GCP with following config
    - n1-standard-1 (1 vCPUs, 3.75 GB Memory)
    - NVIDIA T4
    - Deep Learning VM with CUDA 12.3 M128


### **Compile and Run:**
```bash
make
./Test
```


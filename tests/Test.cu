#include "../src/include/BitonicSort.h"
#include "../src/include/PraefixSum.h"
#include "../src/include/Reduce.h"
#include "../src/include/SplitNode.h"
#include "../src/include/CudaDecisionTree.h"
#include "../src/include/CudaAdaBoost.h"
#include "Data.h"
#include <cassert>
#include <chrono>
#include <string>


void test_reduce_min_on_big_data(long max_n){
    for(int t = 0; t < 10; t++){
        long n = prng(t)%max_n+2;

        std::vector<float> array_host;
        array_host.reserve(n);
        for(long i = 0; i<n; i++){
            array_host.push_back((float)(prng(i + prng(t))%100 + 10));
        }

        long best_index = reduce_index_of_min_big_data(array_host);

        for(long i = 0; i<n; i++){
            assert(array_host[i]>=array_host[best_index]);
        }
        
    }
    std::cout<<"Reduce test successfull " <<std::endl;
}


//######-----------Test sorting-----------######

void test_sort_on_big_data(long max_n){
    
    for(int t = 0; t< 10; t++){
        long n = (prng(t)%max_n);
        int features = prng(t)%10+1;
        int classes = prng(t*t)%10+2;
        std::vector<float> X;
        std::vector<float> Y;
        X.reserve(n * features);
        Y.reserve(n * classes);
        int selected_feature = prng(t*t*t)%features;

        for(long i = 0; i<n; i++){
            for(int j = 0; j < features; j++){
                if(j == selected_feature) X.push_back((float)(prng(t+i)%1000));
                else X.push_back(0.0);
            }
            for(int j = 0; j < classes; j++){
                Y.push_back(0.0);
            }
        }
        sort_big_data(X.begin(), Y.begin(), features, classes, n, selected_feature);
        for(long i = 1; i<n; i++){
            assert(X[i*features+selected_feature]>=X[(i-1)*features+selected_feature]);
        }
        
    }

    std::cout<<"Sort test successfull " <<std::endl;
}

//######-----------Test präfix-----------######

void test_praefix_sum_on_big_data(long max_n){
    for(int t = 0; t< 10; t++){
        long n = prng(t)%max_n;
        int classes = prng(t*t)%10+2;
        std::vector<float> Y;

        for(long i = 0; i<n; i++){
            for(int j = 0; j < classes; j++){
                Y.push_back(1.0);
            }
        }
        praefix_sum_big_data(Y.begin(), classes, n);
        for(long i = 0; i<n; i++){
            for(int c = 0; c<classes; c++){
                assert(Y[i*classes+c]==i+1);
            }
        }
    }
    std::cout<<"Praefix-Sum test successfull " <<std::endl;
}

//######-----------Test SplitNode-----------######
void test_split_node(long max_n){

    std::vector<float> X;
    std::vector<float> Y;

    for(int t = 0; t< 10; t++){
        long n = prng(t)%max_n + 10;//2048<<16; //13 millionen
        X.clear();
        Y.clear();
        int features = prng(t)%10+1;
        int classes = prng(t*t)%10+2;
        int selected_feature = prng(t*t*t)%features;
        int selected_class = prng(2*t*t)%classes;
        int max_value = prng(t)%1000;
        float selected_threshold = max_value/2;

        int count = 0;
        for(long i = 0; i<n; i++){
            for(int j = 0; j < features; j++){
                X.push_back((float)(prng(t+prng(i+prng(j)))%max_value));
            }
            for(int j = 0; j<classes; j++){
                if(X[i*features+selected_feature] > selected_threshold && j == selected_class) Y.push_back(1.0);
                else if(X[i*features+selected_feature] <= selected_threshold && j == (selected_class + 1)%classes) {
                    Y.push_back(1.0);
                    count++;
                }
                else Y.push_back(0.0);
            }
        }
 
        SplitPoint res;
  
        split_node_small_data(X.begin(), Y.begin(), features, classes, n, &res);

        assert(res.feature == selected_feature);
        assert(res.index == count-1);
        assert(res.evaluation == 0); //we construct it perfect classifiable with one split
    }
    std::cout<<"Split-Node test successfull " <<std::endl;
}

void test_cuda_decision_tree(std::string data_set_name, int n, int features, int classes, float* X, float* Y){
    constexpr int depth = 6;
    const float train_factor = 0.8;

    std::cout<<"######---------- Decision Tree "<<data_set_name<<" ----------######"<<std::endl;
    std::cout<<"n "<<n<<", features: "<<features<<", classes: "<<classes<<std::endl;
    std::cout<<"tree depth: "<<depth<<std::endl;

    int train_size = (int)(train_factor * n);
    int test_size = n - train_size;
    std::vector<float> X_train(X, X + train_size * features);
    std::vector<float> Y_train(Y, Y + train_size * classes);
    std::vector<float> X_test(X, X + test_size * features);
    std::vector<float> Y_test(Y, Y + test_size * classes);

    CudaDecisionTree tree(features, classes, depth);

    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(X_train.begin(), Y_train.begin(), train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


    float acc = tree.accuracy(X_test.begin(), Y_test.begin(), test_size);
    std::cout<<"tree accuracy: "<<acc<<std::endl;
    std::cout<<"tree time for training: "<<duration.count() <<" ms"<<std::endl;
    //tree.printTreeParameter();
    std::cout<<std::endl;
}
void shuffle(std::vector<float>& X, std::vector<float>& Y, int features, int classes, long n){
    for(int i = 0; i<n; i++){
        int s = prng(i)%n;
        for(int j = 0; j<features; j++){
            std::swap(X[i*features+j], X[s*features+j]);
        }
        for(int j = 0; j<classes; j++){
            std::swap(Y[i*classes+j], Y[s*classes+j]);
        }
    }
}

void test_cuda_adaboost(std::string data_set_name, int n, int features, int classes, float* X, float* Y){
    const int depth = 2;
    const int trees = 4;
    const float train_factor = 0.8;

    std::cout<<"######---------- AdaBoost test "<<data_set_name<<" ----------######"<<std::endl;
    std::cout<<"n "<<n<<", features: "<<features<<", classes: "<<classes<<std::endl;
    std::cout<<"adaboost with tree depth: "<<depth<<std::endl;
    std::cout<<"adaboost with trees: "<<trees<<std::endl;

    int train_size = (int)(train_factor * n);
    int test_size = n - train_size;
    std::vector<float> X_train(X, X + train_size * features);
    std::vector<float> Y_train(Y, Y + train_size * classes);

    shuffle(X_train,Y_train, features, classes, train_size); //necessary

    std::vector<float> X_test(X, X + test_size * features);
    std::vector<float> Y_test(Y, Y + test_size * classes);

    CudaAdaBoost model(features, classes, depth, trees);

    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X_train.begin(), Y_train.begin(), train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


    float acc = model.accuracy(X_test.begin(), Y_test.begin(), test_size);
    std::cout<<"adaboost accuracy: "<<acc<<std::endl;
    std::cout<<"adaboost time for training: "<<duration.count() <<" ms"<<std::endl;
    std::cout<<std::endl;
}

int main(){

   long TEST_SIZE = 2000;

    test_praefix_sum_on_big_data(TEST_SIZE);
    test_reduce_min_on_big_data(TEST_SIZE);
    test_sort_on_big_data(TEST_SIZE);
    test_split_node(TEST_SIZE);

    test_cuda_decision_tree("iris", iris_n, iris_dimensions, iris_classes, iris_features, iris_targets);
    test_cuda_decision_tree("wine", wine_n, wine_dimensions, wine_classes, wine_features, wine_targets);
    test_cuda_decision_tree("breast cancer", breast_cancer_n, breast_cancer_dimensions, breast_cancer_classes, breast_cancer_features, breast_cancer_targets);
    test_cuda_decision_tree("digits", digits_n, digits_dimensions, digits_classes, digits_features, digits_targets);

    test_cuda_adaboost("iris", iris_n, iris_dimensions, iris_classes, iris_features, iris_targets);
    test_cuda_adaboost("wine", wine_n, wine_dimensions, wine_classes, wine_features, wine_targets);
    test_cuda_adaboost("breast cancer", breast_cancer_n, breast_cancer_dimensions, breast_cancer_classes, breast_cancer_features, breast_cancer_targets);
    test_cuda_adaboost("digits", digits_n, digits_dimensions, digits_classes, digits_features, digits_targets);

}

# executable
TARGET = Test

# compiler and flags
CXX = nvcc
CXXFLAGS = -std=c++17 -O3 -arch=sm_75  -rdc=true -I./src/include -G -g

# source code
SRC = tests/Test.cu src/cuda/Utils.cu src/cuda/Reduce.cu src/cuda/PraefixSum.cu  src/cuda/BitonicSort.cu src/cuda/Map.cu  src/cuda/SplitNode.cu src/cuda/CudaDecisionTree.cu src/cuda/CudaAdaBoost.cu

# rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
CUDA_INSTALL_PATH := /usr/local/cuda
VPATH = ./src/

CXX := g++ -std=c++11 -O3
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc -std=c++11

# Includes
INCLUDES = -I./include -I$(CUDA_INSTALL_PATH)/include
#
# # Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
#
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
OBJS = hvcRA.cu.o hvcI.cu.o main.cu.o make_network.cpp.o poisson_noise.cpp.o
TARGET = exec
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)
#
.SUFFIXES: .c .cpp .cu .o
#
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
#
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
#
%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
#
$(TARGET): $(OBJS) makefile
	$(LINKLINE)

clean:
	rm -f *.o

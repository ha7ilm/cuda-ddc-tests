SOURCE_FILES=main.cpp ddc_cpu.cpp ddc_gpu.cu
CUDAPLACE=/usr/local/cuda
cuda_test: *.cpp *.cu *.h
	$(CUDAPLACE)/bin/nvcc $(SOURCE_FILES) -o cuda-ddc-tests -Xcompiler="-fpermissive"

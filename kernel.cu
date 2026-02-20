    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include <stdio.h>
    #include <chrono>
    #include <iostream>  // Required for cout
    #include <iomanip>   // Required for setw
    #include <string>    // Required for string

    cudaError_t addWithCuda_int(int *c, const int *a, const int *b, unsigned int N); // prototype of the cuda function
    cudaError_t addWithCuda_float(float *c, const float *a, const float *b, unsigned int N); // prototype of the cuda function
    



    // this is a function defined for single thread

    __global__ void addKernel_int(int *c, const int *a, const int *b, int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // the index of this cubic is defined using linear transform, that's reason why we need to calculate as that
        if(idx >= N)
            return;
        c[idx] = a[idx] + b[idx];
    }

    __global__ void addKernel_float(float *c, const float *a, const float *b, int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // the index of this cubic is defined using linear transform, that's reason why we need to calculate as that
        if(idx >= N)
            return;
        c[idx] = a[idx] + b[idx];
    }





    // here we are inside the RAM of pc, so what we define here is the bridge between ram and gpu memory
    int main()
    {
        const int arraySize = 1024;
        int a[arraySize], b[arraySize], c[arraySize] = { 0 }; 
        float a2[arraySize], b2[arraySize], c2[arraySize] = { 0 }; 

        for (int i = 0; i < arraySize; ++i){
            a2[i] = i * (0.25);
            b2[i] = ((arraySize - i)+a2[i])*(0.75);
        }

        for (int i = 0; i < arraySize; ++i){
            a[i] = i*i;
            b[i] = ((arraySize - i)+a[i])*16;
        }


        std::cout << "\n" << std::string(65, '=') << std::endl;
        std::cout << "INT OPERATION" << std::endl;
        std::cout << std::string(65, '=') << std::endl;

        // Add vectors in parallel.
        cudaError_t cudaStatus1 = addWithCuda_int(c, a, b, arraySize);
        if (cudaStatus1 != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus1 = cudaDeviceReset();
        if (cudaStatus1 != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        std::cout << "\n" << std::string(65, '=') << std::endl;
        std::cout << "FLOAT OPERATION" << std::endl;
        std::cout << std::string(65, '=') << std::endl;

        // Add vectors in parallel.
        cudaError_t cudaStatus2 = addWithCuda_float(c2, a2, b2, arraySize);
        if (cudaStatus2 != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus2 = cudaDeviceReset();
        if (cudaStatus2 != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        return 0;
    }

        cudaError_t addWithCuda_float(float *c, const float *a, const float *b, unsigned int size)
    {
        float *dev_a = 0;
        float *dev_b = 0;
        float *dev_c = 0;
        cudaError_t cudaStatus;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // Example of a benchmarking loop for the exercise
        size_t thread_counts[] = { 32, 64, 128, 256, 512, 1024 }; // Multiples of 32 (warp size) 
        size_t num_elements = std::size(thread_counts);
        size_t trials = 200;

        std::cout << std::left << std::setw(10) << "Threads" 
              << "| " << std::setw(12) << "Avg Time(us)" 
              << "| " << std::setw(15) << "Mem BW (GB/s)" 
              << "| " << "Compute (GOPS)" << std::endl;
        std::cout << std::string(65, '-') << std::endl;

        for (int t : thread_counts) {
            size_t N_threads = t;
            dim3 thread_size(N_threads);
            dim3 block_size((size + N_threads - 1) / N_threads); 

            auto start = std::chrono::high_resolution_clock::now();


            for(int i = 0; i < trials; i++) { // Run 100 times for average ì
                addKernel_float<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, size);
            }

            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            double avg_time_s = diff.count() / trials;
            double bytes_moved = 3.0 * size * sizeof(float); // 2 Reads + 1 Write
            double mem_bw_gbs = (bytes_moved / 1e9) / avg_time_s; 
            double compute_gops = (size / 1e9) / avg_time_s;

            std::cout << std::left << std::setw(10) << t 
                  << "| " << std::setw(12) << std::fixed << std::setprecision(2) << (avg_time_s * 1e6)
                  << "| " << std::setw(15) << mem_bw_gbs 
                  << "| " << compute_gops << std::endl;
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        
        return cudaStatus;
    }

    // Helper function for using CUDA to add vectors in parallel.
    cudaError_t addWithCuda_int(int *c, const int *a, const int *b, unsigned int size)
    {
        int *dev_a = 0;
        int *dev_b = 0;
        int *dev_c = 0;
        cudaError_t cudaStatus;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // Example of a benchmarking loop for the exercise
        size_t thread_counts[] = { 32, 64, 128, 256, 512, 1024 }; // Multiples of 32 (warp size) 
        size_t num_elements = std::size(thread_counts);
        size_t trials = 200;

        std::cout << std::left << std::setw(10) << "Threads" 
              << "| " << std::setw(12) << "Avg Time(us)" 
              << "| " << std::setw(15) << "Mem BW (GB/s)" 
              << "| " << "Compute (GOPS)" << std::endl;
        std::cout << std::string(65, '-') << std::endl;

        for (int t : thread_counts) {
            size_t N_threads = t;
            dim3 thread_size(N_threads);
            dim3 block_size((size + N_threads - 1) / N_threads); 

            auto start = std::chrono::high_resolution_clock::now();


            for(int i = 0; i < trials; i++) { // Run 100 times for average ì
                addKernel_int<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, size);
            }

            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            double avg_time_s = diff.count() / trials;
            double bytes_moved = 3.0 * size * sizeof(int); // 2 Reads + 1 Write
            double mem_bw_gbs = (bytes_moved / 1e9) / avg_time_s; 
            double compute_gops = (size / 1e9) / avg_time_s;

            std::cout << std::left << std::setw(10) << t 
                  << "| " << std::setw(12) << std::fixed << std::setprecision(2) << (avg_time_s * 1e6)
                  << "| " << std::setw(15) << mem_bw_gbs 
                  << "| " << compute_gops << std::endl;
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    Error:
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        
        return cudaStatus;
    }

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <iostream>  // Required for cout
#include <iomanip>   // Required for setw
#include <string>    // Required for string
#include <vector>
#include <fstream> // Required for CSV file output



// Update prototypes
cudaError_t addWithCuda_int(std::ofstream& resultsFile, int* c, const int* a, const int* b, unsigned int N);
cudaError_t addWithCuda_float(std::ofstream& resultsFile, float* c, const float* a, const float* b, unsigned int N);




// this is a function defined for single thread

__global__ void addKernel_int(int* c, const int* a, const int* b, int N, int j, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // the index of this cubic is defined using linear transform, that's reason why we need to calculate as that
    int totalThreads = blockDim.x * gridDim.x;

    for (int elem = 0; elem < j; elem++) {
        int i = idx + elem * totalThreads;
        if (i < N) {
            int res = a[i] + b[i];
            for (int loop = 0; loop < k; loop++) {
                res = res + 1;
            }
            c[i] = res;

        }
    };
}

__global__ void addKernel_float(float* c, const float* a, const float* b, int N, int j, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // the index of this cubic is defined using linear transform, that's reason why we need to calculate as that
    int totalThreads = blockDim.x * gridDim.x;

    for (int elem = 0; elem < j; elem++) {
        int i = idx + elem * totalThreads;
        if (i < N) {
            float res = a[i] + b[i];
            for (int loop = 0; loop < k; loop++) {
                res = res + 1.0f;
            }
            c[i] = res;

        }
    }

}





// here we are inside the RAM of pc, so what we define here is the bridge between ram and gpu memory
int main()
{   

    std::ofstream resultsFile("gpu_benchmarks.csv");
    resultsFile << "Datatype,Size,Mapping_j,Intensity_k,Threads,Time_us,BW_GBs,GOPS,ComputeIntensity\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "==========GPU INFO==========\n";
    std::cout << "Name: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Block dim (max threads per block): " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / 1e9 << " GB" << "\n";
    std::cout << "Memory clock rate: " << prop.memoryClockRate << " KHz" << "\n";
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits" << "\n";
    std::cout << "============================\n";

    std::vector<int> sizes{1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    
    for (int arraySize : sizes) {
        std::vector<int> a(arraySize), b(arraySize), c(arraySize, 0);
        std::vector<float> a2(arraySize), b2(arraySize), c2(arraySize, 0.0f);

        for (int i = 0; i < arraySize; ++i) {
            a2[i] = i * 0.25f; b2[i] = ((arraySize - i) + a2[i]) * 0.75f;
            a[i] = i * i; b[i] = ((arraySize - i) + a[i]) * 16;
        }

        std::cout << "\nSize of array: " << arraySize << "\n";
        
        // Pass the file stream to capture data for plotting 
        addWithCuda_int(resultsFile, c.data(), a.data(), b.data(), arraySize);
        addWithCuda_float(resultsFile, c2.data(), a2.data(), b2.data(), arraySize);
    }

    resultsFile.close();
    std::cout << "\nAutomation Complete. Results saved to gpu_benchmarks.csv\n";
    return 0;
}


cudaError_t addWithCuda_float(std::ofstream& resultsFile, float* c, const float* a, const float* b, unsigned int size)
{
    std::vector<int> j_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::vector<int> k_values = {1, 10, 100, 1000, 10000, 100000};

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
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
    size_t thread_counts[] = {128, 256, 1024 }; // Multiples of 32 (warp size) 
    size_t num_elements = sizeof(thread_counts) / sizeof(thread_counts[0]);
    size_t trials = 200;

    

    std::cout << "\n CASE: float \n";
    std::cout << "\n===========================================================\n";
    std::cout << "j | k | Threads | Time(us) | BW(GB/s) | GOPS | CI(ops/Byte)\n";
    std::cout << "==============================================================\n";

    for (int j : j_values) {
        for (int k : k_values) {
            for (int t : thread_counts) {

                size_t N_threads = t;
                dim3 thread_size(t);
                dim3 block_size((size + (t * j) - 1) / (t * j)); 

                auto start = std::chrono::high_resolution_clock::now();


                for (int i = 0; i < trials; i++) { // Run 100 times for average ì
                    addKernel_float << <block_size, thread_size >> > (dev_c, dev_a, dev_b, size, j, k);
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
                double operations = size * (k+1.0f);
                double computeIntensity = operations / bytes_moved;
                double mem_bw_gbs = (bytes_moved / 1e9) / avg_time_s;
                double compute_gops = (operations / 1e9) / avg_time_s;

                // Save to CSV [cite: 151, 165]
                resultsFile << "float," << size << "," << j << "," << k << "," << t << "," 
                            << avg_time_s * 1e6 << "," << mem_bw_gbs << "," << compute_gops << "," << computeIntensity << "\n";

                std::cout << j << " | " << k << " | " << t << " | " << avg_time_s * 1e6 << " | " << mem_bw_gbs << " | " << compute_gops << " | " << computeIntensity << "\n";
            }
        }
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
cudaError_t addWithCuda_int(std::ofstream& resultsFile, int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    std::vector<int> j_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::vector<int> k_values = {1, 10, 100, 1000, 10000, 100000};
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
    size_t thread_counts[] = {32, 64, 128, 256, 1024 }; // Multiples of 32 (warp size) 
    size_t num_elements = sizeof(thread_counts) / sizeof(thread_counts[0]);
    size_t trials = 100;

    

    
    std::cout << "\nCASE: int\n";
    std::cout << "\n=========================================================\n";
    std::cout << "j | k | Threads | Time(us) | BW(GB/s) | GOPS | CI(ops/Byte)\n";
    std::cout << "============================================================\n";

    for (int j : j_values) {
        for (int k : k_values) {
            for (int t : thread_counts) {

                size_t N_threads = t;
                //dim3 thread_size(N_threads);
                //dim3 block_size((size + N_threads - 1) / N_threads);
                // Correct way to launch threads when each thread handles 'j' elements
                dim3 thread_size(t);
                dim3 block_size((size + (t * j) - 1) / (t * j));    
                

                auto start = std::chrono::high_resolution_clock::now();


                for (int i = 0; i < trials; i++) { // Run 100 times for average ì
                    addKernel_int << <block_size, thread_size >> > (dev_c, dev_a, dev_b, size, j, k);
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
                double operations = size * (k + 1);
                double computeIntensity = operations / bytes_moved;
                double mem_bw_gbs = (bytes_moved / 1e9) / avg_time_s;
                double compute_gops = (operations / 1e9) / avg_time_s;

                // Save to CSV [cite: 151, 165]
                resultsFile << "int," << size << "," << j << "," << k << "," << t << "," 
                            << avg_time_s * 1e6 << "," << mem_bw_gbs << "," << compute_gops << "," << computeIntensity << "\n";

                std::cout << j << " | " << k << " | " << t << " | " << avg_time_s * 1e6 << " | " << mem_bw_gbs << " | " << compute_gops << " | " << computeIntensity << "\n";
            }   
        }
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
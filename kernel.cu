#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <iostream>  
#include <iomanip>   
#include <string>    
#include <vector>
#include <fstream> // Required for CSV logging

// Prototypes updated to accept the results file stream 
cudaError_t addWithCuda_double(std::ofstream& resultsFile, double* c, const double* a, const double* b, unsigned int N);
cudaError_t addWithCuda_float(std::ofstream& resultsFile, float* c, const float* a, const float* b, unsigned int N);

// --- KERNELS ---

__global__ void addKernel_double(double* c, const double* a, const double* b, int N, int j, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (int elem = 0; elem < j; elem++) { // Goal 2: Mapping j [cite: 173]
        int i = idx + elem * totalThreads;
        if (i < N) {
            double val_a = a[i];
            double val_b = b[i];
            double res = 0;
            for (int loop = 0; loop < k; loop++) { // Goal 3: Intensity k [cite: 180]
                res = val_a + val_b; 
            }
            c[i] = res;
        }
    }
}

__global__ void addKernel_float(float* c, const float* a, const float* b, int N, int j, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (int elem = 0; elem < j; elem++) {
        int i = idx + elem * totalThreads;
        if (i < N) {
            float val_a = a[i];
            float val_b = b[i];
            float res = 0;
            for (int loop = 0; loop < k; loop++) {
                res = val_a + val_b;
            }
            c[i] = res;
        }
    }
}

// --- MAIN ---

int main()
{
    // Initialize CSV File and Header [cite: 151]
    std::ofstream resultsFile("gpu_benchmarks.csv");
    resultsFile << "Datatype,Size,Mapping_j,Intensity_k,Threads,Time_us,BW_GBs,GOPS,ComputeIntensity\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "==========GPU INFO==========\n";
    std::cout << "Name: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / 1e9 << " GB" << "\n";
    std::cout << "============================\n";

    // Automated sizes from report [cite: 37, 165]
    std::vector<int> sizes{ 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216 };

    for (int arraySize : sizes) {
        std::vector<double> a(arraySize), b(arraySize), c(arraySize, 0.0);
        std::vector<float> a2(arraySize), b2(arraySize), c2(arraySize, 0.0f);

        // Init based on report rules [cite: 28-31]
        for (int i = 0; i < arraySize; ++i) {
            a2[i] = i * 0.1f; b2[i] = i * 0.2f;
            a[i] = i * 0.1;   b[i] = i * 0.2;
        }

        std::cout << "\nSize of array: " << arraySize << "\n";
        addWithCuda_double(resultsFile, c.data(), a.data(), b.data(), arraySize);
        addWithCuda_float(resultsFile, c2.data(), a2.data(), b2.data(), arraySize);
    }

    resultsFile.close();
    return 0;
}

// --- WRAPPERS ---

cudaError_t addWithCuda_float(std::ofstream& resultsFile, float* c, const float* a, const float* b, unsigned int size)
{
    // Expanded parameters for complete report data [cite: 37]
    std::vector<int> j_values = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    std::vector<int> k_values = { 1, 10, 100, 1000, 10000, 100000 };

    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_c, size * sizeof(float));
    cudaMalloc(&dev_a, size * sizeof(float));
    cudaMalloc(&dev_b, size * sizeof(float));

    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    size_t thread_counts[] = { 128, 1024 }; 
    size_t trials = 10; // Averaged over 10 repetitions [cite: 44]

    std::cout << "\n CASE: float \n";
    for (int j : j_values) {
        for (int k : k_values) {
            for (int t : thread_counts) {
                dim3 thread_size(t);
                dim3 block_size((size + (t * j) - 1) / (t * j)); // Dynamic grid for parameter j [cite: 173]

                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < trials; i++) {
                    addKernel_float<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, size, j, k);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                double time = std::chrono::duration<double>(end - start).count() / (double)trials;
                double bytes_moved = 3.0 * size * sizeof(float);
                double operations = (double)size * k; 
                double ci = operations / bytes_moved; // X-axis for Roofline [cite: 43]
                double bw = (bytes_moved / 1e9) / time; // Memory performance [cite: 41]
                double gops = (operations / 1e9) / time; // Compute performance [cite: 42]

                // Log to CSV 
                resultsFile << "float," << size << "," << j << "," << k << "," << t << "," 
                            << time * 1e6 << "," << bw << "," << gops << "," << ci << "\n";
            }
        }
    }
    cudaFree(dev_c); cudaFree(dev_a); cudaFree(dev_b);
    return cudaSuccess;
}

cudaError_t addWithCuda_double(std::ofstream& resultsFile, double* c, const double* a, const double* b, unsigned int size)
{
    std::vector<int> j_values = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    std::vector<int> k_values = { 1, 10, 100, 1000, 10000, 100000 };
    double *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_c, size * sizeof(double));
    cudaMalloc(&dev_a, size * sizeof(double));
    cudaMalloc(&dev_b, size * sizeof(double));

    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    size_t thread_counts[] = { 128, 1024 }; 
    size_t trials = 10; 

    std::cout << "\n CASE: double \n";
    for (int j : j_values) {
        for (int k : k_values) {
            for (int t : thread_counts) {
                dim3 thread_size(t);
                dim3 block_size((size + (t * j) - 1) / (t * j));

                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < trials; i++) {
                    addKernel_double<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, size, j, k);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                double time = std::chrono::duration<double>(end - start).count() / (double)trials;
                double bytes_moved = 3.0 * size * sizeof(double);
                double operations = (double)size * k;
                double ci = operations / bytes_moved;
                double bw = (bytes_moved / 1e9) / time;
                double gops = (operations / 1e9) / time;

                resultsFile << "double," << size << "," << j << "," << k << "," << t << "," 
                            << time * 1e6 << "," << bw << "," << gops << "," << ci << "\n";
            }
        }
    }
    cudaFree(dev_c); cudaFree(dev_a); cudaFree(dev_b);
    return cudaSuccess;
}
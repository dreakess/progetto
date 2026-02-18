    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"

    #include <stdio.h>

    cudaError_t addWithCuda(int *c, const int *a, const int *b, const int N); // prototype of the cuda function



    // this is a function defined for single thread

    __global__ void addKernel(int *c, const int *a, const int *b, int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // the index of this cubic is defined using linear transform, that's reason why we need to calculate as that
        if(idx >= N)
            return;
        c[idx] = a[idx] + b[idx];
    }



    // here we are inside the RAM of pc, so what we define here is the bridge between ram and gpu memory
    int main()
    {
        const int arraySize = 2048;
        int a[arraySize], b[arraySize], c[arraySize] = { 0 }; // we need to remove the const prefix before because we have to modify this vector
        for (int i = 0; i < arraySize; ++i){
            a[i] = i;
            b[i] = arraySize - i;
        }
        int c[arraySize] = { 0 };


        // Add vectors in parallel.
        cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        return 0;
    }

    // Helper function for using CUDA to add vectors in parallel.
    cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
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

        for (int t : thread_counts) {
            size_t N_threads = t;
            dim3 thread_size(N_threads);
            dim3 block_size((size + N_threads - 1) / N_threads); 

            // Start timer here
            for(int i = 0; i < 100; i++) { // Run 100 times for average ì
                addKernel<<<block_size, thread_size>>>(dev_c, dev_a, dev_b, size);
            }
            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
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

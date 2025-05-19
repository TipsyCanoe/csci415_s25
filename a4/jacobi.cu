#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 16

__global__ void jacobi_compute_kernel(float *current, float *next, int n, bool *fixed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * n + x;

    if (x > 0 && x < n - 1 && y > 0 && y < n - 1 && !fixed[idx]) {
        next[idx] = 0.25f * (current[(y - 1) * n + x] +
                              current[(y + 1) * n + x] +
                              current[y * n + (x - 1)] +
                              current[y * n + (x + 1)]);
    } else if (fixed[idx] || x == 0 || x == n - 1 || y == 0 || y == n - 1) {
        // Keep fixed points unchanged
        next[idx] = current[idx];
    }
}

__global__ void reduction_kernel(float *current, float *next, float *delta, int n) {
    __shared__ float shared_diff[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    shared_diff[idx] = 0.0f;
    if (x < n && y < n) {
        float diff = fabs(next[y * n + x] - current[y * n + x]);
        shared_diff[idx] = diff;
    }
    __syncthreads();

    for (int stride = (BLOCK_SIZE * BLOCK_SIZE) / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            shared_diff[idx] = fmaxf(shared_diff[idx], shared_diff[idx + stride]);
        }
        __syncthreads();
    }

    if (idx == 0) {
        atomicMax((int *)delta, __float_as_int(shared_diff[0]));
    }
}

// Host functions
void initialize_grid(float *grid, bool *fixed, int n) {
    for (int i = 0; i < n * n; ++i) {
        grid[i] = 0.0f;
        fixed[i] = false;
    }
}

void read_input_file(const char *filename, int *n, float *tolerance, float *grid, bool *fixed) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d %f", n, tolerance) != 2) {
        fprintf(stderr, "Error reading grid size and tolerance\n");
        exit(EXIT_FAILURE);
    }

    initialize_grid(grid, fixed, *n);

    float value;
    int x1, y1, x2, y2;
    while (fscanf(file, "%f %d %d %d %d", &value, &x1, &y1, &x2, &y2) == 5) {
        for (int y = y1; y <= y2; ++y) {
            for (int x = x1; x <= x2; ++x) {
                if (x >= 0 && x < *n && y >= 0 && y < *n) {
                    grid[y * (*n) + x] = value;
                    fixed[y * (*n) + x] = true;
                }
            }
        }
    }

    fclose(file);
}

// Function to save results for visualization
void save_results(float *grid, int n, const char *filename) {
    // Create output filename
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s.dat", filename);
    
    FILE *file = fopen(output_filename, "w");
    if (!file) {
        fprintf(stderr, "Error creating output file\n");
        return;
    }
    
    // Write data in a format suitable for gnuplot
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            fprintf(file, "%d %d %f\n", x, y, grid[y * n + x]);
        }
        fprintf(file, "\n"); // Add empty line for gnuplot 3D plotting
    }
    
    fclose(file);
    printf("Results saved to %s\n", output_filename);
}

// Main function
int main(int argc, char **argv) {
    int iterations_per_check = 1;
    bool show_timing = false;
    const char *filename = NULL;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-T") == 0) {
            show_timing = true;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations_per_check = atoi(argv[++i]);
        } else {
            filename = argv[i];
        }
    }

    if (!filename) {
        fprintf(stderr, "Usage: %s [-T] [-i m] filename\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Read input file
    int n;
    float tolerance;
    float *grid;
    bool *fixed;
    
    // Allocate memory for grid and fixed flags
    grid = (float *)malloc(1024 * 1024 * sizeof(float)); // Max size
    fixed = (bool *)malloc(1024 * 1024 * sizeof(bool));
    
    if (!grid || !fixed) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Read the file and get the actual grid size
    read_input_file(filename, &n, &tolerance, grid, fixed);
    printf("Grid size: %d x %d, Tolerance: %g\n", n, n, tolerance);
    
    // Device memory
    float *device_grid1, *device_grid2, *device_delta;
    bool *device_fixed;
    
    // Allocate device memory
    cudaMalloc(&device_grid1, n * n * sizeof(float));
    cudaMalloc(&device_grid2, n * n * sizeof(float));
    cudaMalloc(&device_fixed, n * n * sizeof(bool));
    cudaMalloc(&device_delta, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(device_grid1, grid, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_grid2, grid, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fixed, fixed, n * n * sizeof(bool), cudaMemcpyHostToDevice);
    
    // Setup for computation
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // For timing
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    
    if (show_timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    // Main computation loop
    int total_iterations = 0;
    float max_delta = tolerance * 2; // Start with a value greater than tolerance
    bool converged = false;
    
    while (!converged) {
        // Reset delta before reduction
        float zero = 0.0f;
        cudaMemcpy(device_delta, &zero, sizeof(float), cudaMemcpyHostToDevice);
        
        // Run computation kernel iterations_per_check times
        for (int i = 0; i < iterations_per_check; i++) {
            jacobi_compute_kernel<<<gridSize, blockSize>>>(device_grid1, device_grid2, n, device_fixed);
            cudaDeviceSynchronize();
            
            // Swap grids for next iteration
            float *temp = device_grid1;
            device_grid1 = device_grid2;
            device_grid2 = temp;
            
            total_iterations++;
        }
        
        // Check for convergence
        reduction_kernel<<<gridSize, blockSize>>>(device_grid2, device_grid1, device_delta, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&max_delta, device_delta, sizeof(float), cudaMemcpyDeviceToHost);
        // Need to convert float bits representation back to float value
        max_delta = *(float*)&max_delta;
        
        printf("Iteration %d, Max delta: %g\n", total_iterations, max_delta);
        
        if (max_delta < tolerance) {
            converged = true;
            printf("Converged after %d iterations\n", total_iterations);
        }
    }
    
    // Report timing if requested
    if (show_timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        printf("\nTiming Results:\n");
        printf("Total computation time: %.2f ms\n", elapsed_time);
        printf("Average time per iteration: %.4f ms\n", elapsed_time / total_iterations);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Copy results back to host
    cudaMemcpy(grid, device_grid1, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Save results for visualization
    save_results(grid, n, filename);
    
    // Clean up
    free(grid);
    free(fixed);
    cudaFree(device_grid1);
    cudaFree(device_grid2);
    cudaFree(device_fixed);
    cudaFree(device_delta);
    
    return 0;
}
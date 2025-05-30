#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include "nelson_complete.c" // Nelson's algorithm implementation
#include "prog6-2.c" // Cannon's algorithm implementation

// Function to print usage
void print_usage(const char *prog_name) {
    printf("Usage: %s [-T] [-c] [-n] [-d]\n", prog_name);
    printf("  -T : Time the algorithm\n");
    printf("  -c : Run Cannon's algorithm\n");
    printf("  -n : Run Nelson's algorithm\n");
    printf("  -d : Enable debug printing\n");
}

// Generate a random square matrix of size n × n
void generate_random_matrix(int n, double *matrix) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (double)(random() % 100) / 10.0; // Random values between 0.0 and 10.0
    }
}

// Print matrix
void print_matrix(const char *name, double *matrix, int n) {
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    int time_flag = 0, cannon_flag = 0, nelson_flag = 0, debug_flag = 0;
    int opt;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments
    while ((opt = getopt(argc, argv, "Tcdn")) != -1) {
        switch (opt) {
            case 'T':
                time_flag = 1;
                break;
            case 'c':
                cannon_flag = 1;
                break;
            case 'n':
                nelson_flag = 1;
                break;
            case 'd':
                debug_flag = 1;
                break;
            default:
                if (rank == 0) print_usage(argv[0]);
                MPI_Finalize();
                return EXIT_FAILURE;
        }
    }

    // Ensure at least one algorithm is selected
    if (!cannon_flag && !nelson_flag) {
        if (rank == 0) {
            printf("Error: No algorithm selected. Use -c or -n.\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Debug print for flags
    if (debug_flag && rank == 0) {
        printf("Flags set: Time=%d, Cannon=%d, Nelson=%d, Debug=%d\n",
               time_flag, cannon_flag, nelson_flag, debug_flag);
    }

    // Matrix size (example: 4x4, replace with user input or predefined value)
    int n = 4;
    
    // Check if we have a perfect square number of processes for Cannon's algorithm
    if (cannon_flag) {
        int sqrt_size = (int)sqrt(size);
        if (sqrt_size * sqrt_size != size) {
            if (rank == 0)
                printf("Warning: Cannon's algorithm requires a perfect square number of processes.\n");
        }
    }

    double *a, *b, *c;

    // Allocate memory for matrices
    a = (double *)malloc(n * n * sizeof(double));
    b = (double *)malloc(n * n * sizeof(double));
    c = (double *)calloc(n * n, sizeof(double)); // Initialize C to zeros

    // Generate random matrices
    generate_random_matrix(n, a);
    generate_random_matrix(n, b);

    // Timing variables
    double start_time, end_time;

    // Run Cannon's algorithm
    if (cannon_flag) {
        if (rank == 0) printf("Running Cannon's algorithm...\n");
        if (time_flag) start_time = MPI_Wtime();
        MatrixMatrixMultiply(n, a, b, c, MPI_COMM_WORLD);
        if (time_flag) {
            end_time = MPI_Wtime();
            if (rank == 0) printf("Cannon's algorithm execution time: %f seconds\n", end_time - start_time);
        }
    }

    // Run Nelson's algorithm
    if (nelson_flag) {
        if (rank == 0) printf("Running Nelson's algorithm...\n");
        if (time_flag) start_time = MPI_Wtime();
        NelsonMultiply(n, a, b, c, MPI_COMM_WORLD);
        if (time_flag) {
            end_time = MPI_Wtime();
            if (rank == 0) printf("Nelson's algorithm execution time: %f seconds\n", end_time - start_time);
        }
    }

    // Debug print matrices
    if (debug_flag && rank == 0) {
        print_matrix("Matrix A", a, n);
        print_matrix("Matrix B", b, n);
        print_matrix("Matrix C (Result)", c, n);
    }

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    // Finalize MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
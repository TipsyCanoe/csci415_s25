
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Matrix multiplication function for local submatrices
void MatrixMultiply(int n, double *a, double *b, double *c) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
}

// Nelson's Hypercube Matrix Multiply Algorithm
void NelsonMultiply(int n, double *A, double *B, double *C, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (size == 1) {
        MatrixMultiply(n, A, B, C);
        return;
    }

    int dims = (int)log2(size);
    int mask = 1 << (dims - 1);

    double *tempA = (double *)malloc(n * n * sizeof(double));
    double *tempB = (double *)malloc(n * n * sizeof(double));
    double *tempC = (double *)malloc(n * n * sizeof(double));

    MPI_Status status;

    // Communication step: exchange along the hypercube
    int horizontal_partner = rank ^ (1 << 0); // Horizontal neighbor (bit flip)
    int vertical_partner = rank ^ (1 << 1);   // Vertical neighbor (bit flip)

    // Send and receive A and B along the correct dimensions
    MPI_Sendrecv(A, n * n, MPI_DOUBLE, horizontal_partner, 0, 
                 tempA, n * n, MPI_DOUBLE, horizontal_partner, 0, comm, &status);
    
    MPI_Sendrecv(B, n * n, MPI_DOUBLE, vertical_partner, 1, 
                 tempB, n * n, MPI_DOUBLE, vertical_partner, 1, comm, &status);

    // If in a 2x2 configuration, compute the final products directly
    if (size == 4) {
        if (rank == 0 || rank == 3) {
            MatrixMultiply(n, A, tempB, C);
            MatrixMultiply(n, tempA, B, C);
        } else {
            MatrixMultiply(n, A, B, C);
            MatrixMultiply(n, tempA, tempB, C);
        }
    } else {
        // Recursive step
        MPI_Comm sub_comm;
        MPI_Comm_split(comm, rank < mask, rank, &sub_comm);

        if (rank < mask) {
            NelsonMultiply(n, A, tempB, C, sub_comm);
            NelsonMultiply(n, tempA, B, tempC, sub_comm);
        } else {
            NelsonMultiply(n, A, B, C, sub_comm);
            NelsonMultiply(n, tempA, tempB, tempC, sub_comm);
        }

        // Accumulate the results
        for (int i = 0; i < n * n; ++i) {
            C[i] += tempC[i];
        }

        MPI_Comm_free(&sub_comm);
    }

    free(tempA);
    free(tempB);
    free(tempC);
}

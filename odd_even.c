#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void odd_even_sort(int *local_data, int local_n, int rank, int size) {
    int phase;
    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            if (rank % 2 == 0 && rank + 1 < size) {
                MPI_Sendrecv(local_data, local_n, MPI_INT, rank + 1, 0,
                             local_data, local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank % 2 == 1) {
                MPI_Sendrecv(local_data, local_n, MPI_INT, rank - 1, 0,
                             local_data, local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            if (rank % 2 == 1 && rank + 1 < size) {
                MPI_Sendrecv(local_data, local_n, MPI_INT, rank + 1, 0,
                             local_data, local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank % 2 == 0 && rank > 0) {
                MPI_Sendrecv(local_data, local_n, MPI_INT, rank - 1, 0,
                             local_data, local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000;
    int local_n = n / size;
    int *local_data = malloc(local_n * sizeof(int));
    srand(time(NULL) + rank);

    for (int i = 0; i < local_n; i++) {
        local_data[i] = rand() % 1000;
    }

    double start = MPI_Wtime();
    odd_even_sort(local_data, local_n, rank, size);
    double end = MPI_Wtime();

    if (rank == 0)
        printf("Odd-Even Sort completed in %f seconds\n", end - start);

    free(local_data);
    MPI_Finalize();
    return 0;
}

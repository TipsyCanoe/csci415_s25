#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void custom_prefix_sum(int *local_data, int local_n, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int *recv_buffer = malloc(sizeof(int) * local_n);
    int local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
        recv_buffer[i] = local_sum;
    }

    int total = 0;
    for (int step = 1; step < size; step <<= 1) {
        int send_val = local_sum;
        int recv_val = 0;
        if (rank >= step) {
            MPI_Recv(&recv_val, 1, MPI_INT, rank - step, 0, comm, MPI_STATUS_IGNORE);
        }
        if (rank + step < size) {
            MPI_Send(&send_val, 1, MPI_INT, rank + step, 0, comm);
        }
        total += recv_val;
    }

    for (int i = 0; i < local_n; i++) {
        local_data[i] = recv_buffer[i] + total;
    }

    free(recv_buffer);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            printf("Usage: %s <num_elements> <method: custom|mpi>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    int use_custom = (strcmp(argv[2], "custom") == 0);

    int local_n = n / size;
    int *local_data = malloc(local_n * sizeof(int));
    for (int i = 0; i < local_n; i++)
        local_data[i] = rank * local_n + i + 1;  // simple initial values

    double start_time = MPI_Wtime();

    if (use_custom) {
        custom_prefix_sum(local_data, local_n, MPI_COMM_WORLD);
    } else {
        MPI_Scan(local_data, local_data, local_n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    if (rank == 0)
        printf("Method: %s | Time: %f seconds\n", use_custom ? "Custom" : "MPI_Scan", end_time - start_time);

    free(local_data);
    MPI_Finalize();
    return 0;
}

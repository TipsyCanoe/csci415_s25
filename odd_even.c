#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int my_value;
    double start, end;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get the rank and size of the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Check if the number of processes is even
    srand(time(NULL) + rank);
    // Initialize the value to be sorted
    my_value = rand() % 1000;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // start the odd-even sort
    for (int phase = 0; phase < size; phase++) {
        int partner;
        // Determine the partner process for this phase
        if (phase % 2 == 0) {  // Even phase
            partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
        } else {               // Odd phase
            partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
        }
        // Ensure the partner is within bounds
        if (partner < 0 || partner >= size)
            partner = MPI_PROC_NULL;
        // Send and receive values with the partner
        int recv_value;
        MPI_Sendrecv(&my_value, 1, MPI_INT, partner, 0,
                     &recv_value, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Compare and swap values
        if (partner != MPI_PROC_NULL) {
            if ((rank < partner && my_value > recv_value) ||
                (rank > partner && my_value < recv_value)) {
                my_value = recv_value;
            }
        }
    }
    //
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    // Print the sorted value
    if (rank == 0)
        printf("Odd-Even Sort completed in %f seconds.\n", end - start);

    MPI_Finalize();
    return 0;
}

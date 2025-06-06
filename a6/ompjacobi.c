#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define MAX_ITER 1000
#define TOLERANCE 1e-6

void parse_arguments(int argc, char *argv[], int *timing, int *iterations, int *num_threads, char **filename) {
    *timing = 0;
    *iterations = 1;
    *num_threads = 1;
    *filename = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-T") == 0) {
            *timing = 1;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            *iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            *num_threads = atoi(argv[++i]);
        } else {
            *filename = argv[i];
        }
    }

    if (*filename == NULL) {
        fprintf(stderr, "Error: No input file provided.\n");
        exit(EXIT_FAILURE);
    }
}

void read_input(const char *filename, int *n, double *tolerance, double ***grid) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read grid size and tolerance
    fscanf(file, "%d %lf", n, tolerance);

    // Allocate memory for the grid
    *grid = (double **)malloc(*n * sizeof(double *));
    if (*grid == NULL) {
        fprintf(stderr, "Memory allocation failed for grid.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < *n; i++) {
        (*grid)[i] = (double *)calloc(*n, sizeof(double));
    }

    // Read fixed values and apply them to the grid
    double value;
    int x1, y1, x2, y2;
    while (fscanf(file, "%lf %d %d %d %d", &value, &x1, &y1, &x2, &y2) == 5) {
        for (int i = x1; i <= x2; i++) {
            for (int j = y1; j <= y2; j++) {
                (*grid)[i][j] = value;
            }
        }
    }

    fclose(file);
}

void jacobi_iteration(int n, double **grid, double **new_grid, int max_iter, double tol, int num_threads) {
    int iter = 0;
    double diff;

    do {
        diff = 0.0;

        #pragma omp parallel for reduction(+:diff) num_threads(num_threads) collapse(2)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                new_grid[i][j] = 0.25 * (
                    grid[i - 1][j] +
                    grid[i + 1][j] +
                    grid[i][j - 1] +
                    grid[i][j + 1]
                );
                double delta = new_grid[i][j] - grid[i][j];
                diff += delta * delta;
            }
        }

        // Swap values (manually copy updated values back into original grid)
        #pragma omp parallel for num_threads(num_threads) collapse(2)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                grid[i][j] = new_grid[i][j];
            }
        }

        iter++;
    } while (iter < max_iter && diff > tol);
}

int main(int argc, char *argv[]) {
    int timing, iterations, num_threads;
    char *filename;
    parse_arguments(argc, argv, &timing, &iterations, &num_threads, &filename);

    int n;
    double tolerance;
    double **grid;
    read_input(filename, &n, &tolerance, &grid);

    // Initialize solution arrays
    double **new_grid = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        new_grid[i] = (double *)calloc(n, sizeof(double));
    }

    double start_time = 0.0, end_time = 0.0;

    if (timing) {
        start_time = omp_get_wtime();
    }

    for (int i = 0; i < iterations; i++) {
        jacobi_iteration(n, grid, new_grid, MAX_ITER, tolerance, num_threads);
    }

    if (timing) {
        end_time = omp_get_wtime();
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    printf("Grid size: %d, Tolerance: %f\n", n, tolerance);
    printf("Solution:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", grid[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < n; i++) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    #pragma omp parallel
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }

    return 0;
}
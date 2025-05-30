#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 100000.0

int compare(const void *a, const void *b) {
    double fa = *(const double *)a;
    double fb = *(const double *)b;
    return (fa > fb) - (fa < fb);
}

void fill_array(double *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = ((double) rand() / RAND_MAX) * MAX;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <array size>\n", argv[0]);
        return 1;
    }

    size_t n = atol(argv[1]);
    double *arr = malloc(n * sizeof(double));
    if (!arr) {
        perror("malloc");
        return 1;
    }

    srand((unsigned) time(NULL));
    fill_array(arr, n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    qsort(arr, n, sizeof(double), compare);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C qsort time: %f seconds\n", elapsed);

    free(arr);
    return 0;
}

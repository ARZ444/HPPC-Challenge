// 2D Heat Diffusion Simulation using Jacobi Iteration (OpenMP Version)
// Part 3: shared-memory parallelisation of the serial heat diffusion code.
//
// The approach follows the course material (Module 9):
//   - A team of threads is forked for the Jacobi update with
//     #pragma omp parallel for over the outer (row) loop.
//   - The two grids are shared; the loop indices and the per-point difference
//     are private to each thread.
//   - The maximum change is combined safely with reduction(max:max_diff), so
//     there is no race on that shared value.
//
// Because each Jacobi update reads only the old grid and writes only the new
// grid, the computation is identical regardless of the number of threads, so
// the iteration count and final result match the serial version exactly.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NX 1000       // Grid dimension in X direction (number of rows)
#define NY 1000       // Grid dimension in Y direction (number of columns)
#define MAX_ITER 10000
#define TOLERANCE 1e-6

// Initialize the grid with boundary conditions (same as the serial version).
void initialize(double **grid, double **new_grid) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            if (i == 0 || i == NX-1 || j == 0 || j == NY-1) {
                grid[i][j] = 100.0;       // Hot boundary
                new_grid[i][j] = 100.0;
            } else {
                grid[i][j] = 0.0;         // Interior starts cold
                new_grid[i][j] = 0.0;
            }
        }
    }
    grid[NX/2][NY/2] = 200.0;             // Heat source, set once (not re-imposed)
}

// Allocate a 2D grid (same layout as the serial version).
double **allocate_grid() {
    double **grid = (double **)malloc(NX * sizeof(double *));
    for (int i = 0; i < NX; i++) {
        grid[i] = (double *)malloc(NY * sizeof(double));
    }
    return grid;
}

void free_grid(double **grid) {
    for (int i = 0; i < NX; i++) free(grid[i]);
    free(grid);
}

int main() {
    double **grid = allocate_grid();
    double **new_grid = allocate_grid();

    initialize(grid, new_grid);

    double start = omp_get_wtime();

    int iter = 0;
    double max_diff = 0.0;

    for (iter = 0; iter < MAX_ITER; iter++) {
        max_diff = 0.0;

        // One Jacobi iteration, parallelised over the interior rows. The outer
        // loop is split across the thread team; grid and new_grid are shared,
        // while i, j and diff are private (j and diff are declared inside the
        // region so each thread has its own). reduction(max:max_diff) gives
        // every thread a private copy that is combined into the global maximum
        // at the end of the loop, avoiding a race on max_diff.
        #pragma omp parallel for reduction(max:max_diff)
        for (int i = 1; i < NX-1; i++) {
            for (int j = 1; j < NY-1; j++) {
                new_grid[i][j] = (grid[i-1][j] +
                                  grid[i+1][j] +
                                  grid[i][j-1] +
                                  grid[i][j+1]) / 4.0;
                double diff = fabs(new_grid[i][j] - grid[i][j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        // Swap grids (pointer swap, as in the serial version).
        double **temp = grid;
        grid = new_grid;
        new_grid = temp;

        if (max_diff < TOLERANCE) break;
    }

    double end = omp_get_wtime();
    double time_taken = end - start;

    printf("Converged after %d iterations\n", iter + 1);
    printf("Final maximum change: %e\n", max_diff);
    printf("Execution time: %.4f seconds\n", time_taken);

    free_grid(grid);
    free_grid(new_grid);

    return 0;
}

// 2D Heat Diffusion Simulation using Jacobi Iteration (MPI Version)
// Part 1: 1D row domain decomposition with halo (ghost row) exchange.
//
// The parallel design follows the course material:
//   - SPMD model: MPI_Init / Comm_size / Comm_rank / Finalize     (Module 5)
//   - Blocking point-to-point MPI_Send / MPI_Recv for the halos,
//     ordered by rank parity so the exchange is deadlock-free      (Module 6a)
//   - MPI_Allreduce with MPI_MAX for the global convergence test   (Module 6b/7)
//
// Behaviour is kept identical to the serial reference (heat_serial.c): the
// same constants, the same initial conditions, the single heat source set once
// at initialisation (never re-imposed), and the same output format.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX 1000       // Grid dimension in X direction (number of rows)
#define NY 1000       // Grid dimension in Y direction (number of columns)
#define MAX_ITER 10000
#define TOLERANCE 1e-6

// Allocate a grid of (rows x NY) as an array of row pointers. Each row is a
// single contiguous block of NY doubles, matching the serial double** layout,
// so a whole row can be sent or received in one MPI message.
double **allocate_grid(int rows) {
    double **grid = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        grid[i] = (double *)malloc(NY * sizeof(double));
    }
    return grid;
}

void free_grid(double **grid, int rows) {
    for (int i = 0; i < rows; i++) free(grid[i]);
    free(grid);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Row decomposition -------------------------------------------------
    // Split the NX global rows into contiguous blocks. When NX is not evenly
    // divisible by size, the first (NX % size) ranks take one extra row each
    // so the remainder is spread out (1, 2, 4, 8 all divide 1000 evenly).
    int base = NX / size;
    int rem  = NX % size;
    int nlocal = base + (rank < rem ? 1 : 0);                 // real rows on this rank
    int start  = rank * base + (rank < rem ? rank : rem);     // global index of first real row

    // Local storage holds nlocal real rows plus one ghost row above (index 0)
    // and one ghost row below (index nlocal+1). Real rows are local 1..nlocal.
    int local_rows = nlocal + 2;
    double **cur  = allocate_grid(local_rows);
    double **next = allocate_grid(local_rows);

    // ---- Initialisation (same conditions as the serial version) ------------
    // Global edge cells = 100.0 (hot boundary); interior = 0.0; in both grids.
    for (int li = 1; li <= nlocal; li++) {
        int gi = start + (li - 1);                            // global row of this local row
        for (int j = 0; j < NY; j++) {
            if (gi == 0 || gi == NX - 1 || j == 0 || j == NY - 1) {
                cur[li][j]  = 100.0;                          // Hot boundary
                next[li][j] = 100.0;
            } else {
                cur[li][j]  = 0.0;                            // Interior starts cold
                next[li][j] = 0.0;
            }
        }
    }
    // Single heat source at the global centre, set ONCE (as in the serial code;
    // it is not re-imposed each iteration). It lives on whichever rank owns
    // global row NX/2.
    if (start <= NX / 2 && NX / 2 <= start + nlocal - 1) {
        int li = (NX / 2 - start) + 1;
        cur[li][NY / 2] = 200.0;                             // Heat source
    }

    // Neighbour ranks for the halo exchange. -1 marks "no neighbour": rank 0 has
    // no process above, rank size-1 has no process below.
    int up   = (rank == 0)        ? -1 : rank - 1;           // owns the rows just above us
    int down = (rank == size - 1) ? -1 : rank + 1;           // owns the rows just below us

    // Local interior rows this rank updates, as local indices. The global
    // boundary rows 0 and NX-1 are never updated (they stay at 100.0).
    int gi_lo = (start < 1) ? 1 : start;                              // max(1, start)
    int last  = start + nlocal - 1;
    int gi_hi = (last > NX - 2) ? NX - 2 : last;                      // min(NX-2, last)
    int li_lo = gi_lo - start + 1;
    int li_hi = gi_hi - start + 1;

    // Time only the iteration loop, with a barrier first so all ranks start
    // together; rank 0's elapsed time is reported (MPI_Wtime, not clock()).
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int iter = 0;
    double global_max = 0.0;

    for (iter = 0; iter < MAX_ITER; iter++) {
        // ---- Halo exchange (blocking Send/Recv, parity-ordered) ------------
        // Done in two directional steps. In each step even-rank processes send
        // first then receive, while odd-rank processes receive first then send.
        // That way every MPI_Send meets an already-waiting MPI_Recv, so the
        // blocking exchange cannot deadlock (Module 6a ordering). Each message
        // is one full row of NY doubles. The boundary guards (up/down >= 0)
        // mean rank 0 and rank size-1 simply skip the missing neighbour, and a
        // single-process run (np = 1) does no communication at all.

        // Step 1: send my last real row DOWN; receive my top ghost from UP.
        if (rank % 2 == 0) {
            if (down >= 0) MPI_Send(cur[nlocal], NY, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
            if (up   >= 0) MPI_Recv(cur[0],      NY, MPI_DOUBLE, up,   0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            if (up   >= 0) MPI_Recv(cur[0],      NY, MPI_DOUBLE, up,   0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (down >= 0) MPI_Send(cur[nlocal], NY, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
        }

        // Step 2: send my first real row UP; receive my bottom ghost from DOWN.
        if (rank % 2 == 0) {
            if (up   >= 0) MPI_Send(cur[1],          NY, MPI_DOUBLE, up,   1, MPI_COMM_WORLD);
            if (down >= 0) MPI_Recv(cur[nlocal + 1], NY, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            if (down >= 0) MPI_Recv(cur[nlocal + 1], NY, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (up   >= 0) MPI_Send(cur[1],          NY, MPI_DOUBLE, up,   1, MPI_COMM_WORLD);
        }

        // ---- Jacobi update on local interior points ------------------------
        // Identical arithmetic to the serial code. The local maximum change is
        // accumulated over exactly the same interior point set the serial code
        // uses (global rows 1..NX-2, columns 1..NY-2).
        double local_max = 0.0;
        for (int li = li_lo; li <= li_hi; li++) {
            for (int j = 1; j < NY - 1; j++) {
                next[li][j] = (cur[li - 1][j] +
                               cur[li + 1][j] +
                               cur[li][j - 1] +
                               cur[li][j + 1]) / 4.0;
                double diff = fabs(next[li][j] - cur[li][j]);
                if (diff > local_max) local_max = diff;
            }
        }

        // ---- Global convergence test ---------------------------------------
        // Combine every rank's local maximum into one global maximum that all
        // ranks receive, so they take the same break decision and the loop runs
        // for the same number of iterations as the serial version.
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Swap local grids (pointer swap, as in the serial version).
        double **temp = cur;
        cur = next;
        next = temp;

        if (global_max < TOLERANCE) break;
    }

    double elapsed = MPI_Wtime() - t_start;

    // ---- Output (rank 0 only), same format as the serial program -----------
    if (rank == 0) {
        printf("Converged after %d iterations\n", iter + 1);
        printf("Final maximum change: %e\n", global_max);
        printf("Execution time: %.4f seconds\n", elapsed);
    }

    free_grid(cur, local_rows);
    free_grid(next, local_rows);

    MPI_Finalize();
    return 0;
}

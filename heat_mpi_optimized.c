// 2D Heat Diffusion Simulation using Jacobi Iteration (Optimized MPI Version)
// Part 2: same 1D row decomposition as heat_mpi.c, but the halo exchange is
// overlapped with computation using non-blocking point-to-point messages.
//
// Idea (course material):
//   - Non-blocking MPI_Irecv / MPI_Isend start the halo transfer and return
//     immediately, completed later with MPI_Waitall                 (Module 6a)
//   - While the ghost rows are in flight we update the local INTERIOR rows
//     that do not need any ghost cell, so the halo latency is hidden behind
//     useful computation ("overlap communication with computation",
//     hybrid/performance notes, Module 9)
//   - Only after MPI_Waitall do we update the first and last local rows, which
//     depend on the freshly received ghost rows.
//
// Numerically nothing changes: every interior point is still computed from the
// old grid only, so the iteration count and final result match the serial and
// the Part 1 MPI version exactly.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX 1000       // Grid dimension in X direction (number of rows)
#define NY 1000       // Grid dimension in Y direction (number of columns)
#define MAX_ITER 10000
#define TOLERANCE 1e-6

// Allocate a grid of (rows x NY); each row is one contiguous block of NY
// doubles so a whole row can be sent/received in a single MPI message.
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

// Update one local row li from the old grid and return the maximum change on
// that row. Reads only cur[], writes only next[], exactly like the serial
// jacobi/compute_max_diff combined over interior columns 1..NY-2. Marked
// static inline so the compiler folds it into the loops (no call overhead).
static inline double update_row(double **cur, double **next, int li) {
    double row_max = 0.0;
    for (int j = 1; j < NY - 1; j++) {
        next[li][j] = (cur[li - 1][j] +
                       cur[li + 1][j] +
                       cur[li][j - 1] +
                       cur[li][j + 1]) / 4.0;
        double diff = fabs(next[li][j] - cur[li][j]);
        if (diff > row_max) row_max = diff;
    }
    return row_max;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Row decomposition (identical to Part 1) ---------------------------
    int base = NX / size;
    int rem  = NX % size;
    int nlocal = base + (rank < rem ? 1 : 0);                 // real rows on this rank
    int start  = rank * base + (rank < rem ? rank : rem);     // global index of first real row

    int local_rows = nlocal + 2;                              // + top and bottom ghost rows
    double **cur  = allocate_grid(local_rows);
    double **next = allocate_grid(local_rows);

    // ---- Initialisation (same conditions as the serial version) ------------
    for (int li = 1; li <= nlocal; li++) {
        int gi = start + (li - 1);
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
    // Single heat source at the global centre, set ONCE (not re-imposed).
    if (start <= NX / 2 && NX / 2 <= start + nlocal - 1) {
        int li = (NX / 2 - start) + 1;
        cur[li][NY / 2] = 200.0;                             // Heat source
    }

    // Neighbour ranks (-1 = none): rank 0 has no row above, rank size-1 none below.
    int up   = (rank == 0)        ? -1 : rank - 1;
    int down = (rank == size - 1) ? -1 : rank + 1;

    // Local interior rows this rank updates (global boundary rows 0 and NX-1
    // are never touched), as local indices.
    int gi_lo = (start < 1) ? 1 : start;
    int last  = start + nlocal - 1;
    int gi_hi = (last > NX - 2) ? NX - 2 : last;
    int li_lo = gi_lo - start + 1;
    int li_hi = gi_hi - start + 1;

    // Rows that depend on a received ghost row are the first real row (li == 1,
    // needs the top ghost) when there is a neighbour above, and the last real
    // row (li == nlocal, needs the bottom ghost) when there is a neighbour
    // below. Everything strictly between them is independent of the halo and is
    // computed while the messages are in flight.
    int inner_lo = li_lo;
    int inner_hi = li_hi;
    if (up   >= 0) inner_lo = li_lo + 1;     // defer first real row to after Waitall
    if (down >= 0) inner_hi = li_hi - 1;     // defer last real row to after Waitall

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int iter = 0;
    double global_max = 0.0;

    for (iter = 0; iter < MAX_ITER; iter++) {
        // ---- Start the halo exchange (non-blocking) ------------------------
        // Post the receives for the two ghost rows and the sends of our two
        // edge rows, then carry on without waiting. Tag 0 carries a row moving
        // DOWN (a rank's last row -> the neighbour-below's top ghost); tag 1
        // carries a row moving UP (a rank's first row -> the neighbour-above's
        // bottom ghost). The guards skip missing neighbours, so rank 0 and
        // rank size-1 post fewer messages and np = 1 posts none.
        MPI_Request reqs[4];
        int nreq = 0;
        if (up >= 0) {
            MPI_Irecv(cur[0], NY, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(cur[1], NY, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (down >= 0) {
            MPI_Irecv(cur[nlocal + 1], NY, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(cur[nlocal],     NY, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        // ---- Overlap: update halo-independent interior rows ----------------
        // These rows read only real rows that are already up to date, so they
        // can be computed while the ghost rows travel across the network.
        double local_max = 0.0;
        for (int li = inner_lo; li <= inner_hi; li++) {
            double m = update_row(cur, next, li);
            if (m > local_max) local_max = m;
        }

        // ---- Complete the halo exchange ------------------------------------
        if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // ---- Update the rows that needed the ghost cells -------------------
        if (up >= 0) {
            double m = update_row(cur, next, 1);
            if (m > local_max) local_max = m;
        }
        if (down >= 0) {
            double m = update_row(cur, next, nlocal);
            if (m > local_max) local_max = m;
        }

        // ---- Global convergence test (same as Part 1) ----------------------
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Swap local grids (pointer swap).
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

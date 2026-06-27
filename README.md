# HPPC Challenge

High Performance & Parallel Computing challenge. Implementations use MPI (OpenMPI)
and OpenMP in C/C++ on Linux.

## Build & Run

OpenMP:
    gcc -fopenmp program.c -o program
    OMP_NUM_THREADS=4 ./program

MPI:
    mpicc -o program program.c
    mpirun -np 4 ./program

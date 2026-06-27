# Makefile for Heat Diffusion MPI and OpenMP Programs

CFLAGS = -O2 -Wall
OPENMP_FLAGS = -fopenmp
LIBM = -lm

all: heat_mpi

heat_mpi: heat_mpi.c
	mpicc $(CFLAGS) -o heat_mpi heat_mpi.c

optimized: heat_mpi_optimized.c
	mpicc $(CFLAGS) -o heat_mpi_optimized heat_mpi_optimized.c

openmp: heat_openmp.c
	gcc $(CFLAGS) $(OPENMP_FLAGS) -o heat_openmp heat_openmp.c $(LIBM)

clean:
	rm -f heat_mpi heat_mpi_optimized heat_openmp

run-mpi: all
	mpirun -np 4 ./heat_mpi

run-opt: optimized
	mpirun -np 4 ./heat_mpi_optimized

run-openmp: openmp
	export OMP_NUM_THREADS=4 && ./heat_openmp

.PHONY: all optimized openmp clean run-mpi run-opt run-openmp

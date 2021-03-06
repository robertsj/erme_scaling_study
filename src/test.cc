//----------------------------------*-C++-*----------------------------------//
/*
 * \file   test.cc
 * \author Jeremy Roberts
 * \date   02/03/2012
 * \brief  Test class member definitions.
 */
//---------------------------------------------------------------------------//

#include "test.hh"

#include "petsc.h"
#include "slepceps.h"
#include <iostream>
#include <fstream>
#include <assert.h>

// Constructor
Test::Test(Data &d, int num_time_steps)
  : d_dat(d), d_num_time_steps(num_time_steps)
{
  // Get MPI things
  MPI_Comm_size(PETSC_COMM_WORLD, &d_np);
  MPI_Comm_rank(MPI_COMM_WORLD, &d_rank);

  // Size the time vector and initialize to zero.
  d_times.resize(d_num_time_steps, 0.0);
}

// Tests

// 0. Timing for fixed block size, 1 per process
void Test::test_fixed_block_1pp(int block_size, int number_blocks)
{
  // Define the number of blocks per process.
  int block_per_process = number_blocks;
  // Load it the data.
  d_dat.load_block(block_per_process, 0, block_size);
  // Build it.
  d_dat.build_R();
  // Do the timing loop.
  double temp_time;
  for (int i = 0; i < d_num_time_steps; i++)
  {
    temp_time = MPI_Wtime();
    // Multiply
    MatMult(d_dat.d_R, d_dat.d_J1, d_dat.d_J2);
    d_times[i] = MPI_Wtime() - temp_time;
    d_total_time += d_times[i];
  }
  // Wrap up.
  d_dat.cleanup();
  // Compute mean and standard deviation.
  standard_deviation();
  // Print diagnostic
  print(block_size, block_per_process, d_total_time, d_mean_time, d_std_time);
}

// 1. Timing for varied block size, 1 per process
void Test::test_varied_block_1pp(int number_blocks)
{
  // Define the number of blocks per process.
  int block_per_process = number_blocks;
  // Loop to vary the block size
  for (int j = 0; j < 10; j++)
  {
    // Define the block size.
    int block_size = 200 * (j + 1);
    // Load it the data.
    d_dat.load_block(block_per_process, 0, block_size);
    // Build it.
    d_dat.build_R();
    // Do the timing loop.
    double temp_time;
    for (int i = 0; i < d_num_time_steps; i++)
    {
      temp_time = MPI_Wtime();
      // Multiply
      MatMult(d_dat.d_R, d_dat.d_J1, d_dat.d_J2);
      d_times[i] = MPI_Wtime() - temp_time;
      d_total_time += d_times[i];
    }
    // Wrap up.
    d_dat.cleanup();
    // Compute mean and standard deviation.
    standard_deviation();
    // Print diagnostic
    print(block_size, block_per_process, d_total_time, d_mean_time, d_std_time);
  }
}

// 2. Timing for fixed block size, varied per process
void Test::test_fixed_block_Vpp(int block_size, int number_blocks)
{
  // Define the number of blocks per process.
  int block_per_process = number_blocks / d_np;
  // Load it the data.
  d_dat.load_block(block_per_process, 0, block_size);
  // Build it.
  d_dat.build_R();
  // Do the timing loop.
  double temp_time;
  for (int i = 0; i < d_num_time_steps; i++)
  {
    temp_time = MPI_Wtime();
    // Multiply
    MatMult(d_dat.d_R, d_dat.d_J1, d_dat.d_J2);
    d_times[i] = MPI_Wtime() - temp_time;
    d_total_time += d_times[i];
  }
  // Wrap up.
  d_dat.cleanup();
  // Compute mean and standard deviation.
  standard_deviation();
  // Print diagnostic
  print(block_size, block_per_process, d_total_time, d_mean_time, d_std_time);
}

// 3. Timing for fixed block size, varied per process
void Test::test_varied_block_Vpp(int number_blocks)
{
  // Define the number of blocks per process.
  int block_per_process = number_blocks / d_np;
  // Loop to vary the block size
  for (int j = 0; j < 10; j++)
  {
    // Define the block size.
    int block_size = 200 * (j + 1);
    // Load it the data.
    d_dat.load_block(block_per_process, 0, block_size);
    // Build it.
    d_dat.build_R();
    // Do the timing loop.
    double temp_time;
    for (int i = 0; i < d_num_time_steps; i++)
    {
      temp_time = MPI_Wtime();
      // Multiply
      MatMult(d_dat.d_R, d_dat.d_J1, d_dat.d_J2);
      d_times[i] = MPI_Wtime() - temp_time;
      d_total_time += d_times[i];
    }
    // Wrap up.
    d_dat.cleanup();
    // Compute mean and standard deviation.
    standard_deviation();
    // Print diagnostic
    print(block_size, block_per_process, d_total_time, d_mean_time, d_std_time);
  }
}

// 4. Timing for MR for fixed block size, varying per process
void Test::test_fixed_block_MR(int block_size, int number_blocks)
{

  // Define the number of blocks per process.
  int block_per_process = number_blocks;
  // Load it the data.
  d_dat.load_block(block_per_process, d_dat.d_read, block_size);
  // Build it.
  d_dat.build_R();
  d_dat.build_M();
  // Do the timing loop.
  double temp_time;
  for (int i = 0; i < d_num_time_steps; i++)
  {
    temp_time = MPI_Wtime();
    // Multiply
    //MatMult(d_dat.d_R, d_dat.d_J1, d_dat.d_J2);
    //MatMult(d_dat.d_M, d_dat.d_J2, d_dat.d_J3);
    MatMult(d_dat.d_MR, d_dat.d_J1, d_dat.d_J3);
    d_times[i] = MPI_Wtime() - temp_time;
    d_total_time += d_times[i];
  }
  if (1 == 0)
  {
    write_petsc(d_dat.d_R, "matrix_R");
    write_petsc(d_dat.d_M, "matrix_M");
    write_petsc(d_dat.d_J1, "vector_1");
    write_petsc(d_dat.d_J2, "vector_2");
    write_petsc(d_dat.d_J3, "vector_3");
  }
  if (d_dat.d_rank == 0)
  {
    std::cout << "      # MR's = " << d_dat.d_apply_count << std::endl;
  }

  // Wrap up.
  d_dat.cleanup();
  // Compute mean and standard deviation.
  standard_deviation();
  // Print diagnostic
  print(d_dat.d_block_size, block_per_process, d_total_time, d_mean_time, d_std_time);
}

// 5. Timing for MR for fixed block size, varying per process
void Test::test_fixed_block_eig(int block_size, int number_blocks, int solver)
{
  // Define the number of blocks per process.
  int block_per_process = number_blocks;

  // Load it the data.
  d_dat.load_block(block_per_process, d_dat.d_read, block_size);

  // Build it.
  d_dat.build_R();
  d_dat.build_M();

  // Set the eigensolver
  EPS eps;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, d_dat.d_MR, PETSC_NULL);
  EPSSetProblemType(eps, EPS_NHEP);
  if (solver == 0)
  {
    EPSSetType(eps, EPSKRYLOVSCHUR);
  }
  else if (solver == 1)
  {
    EPSSetType(eps, EPSARNOLDI);
  }
  else if (solver == 2)
  {
    EPSSetType(eps, EPSPOWER);
  }
  else
  {
    cout << " wrong solver type. " << endl;
    return;
  }
  EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
  EPSSetTolerances(eps, 1e-9, 100);
  double lambda_real = 0.0;
  double lambda_imag = 0.0;
  int numit = 0;

  // Do the timing loop.
  double temp_time;
  for (int i = 0; i < d_num_time_steps; i++)
  {
    temp_time = MPI_Wtime();
    //MatMult(d_dat.d_MR, d_dat.d_J1, d_dat.d_J3);
    // Solve the eigenproblem
    EPSSolve(eps);
    // Get output
    EPSGetIterationNumber(eps, &numit);
    EPSGetEigenpair(eps, 0, &lambda_real, &lambda_imag, d_dat.d_J1, d_dat.d_J2);
    d_times[i] = MPI_Wtime() - temp_time;
    d_total_time += d_times[i];
  }
  if (d_dat.d_rank == 0)
  {
    std::cout << "# iterations = " << numit << std::endl;
    std::cout << "      lambda = " << lambda_real << std::endl;
    std::cout << "      # MR's = " << d_dat.d_apply_count << std::endl;
  }
  // Wrap up.
  d_dat.cleanup();
  // Compute mean and standard deviation.
  standard_deviation();
  // Print diagnostic
  print(d_dat.d_block_size, block_per_process, d_total_time, d_mean_time, d_std_time);
}

// Utilities

void Test::write_petsc(Mat M, const char name[])
{
   PetscViewer view;
   PetscViewerBinaryOpen(MPI_COMM_WORLD, name, FILE_MODE_WRITE, &view);
   MatView(M, view);
   PetscViewerDestroy(&view);
}

void Test::write_petsc(Vec V, const char name[])
{
  PetscViewer view;
  PetscViewerBinaryOpen(MPI_COMM_WORLD, name, FILE_MODE_WRITE, &view);
  VecView(V, view);
  PetscViewerDestroy(&view);
}

void Test::standard_deviation()
{
  d_mean_time = d_total_time / d_num_time_steps;
  d_std_time = 0.0;
  for (int i = 0; i < d_num_time_steps; i++)
    d_std_time += (d_times[i] - d_mean_time)*(d_times[i] - d_mean_time);
  d_std_time = sqrt(d_std_time/d_num_time_steps);
}

void Test::print(int i, int j, double t1, double t2, double t3)
{
  if (d_rank == 0)
  {
    std::printf(" %4i %4i %12.9e %12.9e %12.9e \n",
                i, j, t1, t2, t3);
  }
}



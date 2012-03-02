//----------------------------------*-C++-*----------------------------------//
/*
 * \file   data.cc
 * \author Jeremy Roberts
 * \date   02/01/2012
 * \brief  Data class member definitions.
 */
//---------------------------------------------------------------------------//

#include "data.hh"

#include <fstream>
#include <assert.h>



void Data::load_block(int block_per_process, int flag, int block_size)
{

  d_block_per_process = block_per_process;

  // File with sizes and data.
  ifstream datafile;

  // Get process.
  MPI_Comm_rank(MPI_COMM_WORLD, &d_rank);

  if (d_rank == 0)
  {
    if (d_read)
    {
      // Open the block.txt
      cout << "reading block.txt" << endl;
      datafile.open("block.txt");
      // Read the energy, spatial, azimuthal, and polar order
      datafile >> d_number_groups; cout << " number_groups = " << d_number_groups << endl;
      datafile >> d_space_order;   cout << "   space_order = " << d_space_order   << endl;
      datafile >> d_azimuth_order; cout << " azimuth_order = " << d_azimuth_order << endl;
      datafile >> d_polar_order;   cout << "   polar_order = " << d_polar_order   << endl;
      d_block_size = 4*d_number_groups * (1+d_space_order) * (1+d_azimuth_order) * (1+d_polar_order);
    }
    else
    {
      // Just default it for testing.
      d_block_size    = block_size;
      d_number_groups = 0;
      d_space_order   = block_size - 1;
      d_azimuth_order = 0;
      d_polar_order   = 0;
    }
  }

  // Broadcast the block size
  MPI_Bcast(&d_number_groups, 1, MPI_INTEGER, 0, PETSC_COMM_WORLD);
  MPI_Bcast(&d_space_order,   1, MPI_INTEGER, 0, PETSC_COMM_WORLD);
  MPI_Bcast(&d_azimuth_order, 1, MPI_INTEGER, 0, PETSC_COMM_WORLD);
  MPI_Bcast(&d_polar_order,   1, MPI_INTEGER, 0, PETSC_COMM_WORLD);
  MPI_Bcast(&d_block_size,    1, MPI_INTEGER, 0, PETSC_COMM_WORLD);

  // We all allocate the array size.
  d_block = new double[d_block_size*d_block_size];

  if (d_rank == 0)
  {
    if (flag)
    {
      // Read the block -- remember, C is row-major.
      for (int i = 0; i < d_block_size*d_block_size; i++)
        datafile >> d_block[i];
      datafile.close();
//      for (int i = 0; i < d_block_size; i++)
//      {
//        for (int j = 0; j < d_block_size; j++)
//        {
//          cout <<  d_block[j + i*d_block_size] << " ";
//        }
//        cout << endl;
//      }
    }
    else
    {
      for (int i = 0; i < d_block_size*d_block_size; i++)
        d_block[i] = 1.0;
    }
  }
  // Broadcast the block
  MPI_Bcast(d_block, d_block_size*d_block_size, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

  if (~flag)
    assert(d_block[0] == 1.0);

  // Get number of processes.
  MPI_Comm_size(PETSC_COMM_WORLD, &d_num_proc);

  // Total number of blocks.
  d_num_block = d_num_proc * d_block_per_process;

}

void Data::build_R()
{
  PetscErrorCode ierr;

  // Dimension of the matrix (number of rows and columns)
  d_num_rows_local  = d_block_per_process * d_block_size;
  d_num_rows_global = d_num_block * d_block_size;

  if (d_rank == 0)
  {
    cout << "    d_num_rows_local = " << d_num_rows_local << endl
         << "        d_block_size = " << d_block_size << endl
         << " d_block_per_process = " << d_block_per_process << endl
         << "         d_num_block = " << d_num_block << endl
         << "   d_num_rows_global = " << d_num_rows_global << endl;
  }

  // Create the matrix.  Ax = y.  Our first attempt is
  // to break up the matrix as follows.  The system is
  //   | b1    0   0   | |x1|   |y1|
  //   |  0   b2   0   |.|x2| = |y2|
  //   |  0    0  b3   | |x3|   |y3|
  // We'll first try breaking it into columns.  Hence,
  // process 1 gets column 1
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,                 // MPI communicator
                          d_block_size,                     // size of block
                          PETSC_DECIDE, //d_block_per_process*d_block_size, // # local rows (=y local size).  Note, I originally
                                                            // thought this was # of BLOCK rows
                          PETSC_DECIDE, //d_block_per_process*d_block_size, // # local cols (=x local size) (thought it was # Bcols)
                          d_num_block*d_block_size, //PETSC_DETERMINE,                  // # global rows
                          d_num_block*d_block_size, //PETSC_DETERMINE,                  // # global cols
                          1,                                // # nz blocks per brow in diag part
                          PETSC_NULL,                       // array version; not needed
                          0,                                // # nz blocks per brow in offdiag
                          PETSC_NULL,                       // array version; not needed
                          &d_R);
  assert(~ierr); // Check for any error (a nonzero return value).

  // Create the two vectors.
  ierr = VecCreateMPI(PETSC_COMM_WORLD,                     // MPI communicator
                      PETSC_DECIDE, //d_block_per_process*d_block_size,     // local size
                      d_num_block*d_block_size,                      // global size
                      &d_J1);
  assert(~ierr); // Check for any error (a nonzero return value).

  ierr = VecDuplicate(d_J1, &d_J2);
  ierr = VecDuplicate(d_J1, &d_J3);
  assert(~ierr); // Check for any error (a nonzero return value).

  // Initialize the vectors.
  VecSet(d_J1, 1.0);
  VecSet(d_J2, 0.0);
  VecSet(d_J3, 0.0);

  VecAssemblyBegin(d_J1);
  VecAssemblyEnd(d_J1);
  VecAssemblyBegin(d_J2);
  VecAssemblyEnd(d_J2);
  VecAssemblyBegin(d_J3);
  VecAssemblyEnd(d_J3);

  //  MatGetOwnershipRange(Amat,&Istart,&Iend);
  // Set my bounds.
  int Istart, Iend;
  Istart = d_rank*d_block_per_process;
  Iend   = Istart + d_block_per_process;
  //cout << "Me: " << d_rank << " Istart " << Istart << " Iend " << Iend << endl;
  int idx[1];

  // Now set set some values
  for (int I = Istart; I < Iend; I++)
  {
    idx[0] = I;
    ierr = MatSetValuesBlocked(d_R, 1, idx, 1, idx, d_block, INSERT_VALUES);
  }

  MatAssemblyBegin(d_R, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(d_R, MAT_FINAL_ASSEMBLY);

  // print me (debugging)

}


void Data::cleanup()
{
  delete d_block;
  MatDestroy(&d_R);
  VecDestroy(&d_J1);
  VecDestroy(&d_J2);
}


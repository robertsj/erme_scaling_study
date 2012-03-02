//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   driver.cc
 * \author Jeremy Roberts
 * \date   02/01/2012
 * \brief  Driver for SLEPc/PETSc scaling study.
 */
//---------------------------------------------------------------------------//

#include "data.hh"
#include "test.hh"
#include "petsc.h"
#include "slepceps.h"
#include <iostream>
#include <cmath>

using namespace std;

//---------------------------------------------------------------------------//
/*
 * The point of this study is to investigate scaling of the operation
 *    J <-- M*R(k)*J
 * which serves as the backbone for most of the response matrix
 * computations.  Recall that R(k) is a block diagonal matrix, where
 * the blocks are dense and composed of response functions of a single
 * node.  We envision the RMM as a means to decompose domains for parallel
 * transport, and hence we consider the case where one or more complete
 * nodes (i.e. there responses) are handled by a single process.  This study
 * will allow an arbitrary response function block to be read in.  Then,
 * the number of processes and the number of nodes (blocks) per process is
 * to be varied.  Ideally, we wish to develop and substantiate a model
 * for the scalability of the approach.  Hopefully, such a model is
 * shown to be a viable alternative to extant parallel transport approaches.
 *
 */
//---------------------------------------------------------------------------//

int main(int argc, char *args[])
{

  // Initialize PETSc and SLEPc.  Explicit PETSc call is optional here.
  PetscInitialize(&argc, &args, PETSC_NULL, PETSC_NULL);
  SlepcInitialize(&argc, &args, (char*) 0, "");

  // Get some command line parameters
  PetscBool flag;
  int block_size;
  int number_blocks;
  int test_id;
  int nt;            // number steps for timing loop
  int read;
  int solver;

  PetscOptionsGetInt(PETSC_NULL, "-bs", &block_size, &flag);
  if (!flag) block_size = 600;

  PetscOptionsGetInt(PETSC_NULL, "-nb", &number_blocks, &flag);
  if (!flag) number_blocks = 24;

  PetscOptionsGetInt(PETSC_NULL, "-test", &test_id, &flag);
  if (!flag) test_id = 0;

  PetscOptionsGetInt(PETSC_NULL, "-nt", &nt, &flag);
  if (!flag) nt = 100;

  PetscOptionsGetInt(PETSC_NULL, "-read", &read, &flag);
  if (!flag) read = 0;

  PetscOptionsGetInt(PETSC_NULL, "-solver", &solver, &flag);
  if (!flag) solver = 0;

  // Define our data.
  Data dat(read);

  // Define the test.
  Test test(dat, nt);

  // Run a test.
  if (test_id == 0)
    test.test_fixed_block_1pp(block_size, number_blocks);
  else if (test_id == 1)
    test.test_varied_block_1pp(number_blocks);
  else if (test_id == 2)
    test.test_fixed_block_Vpp(block_size, number_blocks);
  else if (test_id == 3)
    test.test_varied_block_Vpp(number_blocks);
  else if (test_id == 4)
    test.test_fixed_block_MR(block_size, number_blocks);
  else if (test_id == 5)
    test.test_fixed_block_eig(block_size, number_blocks, solver);
  else
    cout << "unknown test." << endl;

  SlepcFinalize();
  PetscFinalize();
  return 0;
}

//---------------------------------------------------------------------------//
//                 end of driver.cc
//---------------------------------------------------------------------------//


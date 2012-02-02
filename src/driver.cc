//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   driver.cc
 * \author Jeremy Roberts
 * \date   02/01/2012
 * \brief  Driver for SLEPc/PETSc scaling study.
 */
//---------------------------------------------------------------------------//

#include "data.hh"
#include "petsc.h"
#include <iostream>
#include <cmath>
#include "slepceps.h"
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
int main(int argc, char *args[])
{
  // Initialize PETSc and SLEPc.  Only SLEPc really needs to be
  // initialized, but being explicit is useful.
  PetscInitialize(&argc, &args, PETSC_NULL, PETSC_NULL);
  SlepcInitialize(&argc, &args, (char*) 0, "");

  // Define our data.
  Data dat(1);

  // Load it.
  dat.load_block(0, 2000);

  // Build it.
  dat.build_R();

  // How many times we multiply (to give better timing results).
  int number_loops = 100;
  double time[number_loops];
  double total_time = 0.0, temp_time;

  for (int i = 0; i < number_loops; i++)
  {
    temp_time = MPI_Wtime();
    // Multiply
    MatMult(dat.d_R, dat.d_J1, dat.d_J2);
    time[i] = MPI_Wtime() - temp_time;
    total_time += time[i];
  }

  // View the vector.
  PetscViewer view;
  PetscViewerBinaryOpen(MPI_COMM_WORLD,"testvecout",FILE_MODE_WRITE, &view);
  VecView(dat.d_J2, view);

  // Wrap up.
  dat.cleanup();
  SlepcFinalize();
  PetscFinalize();

  if (dat.rank() == 0)
  {
    // Compute mean and standard deviation
    double mean_time = total_time / number_loops;
    double std_time = 0.0;
    for (int i = 0; i < number_loops; i++)
      std_time += (time[i] - mean_time)*(time[i] - mean_time);
    std_time = sqrt(std_time/number_loops);
    cout << " ------------------------------------------" << endl;
    cout << " TOTAL time = " << total_time << " seconds" << endl;
    cout << "  MEAN time = " << mean_time << " seconds" << endl;
    cout << "   STD time = " << std_time << " seconds" << endl;
    cout << " ------------------------------------------" << endl;
  }



  return 0;
}

//---------------------------------------------------------------------------//
//                 end of driver.cc
//---------------------------------------------------------------------------//


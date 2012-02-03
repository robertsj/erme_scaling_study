//----------------------------------*-C++-*----------------------------------//
/*
 * \file   test.hh
 * \author Jeremy Roberts
 * \date   02/01/2012
 * \brief  Test class definition.
 */
//---------------------------------------------------------------------------//

#ifndef TEST_HH
#define TEST_HH

#include "data.hh"

#include <iostream>
#include <fstream>
#include <vector>


using namespace std;

//===========================================================================//
/*!
 *  \class Test
 *  \brief Defines various scaling tests.
 *
 */
//===========================================================================//
class Test
{

public:

  /// Constructor
  Test(Data& d, int nt);

  /// \name Tests
  /// \{

  /*!
   *  \brief Timing for fixed block size, 1 per process
   *
   *  Weak scaling for the one node per process approach.
   *
   */
  void test_fixed_block_1pp(int block_size, int number_blocks);

  /*!
   *  \brief Timing for varied block size, 1 per process
   *
   *  Weak scaling for the one node per process approach,
   *  but various node sizes (i.e. varying approximation
   *  orders in the underlying response model)
   *
   */
  void test_varied_block_1pp(int number_blocks);

  /*!
   *  \brief Timing for fixed block size, varied per process
   *
   *  Strong scaling for a fixed problem size and a fixed
   *  block size.
   *
   */
  void test_fixed_block_Vpp(int block_size, int number_blocks);

  /*!
   *  \brief Timing for varied block size, varied per process
   *
   *  Strong scaling for a fixed problem size with varying
   *  block size.
   */
  void test_varied_block_Vpp(int number_blocks);

  /// \}

  /// \name Utilities
  /// \{

  /// Write a PETSc matrix to binary
  void write_petsc(Mat M, const char name[]);

  /// Write a PETSc vector to binary
  void write_petsc(Vec V, const char name[]);

  /// Standard deviation of a vector.
  void standard_deviation();

  /// Print diagnostic.
  void print(int i, int j, double t1, double t2, double t3);

  /// \}

private:

  /// The problem data
  Data d_dat;

  /// My rank
  int d_rank;

  /// Comm size
  int d_np;

  /// Time vector
  vector<double> d_times;

  /// Total time
  double d_total_time;

  /// Mean time
  double d_mean_time;

  /// Standard deviation
  double d_std_time;

  /// Number steps for timing loop.
  int d_num_time_steps;

};

#endif // TEST_HH
//---------------------------------------------------------------------------//
//                 end of test.hh
//---------------------------------------------------------------------------//


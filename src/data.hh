//----------------------------------*-C++-*----------------------------------//
/*
 * \file   data.hh
 * \author Jeremy Roberts
 * \date   02/01/2012
 * \brief  Data class definition.
 */
//---------------------------------------------------------------------------//

#ifndef DATA_HH
#define DATA_HH

#include <iostream>
#include <fstream>
#include <vector>
#include "petscvec.h"
#include "petscsys.h"
#include "petscmat.h"
#include "slepceps.h"


using namespace std;

//===========================================================================//
/*!
 *  \class Data
 *  \brief Load data and place into matrices.
 */
//===========================================================================//
class Data
{

public:

  /// Typedefs
  //\{

  //\}

  /// Set problem parameters
  Data(int read = 0) :
    d_rank(0), d_read(read), d_apply_count(0)
  {}

  ~Data()
  {
    // free vec's
  }

  /*!
   *  \brief Load a block response matrix.
   *
   *  If flag = 0, use the default, which is a block size of
   *  100, and values all set to unity.  Otherwise, read
   *  the file "block.txt".
   */
  void load_block(int block_per_process, int flag = 0, int block_size = 100);

  /*!
   *  \brief Build the block matrix R and associated vectors.
   *
   *  The actual construction of R (and vectors) might be
   *  a variable to study.
   */
  void build_R();

  /*!
   *  \brief Build the connectivity matrix.
   *
   *  We're only going to consider square arrays.
   */
  void build_M();
  int elSum( int ii, int II, int j1, int j2 );
  /// Destroy matrix and vectors.
  void cleanup();

  /// Return my rank.
  int rank(){return d_rank;}

  friend inline PetscErrorCode apply_MR(Mat A, Vec V_in, Vec V_out);

public:

  /// Number of blocks per process
  int d_block_per_process;

  /// Block size
  int d_block_size;
  int d_number_groups;
  int d_space_order;
  int d_azimuth_order;
  int d_polar_order;

  /// The response function block element (as a 1-d array)
  double* d_block;

  /// Number processes
  int d_num_proc;

  /// Total number blocks
  int d_num_block;

  /// Me
  int d_rank;

  /// Read from data or not.
  int d_read;

  /// Matrix size (local number rows)
  int d_num_rows_local;

  /// Matrix size (global)
  int d_num_rows_global;

  /// The block matrix R
  Mat d_R;

  /// The connectivity matrix M
  Mat d_M;

  /// Matrix shell for M*R
  Mat d_MR;

  int d_apply_count;

  /// J1, J2, J3, and a temporary Jtmp
  Vec d_J1, d_J2, d_J3, d_Jtmp;

};

#endif // DATA_HH
//---------------------------------------------------------------------------//
//                 end of data.hh
//---------------------------------------------------------------------------//


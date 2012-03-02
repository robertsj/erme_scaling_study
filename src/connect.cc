/*
 * connect.cc
 *
 *  Created on: Mar 1, 2012
 *      Author: robertsj
 */
#include "data.hh"
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
//#include <assert.h>

inline PetscErrorCode apply_MR(Mat A, Vec V_in, Vec V_out);


void Data::build_M()
{

  // WE ALL DO THIS STUFF

  int bcl = 1;
  int bcr = 1;
  int bcb = 1;
  int bct = 1;

  // Require the problem to be a square
  int I = sqrt(d_num_block);
  int J = sqrt(d_num_block);

  int elements[I][J];
  for (int j = 0; j < J; j++)
    for (int i = 0; i < I; i++)
      elements[i][j] = 1;


  // size for the "zeroth order" connectivity matrix, mm
  int mm_size = 4 * d_num_block;
  int mm[mm_size][2]; // mm[row][1]=col, mm[row][2]=value

  for (int j = 0; j < mm_size; j++)
  {
    mm[j][0] = -99;
    mm[j][1] = -99;
  }

  int k = 0;
  int nzrow = 0;
  for (int j = 0; j < J; j++) // "physical rows"
  {
    for (int i = 0; i < I; i++) // "physical columns"
    {
      if (elements[i][j] >= 0)
      {
        k++; // I am the kth element; else, I'm just void so move on



        // If there is somebody (>0) to my right, connect my face
        // number 2 to their face number 1
        if ((i < I - 1) and (elements[i + 1][j] >= 0))
        {
          mm[4 * (k - 1) + 1][0] = 4 * k;
          mm[4 * (k - 1) + 1][1] = 2;
          //cout << 4 * (k - 1) + 1  << endl;
        }



        // If there is somebody to my left, connect my face
        // number 1 to their face number 2
        if ((i > 0) and (elements[i - 1][j] >= 0))
        {
          mm[4 * (k - 1)][0] = 4 * (k - 2) + 1;
          mm[4 * (k - 1)][1] = 3;
          //cout << 4 * (k - 1)  << endl;
        }


        // If there is somebody to my top, connect my face 4 with
        // their face 3.
        if ((j < J - 1) and (elements[i][j + 1] >= 0))
        {
          // nzrow counts elements left in my row and in the next
          // row before my neighbor
          nzrow = elSum(i, I, j, j + 1);
          mm[4 * (k - 1) + 3][0] = 4 * (k - 1 + nzrow) + 2;
          mm[4 * (k - 1) + 3][1] = 4;
          //cout << 4 * (k - 1) + 3  << endl;
        }


        // If somebody is to my bottom, my 3 to their 4
        if ((j > 0) and (elements[i][j - 1] >= 0))
        {
          // nzrow counts elements left in my row and in the previous
          // row before my neighbor
          nzrow = elSum(i, I, j - 1, j); // had been j, j-1
          mm[4 * (k - 1) + 2][0] = 4 * (k - 1 - nzrow) + 3;
          mm[4 * (k - 1) + 2][1] = 5;
          //cout << 4 * (k - 1) + 2  << endl;
        }

        // boundaries

        if ((i == 0) and (bcb == 1))
        {
            mm[4 * (k - 1)][0] = 4 * (k - 1);
            mm[4 * (k - 1)][1] = -1;
        }

        if ((i == I - 1) and (bct == 1)) // right
        {
          mm[4 * (k - 1) + 1][0] = 4 * (k - 1) + 1;
          mm[4 * (k - 1) + 1][1] = -2;
        }

        if ((j == 0)) // bot
        {
          mm[4 * (k - 1) + 2][0] = 4 * (k - 1) + 2;
          mm[4 * (k - 1) + 2][1] = -3;
          //cout << " --- " << 4 * (k - 1) + 2 << endl;
        }
        if (j == J - 1) // top
        {
          mm[4 * (k - 1) + 3][0] = 4 * (k - 1) + 3;
          mm[4 * (k - 1) + 3][1]= -4;
          //cout << " --- " << 4 * (k - 1) + 3  << endl;
        }

      }
    }
  } // end mm build
//  for (int j = 0; j < mm_size; j++)
//    cout << " mm = " << j << " " << mm[j][0] << " " << mm[j][1] << endl;

  // Create reflection
  int refl[(1+d_space_order) * (1+d_azimuth_order) * (1+d_polar_order)];
  int tran[(1+d_space_order) * (1+d_azimuth_order) * (1+d_polar_order)];


  //cout << " so = " << d_space_order << " ao = " << d_azimuth_order << " po = " << d_polar_order << endl;

  int i = 0;
  for (int s = 0; s <= d_space_order; s++)
  {
    for (int a = 0; a <= d_azimuth_order; a++)
    {
      for (int p = 0; p <= d_polar_order; p++)
      {
        int expon = (s + a) % 2;

        //cout << " s = " << s << " a = " << a << " p = " << p << " expon = " << expon << endl;
        refl[i] = pow(-1, expon);
        tran[i] = 1;
        i++;
      }
    }
  }



  // ----------------------
  // BUILD THE PETSC MATRIX
  // ----------------------
  int ierr;
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,                 // MPI communicator
                         PETSC_DECIDE, // number of local rows; same as the local size used in creating the y vector for y = Ax.
                         PETSC_DECIDE,         // number of local cols; same as the local size used in creating the x vector for y = Ax.
                         d_num_block*d_block_size,                  // number global rows
                         d_num_block*d_block_size,         // number global cols
                         1,                                // number of nonzeros per row in DIAGONAL portion of local submatrix
                         PETSC_NULL,                       // array of number of nonzeros in the rows of the DIAGONAL portion of the local submatrix
                         1,                                // number of nonzeros per row in the OFF-DIAGONAL portion of local submatrix
                         PETSC_NULL,                       // array of number of nonzeros in the rows of the OFF-DIAGONAL portion of the local submatrix
                         &d_M);

  // Create the shell here, too.
  MatCreateShell(PETSC_COMM_WORLD,
                 PETSC_DECIDE,
                 PETSC_DECIDE,
                 d_num_block*d_block_size,
                 d_num_block*d_block_size,
                 this, // We need this object as the context, since R and M live here
                 &d_MR);
  MatShellSetOperation(d_MR, MATOP_MULT, (void(*)(void)) apply_MR);

  // Set my bounds.  These are my ROWS.
  int Istart, Iend;
  //MatGetOwnershipRange(d_M,&Istart,&Iend);

  //cout << "My M: " << d_rank << " Istart " << Istart << " Iend " << Iend << endl;


  // Now set set some values
  double val = 0.0;
  int row = 0;
  int col = 0;
  int num_ord = (d_space_order+1)*(d_azimuth_order+1)*(d_polar_order+1);

  Istart = d_rank*d_block_per_process*4;
  Iend   = Istart + d_block_per_process*4;
  for (int I = Istart; I < Iend; I++)
  {

    for (int g = 0; g < d_number_groups; g++)
    {
      i = 0;
      for (int s = 0; s <= d_space_order; s++)
      {
        for (int a = 0; a <= d_azimuth_order; a++)
        {
          for (int p = 0; p <= d_polar_order; p++)
          {
            if (mm[I][1] < 0)
            {
              val = refl[i];
            }
            else
            {
              val = 1;
            }

            row = I*num_ord*d_number_groups + g*num_ord + i;
            col = mm[I][0]*num_ord*d_number_groups + g*num_ord + i;
            //cout << " proc " << d_rank << " doing row = " << row << " col = " << col << endl;
            ierr = MatSetValue( d_M, row, col, val, INSERT_VALUES);
            i++;
          }
        }
      }
    }

  }

  MatAssemblyBegin(d_M, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(d_M, MAT_FINAL_ASSEMBLY);
}


int Data::elSum( int ii, int II, int j1, int j2 )
{
    int sum = 0;
    for ( int i = ii+1; i < II; i++ )
    {
            sum++;
    }
    for ( int i = 0; i <= ii; i++ )
    {
            sum++;
    }
    return sum;
}

// Matrix-vector wrapper for M*R
inline PetscErrorCode apply_MR(Mat A, Vec J_in, Vec J_out)
{
  // Get the PETSc context
  void *ctx;
  MatShellGetContext(A, &ctx);
  // and cast it to the base.  This is okay, since we need only apply
  //   M and R, and the base has these.
  Data *dat = (Data*) ctx;

  // Apply R to X_in and then M to the result.
  MatMult( dat->d_R, J_in,      dat->d_J2 );
  MatMult( dat->d_M, dat->d_J2, J_out     );
  //cout << " here i am " << endl;
  // No errors.
  dat->d_apply_count++;
  return 0;
}


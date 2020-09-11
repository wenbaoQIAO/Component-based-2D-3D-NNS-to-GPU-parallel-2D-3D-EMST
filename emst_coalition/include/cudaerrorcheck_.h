#ifndef CUDAERRORCHECK_H
#define CUDAERRORCHECK_H

/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao
 * Creation date : March. 2017
 * Add for cuda error warnings.
 ***************************************************************************
 */

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>

#include "random_generator.h"
#include <cuda_runtime.h>
#include <cuda.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_ERROR_CHECK

// code from GitHub https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )



namespace operators {

//wb.Q cuda error check, code from GitHub https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

//wb.Q cuda error check, code from GitHub https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

//! wb.Q cuda error check, code from GitHub https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//! WB.Q error check cuda synchronozation
inline bool errorCheckCudaThreadSynchronize(){
    cudaError_t err = cudaThreadSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}


////! WB.Q error check cuda synchronozation
bool errorCheckCudaDeviceSynchronize(){
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}

//! WB.Q error check cuda synchronozation
bool cudaChk(cudaError_t err){
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}

}

#endif // CUDAERRORCHECK_H

#ifndef CUDA_MACRO_EMST_H
#define CUDA_MACRO_EMST_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao
 * Creation date : December. 2016
 *
 ***************************************************************************
 */

#include <helper_cuda.h>
#include <helper_functions.h>

#ifdef CUDA_CODE

// WB.Q add
#define SHARED __shared__

//!WB.Q add for parallel 2-opt
#define KER_SCHED_2opt(w,h) \
    int _x = blockIdx.x * blockDim.x + threadIdx.x;\
    int _y = blockIdx.y * blockDim.y + threadIdx.y;
#define END_KER_SCHED_2opt


//QWB to avoid thread conflition
#define THREADFENCE __threadfence();

//! WB.Q, change thread per block for 1D applications
#define KER_CALL_THREAD_BLOCK_1D(b, t, b_width, b_height, width, height) \
    dim3    t(b_width, 1);\
    dim3    b((width / t.x) +1 , 1);

//! WB.Q, change thread per block for 1D applications with fixed grid blocking
#define KER_CALL_THREAD_BLOCK_1D_fix(b, t, b_width, b_height, width, height) \
    dim3    t(b_width, 1);\
    dim3    b(width, 1);

#else//CUDA_CODE



// WB,Q add
#define SHARED

//!WB.Q aad for 1D applications
#define KER_CALL_THREAD_BLOCK_1D(b, t, b_width, b_height, width, height) \
    int b;\
    int t;\

//!WB.Q aad for 1D applications with fixed grid blocking
#define KER_CALL_THREAD_BLOCK_1D_fix(b, t, b_width, b_height, width, height) \
    int b;\
    int t;\


#endif//CUDA_CODE

#endif // CUDA_MACRO_H


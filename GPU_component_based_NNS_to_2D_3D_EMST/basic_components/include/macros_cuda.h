#ifndef CUDA_MACRO_H
#define CUDA_MACRO_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */

//#ifdef _WIN32
//#  define WINDOWS_LEAN_AND_MEAN
//#  define NOMINMAX
//#  include <windows.h>
//#endif
#ifdef CUDA_CODE
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.h>
#endif

// wb.Q add general block size for cuda computing
#define GENERAL_BLOCK_SIZE 128
#define INITVALUE -1
#define CELLULAR_ADAPTIVE 1

//! HW 08/04/15 : add ABS, MIN, MAX
#define ABS(n) (((n)>=0)?(n):(-(n)))
#define MIN(m,n) (((m)<=(n))?(m):(n))
#define MAX(m,n) (((m)>=(n))?(m):(n))

#define EMST_isInf(id1x,id2x,d1,idd1x,idd2x,d2) (((d1)==(d2))?(((id1x)==(idd1x))?((id2x)<(idd2x)):((id1x)<(idd1x))):((d1)<(d2)))

#define STRIDE_ALIGNMENT 32
// Align up n to the nearest multiple of m
#define ALIGN_UP(n) (((n) % STRIDE_ALIGNMENT) ? ((n) + STRIDE_ALIGNMENT - ((n) % STRIDE_ALIGNMENT)) : (n))

#ifdef CUDA_CODE

// WB.Q add
#define SHARED __shared__

//QWB to avoid thread conflition
#define THREADFENCE __threadfence();

#define GLOBAL

#define KERNEL __global__

#define DEVICE_HOST __device__ __host__

//! HW 29/03/15 : modif
#define DEVICE __device__
#define HOST __host__

#define KER_SCHED_3D(w,h,d) \
    int _x = blockIdx.x * blockDim.x + threadIdx.x;\
    int _y = blockIdx.y * blockDim.y + threadIdx.y;\
    int _z = blockIdx.z * blockDim.z + threadIdx.z;

#define KER_SCHED(w,h) \
    int _x = blockIdx.x * blockDim.x + threadIdx.x;\
    int _y = blockIdx.y * blockDim.y + threadIdx.y;

#define END_KER_SCHED

#define END_KER_SCHED_3D

#define KER_RAND_SCHED(w,h) KER_SCHED(w,h)

#define KER_CALL_THREAD_BLOCK(b, t, b_width, b_height, width, height) \
    dim3    t(b_width, b_height);\
    dim3    b((width + t.x - 1) / t.x,(height + t.y - 1) / t.y);

#define KER_CALL_THREAD_BLOCK_3D(b, t, b_width, b_height, b_depth, width, height, depth) \
    dim3    t(b_width, b_height, b_depth);\
    dim3    b((width + t.x - 1) / t.x,(height + t.y - 1) / t.y,(depth + t.z - 1) / t.z);

#define _KER_CALL_(b,t) <<< b, t >>>

#define GPU_ALLOC_MEM(devPtr,size) (checkCudaErrors(cudaMalloc((void**)&(devPtr),(size))))

#define GPU_FREE_MEM(devPtr) (checkCudaErrors(cudaFree((devPtr))))

#define SYNCTHREADS //__syncthreads();

//! WB.Q, change thread per block for 1D applications
#define KER_CALL_THREAD_BLOCK_1D(b, t, b_width, b_height, width, height) \
    dim3    t(b_width, 1);\
    dim3    b((width + t.x - 1) / t.x, 1);

//b((width / t.x) +1 , 1);

//! WB.Q, change thread per block for 1D applications with fixed grid blocking
#define KER_CALL_THREAD_BLOCK_1D_fix(b, t, b_width, b_height, width, height) \
    dim3    t(b_width, 1);\
    dim3    b(width, 1);

#else//CUDA_CODE

#define GLOBAL

#define KERNEL

#define DEVICE_HOST

// WB,Q add
#define SHARED

//! HW 29/03/15 : modif
#define DEVICE
#define HOST

#define KER_SCHED_3D(w,h,d) \
    for (int _z = 0; _z < (d); ++_z) {\
    for (int _y = 0; _y < (h); ++_y) {\
    for (int _x = 0; _x < (w); ++_x) {

#define KER_SCHED(w,h) \
    for (int _y = 0; _y < (h); ++_y) {\
    for (int _x = 0; _x < (w); ++_x) {

#define KER_RAND_SCHED(w,h) \
    for (int _yr = 0; _yr < (h); ++_yr) {\
    for (int _xr = 0; _xr < (w); ++_xr) {\
    int _x = aleat_int(0, (w));\
    _x = _x >= (w) ? (w) - 1 : _x;\
    int _y = aleat_int(0, (w));\
    _y = _y >= (w) ? (w) - 1 : _y;\

#define END_KER_SCHED     }}

#define END_KER_SCHED_3D     }}}

#define KER_CALL_THREAD_BLOCK(b, t, b_width, b_height, width, height) \
    int b;\
    int t;\

#define KER_CALL_THREAD_BLOCK_3D(b, t, b_width, b_height, b_depth, width, height, depth) \
    int b;\
    int t;\

//!WB.Q aad for 1D applications
#define KER_CALL_THREAD_BLOCK_1D(b, t, b_width, b_height, width, height) \
    int b;\
    int t;\

//!WB.Q aad for 1D applications with fixed grid blocking
#define KER_CALL_THREAD_BLOCK_1D_fix(b, t, b_width, b_height, width, height) \
    int b;\
    int t;\

#define _KER_CALL_(b,t)

#define GPU_ALLOC_MEM(devPtr,size)

#define GPU_FREE_MEM(devPtr)

#define SYNCTHREADS

#endif//CUDA_CODE

#endif // CUDA_MACRO_H

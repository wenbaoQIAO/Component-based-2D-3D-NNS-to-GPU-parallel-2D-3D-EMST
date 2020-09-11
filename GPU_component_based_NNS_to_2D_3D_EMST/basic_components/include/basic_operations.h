#ifndef BASIC_OPERATIONS_H
#define BASIC_OPERATIONS_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#ifdef CUDA_CODE
#include <cuda_runtime.h>"
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif
#include <vector>
#include <iterator>

#include "macros_cuda.h"
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "Objectives.h"

#define TEST_CODE 0

using namespace std;
using namespace components;

/*!
 * \defgroup Distances Distances de matching
 * \brief Distances computed between nodes and windows.
 */
/*! @{*/
namespace components
{

//! Addition map using functor
template<class Point>
struct Addition
{
    // This operator is called for each segment
    DEVICE_HOST inline Point operator()(const Point& p1, const Point& p2)
    {
        Point p;
        p = p1 + p2;
        return p;
    }
};

//! Difference map using functor
template<class Point>
struct Difference
{
    // This operator is called for each segment
    DEVICE_HOST inline Point operator()(const Point& p1, const Point& p2)
    {
        Point p;
        p = p1 - p2;
        return p;
    }
};

//! Multiplication map using functor
template<class Point>
struct Mult
{
    // This operator is called for each segment
    DEVICE_HOST inline Point operator()(const Point& p1, const Point& p2)
    {
        Point p;
        p = p1 * p2;
        return p;
    }
};

//! Multiplication map using functor
template<class Point>
struct ScalarMult
{
    // This operator is called for each segment
    DEVICE_HOST inline Point operator()(const Point& p, const GLfloat& fl)
    {
        return p * fl;
    }
};

template<class Point>
struct ScalarDiv
{
    // This operator is called for each segment
    DEVICE_HOST inline Point operator()(const Point& p, const GLfloat& fl)
    {
        return p * (1/fl);
    }
};

//! Multiplication map using functor
template<class Point>
struct ScalarProduct
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(const Point& p1, const Point& p2)
    {
        GLfloat p;
        p = p1 * p2;
        return p;
     }
};

template<class Point>
struct Identical
{
    // This operator is called for each segment
    DEVICE_HOST inline Point operator()(const Point& p)
    {
        return p;
    }
};

#ifdef CUDA_CODE
//! HW 10/04/15 : add structs for atomicAdd() and atomicExch()
template<typename T>
struct AtomicAdd
{
    // This operator is called for each segment
    DEVICE_HOST inline T operator()(T& p, const T value)
    {
        return (atomicAdd((&p), value));
    }
    // This operator is called for each segment
    DEVICE_HOST inline T operator()(T* p, const T value)
    {
        return (atomicAdd(p, value));
    }
};
template<typename T>
struct AtomicExch
{
    // This operator is called for each segment
    DEVICE_HOST inline T operator()(T& p, const T value)
    {
        return (atomicExch((&p), value));
    }
    // This operator is called for each segment
    DEVICE_HOST inline T operator()(T* p, const T value)
    {
        return (atomicExch(p, value));
    }
};
#endif

//! KERNEL FUNCTION
template<class Grid, class O>
KERNEL void K_scalarOp(Grid g1, GLfloat fl, Grid gr, O op)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        gr[_y][_x] = op(g1[_y][_x], fl);
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
template<template<typename> class Grid,
         class Point, class O>
KERNEL void K_scalarOp(Grid<Point> g1, Grid<GLfloat> g2, Grid<Point> gr, O op)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        gr[_y][_x] = op(g1[_y][_x], g2[_y][_x]);
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
//! Kernel call by object
//! Standard Kernel call with objects
template<template<typename> class Grid, class Point>
KERNEL void K_BO_setValue(Grid<Point> g1, Point p)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        g1[_y][_x] = p;
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
//! Kernel call by object
//! Standard Kernel call with objects
//! HW 19/04/15 : modif
template<template<typename> class Grid,
        class Point,
        template<typename> class Grid2,
        class Point2,
        class O>
KERNEL void K_unaryOp(Grid<Point> g1, Grid2<Point2> gr, O op)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        (Point&)(gr[_y][_x]) = op(g1[_y][_x]);
    }

    END_KER_SCHED

    SYNCTHREADS;
}
//! HW 21/05/15 : modif
template<template<typename> class Grid,
        class Point,
        template<typename> class Grid2,
        class Point2>
KERNEL void K_unaryOp(Grid<Point> g1, Grid2<Point2> gr)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        (Point&)(gr[_y][_x]) = g1[_y][_x];
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
template<class Grid, class O>
KERNEL void K_binaryOp(Grid g1, Grid g2, Grid gr, O op)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        gr[_y][_x] = op(g1[_y][_x], g2[_y][_x]);
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//#if TEST_CODE
//! KERNEL FUNCTION
//! Kernel call by object
//! Standard Kernel call with buffers
template<class Node, class O>
KERNEL void K_BinaryOp(Node* d_g1, Node* d_g2, Node* d_gr, size_t w, size_t h, size_t stride, O op)
{
    KER_SCHED(w, h)

    if (_x < w && _y < h) {
        d_gr[_y * stride +_x] = op(d_g1[_y * stride + _x], d_g2[_y * stride + _x]);
    }

    END_KER_SCHED

    SYNCTHREADS;
}
//#endif

//! HW 08/04/15 : add K_cleanCell
template<template<typename> class Grid,
         class Point>
KERNEL void K_cleanCell(Grid<Point> g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        g1[_y][_x].clearCell();
    }

    END_KER_SCHED

    SYNCTHREADS;
}

/*!
 * \brief K_SO_copy
 *
 */
template <template<typename> class Grid1, class Node1,
          template<typename> class Grid2, class Node2>
KERNEL void K_copyOp(Grid1<Node1> nn_source,
                     Grid2<Node2> nn_cible)
{
    KER_SCHED(nn_source.getWidth(), nn_source.getHeight())

    if (_x < nn_source.getWidth() && _y < nn_source.getHeight())
    {
        nn_cible[_y][_x][0] = nn_source[_y][_x][0];
        nn_cible[_y][_x][1] = nn_source[_y][_x][1];
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_copyOp

class BOp {
public:
    void K_sumReduction(Grid<AMObjectives>& gf, AMObjectives& objs) {
        objs = 0.0f;
        for (int _y = 0; _y < gf.getHeight(); ++_y)
        for (int _x = 0; _x < gf.getWidth(); ++_x)
            objs += gf[_y][_x];
    }

    template<template<typename> class Grid,
             class Point>
    void K_sumReduction(Grid<Point>& gf, Point& fr) {
        fr = 0.0f;
        for (int _y = 0; _y < gf.getHeight(); ++_y)
        for (int _x = 0; _x < gf.getWidth(); ++_x)
            fr += gf[_y][_x];
    }

    template<template<typename> class Grid,
             class Point>
    void K_setValue(Grid<Point>& g1, Point const& p) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g1.getWidth(),
                              g1.getHeight());
        K_BO_setValue _KER_CALL_(b, t) (
                    g1,
                    p);

    }

    //! HW 19/04/15 : modif
    template<template<typename> class Grid,
             class Point,
             template<typename> class Grid2,
             class Point2>
    void K_setIdentical(Grid<Point>& g1, Grid2<Point2>& gr) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g1.getWidth(),
                              g1.getHeight());
//        K_unaryOp _KER_CALL_(b, t) (
//                    g1,
//                    gr,
//                    Identical<Point>());
        K_unaryOp _KER_CALL_(b, t) (
                    g1,
                    gr);
    }

    template<template<typename> class Grid,
             class Point>
    void K_scalarMult(Grid<Point>& g1, GLfloat fl, Grid<Point>& gr) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g1.getWidth(),
                              g1.getHeight());
        K_scalarOp _KER_CALL_(b, t) (
                    g1,
                    fl,
                    gr,
                    ScalarMult<Point>());

    }

    template<template<typename> class Grid,
             class Point>
    void K_scalarMult(Grid<Point>& g1, Grid<GLfloat>& gf, Grid<Point>& gr) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g1.getWidth(),
                              g1.getHeight());
        K_scalarOp _KER_CALL_(b, t) (
                    g1,
                    gf,
                    gr,
                    ScalarMult<Point>());
    }

    template<template<typename> class Grid,
             class Point>
    void K_scalarDiv(Grid<Point>& g1, Grid<GLfloat>& gf, Grid<Point>& gr) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g1.getWidth(),
                              g1.getHeight());
        K_scalarOp _KER_CALL_(b, t) (
                    g1,
                    gf,
                    gr,
                    ScalarDiv<Point>());
    }

    //!  HW 08/04/15 : add K_cleanGridOfCell
    template<template<typename> class Grid,
             class Point>
    void K_cleanGridOfCell(Grid<Point>& g1) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g1.getWidth(),
                              g1.getHeight());
        K_cleanCell _KER_CALL_(b, t) (g1);
    }

    /*!
     * \brief K_copy
     *
     */
    template <template<typename> class Grid1, class Node1,
              template<typename> class Grid2, class Node2>
    GLOBAL inline void K_copy(Grid1<Node1>& nn_source,
                              Grid2<Node2>& nn_cible) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_source.getWidth(),
                              nn_source.getHeight());
        K_copyOp _KER_CALL_(b, t) (nn_source, nn_cible);
    }

}; // BOp


//! Test program
template <class Node,
          size_t SXX,
          size_t SYY,
          class O>
class TestBOp {
    Node initNode1;
    Node initNode2;
    Node initNode3;

public:
    TestBOp(Node n1, Node n2, Node n3) :
        initNode1(n1),
        initNode2(n2),
        initNode3(n3) {}

    void run() {
        int devID = 0;
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&devID);

        if (error != cudaSuccess)
        {
            printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        }

        error = cudaGetDeviceProperties(&deviceProp, devID);

        if (deviceProp.computeMode == cudaComputeModeProhibited)
        {
            fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
            exit(EXIT_SUCCESS);
        }

        if (error != cudaSuccess)
        {
            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        }
        else
        {
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        }

        cout << "debut test GPU Binary Op on float ..." << endl;
        const size_t SX = SXX, SY = SXX;
        // Creation de grille en local
        Grid<Node> gdf(SX, SY), gdf1(SX, SY), gdf2(SX, SY);
        gdf1 = Node(initNode1);
        gdf2 = Node(initNode2);
        gdf = Node(initNode3);

        cout << "Creation de grilles sur device GPU ..." << endl;
        // Creation de grilles sur device GPU
#if TEST_CODE
        Grid<Node> gpu_gdf(SX, SY), gpu_gdf1(SX, SY), gpu_gdf2(SX, SY);
        // clean local memory
        gpu_gdf.freeMem();
        gpu_gdf1.freeMem();
        gpu_gdf2.freeMem();

#else
        Grid<Node> gpu_gdf, gpu_gdf1, gpu_gdf2;
#endif
        gpu_gdf.gpu_resize(SX, SY);
        gpu_gdf1.gpu_resize(SX, SY);
        gpu_gdf2.gpu_resize(SX, SY);

        // cuda timer
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
        double x0;
        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
        double xf;
        x0 = clock();

//#if TEST_CODE
        // Affichage
        (ofstream&) std::cout << "gdf1 = " << endl;
        (ofstream&) std::cout << gdf1 << endl;
        (ofstream&) std::cout << "gdf2 = " << endl;
        (ofstream&) std::cout << gdf2 << endl;
//#endif
        cout << gdf.getWidth() << endl;
        cout << gdf.getHeight() << endl;
        cout << gdf.getStride() << endl;

        cout << gpu_gdf.getWidth() << endl;
        cout << gpu_gdf.getHeight() << endl;
        cout << gpu_gdf.getStride() << endl;

        cout << "Appel du Kernel ..." << endl;
        for (int i = 0; i < 1; ++i) {

            // Copie des grilles CPU -> GPU
            gdf1.gpuCopyHostToDevice(gpu_gdf1);
            gdf2.gpuCopyHostToDevice(gpu_gdf2);
            gdf.gpuCopyHostToDevice(gpu_gdf);

//#if TEST_CODE
            // Kernel call with class parameters
            KER_CALL_THREAD_BLOCK(b, t,
                                  4, 4,
                                  gpu_gdf1.getWidth(),
                                  gpu_gdf1.getHeight());
            K_BinaryOp _KER_CALL_(b, t) (
                        gpu_gdf1,
                        gpu_gdf2,
                        gpu_gdf,
                        O());
//#endif

#if TEST_CODE
            Node* d_g1 = gpu_gdf1.getData();
            Node* d_g2 = gpu_gdf2.getData();
            Node* d_gr = gpu_gdf.getData();
            KER_CALL_THREAD_BLOCK(b2, t2,
                                  4, 4,
                                  gpu_gdf1.getWidth(),
                                  gpu_gdf1.getHeight());
            K_BinaryOp<Node> _KER_CALL_(b2, t2) (
                        d_g1,
                        d_g2,
                        d_gr,
                        gpu_gdf1.getWidth(),
                        gpu_gdf1.getHeight(),
                        gpu_gdf1.getStride(),
                        O());
#endif
            // Copie du resultat GPU -> CPU
            gdf.gpuCopyDeviceToHost(gpu_gdf);
            gdf1 = gdf;
        }//for

//#if TEST_CODE
        cout << "Affichage du resultat a la console ..." << endl;
        // Affichage du resultat à la console
        (ofstream&) std::cout << "gdf = " << endl;
        (ofstream&) cout << gdf << endl;
//#endif
        // cpu timer
        cout << "CPU Time : " << (clock() - x0)/CLOCKS_PER_SEC << endl;

        // cuda timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
        cout << "GPU Execution Time: " <<  elapsedTime << " ms" << endl;
        cout << endl;

        // Explicit
        gpu_gdf.gpuFreeMem();
        gpu_gdf1.gpuFreeMem();
        gpu_gdf2.gpuFreeMem();
        gdf.freeMem();
        gdf1.freeMem();
        gdf2.freeMem();
    }
};

}//namespace components

#endif // BASIC_OPERATIONS_H

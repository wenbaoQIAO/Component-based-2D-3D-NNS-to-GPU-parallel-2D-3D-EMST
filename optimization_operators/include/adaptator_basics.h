#ifndef ADAPTATOR_BASICS_H
#define ADAPTATOR_BASICS_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

#ifdef CUDA_CODE
#include "random_generator.h"
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"

#include "SpiralSearch.h"

#include "ViewGrid.h"
#ifdef TOPOLOGIE_HEXA
#include "ViewGridHexa.h"
#endif
#include "NIter.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif

#include "Objectives.h"
#include "basic_operations.h"
#include "geometry.h"
#include "CellularMatrix.h"

#define SOM3D   0
#define WITH_DENSITY 1

#define LARG_CHAP 1.0f // chapeau mexicain for triggerring
#define DELTA 1.0f   // for controlling the number of activated cells in each iteration

//! HW 29/03/15 : add #define
#define MAX_RAND_BUFFER_SIZE 256
//! HW 30/03/15 : add #define
//! In order to get a unique cell id (coordinate in the cellular matrix) according
//! to the cell::PC, here a hypothetical cellular matrix width is defined.
//! Note that this hypothetically computed cell id is only used for GPU CUDA random
//! number seed/starting state setup.
#define MAX_CM_WIDTH 1000

//! HW 260815 : add disparity range check for stereo
#define DISP_RANGE 64

//! HW 070915 : add the threshold to decide if two displacements are the same
#define SIMILARITY_THRESHOLD 1.0f
// max number of cellular matrices in the dynamic CM pattern
// There are at most 9 cellular matrices for the quad topo
#define MAX_NUM_CM 9

using namespace std;
using namespace components;

namespace operators
{

//! HW 29/03/15 : To generate cuRAND pseudorundom numbers
#ifdef CUDA_CODE

KERNEL void K_seedSetup(curandState *state, unsigned int seed, int w, int h)
{
    KER_SCHED(w, h)

    if (_x < w && _y < h) {
        int cid = _x + _y * w;
        /* Each thread gets same seed, a different sequence
           number, no offset */
        curand_init(seed, cid, 0, &state[cid]);
        /* Each thread gets a different seed, a different sequence
           number, no offset */
//        curand_init(seed + (unsigned)(cid * 123456), cid, 0, &state[cid]);
    }
}

template<class Grid>
KERNEL void K_generateUniform(curandState *state,
                              Grid gRand)
{
    KER_SCHED(gRand.getWidth(), gRand.getHeight())

    if (_x < gRand.getWidth() && _y < gRand.getHeight()) {
        int cid = _x + _y * gRand.getWidth();
        /* Copy state to local memory for efficiency */
        curandState localState = state[cid];
        /* Generate pseudo-random uniforms */
        for(int i = 0; i < MAX_RAND_BUFFER_SIZE; i++) {
            /* It return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded */
            gRand[_y][_x][i] = curand_uniform(&localState);
        }
        /* Copy state back to global memory */
        state[cid] = localState;
    }
}
#endif

template <class Cell>
struct GetAdaptor {//std adaptor
    DEVICE_HOST inline void init(int n) = 0;
    DEVICE_HOST inline void init(Cell& cell) = 0;

    //! HW 29/03/15 : method overloading for calling from device
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) = 0;

    DEVICE_HOST inline bool next(Cell& cell) = 0;
};

template <class Cell>
struct GetStdAdaptor {//std adaptor

    size_t nb;
    size_t niter;

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        cell.init();
    }

    //! HW 29/03/15 : method overloading for calling from device
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        return cell.get(ps);
    }

    //! HW 13/04/15 : method overloading for K_SO_labelSet
    DEVICE_HOST inline bool get(Cell& cell, PointCoord& ps) {
        return cell.get(ps);
    }

    DEVICE_HOST inline bool next(Cell& cell) {
        return cell.next();
    }
};

template <class Cell>
struct GetRandomAdaptor {

    //!JCC 300315 : put device_host default
    DEVICE_HOST GetRandomAdaptor() : nb(), niter(1) {
#ifdef CUDA_CODE
#else
        aleat_initialize();
#endif
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
#ifdef CUDA_CODE
#else
        aleat_initialize();
#endif
    }

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    size_t nb;
    size_t niter;

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

#ifdef CUDA_CODE
    DEVICE inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        GLfloat rand = 0.0f;
        //! HW 29/03/15 : to generate cuRAND pseudorundom numbers using CUDA device API
        //! Note that this function is inefficient because each time a new seed (state)
        //! is set up by curand_init().
        //! Note that calls to curand_init() are slower than calls to curand() or curand_uniform().
        //! Therefore, it is much faster to save and restore random generator state
        //! than to recalculate the starting state repeatedly.
        //! Read more at: http://docs.nvidia.com/cuda/curand/
        curandState devStates;
        //! JCC 300315 : remove code in comment
        //! - response : This comment code maybe useful because it differs the seed of each thread/cell.
        //! - response : The current code uses the same seed for each thread/cell but with different sequence number.
        curand_init((unsigned long long)(clock()),
                    (cell.PC[0] + cell.PC[1] * MAX_CM_WIDTH),
                    0,
                    &devStates);
//        curand_init(((unsigned long long)(clock()) + (unsigned long long)((cell.PC[0] + cell.PC[1] * MAX_CM_WIDTH) * 123456)),
//                    (cell.PC[0] + cell.PC[1] * MAX_CM_WIDTH),
//                    0,
//                    &devStates);
        rand = curand_uniform(&devStates);
        return cell.extractRandom(nn, ps, rand);
    }
#else
    //! JCC a host only function doesn't need be called HOST
    inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        GLfloat rand = aleat_float(0, 1);
        return cell.extractRandom(nn, ps, rand);
    }
#endif

    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < niter;
    }
};

//! JCC 310315 : I create specific RandGridAllocator
//! HW 29/03/15 : add typedef and enum
typedef components::Point<GLfloat, MAX_RAND_BUFFER_SIZE> RandNumberBuffer;
typedef Grid<RandNumberBuffer> RandGrid;

//!
//! \brief The RandGridAlloc struct
//!
struct RandGridAlloc {
    unsigned int seed;

    DEVICE_HOST RandGridAlloc() {
        seed = time(NULL);
#ifdef CUDA_CODE
#else
        aleat_initialize(seed);
#endif
    }

    GLOBAL void K_generateRandNumBuffer(RandGrid gRand) {

        //! JCC changing parameter passing
        size_t w = gRand.getWidth();
        size_t h = gRand.getHeight();

#ifdef CUDA_CODE
        /* Allocate space for prng states on device */
        curandState *devStates;
        cudaMalloc((void **)&devStates, w * h * sizeof(curandState));

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              w, h);
        K_seedSetup _KER_CALL_(b, t) (devStates,
                                      seed,
                                      w, h);
        K_generateUniform<Grid<RandNumberBuffer> >  _KER_CALL_(b, t) (devStates,
                                                                      gRand);
#else
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
                for (int k = 0; k < MAX_RAND_BUFFER_SIZE; ++k)
                {
                    gRand[j][i][k] = aleat_float(0, 1);
                }
#endif
    }
};

//!
//! \brief The RandGridAllocFromCPU struct
//!
struct RandGridAllocFromCPU {

    RandGridAllocFromCPU() {
        aleat_initialize();
    }

    //! JCC 310315 : no "GLOBAL" since only CPU possible
    void generateRandNumBuffer(RandGrid gRand) {

        size_t w = gRand.getWidth();
        size_t h = gRand.getHeight();
        RandGrid gRand_cpu(w, h);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
                for (int k = 0; k < MAX_RAND_BUFFER_SIZE; ++k)
                {
                    gRand_cpu[j][i][k] = aleat_float(0, 1);
                }

        gRand_cpu.gpuCopyHostToDevice(gRand);
        //! JCC 300315 : free tmp memory
        gRand_cpu.freeMem();
    }
};

//! JCC 300315 : dissociation from Random allocation
//! HW 29/03/15 : to generate two grids of rundom numbers
template <class Cell>
struct GetRandomGridAdaptor {

    //! Random numbers
    RandGrid gRandInitiator;
    RandGrid gRandRoulette;

    //! max density cell
    GLfloat max_cell_density;
    GetRandomGridAdaptor() : nb(0), niter(1) {}
    ~GetRandomGridAdaptor() {
//        gRandInitiator.gpuFreeMem();
//        gRandRoulette.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRandInitiator.gpuResize(w, h);
        gRandRoulette.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRandInitiator);
        rga.K_generateRandNumBuffer(gRandRoulette);
        max_cell_density = cm.getMaxCellDensity();
    }

    size_t nb;
    size_t niter;

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = false;
        GLfloat rand1 = 0.0f;
        rand1 = gRandInitiator[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        // Cell activation
        if (cell.density >= rand1 * max_cell_density) {
            GLfloat rand2 = 0.0f;
            rand2 = gRandRoulette[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
            // Roulette wheel random extraction
            ret = cell.extractRandom(nn, ps, rand2);
        }
        return ret;
    }

    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < niter;
    }
};

//! HW 22/04/15 : This get adaptor performs random index extraction, instead of roulette wheel random extraction
template <class Cell>
struct GetStdRandomGridAdaptor : public GetRandomGridAdaptor<Cell> {

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = false;
        GLfloat rand1 = 0.0f;
        rand1 = this->gRandInitiator[cell.PC[1]][cell.PC[0]][((this->niter)*(this->gene)+(this->nb)) % MAX_RAND_BUFFER_SIZE];
        // Cell activation
        if (cell.density >= rand1 * (this->max_cell_density)) {
            GLfloat rand2 = 0.0f;
            rand2 = this->gRandRoulette[cell.PC[1]][cell.PC[0]][((this->niter)*(this->gene)+(this->nb)) % MAX_RAND_BUFFER_SIZE];
            // Roulette wheel random extraction
            //! HW 17/04/15 : The reason of using MIN() is that rand is from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
            float i = MIN((((float)cell.getSize()) * rand2), ((float)(cell.getSize() - 1)));
            ret = cell.get((int)i, ps);
        }
        return ret;
    }
};

//! HW 22/04/15 : This get adaptor performs the opposite roulette wheel random extraction, where
//! the larger the density value is, the smaller chance of being extracted it has.
template <class Cell>
struct GetRandomGridOppositeAdaptor : public GetRandomGridAdaptor<Cell> {

    GLfloat max_cell_edgeNum;

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        GetRandomGridAdaptor<Cell>::initialize(cm);
        max_cell_edgeNum = cm.maxEdgeNum;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = false;
        GLfloat rand1 = 0.0f;
        rand1 = this->gRandInitiator[cell.PC[1]][cell.PC[0]][((this->niter)*(this->gene)+(this->nb)) % MAX_RAND_BUFFER_SIZE];
        // Cell activation
        if (cell.density >= rand1 * (this->max_cell_density)) {
//        if (((GLfloat)(cell.edgeNum)) >= rand1 * (GLfloat)(this->max_cell_edgeNum)) {
            GLfloat rand2 = 0.0f;
            rand2 = this->gRandRoulette[cell.PC[1]][cell.PC[0]][((this->niter)*(this->gene)+(this->nb)) % MAX_RAND_BUFFER_SIZE];
            // Roulette wheel random extraction
            ret = cell.extractRandomOpposite(nn, ps, rand2);
        }
        return ret;
    }
};

//! HW 17/04/15 : Add GetStdRandomAdaptor for batch SOM sampling.
//! This get adaptor has no cell activation and it performs random index extraction, instead of roulette wheel random extraction
template <class Cell>
struct GetStdRandomAdaptor {

    //! Random numbers
    RandGrid gRand;

    GetStdRandomAdaptor() : nb(0), niter(1) {}
    ~GetStdRandomAdaptor() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t niter;

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = false;
        GLfloat rand = 0.0f;
        rand = gRand[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        //! HW 17/04/15 : The reason of using MIN() is that rand is from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
        float i = MIN((((float)cell.getSize()) * rand), ((float)(cell.getSize() - 1)));
        ret = cell.get((int)i, ps);
        return ret;
    }

    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < niter;
    }
};

template <class Cell>
struct GetDivideAdaptor {
    size_t size_slab;
    size_t curPos;

    GetDivideAdaptor() {}

    DEVICE_HOST GetDivideAdaptor(size_t w, size_t h, size_t W, size_t H) {
        size_t snet = w * h;
        size_t sdiv = W * H;

        size_slab = (snet + sdiv - 1) / sdiv;
        curPos = 0;
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
    }

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
    }

    DEVICE_HOST inline void init(Cell& cell) {
        curPos = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = curPos < size_slab;
        if (ret) {
            size_t nnidx = cell.PC[1] * cell.vgd.getWidthDual() * size_slab
                    + cell.PC[0] * size_slab + curPos;
            ps[1] = nnidx / nn.adaptiveMap.getWidth();
            ps[0] = nnidx % nn.adaptiveMap.getWidth();
            if (ps[1] >= nn.adaptiveMap.getHeight())
                ret = false;
        }
        return ret;
    }

    DEVICE_HOST inline bool next(Cell cell) {
        return ++curPos < size_slab;
    }
};

//! QWB add CPU version
template <class Cell>
struct GetDivideAdaptor_CPU {
    size_t size_slab;
    size_t curPos;

    GetDivideAdaptor_CPU() {}

    GetDivideAdaptor_CPU(size_t w, size_t h, size_t W, size_t H) {
        size_t snet = w * h;
        size_t sdiv = W * H;

        size_slab = (snet + sdiv - 1) / sdiv;
        curPos = 0;
    }

    size_t gene;
    inline void setGene(int n) {
        gene = n;
    }

    inline void init(int n) {
    }

    inline void init(Cell& cell) {
        curPos = 0;
    }
    inline void init() {
        curPos = 0;
    }

    template <class NN>
    inline bool get(Cell& cell, NN& nn, PointCoord& ps) {


        bool ret = curPos < size_slab;
        if (ret) {
            size_t nnidx = cell.PC[1] * cell.vgd.getWidthDual_cpu() * size_slab
                    + cell.PC[0] * size_slab + curPos;
            ps[1] = nnidx / nn.adaptiveMap.getWidth_cpu();
            ps[0] = nnidx % nn.adaptiveMap.getWidth_cpu();

            if (ps[1] >= nn.adaptiveMap.getHeight_cpu())
                ret = false;

        }
        return ret;
    }

    inline bool next(Cell cell) {

        return ++curPos < size_slab;
    }

    inline bool next() {
        return ++curPos < size_slab;
    }
};


template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchAdaptor {

    DEVICE_HOST virtual bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) = 0;
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchIdAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {
        minP = ps;
        return true;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchCenterAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {
        minP = cm.vgd.FDual(PC);
        return true;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchCenterToCenterAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {
        minP = cm.vgd.FDual(PC);
        ps = minP;
        return true;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchSpiralAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIterator sps_iter(
                    PC,
                cm.vgd.getWidth() * cm.vgd.getHeight(),
                0,
                MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                1
                );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchFindCellAdaptor {

    template <class NN1, class NN2, class IndexCM>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN1& scher,
                            NN2& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            IndexCM& minP) {

        if (ps[0] < scher.adaptiveMap.getWidth() &&
                ps[1] < scher.adaptiveMap.getHeight())
        minP = cm.vgd.findCell(scher.adaptiveMap[ps[1]][ps[0]]);
        return true;
    }
};

//WB.Q 11/16 reload cpu side
template <class CellularMatrix>
struct SearchFindCellAdaptor_cpu {

    template <class NN1, class NN2>
    bool search(CellularMatrix& cm,
                NN1& scher,
                NN2& sched,
                PointCoord& PC,
                PointCoord& ps,
                PointCoord& minP) {

        if (ps[0] < scher.adaptiveMap.getWidth_cpu() &&
                ps[1] < scher.adaptiveMap.getHeight_cpu()){

            minP = cm.vgd.findCell_cpu(scher.adaptiveMap[ps[1]][ps[0]]);

            if (minP[0] >= 0 && minP[0] < cm.getWidth() && minP[1] >= 0 && minP[1] < cm.getHeight())
                cm[minP[1]][minP[0]].insert_cpu(ps);
        }
        return true;
    }
};


template <class Cell>
struct OperateAdaptor {
    DEVICE_HOST virtual void operate(Cell& cell, NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) = 0;
};

static DEVICE_HOST GLfloat chap(GLfloat d, GLfloat rayon)
{
    return(exp(-(d * d)/(rayon * rayon)));
}

template <class Cell,
          class NIter>
struct OperateTriggerAdaptor {

    GLfloat alpha;
    GLfloat radius;

    DEVICE_HOST OperateTriggerAdaptor() : alpha(), radius(){}

    DEVICE_HOST OperateTriggerAdaptor(GLfloat a, GLfloat r) : alpha(a), radius(r){}

    DEVICE_HOST inline void operate(Cell& cell,  NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {
        PointEuclid p = nn_source.adaptiveMap[p_source[1]][p_source[0]];
        NIter ni(p_cible, 0, radius);

        for (int d = 0; d <= radius; ++d) {

            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            GLfloat alpha_temp = alpha * chap((GLfloat)cd, (GLfloat)radius*LARG_CHAP);
            PointCoord pCoord;
            do {
                pCoord = ni.getNodeIncr();
//                pCoord[0] = pCoord[0] % nn_cible.adaptiveMap.getWidth();
//                pCoord[0] = pCoord[0] < 0 ? nn_cible.adaptiveMap.getWidth() - pCoord[0] : pCoord[0];
                if (pCoord[0] >= 0 && pCoord[0] < nn_cible.adaptiveMap.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < nn_cible.adaptiveMap.getHeight()) {

                    if (!nn_cible.fixedMap[pCoord[1]][pCoord[0]]) {
                        PointEuclid n = nn_cible.adaptiveMap[pCoord[1]][pCoord[0]];
                        n[0] = n[0] + alpha_temp * (p[0] - n[0]);
                        n[1] = n[1] + alpha_temp * (p[1] - n[1]);
#ifdef CUDA_ATOMIC
                        AtomicExch<GLfloatP> ato;
                        ato((nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][0]), (n[0]));
                        ato((nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][1]), (n[1]));
//                        atomicExch(((float*)(&(nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][0]))), (float)(n[0]));
//                        atomicExch(((float*)(&(nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][1]))), (float)(n[1]));
#else
                         nn_cible.adaptiveMap[pCoord[1]][pCoord[0]] = n;
#endif
#if SOM3D
                        nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * 1.0;
                        //nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * (nn_source.densityMap[pCoord[1]][pCoord[0]] - nn_cible.densityMap[pCoord[1]][pCoord[0]]);
                        //nn_cible.colorMap[pCoord[1]][pCoord[0]] += (nn_source.colorMap[pCoord[1]][pCoord[0]] - nn_cible.colorMap[p_source[1]][p_source[0]]) * alpha_temp;
#endif
                    }
                }
            } while (ni.nextContourNodeIncr());
        }//for
        }//if
    }//operate
};//OperateTriggerAdaptor

//! JCC 200415 : add for TSP
template <class Cell,
          class NIter>
struct OperateTriggerTSPAdaptor {

    GLfloat alpha;
    GLfloat radius;
    bool isRing;

    DEVICE_HOST OperateTriggerTSPAdaptor() : alpha(), radius(), isRing(true) {}

    DEVICE_HOST OperateTriggerTSPAdaptor(GLfloat a, GLfloat r) : alpha(a), radius(r), isRing(true) {}

    DEVICE_HOST OperateTriggerTSPAdaptor(GLfloat a, GLfloat r, bool isR) : alpha(a), radius(r), isRing(isR) {}

    DEVICE_HOST inline void operate(Cell& cell,  NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {
        PointEuclid p = nn_source.adaptiveMap[p_source[1]][p_source[0]];
        NIter ni(p_cible, 0, radius);

        for (int d = 0; d <= radius; ++d) {

            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            GLfloat alpha_temp = alpha * chap((GLfloat)cd, (GLfloat)radius*LARG_CHAP);
            PointCoord pCoord;
            do {
                pCoord = ni.getNodeIncr();
                if (isRing) {
                    pCoord[0] = pCoord[0] % nn_cible.adaptiveMap.getWidth();
                    pCoord[0] = pCoord[0] < 0 ? nn_cible.adaptiveMap.getWidth() - pCoord[0] : pCoord[0];
                }
                if (pCoord[0] >= 0 && pCoord[0] < nn_cible.adaptiveMap.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < nn_cible.adaptiveMap.getHeight()) {

                    if (!nn_cible.fixedMap[pCoord[1]][pCoord[0]]) {
                        PointEuclid n = nn_cible.adaptiveMap[pCoord[1]][pCoord[0]];
                        n[0] = n[0] + alpha_temp * (p[0] - n[0]);
                        n[1] = n[1] + alpha_temp * (p[1] - n[1]);
#ifdef CUDA_ATOMIC
                        AtomicExch<GLfloatP> ato;
                        ato((nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][0]), (n[0]));
                        ato((nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][1]), (n[1]));
//                        atomicExch(((float*)(&(nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][0]))), (float)(n[0]));
//                        atomicExch(((float*)(&(nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][1]))), (float)(n[1]));
#else
                         nn_cible.adaptiveMap[pCoord[1]][pCoord[0]] = n;
#endif
#if SOM3D
                        nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * 1.0;
                        //nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * (nn_source.densityMap[pCoord[1]][pCoord[0]] - nn_cible.densityMap[pCoord[1]][pCoord[0]]);
                        //nn_cible.colorMap[pCoord[1]][pCoord[0]] += (nn_source.colorMap[pCoord[1]][pCoord[0]] - nn_cible.colorMap[p_source[1]][p_source[0]]) * alpha_temp;
#endif
                    }
                }
            } while (ni.nextContourNodeIncr());
        }//for
        }//if
    }//operate
};//OperateTriggerAdaptor

//! HW 13/04/15 : add OperateTriggerAdaptorWithColor
template <class Cell,
          class NIter>
struct OperateTriggerAdaptorWithColor {

    GLfloat alpha;
    GLfloat radius;

    DEVICE_HOST OperateTriggerAdaptorWithColor() : alpha(), radius(){}

    DEVICE_HOST OperateTriggerAdaptorWithColor(GLfloat a, GLfloat r) : alpha(a), radius(r){}

    DEVICE_HOST inline void operate(Cell& cell,  NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {
        PointEuclid p = nn_source.adaptiveMap[p_source[1]][p_source[0]];
        Point3D pColor = nn_source.colorMap[p_source[1]][p_source[0]];
#ifdef WITH_DENSITY
        GLfloat pDen = nn_source.densityMap[p_source[1]][p_source[0]];
#endif
        NIter ni(p_cible, 0, radius);

        for (int d = 0; d <= radius; ++d) {

            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            GLfloat alpha_temp = alpha * chap((GLfloat)cd, (GLfloat)radius*LARG_CHAP);
            PointCoord pCoord;
            do {
                pCoord = ni.getNodeIncr();
                if (pCoord[0] >= 0 && pCoord[0] < nn_cible.adaptiveMap.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < nn_cible.adaptiveMap.getHeight()) {

                    if (!nn_cible.fixedMap[pCoord[1]][pCoord[0]]) {
                        PointEuclid n = nn_cible.adaptiveMap[pCoord[1]][pCoord[0]];
                        n[0] = n[0] + alpha_temp * (p[0] - n[0]);
                        n[1] = n[1] + alpha_temp * (p[1] - n[1]);
                        Point3D nColor = nn_cible.colorMap[pCoord[1]][pCoord[0]];
                        nColor[0] = nColor[0] + alpha_temp * (pColor[0] - nColor[0]);
                        nColor[1] = nColor[1] + alpha_temp * (pColor[1] - nColor[1]);
                        nColor[2] = nColor[2] + alpha_temp * (pColor[2] - nColor[2]);
#ifdef CUDA_ATOMIC
                        AtomicExch<GLfloatP> ato;
                        ato((nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][0]), (n[0]));
                        ato((nn_cible.adaptiveMap[pCoord[1]][pCoord[0]][1]), (n[1]));
                        ato((nn_cible.colorMap[pCoord[1]][pCoord[0]][0]), (nColor[0]));
                        ato((nn_cible.colorMap[pCoord[1]][pCoord[0]][1]), (nColor[1]));
                        ato((nn_cible.colorMap[pCoord[1]][pCoord[0]][2]), (nColor[2]));
#else
                        nn_cible.adaptiveMap[pCoord[1]][pCoord[0]] = n;
                        nn_cible.colorMap[pCoord[1]][pCoord[0]] = nColor;
#endif
#ifdef WITH_DENSITY
                        GLfloat nDen = nn_cible.densityMap[pCoord[1]][pCoord[0]];
                        nDen = nDen + alpha_temp * (pDen - nDen);
                        nn_cible.densityMap[pCoord[1]][pCoord[0]] = nDen;
#endif
#if SOM3D
                        nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * 1.0;
                        //nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * (nn_source.densityMap[pCoord[1]][pCoord[0]] - nn_cible.densityMap[pCoord[1]][pCoord[0]]);
#endif
                    }
                }
            } while (ni.nextContourNodeIncr());
        }//for
        }//if
    }//operate
};//OperateTriggerAdaptorWithColor


template <template<typename, typename> class NeuralNet, class Cell>
struct OperateInsertAdaptor {

    DEVICE_HOST OperateInsertAdaptor() {}

    template <class NN1>
    DEVICE_HOST void operate(Cell& cell, NN1& nn_source, NeuralNet<Cell, GLfloat>& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight())
        {
            Cell& ci = (Cell&) nn_cible.adaptiveMap[p_cible[1]][p_cible[0]];
            if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                    && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {
                ci.insert(p_source);
            }
        }

    }//operateInsert
};

template <class Cell>
struct OperateInjectAdaptor {

    DEVICE_HOST OperateInjectAdaptor() {}

    DEVICE_HOST inline void operate(Cell& cell, NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight())
        {
            if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                    && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {

                Point3D& pcs = (Point3D&) nn_source.colorMap[p_source[1]][p_source[0]];
                Point3D& pcc = (Point3D&) nn_cible.colorMap[p_cible[1]][p_cible[0]];
                pcc = pcs;
            }
        }

    }//operateInject
};

template <class Cell>
struct OperateInjectAdaptorDebug {

    DEVICE_HOST OperateInjectAdaptorDebug() {}

    DEVICE_HOST inline void operate(Cell& cell, NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight())
        {
            if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                    && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {

                Point3D& pcs = (Point3D&) nn_source.colorMap[p_source[1]][p_source[0]];
                Point3D& pcc = (Point3D&) nn_cible.colorMap[p_cible[1]][p_cible[0]];
                pcc = pcs;
            }
            else
            {
                printf ("p_source[0] = %d, p_source[1] = %d \n", p_source[0], p_source[1]);
                Point3D& pcc = (Point3D&) nn_cible.colorMap[p_cible[1]][p_cible[0]];
                pcc = Point3D(0.0f,1.0f,0.0f);
            }

//            Point3D& pcc = (Point3D&) nn_cible.colorMap[p_cible[1]][p_cible[0]];
//            pcc = pcc + Point3D(0.0f,0.2f,0.0f);

        }

        else
        {
            printf ("p_cible[0] = %d, p_cible[1] = %d \n", p_cible[0], p_cible[1]);
        }

    }//operateInject
};

template <class Cell>
struct OperateInjectAdaptorAM {

    DEVICE_HOST OperateInjectAdaptorAM() {}

    DEVICE_HOST inline void operate(Cell& cell, NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight())
        {
            if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                    && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {

                Point2D& pcs = (Point2D&) nn_source.adaptiveMap[p_source[1]][p_source[0]];
                Point2D& pcc = (Point2D&) nn_cible.adaptiveMap[p_cible[1]][p_cible[0]];
                pcc = pcs;
            }
        }

    }//OperateInjectAdaptorAM
};

//! HW 13/04/15 : add OperateInjectAdaptorWithColor
template <class Cell>
struct OperateInjectAdaptorWithColor {

    DEVICE_HOST OperateInjectAdaptorWithColor() {}

    DEVICE_HOST inline void operate(Cell& cell, NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {
        operate(nn_source, nn_cible, p_source, p_cible);
    }

    DEVICE_HOST void operate(NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight())
        {
            if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                    && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {

                Point3D& pcs = (Point3D&) nn_source.colorMap[p_source[1]][p_source[0]];
                Point3D& pcc = (Point3D&) nn_cible.colorMap[p_cible[1]][p_cible[0]];
                pcc = pcs;

                Point2D& pss = (Point2D&) nn_source.adaptiveMap[p_source[1]][p_source[0]];
                Point2D& psc = (Point2D&) nn_cible.adaptiveMap[p_cible[1]][p_cible[0]];
                psc = pss;

#ifdef WITH_DENSITY
                nn_cible.densityMap[p_cible[1]][p_cible[0]] = nn_source.densityMap[p_source[1]][p_source[0]];
#endif
            }
        }

    }
};//OperateInjectAdaptorWithColor

//template <template<typename, typename> class NeuralNet, class Cell>
template <class Cell>
struct OperateIdAdaptor {

    DEVICE_HOST OperateIdAdaptor() {}

    DEVICE_HOST void operate(Cell& cell,  NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

    }//operate

    DEVICE_HOST void operate(NN& nnr, PointCoord& pc, PointEuclid& pcValue) {
        if (pc[0] >= 0 && pc[0] < nnr.adaptiveMap.getWidth()
                && pc[1] >= 0 && pc[1] < nnr.adaptiveMap.getHeight()) {
            nnr.adaptiveMap[pc[1]][pc[0]] = pcValue;
        }
    }//operate

    DEVICE_HOST void operate(NN& nnr, PointCoord& pc, PointEuclid& pcValue, AMObjectives& pcObj) {
        if (pc[0] >= 0 && pc[0] < nnr.adaptiveMap.getWidth()
                && pc[1] >= 0 && pc[1] < nnr.adaptiveMap.getHeight()) {
            nnr.adaptiveMap[pc[1]][pc[0]] = pcValue;
            nnr.objectivesMap[pc[1]][pc[0]] = pcObj;
        }
    }//operate

    DEVICE_HOST void operate(NN& nnr, PointCoord& pc, PointEuclid& pcValue,
                             AMObjectives& pcObj, AMObjectives& pcObjLeft, AMObjectives& pcObjUp) {
        if (pc[0] >= 0 && pc[0] < nnr.adaptiveMap.getWidth()
                && pc[1] >= 0 && pc[1] < nnr.adaptiveMap.getHeight()) {
            nnr.adaptiveMap[pc[1]][pc[0]] = pcValue;
            nnr.objectivesMap[pc[1]][pc[0]] = pcObj;
            if (pc[0] - 1 >= 0)
                nnr.objectivesMap[pc[1]][pc[0]-1] = pcObjLeft;
            if (pc[1] - 1 >= 0)
                nnr.objectivesMap[pc[1]-1][pc[0]] = pcObjUp;
        }
    }//operate

};//OperateIdAdaptor


template <class CellularMatrix,
          class SpiralSearchCMIterator,
          class Cell,
          template<typename, typename> class NeuralNet
          >
struct SearchSpiralComputeAvgAdaptor {

    //! Random numbers
    Grid<Cell> gGCenter;
    Grid<Point3D> gColor;
    Grid<AMObjectives> gDistr;
    Grid<GLfloat> gSize;
    Grid<GLfloat> gWeight;

    DEVICE_HOST SearchSpiralComputeAvgAdaptor() {}
    DEVICE_HOST ~SearchSpiralComputeAvgAdaptor() {
//        gGCenter.gpuFreeMem();
//        gDistr.gpuFreeMem();
//        gSize.gpuFreeMem();
    }

    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE_HOST void initialize(NeuralNet2<Cell2, GLfloat>& sched) {
        size_t w = sched.adaptiveMap.getWidth();
        size_t h = sched.adaptiveMap.getHeight();
        gGCenter.gpuResize(w, h);
        gColor.gpuResize(w, h);
        gDistr.gpuResize(w, h);
        gSize.gpuResize(w, h);
        gWeight.gpuResize(w, h);
    }

    DEVICE_HOST void free() {
        gGCenter.gpuFreeMem();
        gColor.gpuFreeMem();
        gDistr.gpuFreeMem();
        gSize.gpuFreeMem();
        gWeight.gpuFreeMem();
    }

    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE_HOST inline void init(NeuralNet2<Cell2, GLfloat>& sched) {
        BOp op;
        Grid<PointEuclid>& ga = (Grid<PointEuclid>&) sched.adaptiveMap;
        op.K_setIdentical(ga, gGCenter);
        Grid<Point3D>& gac = (Grid<Point3D>&) sched.colorMap;
        Grid<Point3D>& gcc = (Grid<Point3D>&) gColor;
        op.K_setIdentical(gac, gcc);
        //! HW 25.04.15 : Here the gWeight grid is used for gradient/density, for superpixel experiments.
        Grid<GLfloat>& gag= (Grid<GLfloat>&) sched.densityMap;
        Grid<GLfloat>& gcg = (Grid<GLfloat>&) gWeight;
        op.K_setIdentical(gag, gcg);

//        op.K_copy(sched.adaptiveMap, gGCenter);
//        op.K_copy(sched.colorMap, gColor);

        //! HW 08/04/15 : The cudaMemset2D function only allows int values.
#ifdef CUDA_CODE
//        gDistr.gpuMemSet(0);
        gWeight.gpuMemSet((GLfloat)0.0f);
        gSize.gpuMemSet((GLfloat)0.0f);
#else
        gDistr.gpuMemSet(AMObjectives(0));
        gSize.gpuMemSet(0.0f);
//        gWeight.gpuMemSet(0.0f);
//        gColor.gpuMemSet(Point3D(255,255,255));

#endif
    }

    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE bool search(CellularMatrix& cm,
                            NN& scher,
                            NeuralNet2<Cell2, GLfloat>& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIterator sps_iter(
                    PC,
                cm.vgd.getWidth() * cm.vgd.getHeight(),
                0,
                MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                1
                );
        ret = sps_iter.search(cm,
                              scher,
                              (NN&)sched,
                              ps,
                              minP);
        if (ret) {
            if (minP[0] >= 0 && minP[0] < sched.adaptiveMap.getWidth()
                    && minP[1] >= 0 && minP[1] < sched.adaptiveMap.getHeight()
                    && ps[0] >= 0 && ps[0] < scher.adaptiveMap.getWidth()
                    && ps[1] >= 0 && ps[1] < scher.adaptiveMap.getHeight())
            {
                GLfloat size = gSize[minP[1]][minP[0]];

                // gDistr update
                GLfloat f = gDistr[minP[1]][minP[0]][obj_distr];
                GLfloat mDist = sps_iter.getMinDistance();
                f *= size;
                f += mDist;
                f /= (size + 1);
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gDistr[minP[1]][minP[0]][obj_distr]))), f);
#else
                gDistr[minP[1]][minP[0]][obj_distr] = f;
#endif

                // gColor update
                Point3D p3d = gColor[minP[1]][minP[0]];
                Point3D pp3d = scher.colorMap[ps[1]][ps[0]];
                p3d = p3d * size;
                p3d = p3d + pp3d;
                p3d = p3d * (1 / (size + 1));
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gColor[minP[1]][minP[0]][0]))), p3d[0]);
                atomicExch(((float*)(&(gColor[minP[1]][minP[0]][1]))), p3d[1]);
                atomicExch(((float*)(&(gColor[minP[1]][minP[0]][2]))), p3d[2]);
#else
                gColor[minP[1]][minP[0]] = p3d;
#endif

                // gGCenter update
#ifdef USE_DENSITY_MAP
                GLfloat weight = gWeight[minP[1]][minP[0]];
                PointEuclid p = gGCenter[minP[1]][minP[0]];
                PointEuclid pp = scher.adaptiveMap[ps[1]][ps[0]];
                GLfloat w = scher.densityMap[ps[1]][ps[0]];
                p = p * weight;
                w = MAX(w, 1.0f);
                p = p + pp * w;
                p = p / (weight + w);
                (PointEuclid&) gGCenter[minP[1]][minP[0]] = p;
                gWeight[minP[1]][minP[0]] += w;
#else
                PointEuclid p = gGCenter[minP[1]][minP[0]];
                PointEuclid pp = scher.adaptiveMap[ps[1]][ps[0]];
                p = p * size;
                p = p + pp;
                p = p * (1 / (size + 1));
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gGCenter[minP[1]][minP[0]][0]))), p[0]);
                atomicExch(((float*)(&(gGCenter[minP[1]][minP[0]][1]))), p[1]);
#else
                (PointEuclid&) gGCenter[minP[1]][minP[0]] = p;
#endif
#endif//#ifdef USE_DENSITY_MAP

#ifdef WITH_DENSITY
                // gWeight update
                GLfloat pwc = gWeight[minP[1]][minP[0]];
                GLfloat pwp = scher.densityMap[ps[1]][ps[0]];
                pwc = pwc * size;
                pwc = pwc + pwp;
                pwc = pwc / (size + 1);
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gWeight[minP[1]][minP[0]]))), pwc);
#else
                gWeight[minP[1]][minP[0]] = pwc;
#endif
#endif//#ifdef WITH_DENSITY

                // gSize update
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gSize[minP[1]][minP[0]]))), 1.0f);
#else
                gSize[minP[1]][minP[0]] += 1;
#endif

                // fixedMap update
                sched.fixedMap[minP[1]][minP[0]] = false;
            }
        }

        return ret;
    }


    //! HW 23/05/15 : overload for superpixel application
    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE bool search(CellularMatrix& cm,
                            NN& scher,
                            NeuralNet2<Cell2, GLfloat>& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP,
                            Grid<PointCoord>& gLabel) {
        bool ret = false;

        SpiralSearchCMIterator sps_iter(
                    PC,
                cm.vgd.getWidth() * cm.vgd.getHeight(),
                0,
                MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                1
                );
        ret = sps_iter.search(cm,
                              scher,
                              (NN&)sched,
                              ps,
                              minP);
        if (ret) {
            if (minP[0] >= 0 && minP[0] < sched.adaptiveMap.getWidth()
                    && minP[1] >= 0 && minP[1] < sched.adaptiveMap.getHeight()
                    && ps[0] >= 0 && ps[0] < scher.adaptiveMap.getWidth()
                    && ps[1] >= 0 && ps[1] < scher.adaptiveMap.getHeight())
            {
                gLabel[ps[1]][ps[0]] = minP;

                GLfloat size = gSize[minP[1]][minP[0]];

                // gDistr update
                GLfloat f = gDistr[minP[1]][minP[0]][obj_distr];
                GLfloat mDist = sps_iter.getMinDistance();
                f *= size;
                f += mDist;
                f /= (size + 1);
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gDistr[minP[1]][minP[0]][obj_distr]))), f);
#else
                gDistr[minP[1]][minP[0]][obj_distr] = f;
#endif

                // gColor update
                Point3D p3d = gColor[minP[1]][minP[0]];
                Point3D pp3d = scher.colorMap[ps[1]][ps[0]];
                p3d = p3d * size;
                p3d = p3d + pp3d;
                p3d = p3d * (1 / (size + 1));
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gColor[minP[1]][minP[0]][0]))), p3d[0]);
                atomicExch(((float*)(&(gColor[minP[1]][minP[0]][1]))), p3d[1]);
                atomicExch(((float*)(&(gColor[minP[1]][minP[0]][2]))), p3d[2]);
#else
                gColor[minP[1]][minP[0]] = p3d;
#endif

                // gGCenter update
#ifdef USE_DENSITY_MAP
                GLfloat weight = gWeight[minP[1]][minP[0]];
                PointEuclid p = gGCenter[minP[1]][minP[0]];
                PointEuclid pp = scher.adaptiveMap[ps[1]][ps[0]];
                GLfloat w = scher.densityMap[ps[1]][ps[0]];
                p = p * weight;
                w = MAX(w, 1.0f);
                p = p + pp * w;
                p = p / (weight + w);
                (PointEuclid&) gGCenter[minP[1]][minP[0]] = p;
                gWeight[minP[1]][minP[0]] += w;
#else
                PointEuclid p = gGCenter[minP[1]][minP[0]];
                PointEuclid pp = scher.adaptiveMap[ps[1]][ps[0]];
                p = p * size;
                p = p + pp;
                p = p * (1 / (size + 1));
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gGCenter[minP[1]][minP[0]][0]))), p[0]);
                atomicExch(((float*)(&(gGCenter[minP[1]][minP[0]][1]))), p[1]);
#else
                (PointEuclid&) gGCenter[minP[1]][minP[0]] = p;
#endif
#endif//#ifdef USE_DENSITY_MAP

#ifdef WITH_DENSITY
                // gWeight update
                GLfloat pwc = gWeight[minP[1]][minP[0]];
                GLfloat pwp = scher.densityMap[ps[1]][ps[0]];
                pwc = pwc * size;
                pwc = pwc + pwp;
                pwc = pwc / (size + 1);
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gWeight[minP[1]][minP[0]]))), pwc);
#else
                gWeight[minP[1]][minP[0]] = pwc;
#endif
#endif//#ifdef WITH_DENSITY

                // gSize update
#ifdef CUDA_ATOMIC
                atomicExch(((float*)(&(gSize[minP[1]][minP[0]]))), 1.0f);
#else
                gSize[minP[1]][minP[0]] += 1;
#endif

                // fixedMap update
                sched.fixedMap[minP[1]][minP[0]] = false;
            }
        }
        else
        {
            printf("minP=(%d,%d), ps=(%d,%d)\n", minP[0], minP[1], ps[0], ps[1]);
        }

        return ret;
    }

};

template <class CellularMatrix,
          class SpiralSearchCMIterator,
          class Cell,
          template<typename, typename> class NeuralNet
          >
struct SearchSpiralComputeAvgBufferAdaptor
        : public SearchSpiralComputeAvgAdaptor<
        CellularMatrix,
        SpiralSearchCMIterator,
        Cell,
        NeuralNet
        >
{

    DEVICE_HOST SearchSpiralComputeAvgBufferAdaptor()
        : SearchSpiralComputeAvgAdaptor<
          CellularMatrix,
          SpiralSearchCMIterator,
          Cell,
          NeuralNet
          >() {}

    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE_HOST void initialize(NeuralNet2<Cell2, GLfloat>& sched) {
        SearchSpiralComputeAvgAdaptor<
                        CellularMatrix,
                        SpiralSearchCMIterator,
                        Cell,
                        NeuralNet
                        >::initialize(sched);
    }

    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE_HOST inline void init(NeuralNet2<Cell2, GLfloat>& sched) {

        SearchSpiralComputeAvgAdaptor<
                        CellularMatrix,
                        SpiralSearchCMIterator,
                        Cell,
                        NeuralNet
                        >::init(sched);

        //! HW 08/04/15 : add a new method K_cleanGridOfCell in BOp
        BOp op;
        op.K_cleanGridOfCell(this->gGCenter);
    }


    template <
            template<typename, typename> class NeuralNet2,
            class Cell2>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NeuralNet2<Cell2, GLfloat>& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        ret = SearchSpiralComputeAvgAdaptor<
                CellularMatrix,
                SpiralSearchCMIterator,
                Cell,
                NeuralNet
                >::search(cm,
                          scher,
                          sched,
                          PC,
                          ps,
                          minP);

        if (ret) {

            if (minP[0] >= 0 && minP[0] < sched.adaptiveMap.getWidth()
                    && minP[1] >= 0 && minP[1] < sched.adaptiveMap.getHeight()
                    && ps[0] >= 0 && ps[0] < scher.adaptiveMap.getWidth()
                    && ps[1] >= 0 && ps[1] < scher.adaptiveMap.getHeight())
            {
                Cell& ci = (Cell&) this->gGCenter[minP[1]][minP[0]];
                ci.insert(ps);
            }
        }

        return ret;
    }

};

//! HW 01.06.15: add new adaptors for distributed local search
template <class Cell, class NIter, int radius = 1>
struct PropagateNeighbor {

    NIter niter;

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        niter.initialize(pc, 0, radius);
    }

    DEVICE_HOST inline void init(PointCoord& pc) {
        niter.initialize(pc, 0, radius);
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        return (get(nnr, nnd, pc, pNew));
    }

    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        bool ret = false;

        PointCoord pco = niter.get();

        if (pco[0] >= 0 && pco[0] < nnr.adaptiveMap.getWidth()
                && pco[1] >= 0 && pco[1] < nnr.adaptiveMap.getHeight())
        {
            ret = true;
            PointEuclid motion = nnr.adaptiveMap[pco[1]][pco[0]];
            motion -= nnd.adaptiveMap[pco[1]][pco[0]];
            pNew = nnd.adaptiveMap[pc[1]][pc[0]];
            pNew += motion;

        }

        return ret;
    }

    DEVICE_HOST inline bool next() {
        return niter.next();
    }
};//PropagateNeighbor

template <class Cell, class NIter, int radius = 1>
struct PropagateNeighborRandom {

    //! Random numbers
    RandGrid gRand;

    PropagateNeighborRandom() : nb(0), niter(1) {}
    ~PropagateNeighborRandom() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t niter;
    size_t gene;

    NIter iter;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        iter.initialize(pc, 0, radius);
        nb = pindex;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        size_t neighborhoodSize = iter.getTotalSizeN(radius);

        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        //! HW 17/04/15 : The reason of using MIN() is that rand is from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
        float i = MIN((((float)neighborhoodSize) * rand), ((float)(neighborhoodSize - 1)));

        PointCoord pco;
        pco = iter.getNode((int)i);

        if (pco[0] >= 0 && pco[0] < nnr.adaptiveMap.getWidth()
                && pco[1] >= 0 && pco[1] < nnr.adaptiveMap.getHeight())
        {
            PointEuclid motion = nnr.adaptiveMap[pco[1]][pco[0]];
            motion -= nnd.adaptiveMap[pco[1]][pco[0]];
            pNew = nnd.adaptiveMap[pc[1]][pc[0]];
            pNew += motion;
        }
        else
            return false;

        return true;
    }

    DEVICE_HOST inline bool next() {
        return false;
    }

//    DEVICE_HOST inline bool next() {
//        return ++nb < niter;
//    }
};//PropagateNeighborRandom

template <class Cell, class NIter, int radius = 2, int scale = 1>
struct GenerateNeighbor {

    NIterQuad niter;

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        niter.initialize(pc, 0, radius);
    }

    DEVICE_HOST inline void init(PointCoord& pc) {
        niter.initialize(pc, 0, radius);
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        return (get(nnr, nnd, pc, pNew));
    }

    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        PointCoord pco = niter.get();
        pco -= pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));
        pNew = nnr.adaptiveMap[pc[1]][pc[0]];
        pNew += dis;

        //! HW 260815 : add disparity range check for stereo
#ifdef DISP_RANGE

        if (fabs(nnd.adaptiveMap[pc[1]][pc[0]] - pNew) >= DISP_RANGE)
            return false;
#endif
        return true;
    }

//    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pOld,
//                                AMObjectives& objOld, AMObjectives& objOldLeft, AMObjectives& objOldUp) {

//        PointCoord pco = niter.get();
//        pco -= pc;
//        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));
//        pOld = nnr.adaptiveMap[pc[1]][pc[0]];
//        objOld = nnr.objectivesMap[pc[1]][pc[0]];
//        if (pc[0] - 1 >= 0)
//            objOldLeft = nnr.objectivesMap[pc[1]][pc[0] - 1];
//        if (pc[1] - 1 >= 0)
//            objOldUp = nnr.objectivesMap[pc[1] - 1][pc[0]];
//        nnr.adaptiveMap[pc[1]][pc[0]] = pOld + dis;
//        return true;

    //    }

    DEVICE_HOST inline bool next() {
        return niter.next();
    }
};//GenerateNeighbor

template <class Cell, class NIter, int radius = 10, int scale = 1>
struct GenerateNeighborRandom {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborRandom() : nb(0), niter(1) {}
    ~GenerateNeighborRandom() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t niter;
    size_t gene;

    NIter iter;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        iter.initialize(pc, 0, radius);
        nb = pindex;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        size_t neighborhoodSize = iter.getTotalSizeN(radius);

        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        //! HW 17/04/15 : The reason of using MIN() is that rand is from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
        float i = MIN((((float)neighborhoodSize) * rand), ((float)(neighborhoodSize - 1)));

        PointCoord pco;
        pco = iter.getNode((int)i);
        pco -= pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));
        pNew = nnr.adaptiveMap[pc[1]][pc[0]];
        pNew += dis;

        //! HW 260815 : add disparity range check for stereo
#ifdef DISP_RANGE

        if (fabs(nnd.adaptiveMap[pc[1]][pc[0]] - pNew) >= DISP_RANGE)
            return false;
#endif
        return true;
    }

    DEVICE_HOST inline bool next() {
        return false;
    }

//    DEVICE_HOST inline bool next() {
//        return ++nb < niter;
//    }
};//GenerateNeighborRandom

//! HW 060815 : the simple one direction move operator for variable neighborhood descent (VND)
template <class Cell, class NIter, int radius = 10, int scale = 1>
struct GenerateNeighborInFixedDirection {

    NIter niter;
    size_t neighborhoodNum;
    size_t direction;

    DEVICE_HOST inline void init(Cell& cell, PointCoord& pc, size_t pindex) {
        niter.initialize(pc, 0, radius);
        neighborhoodNum = niter.getSizeN();
        direction = 0;
    }

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        niter.initialize(pc, 0, radius);
        neighborhoodNum = niter.getSizeN();
        direction = 0;
    }

    DEVICE_HOST inline void init(PointCoord& pc) {
        niter.initialize(pc, 0, radius);
        neighborhoodNum = niter.getSizeN();
        direction = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        return (get(nnr, nnd, pc, pNew));
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew, size_t distance) {

        return (get(nnr, nnd, pc, pNew, distance));
    }

    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        size_t distance = niter.getCurrentDistance();
        PointCoord pco = niter.goTo(pc, direction, distance);
        pco -= pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));
        pNew = nnr.adaptiveMap[pc[1]][pc[0]];
        pNew += dis;

        return true;
    }

    //! HW 060815 : Note that here the radius is not a restrict since the "distance" is given as an argument.
    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew, size_t distance) {

        PointCoord pco = niter.goTo(pc, direction, distance);
        pco -= pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));
        pNew = nnr.adaptiveMap[pc[1]][pc[0]];
        pNew += dis;

        return true;
    }

    DEVICE_HOST inline bool next() {
        return niter.nextDistanceIncr();
    }

    DEVICE_HOST inline bool nextNeighborhood() {
        niter.setCurrentDistance(0);
        return ++direction < neighborhoodNum;
    }
};//GenerateNeighborInFixedDirection

template <class Cell, class NIter, int radius = 10, int scale = 1>
struct GenerateNeighborInFixedDirectionRandom {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborInFixedDirectionRandom() : nb(0), niter(1) {}
    ~GenerateNeighborInFixedDirectionRandom() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t niter;
    size_t gene;

    NIter iter;
    size_t neighborhoodNum;
    size_t direction;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline void init(Cell& cell, PointCoord& pc, size_t pindex) {
        iter.initialize(pc, 0, radius);
        neighborhoodNum = iter.getSizeN();
        nb = pindex;

        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        //! HW 17/04/15 : The reason of using MIN() is that rand is from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
        float i = MIN((((float)neighborhoodNum) * rand), ((float)(neighborhoodNum - 1)));
        direction = (int)i;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        size_t distance = iter.getCurrentDistance();
        PointCoord pco = iter.goTo(pc, direction, distance);
        pco -= pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));
        pNew = nnr.adaptiveMap[pc[1]][pc[0]];
        pNew += dis;

        return true;
    }

    DEVICE_HOST inline bool next() {
        return iter.nextDistanceIncr();
    }

    DEVICE_HOST inline bool nextNeighborhood() {
        return false;
    }
};//GenerateNeighborInFixedDirectionRandom

template <class Cell, int radius = 64, int scale = 1, bool left = true>
struct GenerateNeighborStereo {

    int disp;

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        disp = 0;
    }

    DEVICE_HOST inline void init(PointCoord& pc) {
        disp = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        return (get(nnr, nnd, pc, pNew));
    }

    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        PointEuclid dis((float)disp * (1.0f / (float)scale),  0.0f);
        pNew = nnd.adaptiveMap[pc[1]][pc[0]];
        if (left)
            pNew -= dis;
        else
            pNew += dis;

        return true;
    }

    DEVICE_HOST inline bool next() {
        disp++;

#ifdef DISP_RANGE
        return (disp < DISP_RANGE);
#else
        return (disp < radius);
#endif
    }

};//GenerateNeighborStereo

template <class Cell, int radius = 64, int scale = 1, bool left = true>
struct GenerateNeighborStereoRandom {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborStereoRandom() : nb(0), niter(1) {}
    ~GenerateNeighborStereoRandom() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t niter;
    size_t gene;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {
        nb = pindex;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        //! HW 17/04/15 : The reason of using MIN() is that rand is from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
#ifdef DISP_RANGE
        float i = MIN((((float)DISP_RANGE) * rand), ((float)(DISP_RANGE - 1)));
#else
        float i = MIN((((float)radius) * rand), ((float)(radius - 1)));
#endif
        int disp = (int)i;
        PointEuclid dis((float)disp * (1.0f / (float)scale), 0.0f);

        pNew = nnd.adaptiveMap[pc[1]][pc[0]];
        if (left)
            pNew -= dis;
        else
            pNew += dis;

        return true;
    }

    DEVICE_HOST inline bool next() {
        return false;
    }

};//GenerateNeighborStereoRandom

template <class Cell, int radius = 64, bool left = true>
struct GenerateNeighborStereoPerturbation {


    DEVICE_HOST inline void init(PointCoord& pc, size_t pindex) {

    }

    DEVICE_HOST inline void init(PointCoord& pc) {

    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

        return (get(nnr, nnd, pc, pNew));
    }

    DEVICE_HOST inline bool get(NN& nnr, NN& nnd, PointCoord& pc, PointEuclid& pNew) {

#ifdef DISP_RANGE
        float stride = DISP_RANGE / 2.0f;
#else
        float stride = (float)radius / 2.0f;
#endif
        pNew = nnr.adaptiveMap[pc[1]][pc[0]];

        if (left) {
#ifdef DISP_RANGE
            if ((nnr.adaptiveMap[pc[1]][pc[0]][0] - stride) >= (nnd.adaptiveMap[pc[1]][pc[0]][0] - (DISP_RANGE - 1)))
#else
            if ((nnr.adaptiveMap[pc[1]][pc[0]][0] - stride) >= (nnd.adaptiveMap[pc[1]][pc[0]][0] - (radius - 1)))
#endif
                pNew[0] = nnr.adaptiveMap[pc[1]][pc[0]][0] - stride;
            else
                pNew[0] = nnr.adaptiveMap[pc[1]][pc[0]][0] + stride;
        }
        else
        {
#ifdef DISP_RANGE
            if ((nnr.adaptiveMap[pc[1]][pc[0]][0] + stride) <= (nnd.adaptiveMap[pc[1]][pc[0]][0] + (DISP_RANGE - 1)))
#else
            if ((nnr.adaptiveMap[pc[1]][pc[0]][0] + stride) <= (nnd.adaptiveMap[pc[1]][pc[0]][0] + (radius - 1)))
#endif
                pNew[0] = nnr.adaptiveMap[pc[1]][pc[0]][0] + stride;
            else
                pNew[0] = nnr.adaptiveMap[pc[1]][pc[0]][0] - stride;
        }

        return true;
    }

    DEVICE_HOST inline bool next() {
        return false;
    }

};//GenerateNeighborStereoPerturbation

template <class Cell, class NIter, class NIterWin,
          int windowRadius = 2, int radius = 10, int scale = 1>
struct GenerateNeighborCenterWindow {

    int* backup_PointCoord;
    float* backup_PointEuclid;
    int buffSize;
    NIter niter;
    NIterWin niterWindow;

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell);
    }

    DEVICE_HOST inline void init(Cell& cell) {
        niter.initialize(cell.pc, 0, radius);
        niterWindow.initialize(cell.pc, 0, windowRadius);
        buffSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * buffSize * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * buffSize * sizeof(float));

        niterWindow.init();
        int i = 0;
        do {
            PointCoord ps = niterWindow.get();
            if (i < buffSize)
            {
                backup_PointCoord[2*i] = ps[0];
                backup_PointCoord[2*i+1] = ps[1];
                i++;
            }
        } while (niterWindow.next());
    }

    DEVICE_HOST inline void backup(NN& nnr) {

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        PointCoord pco = niter.get();
        pco -= cell.pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

#ifdef DISP_RANGE
                if (fabs(nnd.adaptiveMap[ps[1]][ps[0]] - (nnr.adaptiveMap[ps[1]][ps[0]] + dis)) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {
        return (niter.next());
    }

    DEVICE_HOST inline void clean() {
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborCenterWindow

template <class Cell, class NIter, class NIterWin,
          int windowRadius = 2, int moveTimes = 20, int radius = 10, int scale = 1>
struct GenerateNeighborCenterWindowRandomMove {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborCenterWindowRandomMove() : nb(0) {}
    ~GenerateNeighborCenterWindowRandomMove() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t gene;

    int* backup_PointCoord;
    float* backup_PointEuclid;
    int buffSize;
    int neighborhoodSize;
    NIter niter;
    NIterWin niterWindow;

    // for addressing random numbers in the dynamic CM pattern
    int curCM;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell, curCellularMatrix);
    }

    DEVICE_HOST inline void init(Cell& cell, size_t curCellularMatrix) {
        niter.initialize(cell.pc, 0, radius);
        neighborhoodSize = niter.getTotalSizeN();
        niterWindow.initialize(cell.pc, 0, windowRadius);
        buffSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * buffSize * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * buffSize * sizeof(float));
        nb = 0;
        curCM = curCellularMatrix;

        niterWindow.init();
        int i = 0;
        do {
            PointCoord ps = niterWindow.get();
            if (i < buffSize)
            {
                backup_PointCoord[2*i] = ps[0];
                backup_PointCoord[2*i+1] = ps[1];
                i++;
            }
        } while (niterWindow.next());

    }

    DEVICE_HOST inline void backup(NN& nnr) {

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*moveTimes+curCM*moveTimes+nb) % MAX_RAND_BUFFER_SIZE];
        float iRand = MIN((((float)neighborhoodSize) * rand), ((float)(neighborhoodSize - 1)));
        PointCoord pco;
        pco = niter.getNode((int)iRand);
        pco -= cell.pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

#ifdef DISP_RANGE
                if (fabs(nnd.adaptiveMap[ps[1]][ps[0]] - (nnr.adaptiveMap[ps[1]][ps[0]] + dis)) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {
        return (++nb < moveTimes);
    }

    DEVICE_HOST inline void clean() {
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborCenterWindowRandomMove

template <class Cell, class NIterWin,
          int windowRadius = 2, int radius = 64, int scale = 1, bool left = true>
struct GenerateNeighborCenterWindowStereo {

    int disp;
    int* backup_PointCoord;
    float* backup_PointEuclid;
    int buffSize;
    NIterWin niterWindow;

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell);
    }

    DEVICE_HOST inline void init(Cell& cell) {
        disp = 0;
        niterWindow.initialize(cell.pc, 0, windowRadius);
        buffSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * buffSize * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * buffSize * sizeof(float));

        niterWindow.init();
        int i = 0;
        do {
            PointCoord ps = niterWindow.get();
            if (i < buffSize)
            {
                backup_PointCoord[2*i] = ps[0];
                backup_PointCoord[2*i+1] = ps[1];
                i++;
            }
        } while (niterWindow.next());
    }

    DEVICE_HOST inline void backup(NN& nnr) {

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        PointEuclid dis((float)disp * (1.0f / (float)scale), 0.0f);

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                if (left)
                    nnr.adaptiveMap[ps[1]][ps[0]] = nnd.adaptiveMap[ps[1]][ps[0]] - dis;
                else
                    nnr.adaptiveMap[ps[1]][ps[0]] = nnd.adaptiveMap[ps[1]][ps[0]] + dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        for (int i = 0; i < buffSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {
        disp++;

#ifdef DISP_RANGE
        return (disp < DISP_RANGE);
#else
        return (disp < radius);
#endif
    }

    DEVICE_HOST inline void clean() {
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborCenterWindowStereo

template <class Cell, class NIterWin,
          int windowRadius = 3, int maxPickTimes = 10, int radius = 64, int scale = 1, bool left = true>
struct GenerateNeighborRandomWindowStereo {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborRandomWindowStereo() {}
    ~GenerateNeighborRandomWindowStereo() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t gene;

    int disp;
    int* backup_PointCoord;
    float* backup_PointEuclid;
    int pickWindowSize;
    NIterWin niterWindow;

    bool pickedSucc;

    // for addressing random numbers in the dynamic CM pattern
    int curCM;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell, curCellularMatrix);
    }

    DEVICE_HOST inline void init(Cell& cell, size_t curCellularMatrix) {

        curCM = curCellularMatrix;
        niterWindow.initialize(cell.pc, 0, (cell.radius-1));
        pickWindowSize = niterWindow.getTotalSizeN();
        PointCoord pickedCenter;

        pickedSucc = false;
        for (int i = 0; i < maxPickTimes; i++)
        {
            GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*maxPickTimes+curCM*maxPickTimes+i) % MAX_RAND_BUFFER_SIZE];
            float iRand = MIN((((float)pickWindowSize) * rand), ((float)(pickWindowSize - 1)));
            pickedCenter = niterWindow.getNode((int)iRand);
            if (((cell.radius-1) - niterWindow.getCurrentDistance()) >= windowRadius)
            {
                pickedSucc = true;
                break;
            }
        }

        if (!pickedSucc)
            return;

        disp = 0;
        niterWindow.initialize(pickedCenter, 0, windowRadius);
        pickWindowSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * pickWindowSize * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * pickWindowSize * sizeof(float));

        niterWindow.init();
        int i = 0;
        do {
            PointCoord ps = niterWindow.get();
            if (i < pickWindowSize)
            {
                backup_PointCoord[2*i] = ps[0];
                backup_PointCoord[2*i+1] = ps[1];
                i++;
            }
        } while (niterWindow.next());
    }

    DEVICE_HOST inline void backup(NN& nnr) {

        if (!pickedSucc)
            return;

        for (int i = 0; i < pickWindowSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        if (!pickedSucc)
            return;

        PointEuclid dis((float)disp * (1.0f / (float)scale), 0.0f);

        for (int i = 0; i < pickWindowSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {

                if (left)
                    nnr.adaptiveMap[ps[1]][ps[0]] = nnd.adaptiveMap[ps[1]][ps[0]] - dis;
                else
                    nnr.adaptiveMap[ps[1]][ps[0]] = nnd.adaptiveMap[ps[1]][ps[0]] + dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        if (!pickedSucc)
            return;

        for (int i = 0; i < pickWindowSize; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {

        if (!pickedSucc)
            return false;

        disp++;

#ifdef DISP_RANGE
        return (disp < DISP_RANGE);
#else
        return (disp < radius);
#endif
    }

    DEVICE_HOST inline void clean() {
        if (!pickedSucc)
            return;
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborRandomWindowStereo

template <class Cell, class NIter, class NIterWin,
          int nbPicked = 20, int radius = 10, int scale = 1>
struct GenerateNeighborRandomPick {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborRandomPick() {}
    ~GenerateNeighborRandomPick() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t gene;

    int* backup_PointCoord;
    float* backup_PointEuclid;
    int neighborhoodSize;
    int pickWindowSize;
    NIter niter;
    NIterWin niterWindow;

    // for addressing random numbers in the dynamic CM pattern
    int curCM;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell, curCellularMatrix);
    }

    DEVICE_HOST inline void init(Cell& cell, size_t curCellularMatrix) {
        niter.initialize(cell.pc, 0, radius);
        neighborhoodSize = niter.getTotalSizeN();
        niterWindow.initialize(cell.pc, 0, (cell.radius-1));
        pickWindowSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * nbPicked * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * nbPicked * sizeof(float));
        curCM = curCellularMatrix;

        for (int i = 0; i < nbPicked; i++)
        {
            GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*nbPicked+curCM*nbPicked+i) % MAX_RAND_BUFFER_SIZE];
            float iRand = MIN((((float)pickWindowSize) * rand), ((float)(pickWindowSize - 1)));
            PointCoord ps;
            ps = niterWindow.getNode((int)iRand);
            backup_PointCoord[2*i] = ps[0];
            backup_PointCoord[2*i+1] = ps[1];
        }
    }

    DEVICE_HOST inline void backup(NN& nnr) {

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        PointCoord pco = niter.get();
        pco -= cell.pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

#ifdef DISP_RANGE
                if (fabs(nnd.adaptiveMap[ps[1]][ps[0]] - (nnr.adaptiveMap[ps[1]][ps[0]] + dis)) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {
        return (niter.next());
    }

    DEVICE_HOST inline void clean() {
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborRandomPick

template <class Cell, class NIter, class NIterWin,
          int nbPicked = 20, int moveTimes = 20, int radius = 10, int scale = 1>
struct GenerateNeighborRandomPickRandomMove {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborRandomPickRandomMove() : nb(0) {}
    ~GenerateNeighborRandomPickRandomMove() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t nb;
    size_t gene;

    int* backup_PointCoord;
    float* backup_PointEuclid;
    int neighborhoodSize;
    int pickWindowSize;
    NIter niter;
    NIterWin niterWindow;

    // for addressing random numbers in the dynamic CM pattern
    int curCM;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell, curCellularMatrix);
    }

    DEVICE_HOST inline void init(Cell& cell, size_t curCellularMatrix) {
        niter.initialize(cell.pc, 0, radius);
        neighborhoodSize = niter.getTotalSizeN();
        niterWindow.initialize(cell.pc, 0, (cell.radius-1));
        pickWindowSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * nbPicked * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * nbPicked * sizeof(float));
        nb = 0;
        curCM = curCellularMatrix;
        for (int i = 0; i < nbPicked; i++)
        {
            GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*(nbPicked+moveTimes)+curCM*(nbPicked+moveTimes)+nb++) % MAX_RAND_BUFFER_SIZE];
            float iRand = MIN((((float)pickWindowSize) * rand), ((float)(pickWindowSize - 1)));
            PointCoord ps;
            ps = niterWindow.getNode((int)iRand);
            backup_PointCoord[2*i] = ps[0];
            backup_PointCoord[2*i+1] = ps[1];
        }
    }

    DEVICE_HOST inline void backup(NN& nnr) {

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*(nbPicked+moveTimes)+curCM*(nbPicked+moveTimes)+nb) % MAX_RAND_BUFFER_SIZE];
        float iRand = MIN((((float)neighborhoodSize) * rand), ((float)(neighborhoodSize - 1)));
        PointCoord pco;
        pco = niter.getNode((int)iRand);
        pco -= cell.pc;
        PointEuclid dis((float)pco[0] * (1.0f / (float)scale), (float)pco[1] * (1.0f / (float)scale));

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

#ifdef DISP_RANGE
                if (fabs(nnd.adaptiveMap[ps[1]][ps[0]] - (nnr.adaptiveMap[ps[1]][ps[0]] + dis)) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {
        return (++nb < (nbPicked+moveTimes));
    }

    DEVICE_HOST inline void clean() {
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborRandomPickRandomMove

template <class Cell, class NIterWin,
          int nbPicked = 20, int radius = 64, int scale = 1, bool left = true>
struct GenerateNeighborRandomPickStereo {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborRandomPickStereo() {}
    ~GenerateNeighborRandomPickStereo() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t gene;

    int disp;
    int* backup_PointCoord;
    float* backup_PointEuclid;
    int pickWindowSize;
    NIterWin niterWindow;

    // for addressing random numbers in the dynamic CM pattern
    int curCM;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {
        init(cell, curCellularMatrix);
    }

    DEVICE_HOST inline void init(Cell& cell, size_t curCellularMatrix) {
        disp = 0;
        niterWindow.initialize(cell.pc, 0, (cell.radius-1));
        pickWindowSize = niterWindow.getTotalSizeN();
        backup_PointCoord = (int*) malloc(2 * nbPicked * sizeof(int));
        backup_PointEuclid = (float*) malloc(2 * nbPicked * sizeof(float));
        curCM = curCellularMatrix;

        for (int i = 0; i < nbPicked; i++)
        {
            GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*nbPicked+curCM*nbPicked+i) % MAX_RAND_BUFFER_SIZE];
            float iRand = MIN((((float)pickWindowSize) * rand), ((float)(pickWindowSize - 1)));
            PointCoord ps;
            ps = niterWindow.getNode((int)iRand);
            backup_PointCoord[2*i] = ps[0];
            backup_PointCoord[2*i+1] = ps[1];
        }
    }

    DEVICE_HOST inline void backup(NN& nnr) {

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                backup_PointEuclid[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
                backup_PointEuclid[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
            }
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        PointEuclid dis((float)disp * (1.0f / (float)scale), 0.0f);

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {

                if (left)
                    nnr.adaptiveMap[ps[1]][ps[0]] = nnd.adaptiveMap[ps[1]][ps[0]] - dis;
                else
                    nnr.adaptiveMap[ps[1]][ps[0]] = nnd.adaptiveMap[ps[1]][ps[0]] + dis;
            }
        }
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        for (int i = 0; i < nbPicked; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord[2*i];
            ps[1] = backup_PointCoord[2*i+1];

            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight()) {

                nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid[2*i];
                nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid[2*i+1];
            }
        }
    }

    DEVICE_HOST inline bool next() {
        disp++;

#ifdef DISP_RANGE
        return (disp < DISP_RANGE);
#else
        return (disp < radius);
#endif
    }

    DEVICE_HOST inline void clean() {
        free (backup_PointCoord);
        free (backup_PointEuclid);
    }

}; // GenerateNeighborRandomPickStereo

template <class Cell, class NIter, int maxNbPicked = 100>
struct GenerateNeighborRandomPickExpansionMoveAndSwapMove {

    //! Random numbers
    RandGrid gRand;

    GenerateNeighborRandomPickExpansionMoveAndSwapMove() {}
    ~GenerateNeighborRandomPickExpansionMoveAndSwapMove() {
//        gRand.gpuFreeMem();
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRand.gpuResize(w, h);
        RandGridAlloc rga;
        rga.K_generateRandNumBuffer(gRand);
    }

    size_t gene;
    int scale = 1;

    int* backup_PointCoord_alpha;
    float* backup_PointEuclid_alpha;
    int* backup_PointCoord_beta;
    float* backup_PointEuclid_beta;
    int nbPicked_alpha;
    int nbPicked_beta;
    NIter niter;
    PointEuclid p_alpha_disp;
    PointEuclid p_beta_disp;
    bool pickedSucc;
    int moveTimes; // only three times, firstly alpha -> beta, secondly beta -> alpha, thirdly alpha <->beta

    // for addressing random numbers in the dynamic CM pattern
    int curCM;

    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(Cell& cell, NN& nnr, NN& nnd, size_t curCellularMatrix) {

        niter.initialize(cell.pc, 0, (cell.radius-1));
        int cellSize = niter.getTotalSizeN();
        moveTimes = 0;
        curCM = curCellularMatrix;

        // Here only 4 random numbers are needed for this operator
        // one for determine the number of alpha-labeled pixels
        // one for determine the number of beta-labeled pixels
        // one for determine the position of the randomly picked first alpha-labeled pixel
        // one for determine the position of the randomly picked first beta-labeled pixel
        GLfloat rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*4+curCM*4) % MAX_RAND_BUFFER_SIZE];
        float iRand = MIN((((float)cellSize) * rand), ((float)(cellSize - 1)));
        PointCoord p_alpha;
        p_alpha = niter.getNode((int)iRand);
        p_alpha -= cell.pc;
//        if (!(p_alpha[0] >= 0
//            && p_alpha[0] < nnr.adaptiveMap.getWidth()
//            && p_alpha[1] >= 0
//            && p_alpha[1] < nnr.adaptiveMap.getHeight()))
//            p_alpha = cell.pc;
//        p_alpha_disp = nnr.adaptiveMap[p_alpha[1]][p_alpha[0]] - nnd.adaptiveMap[p_alpha[1]][p_alpha[0]];

        PointEuclid disA((float)p_alpha[0] * (1.0f / (float)scale), (float)p_alpha[1] * (1.0f / (float)scale));
        p_alpha_disp = disA;
        //p_alpha_disp = nnr.adaptiveMap.fetchIntCoor(p_alpha[0],p_alpha[1]);// - nnd.adaptiveMap.fetchIntCoor(p_alpha[0],p_alpha[1]);

        rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*4+curCM*4+1) % MAX_RAND_BUFFER_SIZE];
        iRand = MIN((((float)cellSize) * rand), ((float)(cellSize - 1)));
        PointCoord p_beta;
        p_beta = niter.getNode((int)iRand);
        p_beta -= cell.pc;


//        if (!(p_beta[0] >= 0
//            && p_beta[0] < nnr.adaptiveMap.getWidth()
//            && p_beta[1] >= 0
//            && p_beta[1] < nnr.adaptiveMap.getHeight()))
//            p_beta = cell.pc;
//        p_beta_disp = nnr.adaptiveMap[p_beta[1]][p_beta[0]] - nnd.adaptiveMap[p_beta[1]][p_beta[0]];

        //p_beta_disp = nnr.adaptiveMap.fetchIntCoor(p_beta[0],p_beta[1]);// - nnd.adaptiveMap.fetchIntCoor(p_beta[0],p_beta[1]);
        PointEuclid disB((float)p_beta[0] * (1.0f / (float)scale), (float)p_beta[1] * (1.0f / (float)scale));
        p_beta_disp = disB;

        components::DistanceManhattan<PointEuclid> distance;
        pickedSucc = (distance(p_alpha_disp, p_beta_disp) < SIMILARITY_THRESHOLD) ? false : true;
        if (!pickedSucc)
            return;

        rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*4+curCM*4+2) % MAX_RAND_BUFFER_SIZE];
        nbPicked_alpha = (int)(MIN((((float)maxNbPicked) * rand), ((float)(maxNbPicked - 1))));
        backup_PointCoord_alpha = (int*) malloc(2 * nbPicked_alpha * sizeof(int));
        backup_PointEuclid_alpha = (float*) malloc(2 * nbPicked_alpha * sizeof(float));

        rand = gRand[cell.PC[1]][cell.PC[0]][(gene*MAX_NUM_CM*4+curCM*4+3) % MAX_RAND_BUFFER_SIZE];
        nbPicked_beta = (int)(MIN((((float)maxNbPicked) * rand), ((float)(maxNbPicked - 1))));
        backup_PointCoord_beta = (int*) malloc(2 * nbPicked_beta * sizeof(int));
        backup_PointEuclid_beta = (float*) malloc(2 * nbPicked_beta * sizeof(float));

        niter.init();
        int nb_a = 0;
        int nb_b = 0;
        do {
            PointCoord ps = niter.get();
            if (ps[0] >= 0
                && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0
                && ps[1] < nnr.adaptiveMap.getHeight())
            {
               PointEuclid disp = nnr.adaptiveMap[ps[1]][ps[0]] - nnd.adaptiveMap[ps[1]][ps[0]];
               if ((distance(disp, p_alpha_disp) < SIMILARITY_THRESHOLD) && (nb_a < nbPicked_alpha))
               {
                   backup_PointCoord_alpha[2*nb_a] = ps[0];
                   backup_PointCoord_alpha[2*nb_a+1] = ps[1];
                   nb_a++;
               }
               if ((distance(disp, p_beta_disp) < SIMILARITY_THRESHOLD) && (nb_b < nbPicked_beta))
               {
                   backup_PointCoord_beta[2*nb_b] = ps[0];
                   backup_PointCoord_beta[2*nb_b+1] = ps[1];
                   nb_b++;
               }
            }
        } while (niter.next() && !(nb_a >= nbPicked_alpha && nb_b >= nbPicked_beta));

        nbPicked_alpha = nb_a;
        nbPicked_beta = nb_b;

//        printf("nbPicked_alpha = %d, nbPicked_beta = %d\n", nbPicked_alpha, nbPicked_beta);

    } // init

    DEVICE_HOST inline void backup(NN& nnr) {

        if (!pickedSucc)
            return;

        for (int i = 0; i < nbPicked_alpha; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord_alpha[2*i];
            ps[1] = backup_PointCoord_alpha[2*i+1];
            backup_PointEuclid_alpha[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
            backup_PointEuclid_alpha[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
        }
        for (int i = 0; i < nbPicked_beta; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord_beta[2*i];
            ps[1] = backup_PointCoord_beta[2*i+1];
            backup_PointEuclid_beta[2*i] = nnr.adaptiveMap[ps[1]][ps[0]][0];
            backup_PointEuclid_beta[2*i+1] = nnr.adaptiveMap[ps[1]][ps[0]][1];
        }
    }

    DEVICE_HOST inline void move(Cell& cell, NN& nnr, NN& nnd) {

        if (!pickedSucc)
            return;





        switch (moveTimes) {
        // alpha -> beta
        case 0 : {
            for (int i = 0; i < nbPicked_alpha; i++)
            {
                PointCoord ps;
                ps[0] = backup_PointCoord_alpha[2*i];
                ps[1] = backup_PointCoord_alpha[2*i+1];

#ifdef DISP_RANGE
                if (fabs(p_beta_disp[0]) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += p_beta_disp;// + nnd.adaptiveMap[ps[1]][ps[0]];
            }
            break;
        }
        // beta -> alpha
        case 1 : {
            for (int i = 0; i < nbPicked_beta; i++)
            {
                PointCoord ps;
                ps[0] = backup_PointCoord_beta[2*i];
                ps[1] = backup_PointCoord_beta[2*i+1];

#ifdef DISP_RANGE
                if (fabs(p_alpha_disp[0]) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += p_alpha_disp;// + nnd.adaptiveMap[ps[1]][ps[0]];
            }
            break;
        }
        // alpha <->beta
        case 2 : {
            for (int i = 0; i < nbPicked_alpha; i++)
            {
                PointCoord ps;
                ps[0] = backup_PointCoord_alpha[2*i];
                ps[1] = backup_PointCoord_alpha[2*i+1];

#ifdef DISP_RANGE
                if (fabs(p_beta_disp[0]) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += p_beta_disp;// + nnd.adaptiveMap[ps[1]][ps[0]];
            }
            for (int i = 0; i < nbPicked_beta; i++)
            {
                PointCoord ps;
                ps[0] = backup_PointCoord_beta[2*i];
                ps[1] = backup_PointCoord_beta[2*i+1];

#ifdef DISP_RANGE
                if (fabs(p_alpha_disp[0]) < DISP_RANGE)
#endif
                nnr.adaptiveMap[ps[1]][ps[0]] += p_alpha_disp;// + nnd.adaptiveMap[ps[1]][ps[0]];
            }
            break;
        }

        defaut : break;

        } //switch
    }

    DEVICE_HOST inline void recover(NN& nnr) {

        if (!pickedSucc)
            return;

        for (int i = 0; i < nbPicked_alpha; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord_alpha[2*i];
            ps[1] = backup_PointCoord_alpha[2*i+1];
            nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid_alpha[2*i];
            nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid_alpha[2*i+1];
        }
        for (int i = 0; i < nbPicked_beta; i++)
        {
            PointCoord ps;
            ps[0] = backup_PointCoord_beta[2*i];
            ps[1] = backup_PointCoord_beta[2*i+1];
            nnr.adaptiveMap[ps[1]][ps[0]][0] = backup_PointEuclid_beta[2*i];
            nnr.adaptiveMap[ps[1]][ps[0]][1] = backup_PointEuclid_beta[2*i+1];
        }
    }

    DEVICE_HOST inline bool next() {
        if (!pickedSucc)
            return false;
        return (++moveTimes < 3);
    }

    DEVICE_HOST inline void clean() {
        if (!pickedSucc)
            return;
        free (backup_PointCoord_alpha);
        free (backup_PointEuclid_alpha);
        free (backup_PointCoord_beta);
        free (backup_PointEuclid_beta);
    }

}; // GenerateNeighborRandomPickExpansionMoveAndSwapMove

}//namespace operators

#endif // ADAPTATOR_BASICS_H

#ifndef CELLULAR_MATRIX_H
#define CELLULAR_MATRIX_H
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

#include "random_generator.h"
//#ifdef CUDA_CODE
//#include <cuda_runtime.h>
//#include <cuda.h>
////#include <helper_functions.h>
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
//#include <sm_20_atomic_functions.h>
//#endif

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"
#include "CellAdaptiveSize.h"
#include "distance_functors.h"
#include "adaptator_basics.h"

#include "SpiralSearch.h"

#include "ViewGrid.h"
#ifdef TOPOLOGIE_HEXA
#include "ViewGridHexa.h"
#endif
#include "NIter.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif

using namespace components;
using namespace std;

namespace operators
{

template <class CellularMatrix,
          class ViewG>
KERNEL void K_CM_initialize(CellularMatrix cm, ViewG vgd)
{
    KER_SCHED_3D(cm.getWidth(), cm.getHeight(), cm.getDepth())

            typedef typename ViewG::index_type index_type;
    typedef typename ViewG::point_type point_type;
    index_type PC(_x, _y, _z);
    if (cm.valideIndex(PC)) {
        index_type pc = vgd.FDual(PC);
        point_type pe = vgd.FEuclid(pc);
        (point_type&) cm(PC) = pe;
        //! HW 16/04/15 : Here the radius of (vgd.getLevel()+1) may be redundant,
        //! but just as a compromised solution to the strange "white line" bug.
        cm(PC).initialize(PC, pc, (vgd.getLevel()), vgd);
    }

    END_KER_SCHED_3D

            SYNCTHREADS
}

// QWB 160916 add cpu version only for refreshCell
template <typename CellularMatrix,
          typename ViewG>
void K_CM_initialize_cpu(CellularMatrix cm, ViewG vgd)
{

    for (int _z = 0; _z < (cm.getDepth()); ++_z) {
        for (int _y = 0; _y < (cm.getHeight()); ++_y) {
            for (int _x = 0; _x < (cm.getWidth()); ++_x) {

                typedef typename ViewG::index_type index_type;
                typedef typename ViewG::point_type point_type;
                index_type PC(_x,_y,_z);
                index_type pc = vgd.FDual(PC);
                point_type pe = vgd.FEuclid(pc);
                (point_type&) cm(PC) = pe;
                cm(PC).initialize_cpu(PC, pc, (vgd.getLevel()), vgd);
            }
        }
    }
}

template <class CellularMatrix,
          class ViewG>
KERNEL void K_CM_initialize_dynamicCM(CellularMatrix cm, ViewG vgd)
{
    KER_SCHED(cm.getWidth(), cm.getHeight())

            if (_x < cm.getWidth() && _y < cm.getHeight()) {
        PointCoord PC(_x,_y);
        PointCoord pc = vgd.FDual(PC);
        PointEuclid pe = vgd.FEuclid(pc);
        (PointEuclid&) cm[_y][_x] = pe;
        //! HW 02.09.15 :
        //! When initializing each CellularMatrix<CSpS>, initialize the cell radius of every cell with R-1 instead of R.
        cm[_y][_x].initialize(PC, pc, (vgd.getLevel()-1), vgd);
    }

    END_KER_SCHED

            SYNCTHREADS
}

template <class CellularMatrix>
KERNEL void K_CM_clearCells(CellularMatrix cm)
{
    typedef typename CellularMatrix::index_type index_type;

    KER_SCHED_3D(cm.getWidth(), cm.getHeight(), cm.getDepth())

            index_type idx(_x,_y,_z);
    if (cm.valideIndex(idx))
    {
        cm(idx).clearCell();
    }

    END_KER_SCHED_3D

            SYNCTHREADS
}

//QWB add cpu version for refreshCell
template <class CellularMatrix>
void CM_clearCells_cpu(CellularMatrix cm)
{
    for (int _y = 0; _y < (cm.getHeight_cpu()); ++_y) {
        for (int _x = 0; _x < (cm.getWidth_cpu()); ++_x) {

            cm[_y][_x].clearCell_cpu();
        }
    }
}

template <class CellularMatrix>
KERNEL void K_CM_cellDensityComputation(CellularMatrix cm, NN nn)
{
    KER_SCHED(cm.getWidth(), cm.getHeight())

            if (_x < cm.getWidth() && _y < cm.getHeight()) {
        cm[_y][_x].computeDensity(nn);
    }

    END_KER_SCHED

            SYNCTHREADS
}

template <class CellularMatrix,
          class CellularMatrix2,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_CM_projector(CellularMatrix cm_source,
                           CellularMatrix2 cm_cible,
                           NN nn_source,
                           NN nn_cible,
                           GetAdaptor getAdaptor,
                           SearchAdaptor searchAdaptor,
                           OperateAdaptor operateAdaptor)
{
    KER_SCHED(cm_source.getWidth(), cm_source.getHeight())

            if (_x < cm_source.getWidth() && _y < cm_source.getHeight())
    {
        getAdaptor.init(cm_source[_y][_x]);
        do {
            PointCoord PC(_x, _y);

            // Partition
            PointCoord ps;
            bool extracted;
            //! HW 29/03/15 : modif
            //! JCC 300315 : adaptator is overload for Som only
            extracted = getAdaptor.get(cm_source[_y][_x], nn_source, ps);

            if (extracted) {
                // Spiral search
                PointCoord minP;
                bool found;

                found = searchAdaptor.search(cm_cible,
                                             nn_source,
                                             nn_cible,
                                             PC,
                                             ps,
                                             minP);

                if (found) {
                    operateAdaptor.operate(cm_cible[_y][_x], nn_source, nn_cible, ps, minP);
                }
            }

        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

            SYNCTHREADS
}

//! HW 16.05.15 : Overload for optical flow projection.
//! HW 16.05.15 : Here nn_cible is the searcher while nn_source is the searched.
template <class CellularMatrix,
          class CellularMatrix2,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_CM_projector2(CellularMatrix cm_source,
                            CellularMatrix2 cm_cible,
                            NN nn_source,
                            NN nn_cible,
                            GetAdaptor getAdaptor,
                            SearchAdaptor searchAdaptor,
                            OperateAdaptor operateAdaptor)
{
    KER_SCHED(cm_source.getWidth(), cm_source.getHeight())

            if (_x < cm_source.getWidth() && _y < cm_source.getHeight())
    {
        getAdaptor.init(cm_source[_y][_x]);
        do {
            PointCoord PC(_x, _y);

            // Partition
            PointCoord ps;
            bool extracted;
            //! HW 29/03/15 : modif
            //! JCC 300315 : adaptator is overload for Som only
            extracted = getAdaptor.get(cm_source[_y][_x], nn_cible, ps);

            if (extracted) {
                // Spiral search
                PointCoord minP;
                bool found;

                //! HW 16.05.15 : Here nn_cible is the searcher while nn_source is the searched.
                found = searchAdaptor.search(cm_cible,
                                             nn_cible,
                                             nn_source,
                                             PC,
                                             ps,
                                             minP);

                if (found) {
                    operateAdaptor.operate(cm_cible[_y][_x], nn_source, nn_cible, minP, ps);
                }
            }

        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

            SYNCTHREADS
}


template<class ViewGrid, Level LEVEL>
KERNEL void K_CM_initializeRegularIntoPlaneWithColor(ViewGrid vg, NN nnr, NN nnd)
{
    KER_SCHED(nnr.adaptiveMap.getWidth(), nnr.adaptiveMap.getHeight())

            if (_x < nnr.adaptiveMap.getWidth() && _y < nnr.adaptiveMap.getHeight())
    {
        PointCoord p(_x,_y); // default LOW_LEVEL

        if (LEVEL == BASE) {

            p = vg.F(PointCoord(_x,_y));

        } else if (LEVEL == DUAL) {

            p = vg.FDual(PointCoord(_x,_y));
        }

        nnr.adaptiveMap[_y][_x] = vg.FEuclid(p);

        int _x0 = MAX(0, p[0]);
        _x0 = MIN((int)(nnd.colorMap.getWidth()-1), _x0);
        int _y0 = MAX(0, p[1]);
        _y0 = MIN((int)(nnd.colorMap.getHeight()-1), _y0);
        nnr.colorMap[_y][_x] = nnd.colorMap[_y0][_x0];

        //! HW 24/04/15 : add density for superpixel experiments
        nnr.densityMap[_y][_x] = nnd.densityMap[_y0][_x0];
    }

    END_KER_SCHED

            SYNCTHREADS;
}

template<class ViewGrid, Level LEVEL>
KERNEL void K_CM_initializeRegularIntoPlaneWithPerturbation(ViewGrid vg, NN nnr, NN nnd)
{
    KER_SCHED(nnr.adaptiveMap.getWidth(), nnr.adaptiveMap.getHeight())

            if (_x < nnr.adaptiveMap.getWidth() && _y < nnr.adaptiveMap.getHeight())
    {
        PointCoord p(_x,_y); // default LOW_LEVEL

        if (LEVEL == BASE) {

            p = vg.F(PointCoord(_x,_y));

        } else if (LEVEL == DUAL) {

            p = vg.FDual(PointCoord(_x,_y));
        }

        int _x0 = MAX(0, p[0]);
        _x0 = MIN((int)(nnd.colorMap.getWidth()-1), _x0);
        int _y0 = MAX(0, p[1]);
        _y0 = MIN((int)(nnd.colorMap.getHeight()-1), _y0);

        // Perform perturbation in order to move the matcher neuron (kmean center)
        // to locations corresponding to the lowest gradient position in
        // a 3*3 neighborhood in the matched iamge.
        const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
        const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

        for (int i = 0; i < 8; i++)
        {
            int nx = _x0 + dx8[i];
            int ny = _y0 + dy8[i];

            if (nx >= 0 && nx < nnd.densityMap.getWidth() && ny >= 0 && ny < nnd.densityMap.getHeight())
            {
                // Here the densityMap is used to store the gradient values.
                if (nnd.densityMap[ny][nx] < nnd.densityMap[_y0][_x0])
                {
                    _x0 = nx;
                    _y0 = ny;
                }
            }
        }

        nnr.adaptiveMap[_y][_x] = vg.FEuclid(PointCoord(_x0,_y0));
        nnr.colorMap[_y][_x] = nnd.colorMap[_y0][_x0];

        //! HW 24/04/15 : add density for superpixel experiments
        nnr.densityMap[_y][_x] = nnd.densityMap[_y0][_x0];
    }

    END_KER_SCHED

            SYNCTHREADS;
}

template<class Distance,
         class Condition,
         class NIter,
         class ViewGrid>
KERNEL void K_CM_initializeClusterCenterSeed(ViewGrid vgd, NN nnr, NN nnd)
{
    KER_SCHED(nnr.adaptiveMap.getWidth(), nnr.adaptiveMap.getHeight())

            if (_x < nnr.adaptiveMap.getWidth() && _y < nnr.adaptiveMap.getHeight())
    {
        PointEuclid pc = nnr.adaptiveMap[_y][_x];
        PointCoord PC = vgd.FRound(pc);
        PointCoord minPCoord;
        SpiralSearchNNIterator<
                Distance,
                Condition,
                NIter
                > sa(PC,
                     vgd.getWidth() * vgd.getHeight(),
                     0,
                     MAX(vgd.getWidthDual(), vgd.getHeightDual()),
                     1);
        sa.search<PointEuclid>(nnr, nnd, pc, minPCoord);

        nnr.adaptiveMap[_y][_x] = nnd.adaptiveMap[minPCoord[1]][minPCoord[0]];
        nnr.colorMap[_y][_x] = nnd.colorMap[minPCoord[1]][minPCoord[0]];

        //! HW 24/04/15 : add density for superpixel experiments
        nnr.densityMap[_y][_x] = nnd.densityMap[minPCoord[1]][minPCoord[0]];
    }

    END_KER_SCHED

            SYNCTHREADS;
}

template<class Distance,
         class Condition,
         class NIter,
         class ViewGrid>
KERNEL void K_CM_AMprojectorNoCM(ViewGrid vgd, NN nnr, NN nnd)
{
    KER_SCHED(nnr.adaptiveMap.getWidth(), nnr.adaptiveMap.getHeight())

            if (_x < nnr.adaptiveMap.getWidth() && _y < nnr.adaptiveMap.getHeight())
    {
        PointCoord PC = PointCoord(_x, _y);
        PointCoord minPCoord;
        SpiralSearchNNIterator<
                Distance,
                Condition,
                NIter
                > sa(PC,
                     vgd.getWidth() * vgd.getHeight(),
                     0,
                     MAX(vgd.getWidthDual(), vgd.getHeightDual()),
                     1);
        sa.search<PointCoord>(nnr, nnd, PC, minPCoord);

        nnr.adaptiveMap[_y][_x] = nnd.adaptiveMap[minPCoord[1]][minPCoord[0]];
    }

    END_KER_SCHED

            SYNCTHREADS;
}


template<class ViewGrid, class NIter>
KERNEL void K_CM_initializeFixedMap(ViewGrid vg, Grid<bool> map)
{
    KER_SCHED(vg.getWidthDual(), vg.getHeightDual())

            if (_x < vg.getWidthDual() && _y < vg.getHeightDual()) {

        PointCoord pc = vg.FDualToBase(PointCoord(_x, _y));
        NIter ni(pc, vg.getLevel()-1, vg.getLevel());
        do {
            PointCoord pCoord;
            pCoord = ni.getNodeIncr();
            if (pCoord[0] >= 0 && pCoord[0] < map.getWidth()
                    && pCoord[1] >= 0 && pCoord[1] < map.getHeight()) {
                map[pCoord[1]][pCoord[0]] = true;
            }
        } while (ni.nextNodeIncr());
    }

    END_KER_SCHED

            SYNCTHREADS;
}


/*!
 * \brief The CellularMatrix class
 */

template <class Cell, class ViewG, size_t DimG>
class CellularMatrixMD : public MultiGrid<Cell, DimG>
{
    typedef MultiGrid<Cell, DimG> super_type;
public:
    typedef typename super_type::index_type index_type;
    typedef typename super_type::extents_type extents_type;

    ViewG vgd;
    GLfloat maxCellDensity;
    GLint maxEdgeNum;

    //! HW 27.04.15
    GLint edgeNumCM;
    GLfloat totalDensity;

    DEVICE_HOST CellularMatrixMD() : maxCellDensity(1) {}
    DEVICE_HOST CellularMatrixMD(ViewG viewgd)
        : maxCellDensity(1), vgd(viewgd) {}

    DEVICE_HOST void setViewG(ViewG& v) { vgd = v; }
    DEVICE_HOST ViewG& getViewG() { return vgd; }
    DEVICE_HOST void setMaxCellDensity(GLfloat m) { maxCellDensity = m; }
    DEVICE_HOST GLfloat getMaxCellDensity() { return maxCellDensity; }

    //! HW 26.03.15 : modif
    //! "A template-parameter shall not be redeclared within its scope (including nested scopes).
    //!  A template-parameter shall not have the same name as the template name."
    //! Otherwise, the redeclared declaration shadows the formerly declared template parameter.
    template<class ViewG1>
    GLOBAL void K_initialize(ViewG1& vg) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 4, 4, 4,
                                 this->getWidth(),
                                 this->getHeight(),
                                 this->getDepth());
        K_CM_initialize _KER_CALL_(b, t) (*this, vg);
    }

    //! QWB add cmd initialize on cpu side only
    template<class ViewG1>
    GLOBAL void K_initialize_cpu(ViewG1& vg) {

        K_CM_initialize_cpu (*this, vg);
    }

    //! HW 02.09.15 : add K_initialize_dynamicCM for the initialization of dynamic cellular matrix execution pattern
    //! When initializing each CellularMatrix<CSpS>, initialize the cell radius of every cell with R-1 instead of R.
    template<class ViewG1>
    GLOBAL void K_initialize_dynamicCM(ViewG1& vg) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_initialize_dynamicCM _KER_CALL_(b, t) (*this, vg);
    }

    //! HW 26.03.15 : modif
    template<Level LEVEL, class ViewG1>
    GLOBAL inline void K_initializeRegularIntoPlane(ViewG1& vg, Grid<PointEuclid>& map) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              map.getWidth(),
                              map.getHeight());
        K_VG_initializeIntoPlane<ViewG1, Grid<PointEuclid>, LEVEL> _KER_CALL_(b, t) (
                    vg, map);
    }

    //! HW 17.04.15 : Add K_initializeRegularIntoPlaneWithColor
    template<Level LEVEL, class ViewG1>
    GLOBAL inline void K_initializeRegularIntoPlaneWithColor(ViewG1& vg, NN& nnr, NN& nnd) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.adaptiveMap.getWidth(),
                              nnr.adaptiveMap.getHeight());
        K_CM_initializeRegularIntoPlaneWithColor<ViewG1, LEVEL> _KER_CALL_(b, t) (
                    vg, nnr, nnd);
    }

    //! HW 17.04.15 : Add K_initializeRegularIntoPlaneWithPerturbation
    template<Level LEVEL, class ViewG1>
    GLOBAL inline void K_initializeRegularIntoPlaneWithPerturbation(ViewG1& vg, NN& nnr, NN& nnd) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.adaptiveMap.getWidth(),
                              nnr.adaptiveMap.getHeight());
        K_CM_initializeRegularIntoPlaneWithPerturbation<ViewG1, LEVEL> _KER_CALL_(b, t) (
                    vg, nnr, nnd);
    }

    //! HW 17.04.15 : Add K_initializeFixedMap
    template<class ViewG1, class NIter>
    GLOBAL inline void K_initializeFixedMap(ViewG1& vg, Grid<bool>& map) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              vg.getWidthDual(),
                              vg.getHeightDual());

        K_CM_initializeFixedMap<ViewG1, NIter> _KER_CALL_(b, t) (vg, map);
    }

    //! HW 24.04.15 : Add K_initializeCenterSeedWithPerturbation
    template<class Distance,
             class Condition,
             class NIter,
             class ViewGrid>
    GLOBAL inline void K_initializeClusterCenterSeed(NN& nnr, NN& nnd) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.adaptiveMap.getWidth(),
                              nnr.adaptiveMap.getHeight());
        K_CM_initializeClusterCenterSeed<Distance, Condition, NIter, ViewGrid> _KER_CALL_(b, t) (
                    vgd, nnr, nnd);
    }

    //! HW 16.05.15 : Add K_AMprojectorNoCM
    template<class Distance,
             class Condition,
             class NIter,
             class ViewGrid>
    GLOBAL inline void K_AMprojectorNoCM(NN& nnr, NN& nnd) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.adaptiveMap.getWidth(),
                              nnr.adaptiveMap.getHeight());
        K_CM_AMprojectorNoCM<Distance, Condition, NIter, ViewGrid> _KER_CALL_(b, t) (
                    vgd, nnr, nnd);
    }

    //! HW 15.06.15 : Add K_initializeRegularIntoPlaneWithFlow
    template<Level LEVEL, class ViewG1>
    GLOBAL inline void K_initializeRegularIntoPlaneWithFlow(ViewG1& vg, Grid<PointEuclid>& map) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              map.getWidth(),
                              map.getHeight());
        K_VG_initializeIntoPlaneWithFlow<ViewG1, Grid<PointEuclid>, LEVEL> _KER_CALL_(b, t) (
                    vg, map);
    }

    GLOBAL void K_clearCells() {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 4, 4, 4,
                                 this->getWidth(),
                                 this->getHeight(),
                                 this->getDepth());
        K_CM_clearCells _KER_CALL_(b, t) (*this);
    }

    //! QWB add cpu version for refreshCell on Cpu
    GLOBAL void clearCells_cpu() {

        CM_clearCells_cpu (*this);
    }


    GLOBAL void K_cellDensityComputation(NN& nn) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_cellDensityComputation _KER_CALL_(b, t) (*this, nn);

        //! HW 01/04/15 : modif
        //! Otherwise GPU verison will crash because here (*this)._data now
        //! points to GPU device memory, and can not be directly accessed on host side.
        CellularMatrixMD<Cell, ViewG, DimG> tmp;
        tmp.resize(this->getWidth(), this->getHeight());
        tmp.gpuCopyDeviceToHost(*this);
        this->maxCellDensity = 0.0f; //! HW 22/04/15 : modif from this->maxCellDensity = 1;
        this->maxEdgeNum = 0;
        this->edgeNumCM = 0;
        this->totalDensity = 0.0f;
        for (int y = 0; y < this->getHeight(); y++) {
            for (int x = 0; x < this->getWidth(); x++)
            {
                this->edgeNumCM += tmp[y][x].edgeNum;
                this->totalDensity += tmp[y][x].density;
                if (tmp[y][x].density >= this->maxCellDensity)
                {
                    this->maxCellDensity = tmp[y][x].density;
                }
                if (tmp[y][x].edgeNum >= this->maxEdgeNum)
                {
                    this->maxEdgeNum = tmp[y][x].edgeNum;
                }
            }
        }

        printf("maxCellDensity = %f, totalDensity = %f, maxEdgeNum = %d, edgeNumCM = %d \n",
               this->maxCellDensity, this->totalDensity, this->maxEdgeNum, this->edgeNumCM);

        //        cout << "maxCellDensity = " << this->maxCellDensity
        //             << " totalDensity = " << this->totalDensity
        //             << " maxEdgeNum = " << this->maxEdgeNum
        //             << " edgeNumCM = " << this->edgeNumCM
        //             << endl;

        tmp.freeMem();
    }

    /*!
     * \brief K_projector
     *
     */
    template <class CellularMatrix,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_projector(CellularMatrix& cible, NN& nn_source, NN& nn_cible, GetAdaptor& ga, SearchAdaptor& sa, OperateAdaptor& oa) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_projector _KER_CALL_(b, t) (
                    *this, cible, nn_source, nn_cible, ga, sa, oa);
    }

    //! HW 16.05.15 : Overload for optical flow projection.
    //! HW 16.05.15 : Here nn_cible is the searcher while nn_source is the searched.
    /*!
     * \brief K_projector
     *
     */
    template <class CellularMatrix,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_projector2(CellularMatrix& cible, NN& nn_source, NN& nn_cible, GetAdaptor& ga, SearchAdaptor& sa, OperateAdaptor& oa) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_projector2 _KER_CALL_(b, t) (
                    *this, cible, nn_source, nn_cible, ga, sa, oa);
    }

}; // class CellularMatrix

template <class Cell, class ViewG>
class CellularMatrix : public CellularMatrixMD<Cell,ViewG,2> {
    typedef CellularMatrixMD<Cell,ViewG,2> super_type;
public:
    typedef typename super_type::index_type index_type;
    typedef typename super_type::extents_type extents_type;
};

// wb.Q search cma_dll from each cell
template <typename CellularMatrix, class Grid1 >
KERNEL inline void K_CM_searchCMA(CellularMatrix cm,  Grid1 g_dll) {

    KER_SCHED_3D(cm.getWidth(), cm.getHeight(), cm.getDepth())

            typename CellularMatrix::index_type pc(_x, _y, _z);
    if (cm.valideIndex(pc) && cm.g_cellular(pc) != INITVALUE)
    {
        typedef typename Grid1::index_type index_type;

        GLint pco = cm.g_cellular(pc);

        while(pco != INITVALUE){

            index_type pco2 = g_dll.back_offset(pco);
            cm(pc).size += 1;

            // Next element of the list
            pco = g_dll(pco);

        }// end while

    }
    END_KER_SCHED_3D

            SYNCTHREADS
}


//! wb.Q 2019 June add adaptive size cellular partition
template <class Cell, class ViewG, size_t DimG>
class CellularMDAdaptiveSize : public CellularMatrixMD<Cell, ViewG, DimG> {

public:
    // wb.Q Disjoint set map as offset values, it is not each cell has a grid<>,
    Grid<GLint> g_dll; // resized same with input points size, second kernel, build dll like what you have done in thesis.

    // wb.Q resized same with cellular size, store the starting point of each dll
    MultiGrid<GLint, DimG> g_cellular;


    //    template <class CellularMatrix>
    GLOBAL inline void K_searchCMA() {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 GENERAL_BLOCK_SIZE, 1, 1,
                                 this->getWidth(),
                                 this->getHeight(),
                                 this->getDepth());

        K_CM_searchCMA _KER_CALL_(b, t)(*this, this->g_dll);
    }

};

}//namespace operators

#endif // CELLULAR_MATRIX_H

#ifndef CELLADAPTIVESIZE_H
#define CELLADAPTIVESIZE_H
/*
 ***************************************************************************
 *
 * Author : Wenbao.Qiao, J.C. Creput
 * Creation date : Mar. 2019
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

//#ifdef CUDA_CODE
//#include <cuda_runtime.h>
//#include <cuda.h>
//#include <helper_functions.h>
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
//#include <sm_20_atomic_functions.h>
//#endif

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"

#include "NeuralNet.h"
#include "distances_matching.h"

#include "ViewGrid.h"
#ifdef TOPOLOGIE_HEXA
#include "ViewGridHexa.h"
#endif
#include "NIter.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif

using namespace std;

namespace components
{

/*!
 * \brief wb.Q, JCC, June 2019 The Adaptive Size Cell struct. There is not static or dynamic buffer for each cell.
 * Each cell has a dynamic linked list, insert and search operations are done on this dll.
 */
template <class Distance,
          class Condition,
          class NIter,
          class ViewGrid,
          class PointCoord>
struct AdaptiveCell : public ViewGrid::point_type {

    typedef typename ViewGrid::index_type index_type;
    typedef typename ViewGrid::point_type point_type;

    index_type PC;//in gd dual/cell level
    index_type pc;//in gd low level
    size_t radius;
    ViewGrid vgd;

    PointCoord minPCoord;//in cible grid
    GLfloat minDistance;

    int size;
    // wb.Q June 2019
    int startPointer; // wb.Q start node in dll, can be root of a component or not
    int endPointer; // size_count wb.Q end node in dll, should be refreshed when inserting new nodes
    GLfloat density;

    NIter iter;

    DEVICE_HOST AdaptiveCell() {}

    DEVICE_HOST AdaptiveCell(GLfloat const& v0, GLfloat const& v1) : point_type(v0, v1) {}

    DEVICE_HOST AdaptiveCell(
            PointCoord PC,
            PointCoord pc,
            size_t radius,
            ViewGrid vg
            )
        :
          PC(PC),
          pc(pc),
          radius(radius),
          vgd(vg),
          iter(pc, 0, radius)
    {}

    DEVICE_HOST void initialize(
            index_type PPC,
            index_type ppc,
            size_t rradius,
            ViewGrid& vvg
            ) {
        PC = PPC;
        pc = ppc;
        radius = rradius;
        vgd = vvg;
        iter.initialize(PointCoord(pc[0],pc[1]), 0, radius);
        startPointer = -1; // wb.Q 2019 add startPointer endPointer for adaptive size cellular partition
        endPointer = -1;
        size = 0;
    }

    DEVICE_HOST inline PointCoord getMinPCoord() { return minPCoord; }
    DEVICE_HOST inline GLfloat getMinDistance() { return minDistance; }
    DEVICE_HOST inline void clearCell() { startPointer = -1; endPointer = -1; size = 0; }
    // Size of the cell
    DEVICE_HOST inline size_t getSize() { return size; }

};

//!  Wenbao Qiao 19.06.28 : add
/*!
 * \brief The Adaptive Size Cell struct
 */
template <class Distance,
          class Condition,
          class NIter,
          class ViewG,
          class PointCoord >
struct CellMDAaptive : public AdaptiveCell<Distance, Condition, NIter, ViewG, PointCoord> {

    typedef AdaptiveCell<Distance, Condition, NIter, ViewG, PointCoord> super_type;
    typedef typename super_type::point_type point_type;
    typedef typename super_type::index_type index_type;

//    operators::CellularMatrixAdaptiveSize<AaptiveCellMD<Distance, Condition, NIter, ViewG, PointCoord>,
//    ViewG, 2> dd; // here, do not solve your pb, because it is a new cellular, has nothing to do with original input.

    size_t curPos;

    DEVICE_HOST CellMDAaptive() {}

    DEVICE_HOST CellMDAaptive(GLfloat const& v0, GLfloat const& v1) : AdaptiveCell<Distance, Condition, NIter, ViewG, PointCoord>(v0, v1) { bCell.init(); }

    DEVICE_HOST CellMDAaptive(
            index_type PC,
            index_type pc,
            size_t radius,
            ViewG vg
            )
        :
          AdaptiveCell<Distance, Condition, NIter, ViewG, PointCoord>(
              PC,
              pc,
              radius,
              vg
              )
    { }

    DEVICE_HOST inline void clearCell() {
        this->startPointer = -1;
        this->endPointer = -1;
        this->size = 0;
        // wb.Q here need to clean DLL into -1 for each node
//        printf("Initialize adaptive cell !\n");
    }


    //! To iterate
    DEVICE_HOST inline void init() { curPos = 0; }
    DEVICE_HOST inline bool get(PointCoord& ps) {
        bool ret = curPos < this->size;
        return (ret);
    }

    DEVICE_HOST inline bool next() {
        return -1;
    }

    // wb.Q here insert and search in adaptive cell are temporary operations adapted to old version cm(pc).insert(ps);
    DEVICE_HOST bool insert(PointCoord& pc) {
        bool ret = false;


        return ret;
    }

    //! Search coordinate level
    template <class NN>
    DEVICE_HOST bool search(NN& scher, NN& sched, PointCoord ps) {
        bool ret = false;

        return ret;
    }

    //! From density map
    template <class NN>
    DEVICE_HOST bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat random) {
        bool ret = false;

        return ret;
    }


    //! QWB 041216 add to search closest node with different component Id and compare equale distance
    template <class NN>
    DEVICE_HOST bool searchClosestDiffIdEqualWeightAuMinimum(NN& scher, NN& sched, PointCoord ps,
                                                             PointCoord& minPInCell) {

        bool ret = false;

        return ret;
    }

    //! From density map
    template <class NN>
    DEVICE_HOST void computeDensity(NN& neuralNet) {

    }

}; //AaptiveCellMD


}//namespace components

#endif // CELL_H

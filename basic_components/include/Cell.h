#ifndef CELL_H
#define CELL_H
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


#define TEST_CODE 0
//  32 for uy734  84/128for zi929  9/32 for lu980/rw1621
// 73/128 for mu1979  21/32 for nu3496, 95/128 for ca4663. 51/128 for tz6117 72/128 for eg7146
//49/64 for ym7663 28/64 for pm8079  9/32 for ei8246  115/128 for ar9152 113/128 for ja9847
// 24/32 for gr9882
// 44/64 for kz9976  21/32 for fi10639  25/32 for mo14185 26/32 for it16862  25/32 for vm22775/sw24978 23/32 for bm33708
// 256 for ch71009
// 41/64 for lra498378 r=1 256 for lra498378 r=2  10/32 for lrb744710
//! Maximum buffer size
#define MAX_CELL_SIZE 32 // 32 for 700000
//#define MAX_CELL_SIZE 1024
//#define MAX_CELL_SIZE 2048
//#define MAX_CELL_SIZE 4096
//#define MAX_CELL_SIZE 8192
//#define MAX_CELL_SIZE 16384
//#define MAX_CELL_SIZE 32768

#define BUFFER_INCREMENT 256

#define OPPOSITE(a) (1.0f / (1.0f + (a) * (a)))

using namespace std;

namespace components
{

//! HW 12.05.15 : modif for overloading
template <class Node>
class Buffer : public Point<Node, MAX_CELL_SIZE> {

public:

    int length;

public:

    DEVICE_HOST Buffer() {
       length = MAX_CELL_SIZE;
    }

    DEVICE_HOST bool init() {
        length = MAX_CELL_SIZE;
        return true;
    }
    DEVICE_HOST bool init(int size) {
        length = MAX_CELL_SIZE;
        return true;
    }
    DEVICE_HOST bool incre() {
        return false;
    }
    DEVICE_HOST bool incre(int size) {
        return false;
    }

};

//! HW 12.05.15 : Add dynamic buffer
template <class Node>
struct BufferDy {

    Node *elem;
    int length;
    int unitSize;

    DEVICE_HOST BufferDy() {
       unitSize = BUFFER_INCREMENT;
       init();
    }
    DEVICE_HOST BufferDy(int l, int u = BUFFER_INCREMENT) {
       unitSize = u;
       init(l);
    }

    DEVICE_HOST bool init() {
        unitSize = BUFFER_INCREMENT;
        if (elem != NULL) free(elem);
        elem = (Node*)malloc(unitSize * sizeof(Node));
        if (!elem) return (false);
        length = unitSize;
        return true;
    }
    DEVICE_HOST bool init(int size) {
        unitSize = BUFFER_INCREMENT;
        if (elem != NULL) free(elem);
        elem = (Node*)malloc(size * sizeof(Node));
        if (!elem) return (false);
        length = size;
        return true;
    }
    DEVICE_HOST bool incre() {
        Node *newBase;
        newBase = (Node*)realloc(elem, (length + unitSize) * sizeof(Node));
        if (!newBase) return (false);
        elem = newBase;
        length += unitSize;
        return true;
    }
    DEVICE_HOST bool incre(int size) {
        Node *newBase;
        newBase = (Node*)realloc(elem, (length + size) * sizeof(Node));
        if (!newBase) return (false);
        elem = newBase;
        length += size;
        return true;
    }
    DEVICE_HOST bool decre() {
        Node *newBase;
        newBase = (Node*)realloc(elem, (length - unitSize) * sizeof(Node));
        if (!newBase) return (false);
        elem = newBase;
        length -= unitSize;
        return true;
    }
    DEVICE_HOST bool decre(int size) {
        Node *newBase;
        newBase = (Node*)realloc(elem, (length - size) * sizeof(Node));
        if (!newBase) return (false);
        elem = newBase;
        length -= size;
        return true;
    }

    DEVICE_HOST inline Node const& operator[](int i) {
        return elem[i];
    }
};

/*!
 * \brief The Cell struct
 */

template <class Distance,
          class Condition,
          class NIter,
          class ViewGrid,
          class PointCoord>
struct Cell : public ViewGrid::point_type {

    typedef typename ViewGrid::index_type index_type;
    typedef typename ViewGrid::point_type point_type;

    index_type PC;//in gd dual/cell level
    index_type pc;//in gd low level
    size_t radius;
    ViewGrid vgd;

    PointCoord minPCoord;//in cible grid
    GLfloat minDistance;

    int size;
    int size_count;
    GLfloat density;
    //! HW 22/04/15 : The larger the density value is, the smaller chance of being extracted it has.
    GLfloat densityOpp;
    int edgeNum;

    NIter iter;

    DEVICE_HOST Cell() {}

    //! HW 13/04/15 : add a constructor for Cell value
    DEVICE_HOST Cell(GLfloat const& v0, GLfloat const& v1) : point_type(v0, v1) {}

    DEVICE_HOST Cell(
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

    //! HW 28/03/15 : I remove the "virtual" keyword, otherwise GPU version will go wrong.
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
        size_count = -1;
        size = 0;
    }

    //QWB add cpu version for refreshCell
    void initialize_cpu(
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
        size_count = -1;
        size = 0;
    }

    //! HW 28/03/15 : I comment the virtual method declarations, otherwise GPU version will go wrong.
#ifdef VIRTUAL
    // To iterate
    DEVICE_HOST virtual  void init() = 0;
    DEVICE_HOST virtual  bool get(PointCoord& ps) = 0;
    DEVICE_HOST virtual  bool get(int i, PointCoord& ps) = 0;
    DEVICE_HOST virtual  bool next() = 0;

    // Get one point random
    DEVICE_HOST virtual bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat rand) = 0;

    // Search Closest point
    DEVICE_HOST virtual bool search(NN& scher, NN& sched, PointCoord ps) = 0;

    // Insertion in buffer
    DEVICE_HOST virtual bool insert(PointCoord& pc) = 0;

    // Utilities
    DEVICE_HOST virtual void computeDensity(NN& neuralNet) = 0;
#endif
    DEVICE_HOST inline PointCoord getMinPCoord() { return minPCoord; }
    DEVICE_HOST inline GLfloat getMinDistance() { return minDistance; }
    DEVICE_HOST inline void clearCell() { size_count = -1; size = 0; }
    // Size of the cell
    DEVICE_HOST inline size_t getSize() { return size; }
    //QWB add cpu version for refreshCell
    inline void clearCell_cpu() { size_count = -1; size = 0; }

};

//!  HW 12.05.15 : modif
/*!
 * \brief The CellB struct
 */
template <class Distance,
          class Condition,
          class NIter,
          class ViewG,
          template<typename > class BufferType,
          class PointCoord >
struct CellBMD : public Cell<Distance, Condition, NIter, ViewG, PointCoord> {

    typedef Cell<Distance, Condition, NIter, ViewG, PointCoord> super_type;
    typedef typename super_type::point_type point_type;
    typedef typename super_type::index_type index_type;

    BufferType<PointCoord> bCell;
    size_t curPos;

    DEVICE_HOST CellBMD() {}

    //! HW 13/04/15 : add a constructor for Cell value
    DEVICE_HOST CellBMD(GLfloat const& v0, GLfloat const& v1) : Cell<Distance, Condition, NIter, ViewG, PointCoord>(v0, v1) { bCell.init(); }

    DEVICE_HOST CellBMD(
            index_type PC,
            index_type pc,
            size_t radius,
            ViewG vg
            )
        :
          Cell<Distance, Condition, NIter, ViewG, PointCoord>(
              PC,
              pc,
              radius,
              vg
              )
    { bCell.init(); }

    DEVICE_HOST inline void clearCell() {
        this->size_count = -1;
        this->size = 0;
        if (!(bCell.init()))
            printf("Cell buffer alloc failed !\n");
    }

    // qwb add cpu version
    inline void clearCell_cpu() {
        this->size_count = -1;
        this->size = 0;
        if (!(bCell.init()))
            printf("Cell buffer alloc failed !\n");
    }


    //! To iterate
    DEVICE_HOST inline void init() { curPos = 0; }
    DEVICE_HOST inline bool get(PointCoord& ps) {
        bool ret = curPos < this->size;
        if (ret)
            ps = bCell[curPos];
        return (ret);
    }
    //! HW 17/04/15 : method overloading for GetStdRandomAdaptor
    DEVICE_HOST inline bool get(int i, PointCoord& ps) {
        bool ret = i < this->size;
        if (ret)
            ps = bCell[i];
        return (ret);
    }
    DEVICE_HOST inline bool next() {
        return ++curPos < this->size;
    }

    //! From density map
    template <class NN>
    DEVICE_HOST bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat random) {
        bool ret = false;
        size_t count = 0;
        GLfloat sum = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
            sum += value;
            if (sum >= random * this->density)
            {
                ret = true;
                ps = pco;
                break;
            }
            count++;
        }
        return ret;
    }//searchB

    //! HW 22/04/15 : The larger the density value is, the smaller chance of being extracted it has.
    template <class NN>
    DEVICE_HOST bool extractRandomOpposite(NN& neuralNet, PointCoord& ps, GLfloat random) {
        bool ret = false;
        size_t count = 0;
        GLfloat sum = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
            sum += OPPOSITE(value);
            if (sum >= random * this->densityOpp)
            {
                ret = true;
                ps = pco;
                break;
            }
            count++;
        }
        return ret;
    }//searchB

    //! Search coordinate level
    template <class NN>
    DEVICE_HOST bool search(NN& scher, NN& sched, PointCoord ps) {
        bool ret = false;
        size_t count = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            Distance dist;
            Condition cond;
            if (pco[0] >= 0 && pco[0] < sched.adaptiveMap.getWidth()
                    && pco[1] >= 0 && pco[1] < sched.adaptiveMap.getHeight())
            {
                GLfloat v = dist(ps, pco, scher, sched);
                bool c = cond(ps, pco, scher, sched);
                if (v < this->minDistance && c)
                {
                    ret = true;
                    this->minDistance = v;
                    this->minPCoord = pco;
                }
            }
            count++;
        }
        return ret;
    }//searchB

    // Insertion in buffer
    DEVICE_HOST bool insert(PointCoord& pc) {
        bool ret = false;
#ifdef CUDA_CODE
        // get unique exclusive index
        int pos = atomicAdd(&(this->size), 1);
        if (pos < MAX_CELL_SIZE) {
            // access to the cell buffer
            ret = true;
            (PointCoord&)bCell[pos] = pc;
//            (PointCoord&)bCell[atomicAdd(&(this->size), 1)] = pc;
        }
#else

        if (this->size < bCell.length) {
            ret = true;
            (PointCoord&)bCell[this->size] = pc;
            this->size += 1;
        }
#endif
        return ret;
    }

//    nn_source.networkLinks[couple2[1]][couple2[0]].bCell[atomicAdd(&(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks), 1)] = ps_; // WB.Q this way can work for multi-thread operation
//    rootMergingGraphGpu[0][rootCouple2].bCell[atomicAdd(&(rootMergingGraphGpu[0][rootCouple2].numLinks), 1)] = pTemp2;


    // QWB add Insertion only work on cpu side
    bool insert_cpu(PointCoord& pc) {
        bool ret = false;
        if (this->size < bCell.length) {
            ret = true;
            (PointCoord&)bCell[this->size] = pc;
            this->size += 1;
        }
        else if (bCell.incre()) {
            ret = true;
            (PointCoord&)bCell[this->size] = pc;
            this->size += 1;
        }
        else
            printf("Cell is full, insert failed ! this->PC(%d, %d), this->pc(%d, %d), pc(%d, %d), bCell.length=%d, cellSize=%d \n",
                   this->PC[0], this->PC[1], this->pc[0], this->pc[1], pc[0], pc[1], bCell.length, (int)this->size);
        return ret;
    }


    //! QWB 041216 add to search closest node with different component Id and compare equale distance
    template <class NN>
    DEVICE_HOST bool searchClosestDiffIdEqualWeightAuMinimum(NN& scher, NN& sched, PointCoord ps,
                                                    PointCoord& minPInCell) {

        bool ret = false;
        size_t count = 0;
        while (count < this->size)//230616 qiao test, on gpu side, this->size = 0
        {
            PointCoord pco = bCell[count];
            //if(scher.disjointSetMap[pco[1]][pco[0]] != scher.disjointSetMap[ps[1]][ps[0]]){
            if(scher.disjointSetMap.findRoot(scher.disjointSetMap, scher.disjointSetMap.compute_offset(pco))
                    != scher.disjointSetMap.findRoot(scher.disjointSetMap, scher.disjointSetMap.compute_offset(ps))){

                Distance dist;
                if (pco[0] >= 0 && pco[0] < sched.adaptiveMap.getWidth()
                        && pco[1] >= 0 && pco[1] < sched.adaptiveMap.getHeight())
                {
                    double v = dist(ps, pco, scher, sched);

                    if (this->minPCoord[0] == -1)
                    {
                        ret = true;
                        minPInCell = pco;
                        this->minPCoord = pco;
                        this->minDistance = v;
                    }
                    else
                    if (v < this->minDistance )
                    {
                        ret = true;
                        minPInCell = pco;
                        this->minPCoord = pco;
                        this->minDistance = v;
                    }
                    else
                    if (v == this->minDistance && pco[0] < this->minPCoord[0])
                    {
                        ret = true;
                        minPInCell = pco;
                        this->minPCoord = pco;
                        this->minDistance = v;
                    }
//                  printf("ps %d, %d, pco, %d, %d,  \n", ps[0], ps[1], pco[0], pco[1]);
                }
            }
            count++;
        }
        return ret;
    }//searchB

    //! From density map
    template <class NN>
    DEVICE_HOST void computeDensity(NN& neuralNet) {
        size_t count = 0;
        GLfloat sum = 0.0f;
        //! HW 22/04/15 : The larger the density value is, the smaller chance of being extracted it has.
        GLfloat sumOpp = 0.0f;
        this->edgeNum = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
            sum += value;
            sumOpp += OPPOSITE(value);
            if (neuralNet.activeMap[pco[1]][pco[0]] == true)
                this->edgeNum += 1;

            count++;
        }
        this->density = sum;
        this->densityOpp = sumOpp;
    }//searchB
}; //CellB

template <class Distance,
          class Condition,
          class NIter,
          class ViewG>
class CellB : public CellBMD<Distance, Condition, NIter, ViewG, Buffer, PointCoord > {};

/*!
 * \brief The CellSpS struct
 */
template <class Distance,
          class Condition,
          class NIter,
          class ViewG
          >
struct CellSpS : public Cell<Distance, Condition, NIter, ViewG, PointCoord> {

    typedef Cell<Distance, Condition, NIter, ViewG, PointCoord> super_type;
    typedef typename super_type::point_type point_type;
    typedef typename super_type::index_type index_type;

    DEVICE_HOST CellSpS() {}

    //! HW 13/04/15 : add a constructor for Cell value
    DEVICE_HOST CellSpS(GLfloat const& v0, GLfloat const& v1) : Cell<Distance, Condition, NIter, ViewG, PointCoord>(v0, v1) {}

    //! HW 09/03/15 : I modify at lines 184-191.
    DEVICE_HOST CellSpS(
            index_type PC,
            index_type pc,
            size_t radius,
            ViewG vg
            )
        :
          Cell<Distance, Condition, NIter, ViewG, PointCoord>(
              PC,
              pc,
              radius,
              vg
              )
    {}

//    DEVICE_HOST inline size_t getSize() { return 0; }
    //! HW 17/04/15 : method rewriting for GetStdRandomAdaptor
    DEVICE_HOST inline size_t getSize() {
        return (this->iter.getTotalSizeN(this->radius));
    }

    // To iterate
    DEVICE_HOST inline void init() { this->iter.init(); }
    DEVICE_HOST inline bool get(PointCoord& ps) {
        ps = this->iter.get();
//!JCC to see
        return (ps[0] >= 0 && ps[0] < this->vgd.getWidth()
                && ps[1] >= 0 && ps[1] < this->vgd.getHeight());
//        return (true);
    }
    //! HW 17/04/15 : method overloading for GetStdRandomAdaptor
    DEVICE_HOST inline bool get(int i, PointCoord& ps) {
        //!HW not the efficient way
        this->iter.init();
        for (int j = 0; j < i; j++)
            this->iter.get();
        ps = this->iter.get();
//!JCC to see
        return (ps[0] >= 0 && ps[0] < this->vgd.getWidth()
                && ps[1] >= 0 && ps[1] < this->vgd.getHeight());
//        return (true);
    }
    DEVICE_HOST inline bool next() { return this->iter.next(); }

    //! Euclidean/Value level
    DEVICE_HOST bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat random) {
        bool ret = false;
        NIter ni(this->pc, 0, this->radius);
        GLfloat sum = 0;
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < neuralNet.densityMap.getWidth()
                    && pco[1] >= 0 && pco[1] < neuralNet.densityMap.getHeight()) {
                GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
                sum += value;
                if (sum >= random * this->density)
                {
                    ret = true;
                    ps = pco;
                    break;
                }
            }
        } while (ni.nextNodeIncr());
        return ret;
    }

    //! HW 22/04/15 : The larger the density value is, the smaller chance of being extracted it has.
    DEVICE_HOST bool extractRandomOpposite(NN& neuralNet, PointCoord& ps, GLfloat random) {
        bool ret = false;
        NIter ni(this->pc, 0, this->radius);
        GLfloat sum = 0;
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < neuralNet.densityMap.getWidth()
                    && pco[1] >= 0 && pco[1] < neuralNet.densityMap.getHeight()) {
                GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
                sum += OPPOSITE(value);
                if (sum >= random * this->densityOpp)
                {
                    ret = true;
                    ps = pco;
                    break;
                }
            }
        } while (ni.nextNodeIncr());
        return ret;
    }

    //! Search function at coordinate level
    DEVICE_HOST bool search(NN& scher, NN& sched, PointCoord ps) {
        bool ret = true;
        NIter ni(this->pc, 0, this->radius);
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < sched.adaptiveMap.getWidth()
                    && pco[1] >= 0 && pco[1] < sched.adaptiveMap.getHeight()) {
                Distance dist;
                Condition cond;
                GLfloat v = dist(ps, pco, scher, sched);
                bool c = cond(ps, pco, scher, sched);
                if (v < this->minDistance && c)
                {
                    ret = true;
                    this->minDistance = v;
                    this->minPCoord = pco;
                }
            }
        } while (ni.nextNodeIncr());
        return ret;
    }

    // Insertion in buffer
    DEVICE_HOST bool insert(PointCoord& pc) {
        return true;
    }

    DEVICE_HOST void computeDensity(NN& nn) {
        NIter ni(this->pc, 0, this->radius);
        GLfloat sum = 0;
        //! HW 22/04/15 : The larger the density value is, the smaller chance of being extracted it has.
        GLfloat sumOpp = 0.0f;
        this->edgeNum = 0;
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < nn.densityMap.getWidth()
                    && pco[1] >= 0 && pco[1] < nn.densityMap.getHeight()) {
                GLfloat value = nn.densityMap[pco[1]][pco[0]];
                sum += value;
                sumOpp += OPPOSITE(value);
                if (nn.activeMap[pco[1]][pco[0]] == true)
                    this->edgeNum += 1;
            }

        } while (ni.nextNodeIncr());
        this->density = sum;
        this->densityOpp = sumOpp;
    }
};

}//namespace components

#endif // CELL_H

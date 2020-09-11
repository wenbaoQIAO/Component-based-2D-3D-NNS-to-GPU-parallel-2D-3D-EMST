#ifndef NIter_H
#define NIter_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#ifdef CUDA_CODE
#include <cuda_runtime.h>"
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

#include "macros_cuda.h"
#include "ConfigParams.h"
//#include "Node.h"
//#include "GridOfNodes.h"

using namespace std;
using namespace components;

namespace components
{

/*!
 * \brief The NeighborhoodIterator class
 */
class NeighborhoodIterator
{
protected:
    //! Central point of neighborhood
    PointCoord pCenter;

    //! Current location in the grid
    PointCoord pCur;

    //! Max distance : neighborhood size or contour
    size_t max_distance;
    //! Current distance
    size_t current_distance;

    //! Current position
    size_t offset_start;
    size_t current_contour_pos;
    size_t cpt_contour_pos;

    size_t sizeN;

    bool dual;// if it is a dual graph

public:
    //!JCC 040315 : added getSizeN
    DEVICE_HOST size_t getSizeN() {
        return sizeN;
    }

    DEVICE_HOST size_t getTotalSizeN() {
        return (((max_distance + 1) * max_distance * sizeN) / 2 + 1);
    }

    DEVICE_HOST size_t getTotalSizeN(size_t d) {
        return (((d + 1) * d * sizeN) / 2 + 1);
    }

    DEVICE_HOST NeighborhoodIterator() {}

    /*!
     * \brief NeighborhoodIterator
     * \param pc
     * \param d_min
     * \param d_max
     * \param dual_g
     * \param offset
     * \param size
     * \return
     */
    DEVICE_HOST NeighborhoodIterator(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0, size_t size=6)
        :
        current_distance(d_min),
        max_distance(d_max),
        offset_start(offset),
        sizeN(size),
        dual(dual_g)
    {
        pCenter = pc;
        offset_start = 0;//current_distance == 0 ? 0 : offset_start %= (sizeN * current_distance);
        current_contour_pos = offset_start;
        cpt_contour_pos = 0;
        pCur[1] = pCenter[1] + (dual?2*current_distance:0);
        pCur[0] = pCenter[0] + (!dual?current_distance:0);
    }

    //! HW 26.05.15 : this method is suppressed in the base class and implemented in child bases.
//    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0, size_t size=6) {
//        current_distance = d_min;
//        max_distance = d_max;
//        offset_start = offset;
//        sizeN = size;
//        dual = dual_g;
//        pCenter = pc;
//        this->setCurrentDistance(0);
//    }

    DEVICE_HOST inline void init() {
        this->setCurrentDistance(0);
    }

    //! HW 09/04/15: In CUDA_CODE, get() method is put directly into the inherited class.
    //! Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifndef CUDA_CODE
    DEVICE_HOST inline PointCoord get() {
        return this->getNodeIncr();
    }
#endif

    DEVICE_HOST inline bool next() {
        return this->nextNodeIncr();
    }

    DEVICE_HOST void setMaxDistance(size_t d) {
        this->max_distance = d;
    }

    DEVICE_HOST void setCurrentDistance(size_t d, size_t offset=0) {
        current_distance = d;
        offset_start = offset;
        offset_start = 0;//current_distance == 0 ? 0 : offset_start %= (sizeN * current_distance);
        current_contour_pos = offset_start;
        cpt_contour_pos = 0;
        pCur[1] = pCenter[1] + (dual?2*current_distance:0);
        pCur[0] = pCenter[0] + (!dual?current_distance:0);
    }

    //! HW 14.06.15: Add setCurrentIndex(size_t id) for direct access
    DEVICE_HOST void setCurrentIndex(size_t id) {
        size_t d = 0;
        size_t s = 0;
        while (s < id)
        {
            d++;
            s += d * sizeN;
        }
        // Be careful here. If id = 0, then current_distance = 0, and the current_contour_pos will not be used.
        current_distance = d;
        current_contour_pos = id - (s - d * sizeN) - 1;
    }

    DEVICE_HOST size_t getCurrentDistance() {
        return current_distance;
    }

    DEVICE_HOST bool nextNode() {
        bool ret = true;
        current_contour_pos += 1;
        if (current_contour_pos >= sizeN * current_distance)
            current_contour_pos = 0;
        cpt_contour_pos += 1;
        if (cpt_contour_pos >= sizeN * current_distance) {
            current_distance += 1;
            offset_start = current_distance == 0 ? 0 : offset_start %= (sizeN * current_distance);
            current_contour_pos = offset_start;
            cpt_contour_pos = 0;
            pCur[1] = pCenter[1] + (dual?2*current_distance:0);
            pCur[0] = pCenter[0] + (!dual?current_distance:0);
            if (current_distance > max_distance)
                ret = false;
        }
        return ret;
    }

    //! HW 060815 : add nextDistanceIncr() for distance increment
    DEVICE_HOST bool nextDistanceIncr() {
        bool ret = true;
        current_distance += 1;
        if (current_distance > max_distance)
            ret = false;
        return ret;
    }

    DEVICE_HOST bool nextNodeIncr() {
        bool ret = true;
        current_contour_pos += 1;
        if (current_contour_pos >= sizeN * current_distance) {
            current_distance += 1;
            current_contour_pos = 0;
            pCur[1] = pCenter[1] + (dual?2*current_distance:0);
            pCur[0] = pCenter[0] + (!dual?current_distance:0);
            if (current_distance > max_distance)
                ret = false;
        }
        return ret;
    }

    DEVICE_HOST bool nextContourNode() {
        bool ret = true;
        current_contour_pos += 1;
        if (current_contour_pos >= sizeN * current_distance)
            current_contour_pos = 0;
        cpt_contour_pos += 1;
        if (cpt_contour_pos >= sizeN * current_distance) {
            ret = false;
            current_distance += 1;
            offset_start = current_distance == 0 ? 0 : offset_start %= (sizeN * current_distance);
            current_contour_pos = offset_start;
            cpt_contour_pos = 0;
            pCur[1] = pCenter[1] + (dual?2*current_distance:0);
            pCur[0] = pCenter[0] + (!dual?current_distance:0);
        }
        return ret;
    }

    DEVICE_HOST bool nextContourNodeIncr() {
        bool ret = true;
        current_contour_pos += 1;
        if (current_contour_pos >= sizeN * current_distance) {
            current_distance += 1;
            current_contour_pos = 0;
            ret = false;
            pCur[1] = pCenter[1] + (dual?2*current_distance:0);
            pCur[0] = pCenter[0] + (!dual?current_distance:0);
        }
        return ret;
    }

    //! HW 09/04/15: Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifndef CUDA_CODE
    /*!
     * Direct access
     */
    DEVICE_HOST virtual PointCoord getNode() = 0;
    /*!
     * Incremental access
     */
    DEVICE_HOST virtual PointCoord getNodeIncr() = 0;
#endif

};//NeighborhoodIterator

/*!
 * \brief The NIterTetra class
 * Turn CounterClokwise starting from right point (dir=0)
 */
class NIterTetra : public NeighborhoodIterator
{
public:
    DEVICE_HOST NIterTetra() : NeighborhoodIterator() {}

    /*!
     * \brief NIterTetra
     * \param pc
     * \param d_min
     * \param d_max
     * \param dual_g
     * \param offset
     * \return
     */
    DEVICE_HOST NIterTetra(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0) :
        NeighborhoodIterator(pc, d_min, d_max, dual_g, offset, 4) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0, size_t size=4) {
        current_distance = d_min;
        max_distance = d_max;
        offset_start = offset;
        sizeN = size;
        dual = dual_g;
        pCenter = pc;
        this->setCurrentDistance(0);
    }

    //! HW 060815 : overload goTo()
    /*!
     * Direct access on one direction
     */
    DEVICE_HOST PointCoord goTo(PointCoord pc, size_t DIR, size_t d) {

        int Y = pc[1];
        int X = pc[0];

        // Turn CounterClockwise starting from right
        switch (DIR) {
        case 0 :
            // right
            pCur[1] = Y;
            pCur[0] = X + d;
            break;
        case 1 :
            // up
            pCur[1] = Y - d;
            pCur[0] = X;
            break;
        case 2 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 3 :
            // down
            pCur[1] = Y + d;
            pCur[0] = X;
            break;
        default :
            break;
        }//switch

        return pCur;
    }//goTo

    /*!
     * Direct access on one direction
     */   
    template<size_t DIR>
    DEVICE_HOST PointCoord goTo(PointCoord pc, size_t d) {

        int Y = pc[1];
        int X = pc[0];

        // Turn CounterClockwise starting from right
        switch (DIR) {
        case 0 :
            // right
            pCur[1] = Y;
            pCur[0] = X + d;
            break;
        case 1 :
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d;
            break;
        case 2 :
            // up
            pCur[1] = Y - d;
            pCur[0] = X;
            break;
        case 3 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d;
            break;
        case 4 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 5 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d;
            break;
        case 6 :
            // down
            pCur[1] = Y + d;
            pCur[0] = X;
            break;
        case 7 :
            // down right
            pCur[1] = Y + d;
            pCur[0] = X + d;
            break;
        default :
            break;
        }//switch

        return pCur;
    }//goTo

    /*!
     * Direct access
     */
    DEVICE_HOST PointCoord getNode() {

        if (current_distance != 0)
        {
            int p = current_contour_pos % current_distance;

            switch (current_contour_pos / current_distance) {
            case 0 :
                goTo<3>(goTo<0>(pCenter, current_distance), p);
                break;
            case 1 :
                goTo<5>(goTo<2>(pCenter, current_distance), p);
                break;
            case 2 :
                goTo<7>(goTo<4>(pCenter, current_distance), p);
                break;
            case 3 :
                goTo<1>(goTo<6>(pCenter, current_distance), p);
                break;
            default :
                break;
            }//switch
        }
        else {// dist == 0
            // Current location in the grid
            pCur = pCenter;
        }

        return pCur;
    }//getNode

    //! HW 14.06.15: Add direct access with specific index
    /*!
     * Direct access
     */
    DEVICE_HOST PointCoord getNode(size_t id) {

        size_t d = 0;
        size_t s = 0;
        while (s < id)
        {
            d++;
            s += d * sizeN;
        }
        // Be careful here. If id = 0, then current_distance = 0, and the current_contour_pos will not be used.
        current_distance = d;
        current_contour_pos = id - (s - d * sizeN) - 1;

        return (getNode());
    }

    /*!
     * Incremental access
     */
    DEVICE_HOST PointCoord getNodeIncr() {

        if (current_distance != 0)
        {
            int Y = pCur[1];
            int X = pCur[0];

            switch (current_contour_pos / current_distance) {
            case 0 :
                Y = Y - 1;
                X = X - 1;
                break;
            case 1 :
                Y = Y + 1;
                X = X - 1;
                break;
            case 2 :
                Y = Y + 1;
                X = X + 1;
                break;
            case 3 :
                Y = Y - 1;
                X = X + 1;
                break;
            default :
                break;
            }//switch
            pCur[1] = Y;
            pCur[0] = X;
        }
        else {
            pCur = pCenter;
        }

        return pCur;
    }//getNodeIncr

    //! HW 09/04/15: In CUDA_CODE, get() method is put directly into the inherited class.
    //! Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifdef CUDA_CODE
    DEVICE_HOST inline PointCoord get() {
        return this->getNodeIncr();
    }
#endif

};//NIterTetra

/*!
 * \brief The NIterTetraDual class
 */
//! NIter (couterclokwise) in Dual graph since
//! there is no problem of odd/even line and problem of
//! changing representation (in table) when passing to
//! dual.

//! HW 20/03/15 : add new NIterTetraDual class
/*!
 * \brief The NIterTetraDual class
 * Turn Clokwise starting from bottom point (dir=0)
 */
class NIterTetraDual : public NeighborhoodIterator
{
public:

    DEVICE_HOST NIterTetraDual() : NeighborhoodIterator() {}

    /*!
     * \brief NIterTetraDual
     * \param pc
     * \param d_min
     * \param d_max
     * \param offset
     * \return
     */
    DEVICE_HOST NIterTetraDual(PointCoord pc, size_t d_min, size_t d_max, size_t offset=0) :
        NeighborhoodIterator(pc, d_min, d_max, true, offset, 8) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = true, size_t offset=0, size_t size=8) {
        current_distance = d_min;
        max_distance = d_max;
        offset_start = offset;
        sizeN = size;
        dual = dual_g;
        pCenter = pc;
        this->setCurrentDistance(0);
    }

    /*!
     * Direct access on one direction
     */
    template<size_t DIR>
    DEVICE_HOST PointCoord goTo(PointCoord pc, size_t d) {

        int Y = pc[1];
        int X = pc[0];

        // Turn Clockwise starting from bottom
        switch (DIR) {
        case 0 :
            // down
            pCur[1] = Y + 2 * d;
            pCur[0] = X;
            break;
        case 1 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d / 2  - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 2 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 3 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 4 :
            // up
            pCur[1] = Y - 2 * d;
            pCur[0] = X;
            break;
        case 5 :
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d / 2 + (d % 2)*((Y+1) % 2)*((Y+1) % 2);
            break;
        case 6 :
            // right
            pCur[1] = Y;
            pCur[0] = X + d;
            break;
        case 7 :
            // down right
            pCur[1] = Y + d;
            pCur[0] = X + d / 2 + (d % 2)*((Y+1) % 2)*((Y+1) % 2);
            break;
        default :
            break;
        }//switch

        return pCur;
    }//goTo

    /*!
     * Direct access
     */
    DEVICE_HOST PointCoord getNode() {

        if (current_distance != 0)
        {
            int p = current_contour_pos % current_distance;

            switch (current_contour_pos / current_distance) {
            case 0 :
                // down
                goTo<3>(goTo<0>(pCenter, current_distance), p);
                break;
            case 1 :
                // down left
                goTo<3>(goTo<1>(pCenter, current_distance), p);
                break;
            case 2 :
                // left
                goTo<5>(goTo<2>(pCenter, current_distance), p);
                break;
            case 3 :
                // up left
                goTo<5>(goTo<3>(pCenter, current_distance), p);
                break;
            case 4 :
                // up
                goTo<7>(goTo<4>(pCenter, current_distance), p);
                break;
            case 5 :
                // up right
                goTo<7>(goTo<5>(pCenter, current_distance), p);
                break;
            case 6 :
                // right
                goTo<1>(goTo<6>(pCenter, current_distance), p);
                break;
            case 7 :
                // down right
                goTo<1>(goTo<7>(pCenter, current_distance), p);
                break;
            default :
                break;
            }//switch
        }
        else {// dist == 0
            // Current location in the grid
            pCur = pCenter;
        }

        return pCur;
    }//getNode

    /*!
     * Incremental access
     */
    DEVICE_HOST PointCoord getNodeIncr() {

        if (current_distance != 0)
        {
            int Y = pCur[1];
            int X = pCur[0];

            switch (current_contour_pos / current_distance) {
            case 0 :
                // up left
                pCur[1] = Y - 1;
                pCur[0] = X - (Y % 2)*(Y % 2);
                break;
            case 1 :
                // up left
                pCur[1] = Y - 1;
                pCur[0] = X - (Y % 2)*(Y % 2);
                break;
            case 2 :
                // up right
                pCur[1] = Y - 1;
                pCur[0] = X + ((Y+1) % 2)*((Y+1) % 2);
                break;
            case 3 :
                // up right
                pCur[1] = Y - 1;
                pCur[0] = X + ((Y+1) % 2)*((Y+1) % 2);
                break;
            case 4 :
                // down right
                pCur[1] = Y + 1;
                pCur[0] = X + ((Y+1) % 2)*((Y+1) % 2);
                break;
            case 5 :
                // down right
                pCur[1] = Y + 1;
                pCur[0] = X + ((Y+1) % 2)*((Y+1) % 2);
                break;
            case 6 :
                // down left
                pCur[1] = Y + 1;
                pCur[0] = X - (Y % 2)*(Y % 2);
                break;
            case 7 :
                // down left
                pCur[1] = Y + 1;
                pCur[0] = X - (Y % 2)*(Y % 2);
                break;
            default :
                break;
            }//switch
        }
        else {
            pCur = pCenter;
        }
        return pCur;
    }//getNodeInc

    //! HW 09/04/15: In CUDA_CODE, get() method is put directly into the inherited class.
    //! Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifdef CUDA_CODE
    DEVICE_HOST inline PointCoord get() {
        return this->getNodeIncr();
    }
#endif

};//NIterTetraDual


/*!
 * \brief The NIterQuad class
 */
class NIterQuad : public NeighborhoodIterator
{
public:
    DEVICE_HOST NIterQuad() : NeighborhoodIterator() {}

    DEVICE_HOST NIterQuad(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0) :
        NeighborhoodIterator(pc, d_min, d_max, dual_g, offset, 8) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0, size_t size=8) {
        current_distance = d_min;
        max_distance = d_max;
        offset_start = offset;
        sizeN = size;
        dual = dual_g;
        pCenter = pc;
        this->setCurrentDistance(d_min);
    }

    //! HW 060815 : overload goTo()
    /*!
     * Direct access on one direction
     */
    DEVICE_HOST PointCoord goTo(PointCoord pc, size_t DIR, size_t d) {

        int Y = pc[1];
        int X = pc[0];

        // Turn CounterClockwise starting from right
        switch (DIR) {
        case 0 :
            // right
            pCur[1] = Y;
            pCur[0] = X + d;
            break;
        case 1 :
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d;
            break;
        case 2 :
            // up
            pCur[1] = Y - d;
            pCur[0] = X;
            break;
        case 3 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d;
            break;
        case 4 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 5 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d;
            break;
        case 6 :
            // down
            pCur[1] = Y + d;
            pCur[0] = X;
            break;
        case 7 :
            // down right
            pCur[1] = Y + d;
            pCur[0] = X + d;
            break;
        default :
            break;
        }//switch

        return pCur;
    }//goTo

    /*!
     * Direct access on one direction
     */
    template<size_t DIR>
    DEVICE_HOST PointCoord goTo(PointCoord pc, size_t d) {

        int Y = pc[1];
        int X = pc[0];

        // Turn CounterClockwise starting from right
        switch (DIR) {
        case 0 :
            // right
            pCur[1] = Y;
            pCur[0] = X + d;
            break;
        case 1 :
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d;
            break;
        case 2 :
            // up
            pCur[1] = Y - d;
            pCur[0] = X;
            break;
        case 3 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d;
            break;
        case 4 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 5 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d;
            break;
        case 6 :
            // down
            pCur[1] = Y + d;
            pCur[0] = X;
            break;
        case 7 :
            // down right
            pCur[1] = Y + d;
            pCur[0] = X + d;
            break;
        default :
            break;
        }//switch

        return pCur;
    }//goTo

    /*!
     * Direct access
     */
    DEVICE_HOST PointCoord getNode() {

        if (current_distance != 0)
        {
            int p = current_contour_pos % current_distance;

            switch (current_contour_pos / current_distance) {
            case 0 :
                goTo<2>(goTo<0>(pCenter, current_distance), p);
                break;
            case 1 :
                goTo<4>(goTo<1>(pCenter, current_distance), p);
                break;
            case 2 :
                goTo<4>(goTo<2>(pCenter, current_distance), p);
                break;
            case 3 :
                goTo<6>(goTo<3>(pCenter, current_distance), p);
                break;
            case 4 :
                goTo<6>(goTo<4>(pCenter, current_distance), p);
                break;
            case 5 :
                goTo<0>(goTo<5>(pCenter, current_distance), p);
                break;
            case 6 :
                goTo<0>(goTo<6>(pCenter, current_distance), p);
                break;
            case 7 :
                goTo<2>(goTo<7>(pCenter, current_distance), p);
                break;
            default :
                break;
            }//switch
        }
        else {// dist == 0
            // Current location in the grid
            pCur = pCenter;
        }

        return pCur;
    }//getNode

    //! HW 14.06.15: Add direct access with specific index
    /*!
     * Direct access
     */
    DEVICE_HOST PointCoord getNode(size_t id) {

        size_t d = 0;
        size_t s = 0;
        while (s < id)
        {
            d++;
            s += d * sizeN;
        }
        // Be careful here. If id = 0, then current_distance = 0, and the current_contour_pos will not be used.
        current_distance = d;
        current_contour_pos = id - (s - d * sizeN) - 1;

        return (getNode());
    }

    /*!
     * Incremental access
     */
    DEVICE_HOST PointCoord getNodeIncr() {

        if (current_distance != 0)
        {
            int Y = pCur[1];
            int X = pCur[0];

            switch (current_contour_pos / current_distance) {
            case 0 :
                Y = Y - 1;
                break;
            case 1 :
                X = X - 1;
                break;
            case 2 :
                X = X - 1;
                break;
            case 3 :
                Y = Y + 1;
                break;
            case 4 :
                Y = Y + 1;
                break;
            case 5 :
                X = X + 1;
                break;
            case 6 :
                X = X + 1;
                break;
            case 7 :
                Y = Y - 1;
                break;
            default :
                break;
            }//switch
            pCur[1] = Y;
            pCur[0] = X;
        }
        else {
            pCur = pCenter;
        }
        return pCur;
    }//getNodeInc

    //! HW 09/04/15: In CUDA_CODE, get() method is put directly into the inherited class.
    //! Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifdef CUDA_CODE
    DEVICE_HOST inline PointCoord get() {
        return this->getNodeIncr();
    }
#endif

};//NIterQuad

/*!
 * \brief The NIterQuadDual class
 */
//! NIter (couterclokwise) in Dual graph since
//! there is no problem of odd/even line and problem of
//! changing representation (in table) when passing to
//! dual.

//! HW 20/03/15 : add new NIterQuadDual class
/*!
 * \brief The NIterQuadDual class
 * Turn Clokwise starting from bottom point (dir=0)
 */
class NIterQuadDual : public NIterQuad
{
public:

    DEVICE_HOST NIterQuadDual() : NIterQuad() {}

    /*!
     * \brief NIterQuadDual
     * \param pc
     * \param d_min
     * \param d_max
     * \param offset
     * \return
     */
    DEVICE_HOST NIterQuadDual(PointCoord pc, size_t d_min, size_t d_max, size_t offset=0) :
        NIterQuad(pc, d_min, d_max, true, offset) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = true, size_t offset=0, size_t size=8) {
        current_distance = d_min;
        max_distance = d_max;
        offset_start = offset;
        sizeN = size;
        dual = dual_g;
        pCenter = pc;
        this->setCurrentDistance(d_min);
    }

    DEVICE_HOST bool nextNodeIncr() {
        bool ret = true;
        current_contour_pos += 1;
        if (current_contour_pos >= sizeN * current_distance) {
            current_distance += 1;
            current_contour_pos = 0;
            pCur[1] = pCenter[1] + current_distance;
            if (current_distance > max_distance)
                ret = false;
        }
        return ret;
    }

    /*!
     * Direct access on one direction
     */
    template<size_t DIR>
    DEVICE_HOST PointCoord goTo(PointCoord pc, size_t d) {

        int Y = pc[1];
        int X = pc[0];

        // Turn Clockwise starting from bottom
        switch (DIR) {
        case 0 :
            // down
            pCur[1] = Y + d;
            pCur[0] = X;
            break;
        case 1 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d;
            break;
        case 2 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 3 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d;
            break;
        case 4 :
            // up
            pCur[1] = Y - d;
            pCur[0] = X;
            break;
        case 5 :
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d;
            break;
        case 6 :
            // right
            pCur[1] = Y;
            pCur[0] = X + d;
            break;
        case 7 :
            // down right
            pCur[1] = Y + d;
            pCur[0] = X + d;
            break;
        default :
            break;
        }//switch

        return pCur;
    }//goTo

    /*!
     * Incremental access
     */
    DEVICE_HOST PointCoord getNodeIncr() {

        if (current_distance != 0)
        {
            int Y = pCur[1];
            int X = pCur[0];

            switch (current_contour_pos / current_distance) {
            case 0 :
                // left
                X = X - 1;
                break;
            case 1 :
                // up
                Y = Y - 1;
                break;
            case 2 :
                // up
                Y = Y - 1;
                break;
            case 3 :
                // right
                X = X + 1;
                break;
            case 4 :
                // right
                X = X + 1;
                break;
            case 5 :
                // down
                Y = Y + 1;
                break;
            case 6 :
                // down
                Y = Y + 1;
                break;
            case 7 :
                // left
                X = X - 1;
                break;
            default :
                break;
            }//switch
            pCur[1] = Y;
            pCur[0] = X;
        }
        else {
            pCur = pCenter;
        }
        return pCur;
    }//getNodeInc

    //! HW 09/04/15: In CUDA_CODE, get() method is put directly into the inherited class.
    //! Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifdef CUDA_CODE
    DEVICE_HOST inline PointCoord get() {
        return this->getNodeIncr();
    }
#endif

}; //NIterQuadDual


//! HW 06.08.15: Add K_NIter_goTo
template<class Grid, class Node,
         size_t MIN_DIST,
         size_t MAX_DIST,
         class NIter>
KERNEL void K_NIter_goTo(Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x == g1.getWidth()/2 && _y == g1.getHeight()/2) {

        PointCoord pc(_x,_y);
        NIter ni(pc, MIN_DIST, MAX_DIST);
        PointCoord pCoord;
        size_t direction = 0;
        do {
            ni.setCurrentDistance(MIN_DIST);
            do {
                size_t distance = ni.getCurrentDistance();
                pCoord = ni.goTo(pc, direction, distance);
                if (pCoord[0] >= 0 && pCoord[0] < g1.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < g1.getHeight()) {
                    g1[pCoord[1]][pCoord[0]] = Node(ni.getCurrentDistance());
                }
            } while (ni.nextDistanceIncr());
            direction++;
        } while (direction < ni.getSizeN());
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 14.06.15: Add K_NIterDirectAccess
//! KERNEL FUNCTION
template<class Grid, class Node,
         size_t MIN_DIST,
         size_t MAX_DIST,
         class NIter>
KERNEL void K_NIterDirectAccess(Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x == g1.getWidth()/2 && _y == g1.getHeight()/2) {

        NIter ni(PointCoord(_x,_y), MIN_DIST, MAX_DIST);
        PointCoord pCoord;
        size_t id = 0;
        do {
            pCoord = ni.getNode(id);
            if (pCoord[0] >= 0 && pCoord[0] < g1.getWidth()
                    && pCoord[1] >= 0 && pCoord[1] < g1.getHeight()) {
                g1[pCoord[1]][pCoord[0]] = Node(id);
            }
            id++;
        } while (id < ni.getTotalSizeN(MAX_DIST));
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
template<class Grid, class Node,
         size_t MIN_DIST,
         size_t MAX_DIST,
         class NIter>
KERNEL void K_NIter(Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x == g1.getWidth()/2 && _y == g1.getHeight()/2) {

        NIter ni(PointCoord(_x,_y), MIN_DIST, MAX_DIST);
        PointCoord pCoord;
        do {
            pCoord = ni.getNodeIncr();
            if (pCoord[0] >= 0 && pCoord[0] < g1.getWidth()
                    && pCoord[1] >= 0 && pCoord[1] < g1.getHeight()) {
                g1[pCoord[1]][pCoord[0]] = Node(ni.getCurrentDistance());
            }
        } while (ni.nextNode());
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
template<class Grid, class Node,
         size_t MIN_DIST,
         size_t MAX_DIST,
         class NIter>
KERNEL void K_NIterByDistanceStep(Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x == g1.getWidth()/2 && _y == g1.getHeight()/2) {

        NIter ni(PointCoord(_x,_y), MIN_DIST, MAX_DIST);
        for (int d = MIN_DIST; d <= MAX_DIST; ++d) {
            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            PointCoord pCoord;
            do {
//                pCoord = ni.getNode();
                pCoord = ni.getNodeIncr();
                if (pCoord[0] >= 0 && pCoord[0] < g1.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < g1.getHeight()) {
                    g1[pCoord[1]][pCoord[0]] = Node(cd);
                }
            } while (ni.nextContourNode());
        }
    }

    END_KER_SCHED

    SYNCTHREADS;
}

// Hongjian's Debug
//! Test program
template <class Node,
          size_t SXX,
          size_t SYY,
          size_t MIN_DIST,
          size_t MAX_DIST,
          class NIter>
class TestNiter {

    Node initNode1;

public:

    TestNiter(Node n1) : initNode1(n1) {}

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

        cout << "debut test Neighborhood Iterator ..." << endl;
        const size_t SX = SXX, SY = SYY;
        // Creation de grille en local
        Grid<Node> gd(SX, SY);
        gd = Node(initNode1);

        //! HW 11/03/15: concise notation test
        PointCoord p1(1,2);
        Node pp1 = gd[p1[1]][p1[0]];
        (ofstream&) std::cout << "pp1 = " << endl;
        (ofstream&) cout << pp1 << endl;

        cout << "Creation de grilles sur device GPU ..." << endl;
        // Creation de grilles sur device GPU
        Grid<Node> gpu_gd;
        gpu_gd.gpuResize(SX, SY);

        // cuda timer
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
        double x0;
        x0 = clock();

        // Affichage
#if TEST_CODE
        (ofstream&) std::cout << "gd1 = " << endl;
        (ofstream&) std::cout << gd << endl;
#endif
        cout << "Appel du Kernel ..." << endl;

        // Copie des grilles CPU -> GPU
        gd.gpuCopyHostToDevice(gpu_gd);

        // Kernel call with class parameters
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              gpu_gd.getWidth(),
                              gpu_gd.getHeight());
//        K_NIter<Grid<Node>, Node, MIN_DIST, MAX_DIST, NIter> _KER_CALL_(b, t) (gpu_gd);
        K_NIterByDistanceStep<Grid<Node>, Node, MIN_DIST, MAX_DIST, NIter> _KER_CALL_(b, t) (gpu_gd);
//        K_NIterDirectAccess<Grid<Node>, Node, MIN_DIST, MAX_DIST, NIter> _KER_CALL_(b, t) (gpu_gd);
//        K_NIter_goTo<Grid<Node>, Node, MIN_DIST, MAX_DIST, NIter> _KER_CALL_(b, t) (gpu_gd);

        // Copie du resultat GPU -> CPU
        gd.gpuCopyDeviceToHost(gpu_gd);

        cout << "Affichage du resultat a la console ..." << endl;
        // Affichage du resultat à la console
        (ofstream&) std::cout << "gd = " << endl;
        (ofstream&) cout << gd << endl;

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
        gpu_gd.gpuFreeMem();
        gd.freeMem();
    }
};

}//components
#endif // NITER_H

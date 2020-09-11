#ifndef NITER1D_H
#define NITER1D_H
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
#include "Node.h"
#include "GridOfNodes.h"

#include "NIter.h"


using namespace std;
using namespace components;

namespace components
{

/*!
 * \brief The NIter1D class
 * Turn CounterClokwise starting from right point (dir=0)
 */
class NIter1D : public NeighborhoodIterator
{
public:

    DEVICE_HOST NIter1D() : NeighborhoodIterator() {}

    /*!
     * \brief NIterHexa
     * \param pc
     * \param d_min
     * \param d_max
     * \param dual_g
     * \param offset
     * \return
     */
    DEVICE_HOST NIter1D(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0) :
        NeighborhoodIterator(pc, d_min, d_max, dual_g, offset, 2) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0, size_t size=2) {
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
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
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
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
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
            switch (current_contour_pos % 2) {
            case 0 :
                goTo<0>(pCenter, current_distance);
                break;
            case 1 :
                goTo<1>(pCenter, current_distance);
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

            switch (current_contour_pos % 2) {
            case 0 :
                X = X - 2 * current_distance;
                break;
            case 1 :
                X = X + 2 * current_distance;
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

};//NIter1D

}//components
#endif // NITER1D_H

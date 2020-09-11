#ifndef NITERHEXA_H
#define NITERHEXA_H
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
 * \brief The NIterHexa class
 * Turn CounterClokwise starting from right point (dir=0)
 */
class NIterHexa : public NeighborhoodIterator
{
public:

    DEVICE_HOST NIterHexa() : NeighborhoodIterator() {}

    /*!
     * \brief NIterHexa
     * \param pc
     * \param d_min
     * \param d_max
     * \param dual_g
     * \param offset
     * \return
     */
    DEVICE_HOST NIterHexa(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0) :
        NeighborhoodIterator(pc, d_min, d_max, dual_g, offset, 6) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = false, size_t offset=0, size_t size=6) {
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
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d / 2 + (d % 2)*((Y+1) % 2)*((Y+1) % 2);
            break;
        case 2 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 3 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 4 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 5 :
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
            pCur[0] = X + d / 2 + (d % 2)*((Y+1) % 2)*((Y+1) % 2);
            break;
        case 2 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 3 :
            // left
            pCur[1] = Y;
            pCur[0] = X - d;
            break;
        case 4 :
            // down left
            pCur[1] = Y + d;
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 5 :
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
                goTo<2>(goTo<0>(pCenter, current_distance), p);
                break;
            case 1 :
                goTo<3>(goTo<1>(pCenter, current_distance), p);
                break;
            case 2 :
                goTo<4>(goTo<2>(pCenter, current_distance), p);
                break;
            case 3 :
                goTo<5>(goTo<3>(pCenter, current_distance), p);
                break;
            case 4 :
                goTo<0>(goTo<4>(pCenter, current_distance), p);
                break;
            case 5 :
                goTo<1>(goTo<5>(pCenter, current_distance), p);
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
                X = X - ((Y + 1) % 2) * ((Y + 1) % 2);
                break;
            case 1 :
                X = X - 1;
                break;
            case 2 :
                Y = Y + 1;
                X = X - ((Y + 1) % 2) * ((Y + 1) % 2);
                break;
            case 3 :
                Y = Y + 1;
                X = X + (Y % 2) * (Y % 2);
                break;
            case 4 :
                X = X + 1;
                break;
            case 5 :
                Y = Y - 1;
                X = X + (Y % 2) * (Y % 2);
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

};//NIterHexa

/*!
 * \brief The NIterHexaDual class
 * Turn Clokwise starting from bottom point (dir=0)
 */
class NIterHexaDual : public NIterHexa
{
public:

    DEVICE_HOST NIterHexaDual() : NIterHexa() {}

    /*!
     * \brief NIterHexaDual
     * \param pc
     * \param d_min
     * \param d_max
     * \param offset
     * \return
     */
    DEVICE_HOST NIterHexaDual(PointCoord pc, size_t d_min, size_t d_max, size_t offset=0) :
        NIterHexa(pc, d_min, d_max, true, offset) { }

    DEVICE_HOST inline void initialize(PointCoord pc, size_t d_min, size_t d_max, bool dual_g = true, size_t offset=0, size_t size=6) {
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
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 2 :
            // up left
            pCur[1] = Y - d;
            pCur[0] = X - d / 2 - (d % 2)*(Y % 2)*(Y % 2);
            break;
        case 3 :
            // up
            pCur[1] = Y - 2 * d;
            pCur[0] = X;
            break;
        case 4 :
            // up right
            pCur[1] = Y - d;
            pCur[0] = X + d / 2 + (d % 2)*((Y+1) % 2)*((Y+1) % 2);
            break;
        case 5 :
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
                // up
                pCur[1] = Y - 2;
                break;
            case 2 :
                // up right
                pCur[1] = Y - 1;
                pCur[0] = X + ((Y+1) % 2)*((Y+1) % 2);
                break;
            case 3 :
                // down right
                pCur[1] = Y + 1;
                pCur[0] = X + ((Y+1) % 2)*((Y+1) % 2);
                break;
            case 4 :
                // down
                pCur[1] = Y + 2;
                break;
            case 5 :
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

};//NIterHexaDual


}//components
#endif // NITERHEXA_H

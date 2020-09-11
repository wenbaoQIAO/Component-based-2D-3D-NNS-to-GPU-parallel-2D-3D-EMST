#ifndef LOCALSEARCH_H
#define LOCALSEARCH_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : May 2015
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
#include <cuda_runtime.h>
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
#include "NeuralNet.h"
#include "distances_matching.h"
#include "basic_operations.h"
#include "Cell.h"
#include "SpiralSearch.h"
#include "adaptator_basics.h"
#include "CellularMatrix.h"
#include "distance_functors.h"
#include "Evaluation.h"
//#include "SOM3DOperators.h"
//#include "Trace.h"

#include "ViewGrid.h"
#ifdef TOPOLOGIE_HEXA
#include "ViewGridHexa.h"
#endif
#include "NIter.h"

//! JCC 200415 : TSP
#include "NIter1D.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif

//#define SYNCHRONIZED_EXECUTION
#ifdef SYNCHRONIZED_EXECUTION
#else
#define DYNAMIC_CELLULAR_MATRIX
#endif

#define COMPLETE_LS_UNTIL_CONVERGENCY 1

#define STEREO_OPERATOR         1
#define CELL_SMALL_MOVE_RATE    100
#define CELL_PROPAGATION_RATE   1
#define CELL_PERTURBATION_RATE  100


#define DY_CENTER_WINDOW_MOVE_RATE                   100
#define DY_CENTER_WINDOW_RANDOM_MOVE_RATE            100
#define DY_RANDOM_WINDOW_MOVE_RATE                   100
#define DY_RANDOM_PICK_MOVE_RATE                     100
#define DY_RANDOM_PICK_RANDOM_MOVE_RATE              100
#define DY_RANDOM_PICK_EXPANSION_AND_SWAP_MOVE_RATE  1

#define STEREO_CONS_WTA 1

using namespace std;
using namespace components;

namespace operators
{

#ifdef DYNAMIC_CELLULAR_MATRIX

struct flowResultInfo
{
    int leftRight; // 0-left; 1-right; 2-post-processing
    int initMode;
    int iteNum;
    int cellRadius;
    int lsMode;
    float costWei;
    float costWindowWei;
    float smoothWei;
    float objValBefore;
    float objValAfter;
    float objValDataBefore;
    float objValDataAfter;
    float objValSmoothBefore;
    float objValSmoothAfter;
    float GTErrBefore;
    float GTErrAfter;
    float runtime;
    float CUDA_runtime;
};

/*!
 * \brief K_DLS_WTA_Data
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class GetAdaptor,
          class NeighborAdaptor
          >
KERNEL void K_DLS_WTA_Data(CellularMatrix cmr,
                           NN nnr,
                           NN nnd,
                           Evaluation evaluate,
                           GetAdaptor getAdaptor,
                           NeighborAdaptor neighborAdaptor)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        getAdaptor.init(cmr[_y][_x]);

        do {
            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cmr[_y][_x],
                                       nnr,
                                       ps);

            if (extracted
                    && ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {

                neighborAdaptor.init(ps);

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                      nnr,
                                                      nnd,
                                                      ps,
                                                      pNew);
                    if (getNeighbor) {

                        GLfloat currentCost, newCost;
                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                        newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                        if (newCost < currentCost)
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                        }
                    }

                } while (neighborAdaptor.next());
            }

        } while (getAdaptor.next(cmr[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_WTA_Data

/*!
 * \brief K_DLS_localSearchLength
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class GetAdaptor,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchLength(CellularMatrix cmr,
                                    NN nnr,
                                    NN nnd,
                                    Evaluation evaluate,
                                    GetAdaptor getAdaptor,
                                    NeighborAdaptor neighborAdaptor,
                                    bool is_FI,
                                    size_t CMid)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        getAdaptor.init(cmr[_y][_x]);
        size_t pindex = 0;

        do {

            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cmr[_y][_x],
                                       nnr,
                                       ps);

            if (extracted
                    && ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {


                bool improvementFound = false;
                neighborAdaptor.init(ps, pindex+CMid);

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                     nnr,
                                                     nnd,
                                                     ps,
                                                     pNew);

                    if (getNeighbor) {

                        GLfloat currentCost, currentLen, newCost, newLen;
                        GLfloat newLenCenter, newLenUp, newLenLeft, newLenUpLeft;

                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                        currentLen = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_length];
                        // left
                        if (ps[0] - 1 >= 0)
                            currentLen += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_length];
                        // up
                        if (ps[1] - 1 >= 0)
                            currentLen += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_length];
                        if (cmr[_y][_x].iter.getSizeN() == 6) // for hexa only
                        {
                            // upper left
                            if (ps[0] - 1 >= 0 && ps[1] - 1 >= 0)
                                currentLen += nnr.objectivesMap[ps[1]-1][ps[0]-1][(int)obj_length];
                        }

                        newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                        newLen = evaluate.evaluateNodeLength(nnr, ps, pNew, newLenCenter, newLenUp, newLenLeft, newLenUpLeft);

                        if ((evaluate.actObj.cost * newCost + evaluate.actObj.length * newLen)
                                < (evaluate.actObj.cost * currentCost + evaluate.actObj.length * currentLen))
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_length] = newLenCenter;
                            // left
                            if (ps[0] - 1 >= 0)
                                nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_length] = newLenLeft;
                            // up
                            if (ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_length] = newLenUp;
                            if (cmr[_y][_x].iter.getSizeN() == 6) // for hexa only
                            {
                                // upper left
                                if (ps[0] - 1 >= 0 && ps[1] - 1 >= 0)
                                    nnr.objectivesMap[ps[1]-1][ps[0]-1][(int)obj_length] = newLenUpLeft;
                            }
                            improvementFound = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
            }

            pindex++;

        } while (getAdaptor.next(cmr[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchLength

/*!
 * \brief K_DLS_localSearchSmooth
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class GetAdaptor,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchSmooth(CellularMatrix cmr,
                                    NN nnr,
                                    NN nnd,
                                    Evaluation evaluate,
                                    GetAdaptor getAdaptor,
                                    NeighborAdaptor neighborAdaptor,
                                    bool is_FI,
                                    size_t CMid)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        getAdaptor.init(cmr[_y][_x]);
        size_t pindex = 0;

        do {

            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cmr[_y][_x],
                                       nnr,
                                       ps);

            if (extracted
                    && ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {



                bool improvementFound = false;
                neighborAdaptor.init(ps, pindex+CMid);

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                     nnr,
                                                     nnd,
                                                     ps,
                                                     pNew);

                    if (getNeighbor) {

                        GLfloat currentCost, currentSmo, newCost, newSmo;
                        GLfloat newSmoCenter, newSmoUp, newSmoLeft;

                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                        currentSmo = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing];
                        // left
                        if (ps[0] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing];
                        // up
                        if (ps[1] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing];

                        newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                        newSmo = evaluate.evaluateNodeSmooth(nnr, nnd, ps, pNew, newSmoCenter, newSmoUp, newSmoLeft);

                        if ((evaluate.actObj.cost * newCost + evaluate.actObj.smoothing * newSmo)
                                < (evaluate.actObj.cost * currentCost + evaluate.actObj.smoothing * currentSmo))
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing] = newSmoCenter;
                            // left
                            if (ps[0] - 1 >= 0)
                                nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing] = newSmoLeft;
                            // up
                            if (ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing] = newSmoUp;

                            improvementFound = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
            }

            pindex++;

        } while (getAdaptor.next(cmr[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchSmooth

/*!
 * \brief K_DLS_localSearchWindow
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class GetAdaptor,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchWindow(CellularMatrix cmr,
                                    NN nnr,
                                    NN nnd,
                                    Evaluation evaluate,
                                    GetAdaptor getAdaptor,
                                    NeighborAdaptor neighborAdaptor,
                                    bool is_FI,
                                    size_t CMid)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        getAdaptor.init(cmr[_y][_x]);
        size_t pindex = 0;

        do {

            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cmr[_y][_x],
                                       nnr,
                                       ps);


            if (extracted
                    && ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {

                bool improvementFound = false;
                neighborAdaptor.init(ps, pindex+CMid);

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                     nnr,
                                                     nnd,
                                                     ps,
                                                     pNew);
                    if (getNeighbor) {
                        GLfloat currentCost, newCost;
                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window];
                        PointEuclid pMotion = pNew - nnd.adaptiveMap[ps[1]][ps[0]];
                        newCost = evaluate.evaluateNodeCostWindow(nnr, nnd, ps, pMotion);
                        if (newCost < currentCost)
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window] = newCost;
                            improvementFound = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
            }

            pindex++;

        } while (getAdaptor.next(cmr[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchWindow

/*!
 * \brief K_DLS_localSearchWindowSmooth
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class GetAdaptor,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchWindowSmooth(CellularMatrix cmr,
                                          NN nnr,
                                          NN nnd,
                                          Evaluation evaluate,
                                          GetAdaptor getAdaptor,
                                          NeighborAdaptor neighborAdaptor,
                                          bool is_FI,
                                          size_t CMid)

{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        getAdaptor.init(cmr[_y][_x]);
        size_t pindex = 0;

        do {

            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cmr[_y][_x],
                                       nnr,
                                       ps);


            if (extracted
                    && ps[0] >= 0
                    && ps[0] < nnr.adaptiveMap.getWidth()
                    && ps[1] >= 0
                    && ps[1] < nnr.adaptiveMap.getHeight()) {

                bool improvementFound = false;
                neighborAdaptor.init(ps, pindex+CMid);

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                      nnr,
                                                      nnd,
                                                      ps,
                                                      pNew);
                    if (getNeighbor) {

                        GLfloat currentCost, currentSmo, newCost, newSmo;
                        GLfloat newSmoCenter, newSmoUp, newSmoLeft;


                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window];
                        currentSmo = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing];
                        // left
                        if (ps[0] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing];
                        // up
                        if (ps[1] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing];


                        PointEuclid pMotion = pNew - nnd.adaptiveMap[ps[1]][ps[0]];
                        newCost = evaluate.evaluateNodeCostWindow(nnr, nnd, ps, pMotion);
                        newSmo = evaluate.evaluateNodeSmooth(nnr, nnd, ps, pNew, newSmoCenter, newSmoUp, newSmoLeft);

                        if ((evaluate.actObj.cost_window * newCost + evaluate.actObj.smoothing * newSmo)
                                < (evaluate.actObj.cost_window * currentCost + evaluate.actObj.smoothing * currentSmo))
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window] = newCost;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing] = newSmoCenter;
                            // left
                            if (ps[0] - 1 >= 0)
                                nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing] = newSmoLeft;
                            // up
                            if (ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing] = newSmoUp;

                            improvementFound = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
            }

            pindex++;

        } while (getAdaptor.next(cmr[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchWindowSmooth

/*!
 * \brief K_DLS_localSearchLargeMove
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchLargeMove(CellularMatrix cmr,
                                       NN nnr,
                                       NN nnd,
                                       Evaluation evaluate,
                                       NeighborAdaptor neighborAdaptor,
                                       bool is_FI,
                                       size_t CMid)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {



        neighborAdaptor.init(cmr[_y][_x],
                             nnr,
                             nnd,
                             CMid);

        bool improvementFound = false;

        do {
            float costBefore = evaluate.evaluateCell(cmr[_y][_x], nnr, nnd);
            neighborAdaptor.backup(nnr);
            neighborAdaptor.move(cmr[_y][_x], nnr, nnd);
            float costAfter = evaluate.evaluateCell(cmr[_y][_x], nnr, nnd);

//            if (_x > 10 && _x < 20 && _y > 10 && _y < 20)
//            {
//                printf("(%d,%d) : costBefore = %f, costAfter = %f\n", _x, _y, costBefore, costAfter);
//            }

            if (costBefore < costAfter)
                neighborAdaptor.recover(nnr);
            else if (costBefore > costAfter)
                improvementFound = true;

        } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));

        neighborAdaptor.clean();
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchLargeMove

#endif // #ifdef DYNAMIC_CELLULAR_MATRIX

//! Type of comportment for decreasing parameters
enum LS_ModeCalcul {
    LS_STOCHASTIC,
    LS_FIRST,
    LS_BEST,
    VND,
    VNDrand
};

//! Type of comportment for decreasing parameters
struct LocalSearchParams {
    int niter;
    size_t nGene;
    LS_ModeCalcul modeCalcul;//online/batch
    bool buffered;//the matcher is buffered via the savgab

    DEVICE_HOST LocalSearchParams() {
        niter = 1;
        nGene = 100;
        modeCalcul = LS_FIRST;
        buffered = false;
    }

    /*!
     * \brief readParameters
     * \param name
     */
    DEVICE_HOST void readParameters(std::string const& name, ConfigParams& params) {
        params.readConfigParameter(name,"niter", niter);
        params.readConfigParameter(name,"nGene", nGene);
        params.readConfigParameter(name,"modeCalcul", (int&)modeCalcul);
        params.readConfigParameter(name,"buffered", buffered);
    }//readParameters

};

/*!
 * \brief Class Local Search.
 * It is a mixte CPU/GPU class, then with
 * default copy constructor for allowing
 * parameter passing to GPU Kernel.
 **/
template <class CellularMatrixR,
          class CellR,
          class CellD,
          class NIter,
          class NIterDual,
          class NIterNeighborhood,
          class Evaluation
#ifdef DYNAMIC_CELLULAR_MATRIX
          , class EvaluationCM
          , class ViewG
#endif
          >
class LocalSearchOperator
{
protected:
//public:
    //! Neural net that matches
    NN mr;
    //! Neural net that is matched
    NN md;
    //! Cellular matrix with neural net that matches
    CellularMatrixR cmr;
    //! Evaluation
    Evaluation eva;
#ifdef DYNAMIC_CELLULAR_MATRIX
    EvaluationCM evaCM;
#endif
    //! Standard Som parameters
    LocalSearchParams localSearchParams;
    //! Generation courante
    size_t gene;
    //! HW 19.06.15 : add for the SYNCHRONIZED_EXECUTION case
    size_t cellSize;

    //! Adaptator
    GetStdAdaptor<CellD> gsdta;
//    GetStdAdaptor<CellR> gsdtaR;

    //! Adaptator which needs initialize()
#if STEREO_OPERATOR
    GenerateNeighborStereoRandom<CellD> gnra;
#else
    GenerateNeighborRandom<CellD, NIterNeighborhood> gnra;
#endif
    GenerateNeighborInFixedDirectionRandom<CellD, NIter> vnara;

#ifdef DYNAMIC_CELLULAR_MATRIX
    GenerateNeighborCenterWindowRandomMove<CellD, NIterNeighborhood, NIter> gnwarm;
#if STEREO_OPERATOR
    GenerateNeighborRandomWindowStereo<CellD, NIter> gnrwa;
    GenerateNeighborRandomPickStereo<CellD, NIter> gnrpa;
#else
    GenerateNeighborRandomWindowStereo<CellD, NIter> gnrwa; // To add adaptator GenerateNeighborRandomWindow
    GenerateNeighborRandomPick<CellD, NIterNeighborhood, NIter> gnrpa;
#endif
    GenerateNeighborRandomPickRandomMove<CellD, NIterNeighborhood, NIter> gnrparm;
    GenerateNeighborRandomPickExpansionMoveAndSwapMove<CellD, NIter> gnrpaes;
#endif

public:
    flowResultInfo _flowResultInfo;

    DEVICE_HOST explicit LocalSearchOperator() {}

#ifdef DYNAMIC_CELLULAR_MATRIX
    //! \brief Constructeur par defaut.
    DEVICE_HOST explicit LocalSearchOperator(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            LocalSearchParams& p,
            Evaluation& eval,
            EvaluationCM& evalcm
            ) :
        mr(nnr),
        md(nnd),
        cmr(cr),
        localSearchParams(p),
        eva(eval),
        evaCM (evalcm) {}

    /*!
     * \brief initialize
     */
    GLOBAL void initialize(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            LocalSearchParams& p,
            Evaluation& eval,
            EvaluationCM& evalcm
            ) {

        mr = nnr;
        md = nnd;
        cmr = cr;
        localSearchParams = p;
        eva = eval;
        evaCM = evalcm;

        initialize();
    }

    /*!
     * \brief initialize
     */
    GLOBAL void initialize(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            LocalSearchParams& p,
            Evaluation& eval,
            EvaluationCM& evalcm,
            size_t _cellSize
            ) {

        mr = nnr;
        md = nnd;
        cmr = cr;
        localSearchParams = p;
        eva = eval;
        evaCM = evalcm;
        cellSize = _cellSize;

        initialize();
    }
#else
    //! \brief Constructeur par defaut.
    DEVICE_HOST explicit LocalSearchOperator(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            LocalSearchParams& p,
            Evaluation& eval
            ) :
        mr(nnr),
        md(nnd),
        cmr(cr),
        localSearchParams(p),
        eva(eval) {}

    /*!
     * \brief initialize
     */
    GLOBAL void initialize(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            LocalSearchParams& p,
            Evaluation& eval
            ) {

        mr = nnr;
        md = nnd;
        cmr = cr;
        localSearchParams = p;
        eva = eval;

        initialize();
    }

    /*!
     * \brief initialize
     */
    GLOBAL void initialize(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            LocalSearchParams& p,
            Evaluation& eval,
            size_t _cellSize
            ) {

        mr = nnr;
        md = nnd;
        cmr = cr;
        localSearchParams = p;
        eva = eval;
        cellSize = _cellSize;

        initialize();
    }
#endif

    /*!
     * \brief initialize
     */
    GLOBAL void initialize() {

        gnra.initialize(cmr);
        vnara.initialize(cmr);

#ifdef DYNAMIC_CELLULAR_MATRIX
        gnwarm.initialize(cmr);
        gnrwa.initialize(cmr);
        gnrpa.initialize(cmr);
        gnrparm.initialize(cmr);
        gnrpaes.initialize(cmr);
#endif
        init();

    }//initialize()

    /*!
     * \brief init
     */
    void init() {
        gene = 0;

#ifdef DYNAMIC_CELLULAR_MATRIX
        gnra.init(cellSize * MAX_NUM_CM);
        vnara.init(cellSize * MAX_NUM_CM);
#else
        gnra.init(cellSize);
        vnara.init(cellSize);
#endif
    }

    /*!
     * \brief run
     */
    GLOBAL float evaluation() {
        eva.K_evaluate(mr, md, DEFAULT_WINDOW_RADIUS);
        GLfloat evaCost = eva.K_reductionForSum(mr, md);
        cout << "evaCost = " << evaCost << "(" << (int)evaCost << ")" << endl;
        return evaCost;
    } // evaluation()

    int gettimeofday(struct timeval * tp, struct timezone * tzp)
    {
//        // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
//        static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

//        SYSTEMTIME  system_time;
//        FILETIME    file_time;
//        uint64_t    time;

//        GetSystemTime( &system_time );
//        SystemTimeToFileTime( &system_time, &file_time );
//        time =  ((uint64_t)file_time.dwLowDateTime )      ;
//        time += ((uint64_t)file_time.dwHighDateTime) << 32;

//        tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
//        tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
        return 0;
    }

    /*!
     * \brief run
     */
    GLOBAL void run() {

        //! WH 040815 : for optical flow result statistics
#if COMPLETE_LS_UNTIL_CONVERGENCY
        _flowResultInfo.iteNum += localSearchParams.nGene;
#else
        _flowResultInfo.iteNum = localSearchParams.nGene;
#endif
        _flowResultInfo.lsMode = localSearchParams.modeCalcul;
        _flowResultInfo.costWei = eva.actObj.cost;
        _flowResultInfo.costWindowWei = eva.actObj.cost_window;
        _flowResultInfo.smoothWei = eva.actObj.smoothing;

        cout << "================= run =================" << endl;
        cout << "Before DLS:" << endl;
        _flowResultInfo.objValBefore = evaluation();
        _flowResultInfo.objValDataBefore = eva.getObjectives(obj_cost) * eva.actObj.cost;
        _flowResultInfo.objValSmoothBefore = eva.getObjectives(obj_smoothing) * eva.actObj.smoothing;

        // timer
        timeval tv, tv1;
        double __timePreprocessing__;
        gettimeofday(&tv, 0);
        // cuda timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);


#ifdef DYNAMIC_CELLULAR_MATRIX

        PointCoord pcCur = cmr.vgd.getCenter();
        size_t _RCur = cmr.vgd.getLevel();
        NIter iteTem(pcCur, 0, _RCur);
        size_t neighborhoodNum = iteTem.getSizeN();
        size_t wid = cmr.vgd.getWidthLowLevel();
        size_t hei = cmr.vgd.getHeightLowLevel();
        ViewG vgdDyn;
        vgdDyn = ViewG(pcCur, wid, hei, _RCur);

        cmr.setViewG(vgdDyn);
        cmr.gpuResize(vgdDyn.getWidthDual(), vgdDyn.getHeightDual());
        cmr.K_initialize_dynamicCM(vgdDyn);
        evaCM.initCM(cmr);

        for (int i = 0; i < localSearchParams.nGene; i++) {

            // run with the original cellular matrix
            bool ret = this->activate(0);
            if (!ret)
                break;

            size_t direction = 0;

            do {

                // change the center of cellular matrix
                PointCoord pco = iteTem.goTo(pcCur, direction, 4);
//                PointCoord pco = iteTem.goTo(pcCur, direction, _RCur);
                vgdDyn = ViewG(pco, wid, hei, _RCur);
             //   cout << " pco = " << pco[0] << " " << pco[1] << endl;

                cmr.setViewG(vgdDyn);
                cmr.gpuResize(vgdDyn.getWidthDual(), vgdDyn.getHeightDual());
                cmr.K_initialize_dynamicCM(vgdDyn);
                evaCM.initCM(cmr);

                // run with the new cellular matrix
                this->activate(direction+1);


            } while (++direction < neighborhoodNum);
//            } while (++direction < 2);

            // reset to the original cellular matrix
            vgdDyn = ViewG(pcCur, wid, hei, _RCur);
            cmr.setViewG(vgdDyn);
            cmr.gpuResize(vgdDyn.getWidthDual(), vgdDyn.getHeightDual());
            cmr.K_initialize_dynamicCM(vgdDyn);
            evaCM.initCM(cmr);

            gene += 1;
        }
#else
        for (int i = 0; i < localSearchParams.nGene; i++) {
            bool ret = this->activate(0);
            gene += 1;
            if (!ret)
                break;
        }
#endif

        // timer
        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        _flowResultInfo.CUDA_runtime = elapsedTime;
        gettimeofday(&tv1, 0);

        __timePreprocessing__ = tv1.tv_sec - tv.tv_sec + (double)(tv1.tv_usec - tv.tv_usec) / CLOCKS_PER_SEC;
        _flowResultInfo.runtime = __timePreprocessing__;

        cout << "After DLS: (" << localSearchParams.nGene << " iterations)" << endl;
        _flowResultInfo.objValAfter = evaluation();
        _flowResultInfo.objValDataAfter = eva.getObjectives(obj_cost) * eva.actObj.cost;
        _flowResultInfo.objValSmoothAfter = eva.getObjectives(obj_smoothing) * eva.actObj.smoothing;

#if COMPLETE_LS_UNTIL_CONVERGENCY
#endif

    } // run()


    /*!
     * \brief construct
     * \return
     */
    GLOBAL void construct() {

#ifdef SYNCHRONIZED_EXECUTION
        size_t pindex = 0;
        do {
            GenerateNeighborStereo<CellD> disp;
            K_WTA_Data(cmr, mr, md, eva, disp, pindex);
            pindex++;
        } while (pindex < cellSize);
#endif

#ifdef DYNAMIC_CELLULAR_MATRIX
        GenerateNeighborStereo<CellD> disp;
        K_WTA_Data(cmr, mr, md, eva, gsdta, disp);
#endif

    }

    /*!
     * \brief activate
     * \return
     */
    GLOBAL bool activate(size_t CMid) {
        bool ret = true;

        if (gene == localSearchParams.nGene)
            return false;


#ifdef SYNCHRONIZED_EXECUTION

        size_t pindex = 0;

        do {

            if (localSearchParams.modeCalcul == VND) {

                GenerateNeighborInFixedDirection<CellD, NIter, 10, 1> vna;



                if (eva.actObj.cost_window != 0) {

                    if (eva.actObj.smoothing != 0)
                        K_VND_WindowSmooth(cmr, mr, md, eva, vna, false, pindex);
                    else
                        K_VND_Window(cmr, mr, md, eva, vna, false, pindex);
                }

                else {

                    if (eva.actObj.smoothing != 0)
                        K_VND_Smooth(cmr, mr, md, eva, vna, false, pindex);
                    else
                        K_VND_Length(cmr, mr, md, eva, vna, false, pindex);
                }
            }

            else if (localSearchParams.modeCalcul == VNDrand) {


                vnara.setGene(gene);

                if (eva.actObj.cost_window != 0) {

                    if (eva.actObj.smoothing != 0)
                        K_VND_WindowSmooth(cmr, mr, md, eva, vnara, false, pindex);
                    else
                        K_VND_Window(cmr, mr, md, eva, vnara, false, pindex);
                }

                else {

                    if (eva.actObj.smoothing != 0)
                        K_VND_Smooth(cmr, mr, md, eva, vnara, false, pindex);
                    else
                        K_VND_Length(cmr, mr, md, eva, vnara, false, pindex);
                }
            }

            else {

                if ((gene+1) % CELL_PROPAGATION_RATE == 0) {

                    PropagateNeighbor<CellD, NIter> pa;


                    if (eva.actObj.cost_window != 0) {

                        if (eva.actObj.smoothing != 0)
                            K_localSearchWindowSmooth(cmr, mr, md, eva, pa, false, pindex);
                        else
                            K_localSearchWindow(cmr, mr, md, eva, pa, false, pindex);
                    }

                    else {

                        if (eva.actObj.smoothing != 0)
                            K_localSearchSmooth(cmr, mr, md, eva, pa, false, pindex);
                        else
                            K_localSearchLength(cmr, mr, md, eva, pa, false, pindex);
                    }
                }

                if ((gene+1) % CELL_PERTURBATION_RATE == 0) {

                    GenerateNeighborStereoPerturbation<CellD> npa;


                    K_perturbation(cmr, mr, md, npa, pindex);

                    eva.K_evaluate(mr, md, DEFAULT_WINDOW_RADIUS);
                }

                if ((gene+1) % CELL_SMALL_MOVE_RATE == 0) {

                    if (localSearchParams.modeCalcul == LS_STOCHASTIC) {

                        gnra.setGene(gene);

                        if (eva.actObj.cost_window != 0) {

                            if (eva.actObj.smoothing != 0)
                                K_localSearchWindowSmooth(cmr, mr, md, eva, gnra, false, pindex);
                            else
                                K_localSearchWindow(cmr, mr, md, eva, gnra, false, pindex);
                        }

                        else {

                            if (eva.actObj.smoothing != 0)
                                K_localSearchSmooth(cmr, mr, md, eva, gnra, false, pindex);
                            else
                                K_localSearchLength(cmr, mr, md, eva, gnra, false, pindex);
                        }

                    } else if (localSearchParams.modeCalcul == LS_FIRST) {


#if STEREO_OPERATOR
                        GenerateNeighborStereo<CellD> na;
#else
                        GenerateNeighbor<CellD, NIterNeighborhood, 2, 1> na;
#endif

                        if (eva.actObj.cost_window != 0) {

                            if (eva.actObj.smoothing != 0)

                                K_localSearchWindowSmooth(cmr, mr, md, eva, na, true, pindex);



                            else
                                K_localSearchWindow(cmr, mr, md, eva, na, true, pindex);
                        }

                        else {

                            if (eva.actObj.smoothing != 0)
                                K_localSearchSmooth(cmr, mr, md, eva, na, true, pindex);
                            else
                                K_localSearchLength(cmr, mr, md, eva, na, true, pindex);
                        }

                    } else if (localSearchParams.modeCalcul == LS_BEST) {

#if STEREO_OPERATOR
                        GenerateNeighborStereo<CellD> na;
#else
                        GenerateNeighbor<CellD, NIterNeighborhood, 2, 1> na;
#endif

                        if (eva.actObj.cost_window != 0) {

                            if (eva.actObj.smoothing != 0)
                                K_localSearchWindowSmooth(cmr, mr, md, eva, na, false, pindex);
                            else
                                K_localSearchWindow(cmr, mr, md, eva, na, false, pindex);
                        }

                        else {

                            if (eva.actObj.smoothing != 0)
                                K_localSearchSmooth(cmr, mr, md, eva, na, false, pindex);
                            else
                                K_localSearchLength(cmr, mr, md, eva, na, false, pindex);
                        }
                    }
                } // if ((gene+1) % CELL_SMALL_MOVE_RATE == 0)
            }

            pindex++;


        } while (pindex < cellSize);


#endif // #ifdef SYNCHRONIZED_EXECUTION

#ifdef DYNAMIC_CELLULAR_MATRIX


        if ((gene+1) % CELL_PROPAGATION_RATE == 0) {

            PropagateNeighbor<CellD, NIter> pa;

            if (eva.actObj.cost_window != 0) {

                if (eva.actObj.smoothing != 0)
                    K_localSearchWindowSmooth(cmr, mr, md, eva, gsdta, pa, false, CMid*cellSize);
                else
                    K_localSearchWindow(cmr, mr, md, eva, gsdta, pa, false, CMid*cellSize);
            }

            else {

                if (eva.actObj.smoothing != 0)
                    K_localSearchSmooth(cmr, mr, md, eva, gsdta, pa, false, CMid*cellSize);
                else
                    K_localSearchLength(cmr, mr, md, eva, gsdta, pa, false, CMid*cellSize);
            }
        }

        if ((gene+1) % CELL_SMALL_MOVE_RATE == 0) {

            if (localSearchParams.modeCalcul == LS_STOCHASTIC) {

                gnra.setGene(gene);

                if (eva.actObj.cost_window != 0) {

                    if (eva.actObj.smoothing != 0)
                        K_localSearchWindowSmooth(cmr, mr, md, eva, gsdta, gnra, false, CMid*cellSize);

                    else
                        K_localSearchWindow(cmr, mr, md, eva, gsdta, gnra, false, CMid*cellSize);
                }

                else {

                    if (eva.actObj.smoothing != 0)
                        K_localSearchSmooth(cmr, mr, md, eva, gsdta, gnra, false, CMid*cellSize);
                    else
                        K_localSearchLength(cmr, mr, md, eva, gsdta, gnra, false, CMid*cellSize);
                }

            } else if (localSearchParams.modeCalcul == LS_FIRST) {

#if STEREO_OPERATOR
                GenerateNeighborStereo<CellD> na;
#else
                GenerateNeighbor<CellD, NIterNeighborhood, 2, 1> na;
             //   GenerateNeighbor<CellD> na;


#endif

                if (eva.actObj.cost_window != 0) {

                    if (eva.actObj.smoothing != 0)
                        K_localSearchWindowSmooth(cmr, mr, md, eva, gsdta, na, true, CMid*cellSize);


                    else
                        K_localSearchWindow(cmr, mr, md, eva, gsdta, na, true, CMid*cellSize);

                }

                else {

                    if (eva.actObj.smoothing != 0)
                        K_localSearchSmooth(cmr, mr, md, eva, gsdta, na, true, CMid*cellSize);

                    else
                        K_localSearchLength(cmr, mr, md, eva, gsdta, na, true, CMid*cellSize);
                }

            } else if (localSearchParams.modeCalcul == LS_BEST) {

#if STEREO_OPERATOR
                GenerateNeighborStereo<CellD> na;
#else
                GenerateNeighbor<CellD, NIterNeighborhood, 2, 1> na;
             //   GenerateNeighbor<CellD> na;

#endif

                if (eva.actObj.cost_window != 0) {

                    if (eva.actObj.smoothing != 0)
                        K_localSearchWindowSmooth(cmr, mr, md, eva, gsdta, na, false, CMid*cellSize);
                    else
                        K_localSearchWindow(cmr, mr, md, eva, gsdta, na, false, CMid*cellSize);
                }

                else {

                    if (eva.actObj.smoothing != 0)
                        K_localSearchSmooth(cmr, mr, md, eva, gsdta, na, false, CMid*cellSize);
                    else
                        K_localSearchLength(cmr, mr, md, eva, gsdta, na, false, CMid*cellSize);
                }
            }

        } // if ((gene+1) % CELL_SMALL_MOVE_RATE == 0)

        // ============== large moves ==============

        if ((gene+1) % DY_CENTER_WINDOW_MOVE_RATE == 0) {

#if STEREO_OPERATOR
            GenerateNeighborCenterWindowStereo<CellD, NIter> gnwa;
#else
            GenerateNeighborCenterWindow<CellD, NIterNeighborhood, NIter> gnwa;
#endif

            if (localSearchParams.modeCalcul == LS_FIRST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnwa, true, CMid);

            } else if (localSearchParams.modeCalcul == LS_BEST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnwa, false, CMid);
            }
        }

        if ((gene+1) % DY_CENTER_WINDOW_RANDOM_MOVE_RATE == 0) {

            gnwarm.setGene(gene);

            if (localSearchParams.modeCalcul == LS_FIRST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnwarm, true, CMid);

            } else if (localSearchParams.modeCalcul == LS_BEST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnwarm, false, CMid);
            }
        }

        if ((gene+1) % DY_RANDOM_WINDOW_MOVE_RATE == 0) {

            gnrwa.setGene(gene);

            if (localSearchParams.modeCalcul == LS_FIRST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnrwa, true, CMid);

            } else if (localSearchParams.modeCalcul == LS_BEST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnrwa, false, CMid);
            }
        }

        if ((gene+1) % DY_RANDOM_PICK_MOVE_RATE == 0) {

            gnrpa.setGene(gene);

            if (localSearchParams.modeCalcul == LS_FIRST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnrpa, true, CMid);

            } else if (localSearchParams.modeCalcul == LS_BEST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnrpa, false, CMid);
            }
        }

        if ((gene+1) % DY_RANDOM_PICK_RANDOM_MOVE_RATE == 0) {

            gnrparm.setGene(gene);

            if (localSearchParams.modeCalcul == LS_FIRST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnrparm, true, CMid);

            } else if (localSearchParams.modeCalcul == LS_BEST) {

                K_localSearchLargeMove(cmr, mr, md, evaCM, gnrparm, false, CMid);
            }
        }

        if ((gene+1) % DY_RANDOM_PICK_EXPANSION_AND_SWAP_MOVE_RATE == 0) {

            gnrpaes.setGene(gene);
       //     K_detectColorEdges<sobelEdge>(mr);
            K_localSearchLargeMove(cmr, mr, md, evaCM, gnrpaes, false, CMid);
        }

#endif // #ifdef DYNAMIC_CELLULAR_MATRIX

        return ret;
    }//activate




#ifdef SYNCHRONIZED_EXECUTION

    //! HW 060815 : add the following four implementations of variable neighborhood descent (VND)
    //!             using four different objectives :
    //!             (1) cost+length; (2) cost+smooth; (3) costWindow; (4) costWindow+smooth
    /*!
     * \brief K_VND_Length
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_VND_Length(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_VND_Length _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_VND_Smooth
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_VND_Smooth(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_VND_Smooth _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_VND_Window
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_VND_Window(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_VND_Window _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_VND_WindowSmooth
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_VND_WindowSmooth(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_VND_WindowSmooth _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_localSearchLength
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchLength(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchLength _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_localSearchSmooth
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchSmooth(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchSmooth _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_localSearchWindow
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchWindow(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchWindow _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_localSearchWindowSmooth
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchWindowSmooth(CellularMatrix& cmr,
                                                 NN& nnr,
                                                 NN& nnd,
                                                 EvaluationType& eva,
                                                 NeighborAdaptor& na,
                                                 bool is_FI,
                                                 size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchWindowSmooth _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, pindex);
    }

    /*!
     * \brief K_WTA_Data
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_WTA_Data(CellularMatrix& cmr,
                                  NN& nnr,
                                  NN& nnd,
                                  EvaluationType& eva,
                                  NeighborAdaptor& na,
                                  size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_WTA_Data _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, pindex);
    }

    /*!
     * \brief K_perturbation
     *
     */
    template <class CellularMatrix,
              class NeighborAdaptor
              >
    GLOBAL inline void K_perturbation(CellularMatrix& cmr,
                                      NN& nnr,
                                      NN& nnd,
                                      NeighborAdaptor& na,
                                      size_t pindex) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_perturbation _KER_CALL_(b, t) (cmr, nnr, nnd, na, pindex);
    }

#endif // #ifdef SYNCHRONIZED_EXECUTION

#ifdef DYNAMIC_CELLULAR_MATRIX

    /*!
     * \brief K_WTA_Data
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class GetAdaptor,
              class NeighborAdaptor
              >
    GLOBAL inline void K_WTA_Data(CellularMatrix& cmr,
                                  NN& nnr,
                                  NN& nnd,
                                  EvaluationType& eva,
                                  GetAdaptor& ga,
                                  NeighborAdaptor& na) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_WTA_Data _KER_CALL_(b, t) (cmr, nnr, nnd, eva, ga, na);
    }

    /*!
     * \brief K_localSearchLength
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class GetAdaptor,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchLength(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           GetAdaptor& ga,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t CMid) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchLength _KER_CALL_(b, t) (cmr, nnr, nnd, eva, ga, na, is_FI, CMid);
    }

    /*!
     * \brief K_localSearchSmooth
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class GetAdaptor,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchSmooth(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           GetAdaptor& ga,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t CMid) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchSmooth _KER_CALL_(b, t) (cmr, nnr, nnd, eva, ga, na, is_FI, CMid);
    }

    /*!
     * \brief K_localSearchWindow
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class GetAdaptor,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchWindow(CellularMatrix& cmr,
                                           NN& nnr,
                                           NN& nnd,
                                           EvaluationType& eva,
                                           GetAdaptor& ga,
                                           NeighborAdaptor& na,
                                           bool is_FI,
                                           size_t CMid) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchWindow _KER_CALL_(b, t) (cmr, nnr, nnd, eva, ga, na, is_FI, CMid);
    }

    /*!
     * \brief K_localSearchWindow
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class GetAdaptor,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchWindowSmooth(CellularMatrix& cmr,
                                                 NN& nnr,
                                                 NN& nnd,
                                                 EvaluationType& eva,
                                                 GetAdaptor& ga,
                                                 NeighborAdaptor& na,
                                                 bool is_FI,
                                                 size_t CMid) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchWindowSmooth _KER_CALL_(b, t) (cmr, nnr, nnd, eva, ga, na, is_FI, CMid);
    }

    /*!
     * \brief K_localSearchLargeMove
     *
     */
    template <class CellularMatrix,
              class EvaluationType,
              class NeighborAdaptor
              >
    GLOBAL inline void K_localSearchLargeMove(CellularMatrix& cmr,
                                              NN& nnr,
                                              NN& nnd,
                                              EvaluationType& eva,
                                              NeighborAdaptor& na,
                                              bool is_FI,
                                              size_t CMid) {


        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_DLS_localSearchLargeMove _KER_CALL_(b, t) (cmr, nnr, nnd, eva, na, is_FI, CMid);
    }

#endif // #ifdef DYNAMIC_CELLULAR_MATRIX

    /*!
     * \brief K_debug
     *
     */
    template <class Operator>
    GLOBAL inline void K_debug(Operator& nn_cible) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_cible.gGCenter.getWidth(),
                              nn_cible.gGCenter.getHeight());
        K_SO_debug _KER_CALL_(b, t) (nn_cible);
    }
    template <class Operator>
    GLOBAL inline void K_debugNN(Operator& nn_cible) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_cible.adaptiveMap.getWidth(),
                              nn_cible.adaptiveMap.getHeight());
        K_SO_debugNN _KER_CALL_(b, t) (nn_cible);
    }
    template <class Operator>
    GLOBAL inline void K_debugNoBuffer(Operator& nn_cible) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_cible.gGCenter.getWidth(),
                              nn_cible.gGCenter.getHeight());
        K_SO_debugNoBuffer _KER_CALL_(b, t) (nn_cible);
    }
    template <class Grid>
    GLOBAL inline void K_debugGrid(Grid& g) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              g.getWidth(),
                              g.getHeight());
        K_SO_debugGrid _KER_CALL_(b, t) (g);
    }


};//Class LocalSearchOperator

#ifdef SYNCHRONIZED_EXECUTION

/*!
 * \brief K_DLS_WTA_Data
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_WTA_Data(CellularMatrix cmr,
                           NN nnr,
                           NN nnd,
                           Evaluation evaluate,
                           NeighborAdaptor neighborAdaptor,
                           size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            neighborAdaptor.init(ps, pindex);

            do {
                PointEuclid pNew;
                bool getNeighbor = false;
                getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                  nnr,
                                                  nnd,
                                                  ps,
                                                  pNew);
                if (getNeighbor) {

                    GLfloat currentCost, newCost;
                    currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                    newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                    if (newCost < currentCost)
                    {
                        nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                    }
                }

            } while (neighborAdaptor.next());
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_WTA_Data

//! HW 060815 : add the following four implementations of variable neighborhood descent (VND)
//!             using four different objectives :
//!             (1) cost+length; (2) cost+smooth; (3) costWindow; (4) costWindow+smooth
/*!
 * \brief K_DLS_VND_Length
 *
 */


template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_VND_Length(CellularMatrix cmr,
                             NN nnr,
                             NN nnd,
                             Evaluation evaluate,
                             NeighborAdaptor neighborAdaptor,
                             bool is_FI,
                             size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFoundVND = false;
            neighborAdaptor.init(cmr[_y][_x], ps, pindex);

            do {
                bool improvementFoundLS = false;

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                      nnr,
                                                      nnd,
                                                      ps,
                                                      pNew);

                    if (getNeighbor) {

                        GLfloat currentCost, currentLen, newCost, newLen;
                        GLfloat newLenCenter, newLenUp, newLenLeft, newLenUpLeft;

                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                        currentLen = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_length];
                        // left
                        if (ps[0] - 1 >= 0)
                            currentLen += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_length];
                        // up
                        if (ps[1] - 1 >= 0)
                            currentLen += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_length];
                        if (cmr[_y][_x].iter.getSizeN() == 6) // for hexa only
                        {
                            // upper left
                            if (ps[0] - 1 >= 0 && ps[1] - 1 >= 0)
                                currentLen += nnr.objectivesMap[ps[1]-1][ps[0]-1][(int)obj_length];
                        }

                        newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                        newLen = evaluate.evaluateNodeLength(nnr, ps, pNew, newLenCenter, newLenUp, newLenLeft, newLenUpLeft);

                        if ((evaluate.actObj.cost * newCost + evaluate.actObj.length * newLen)
                                < (evaluate.actObj.cost * currentCost + evaluate.actObj.length * currentLen))
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_length] = newLenCenter;
                            // left
                            if (ps[0] - 1 >= 0)
                                nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_length] = newLenLeft;
                            // up
                            if (ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_length] = newLenUp;
                            if (cmr[_y][_x].iter.getSizeN() == 6) // for hexa only
                            {
                                // upper left
                                if (ps[0] - 1 >= 0 && ps[1] - 1 >= 0)
                                    nnr.objectivesMap[ps[1]-1][ps[0]-1][(int)obj_length] = newLenUpLeft;
                            }

                            improvementFoundLS = true;
                            improvementFoundVND = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFoundLS : true));

            } while (neighborAdaptor.nextNeighborhood() && !improvementFoundVND);
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_VND_Length

/*!
 * \brief K_DLS_VND_Smooth
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_VND_Smooth(CellularMatrix cmr,
                             NN nnr,
                             NN nnd,
                             Evaluation evaluate,
                             NeighborAdaptor neighborAdaptor,
                             bool is_FI,
                             size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFoundVND = false;
            neighborAdaptor.init(cmr[_y][_x], ps, pindex);

            do {
                bool improvementFoundLS = false;

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                      nnr,
                                                      nnd,
                                                      ps,
                                                      pNew);

                    if (getNeighbor) {

                        GLfloat currentCost, currentSmo, newCost, newSmo;
                        GLfloat newSmoCenter, newSmoUp, newSmoLeft;

                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                        currentSmo = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing];
                        // left
                        if (ps[0] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing];
                        // up
                        if (ps[1] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing];

                        newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                        newSmo = evaluate.evaluateNodeSmooth(nnr, nnd, ps, pNew, newSmoCenter, newSmoUp, newSmoLeft);

                        if ((evaluate.actObj.cost * newCost + evaluate.actObj.smoothing * newSmo)
                                < (evaluate.actObj.cost * currentCost + evaluate.actObj.smoothing * currentSmo))
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing] = newSmoCenter;
                            // left
                            if (ps[0] - 1 >= 0)
                                nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing] = newSmoLeft;
                            // up
                            if (ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing] = newSmoUp;

                            improvementFoundLS = true;
                            improvementFoundVND = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFoundLS : true));

            } while (neighborAdaptor.nextNeighborhood() && !improvementFoundVND);
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_VND_Smooth

/*!
 * \brief K_DLS_VND_Window
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_VND_Window(CellularMatrix cmr,
                             NN nnr,
                             NN nnd,
                             Evaluation evaluate,
                             NeighborAdaptor neighborAdaptor,
                             bool is_FI,
                             size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFoundVND = false;
            neighborAdaptor.init(cmr[_y][_x], ps, pindex);

            do {
                bool improvementFoundLS = false;

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                      nnr,
                                                      nnd,
                                                      ps,
                                                      pNew);

                    if (getNeighbor) {
                        GLfloat currentCost, newCost;
                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window];
                        PointEuclid pMotion = pNew - nnd.adaptiveMap[ps[1]][ps[0]];
                        newCost = evaluate.evaluateNodeCostWindow(nnr, nnd, ps, pMotion);
                        if (newCost < currentCost)
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window] = newCost;

                            improvementFoundLS = true;
                            improvementFoundVND = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFoundLS : true));

            } while (neighborAdaptor.nextNeighborhood() && !improvementFoundVND);
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_VND_Window
#endif
#ifdef SYNCHRONIZED_EXECUTION

/*!
 * \brief K_DLS_VND_WindowSmooth
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_VND_WindowSmooth(CellularMatrix cmr,
                                   NN nnr,
                                   NN nnd,
                                   Evaluation evaluate,
                                   NeighborAdaptor neighborAdaptor,
                                   bool is_FI,
                                   size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFoundVND = false;
            neighborAdaptor.init(cmr[_y][_x], ps, pindex);

            do {
                bool improvementFoundLS = false;

                do {
                    PointEuclid pNew;
                    bool getNeighbor = false;
                    getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                      nnr,
                                                      nnd,
                                                      ps,
                                                      pNew);

                    if (getNeighbor) {

                        GLfloat currentCost, currentSmo, newCost, newSmo;
                        GLfloat newSmoCenter, newSmoUp, newSmoLeft;

                        currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window];
                        currentSmo = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing];
                        // left
                        if (ps[0] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing];
                        // up
                        if (ps[1] - 1 >= 0)
                            currentSmo += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing];


                        PointEuclid pMotion = pNew - nnd.adaptiveMap[ps[1]][ps[0]];
                        newCost = evaluate.evaluateNodeCostWindow(nnr, nnd, ps, pMotion);
                        newSmo = evaluate.evaluateNodeSmooth(nnr, nnd, ps, pNew, newSmoCenter, newSmoUp, newSmoLeft);

                        if ((evaluate.actObj.cost_window * newCost + evaluate.actObj.smoothing * newSmo)
                                < (evaluate.actObj.cost_window * currentCost + evaluate.actObj.smoothing * currentSmo))
                        {
                            nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window] = newCost;
                            nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing] = newSmoCenter;
                            // left
                            if (ps[0] - 1 >= 0)
                                nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing] = newSmoLeft;
                            // up
                            if (ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing] = newSmoUp;

                            improvementFoundLS = true;
                            improvementFoundVND = true;
                        }
                    }

                } while (neighborAdaptor.next() && (is_FI ? !improvementFoundLS : true));

            } while (neighborAdaptor.nextNeighborhood() && !improvementFoundVND);
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_VND_WindowSmooth


/*!
 * \brief K_DLS_localSearchLength
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchLength(CellularMatrix cmr,
                                    NN nnr,
                                    NN nnd,
                                    Evaluation evaluate,
                                    NeighborAdaptor neighborAdaptor,
                                    bool is_FI,
                                    size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFound = false;
            neighborAdaptor.init(ps, pindex);

            do {
                PointEuclid pNew;
                bool getNeighbor = false;
                getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                  nnr,
                                                  nnd,
                                                  ps,
                                                  pNew);

                if (getNeighbor) {

                    GLfloat currentCost, currentLen, newCost, newLen;
                    GLfloat newLenCenter, newLenUp, newLenLeft, newLenUpLeft;

                    currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                    currentLen = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_length];
                    // left
                    if (ps[0] - 1 >= 0)
                        currentLen += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_length];
                    // up
                    if (ps[1] - 1 >= 0)
                        currentLen += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_length];
                    if (cmr[_y][_x].iter.getSizeN() == 6) // for hexa only
                    {
                        // upper left
                        if (ps[0] - 1 >= 0 && ps[1] - 1 >= 0)
                            currentLen += nnr.objectivesMap[ps[1]-1][ps[0]-1][(int)obj_length];
                    }

                    newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                    newLen = evaluate.evaluateNodeLength(nnr, ps, pNew, newLenCenter, newLenUp, newLenLeft, newLenUpLeft);

                    if ((evaluate.actObj.cost * newCost + evaluate.actObj.length * newLen)
                            < (evaluate.actObj.cost * currentCost + evaluate.actObj.length * currentLen))
                    {
                        nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_length] = newLenCenter;
                        // left
                        if (ps[0] - 1 >= 0)
                            nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_length] = newLenLeft;
                        // up
                        if (ps[1] - 1 >= 0)
                            nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_length] = newLenUp;
                        if (cmr[_y][_x].iter.getSizeN() == 6) // for hexa only
                        {
                            // upper left
                            if (ps[0] - 1 >= 0 && ps[1] - 1 >= 0)
                                nnr.objectivesMap[ps[1]-1][ps[0]-1][(int)obj_length] = newLenUpLeft;
                        }
                        improvementFound = true;
                    }
                }

            } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchLength

/*!
 * \brief K_DLS_localSearchSmooth
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchSmooth(CellularMatrix cmr,
                                    NN nnr,
                                    NN nnd,
                                    Evaluation evaluate,
                                    NeighborAdaptor neighborAdaptor,
                                    bool is_FI,
                                    size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFound = false;
            neighborAdaptor.init(ps, pindex);

            do {
                PointEuclid pNew;
                bool getNeighbor = false;
                getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                  nnr,
                                                  nnd,
                                                  ps,
                                                  pNew);

                if (getNeighbor) {

                    GLfloat currentCost, currentSmo, newCost, newSmo;
                    GLfloat newSmoCenter, newSmoUp, newSmoLeft;

                    currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost];
                    currentSmo = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing];
                    // left
                    if (ps[0] - 1 >= 0)
                        currentSmo += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing];
                    // up
                    if (ps[1] - 1 >= 0)
                        currentSmo += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing];

                    newCost = evaluate.evaluateNodeCost(nnr, nnd, ps, pNew);
                    newSmo = evaluate.evaluateNodeSmooth(nnr, nnd, ps, pNew, newSmoCenter, newSmoUp, newSmoLeft);

                    if ((evaluate.actObj.cost * newCost + evaluate.actObj.smoothing * newSmo)
                            < (evaluate.actObj.cost * currentCost + evaluate.actObj.smoothing * currentSmo))
                    {
                        nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost] = newCost;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing] = newSmoCenter;
                        // left
                        if (ps[0] - 1 >= 0)
                            nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing] = newSmoLeft;
                        // up
                        if (ps[1] - 1 >= 0)
                            nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing] = newSmoUp;

                        improvementFound = true;
                    }
                }

            } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchSmooth

/*!
 * \brief K_DLS_localSearchWindow
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchWindow(CellularMatrix cmr,
                                    NN nnr,
                                    NN nnd,
                                    Evaluation evaluate,
                                    NeighborAdaptor neighborAdaptor,
                                    bool is_FI,
                                    size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFound = false;
            neighborAdaptor.init(ps, pindex);

            do {
                PointEuclid pNew;
                bool getNeighbor = false;
                getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                  nnr,
                                                  nnd,
                                                  ps,
                                                  pNew);
                if (getNeighbor) {
                    GLfloat currentCost, newCost;
                    currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window];
                    PointEuclid pMotion = pNew - nnd.adaptiveMap[ps[1]][ps[0]];
                    newCost = evaluate.evaluateNodeCostWindow(nnr, nnd, ps, pMotion);
                    if (newCost < currentCost)
                    {
                        nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window] = newCost;
                        improvementFound = true;
                    }
                }

            } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchWindow

/*!
 * \brief K_DLS_localSearchWindowSmooth
 *
 */
template <class CellularMatrix,
          class Evaluation,
          class NeighborAdaptor
          >
KERNEL void K_DLS_localSearchWindowSmooth(CellularMatrix cmr,
                                          NN nnr,
                                          NN nnd,
                                          Evaluation evaluate,
                                          NeighborAdaptor neighborAdaptor,
                                          bool is_FI,
                                          size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight()) {

            bool improvementFound = false;
            neighborAdaptor.init(ps, pindex);

            do {
                PointEuclid pNew;
                bool getNeighbor = false;
                getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                  nnr,
                                                  nnd,
                                                  ps,
                                                  pNew);
                if (getNeighbor) {

                    GLfloat currentCost, currentSmo, newCost, newSmo;
                    GLfloat newSmoCenter, newSmoUp, newSmoLeft;

                    currentCost = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window];
                    currentSmo = nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing];
                    // left
                    if (ps[0] - 1 >= 0)
                        currentSmo += nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing];
                    // up
                    if (ps[1] - 1 >= 0)
                        currentSmo += nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing];


                    PointEuclid pMotion = pNew - nnd.adaptiveMap[ps[1]][ps[0]];
                    newCost = evaluate.evaluateNodeCostWindow(nnr, nnd, ps, pMotion);
                    newSmo = evaluate.evaluateNodeSmooth(nnr, nnd, ps, pNew, newSmoCenter, newSmoUp, newSmoLeft);

                    if ((evaluate.actObj.cost_window * newCost + evaluate.actObj.smoothing * newSmo)
                            < (evaluate.actObj.cost_window * currentCost + evaluate.actObj.smoothing * currentSmo))
                    {
                        nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_cost_window] = newCost;
                        nnr.objectivesMap[ps[1]][ps[0]][(int)obj_smoothing] = newSmoCenter;
                        // left
                        if (ps[0] - 1 >= 0)
                            nnr.objectivesMap[ps[1]][ps[0]-1][(int)obj_smoothing] = newSmoLeft;
                        // up
                        if (ps[1] - 1 >= 0)
                            nnr.objectivesMap[ps[1]-1][ps[0]][(int)obj_smoothing] = newSmoUp;

                        improvementFound = true;
                    }
                }

            } while (neighborAdaptor.next() && (is_FI ? !improvementFound : true));
        }
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_DLS_localSearchWindowSmooth

/*!
 * \brief K_DLS_perturbation
 *
 */
template <class CellularMatrix,
          class NeighborAdaptor
          >
KERNEL void K_DLS_perturbation(CellularMatrix cmr,
                               NN nnr,
                               NN nnd,
                               NeighborAdaptor neighborAdaptor,
                               size_t pindex)
{
    KER_SCHED(cmr.getWidth(), cmr.getHeight())

    if (_x < cmr.getWidth() && _y < cmr.getHeight()) {

        PointCoord ps = cmr[_y][_x].iter.getNode(pindex);

        if (ps[0] >= 0 && ps[0] < nnr.adaptiveMap.getWidth()
                && ps[1] >= 0 && ps[1] < nnr.adaptiveMap.getHeight())
        {
            neighborAdaptor.init(ps, pindex);

            do {
                PointEuclid pNew;
                bool getNeighbor = false;
                getNeighbor = neighborAdaptor.get(cmr[_y][_x],
                                                  nnr,
                                                  nnd,
                                                  ps,
                                                  pNew);
                if (getNeighbor) {

                    nnr.adaptiveMap[ps[1]][ps[0]] = pNew;
                }

            } while (neighborAdaptor.next());
        }
    }

    END_KER_SCHED

    SYNCTHREADS

} //K_DLS_perturbation

#endif //#ifdef SYNCHRONIZED_EXECUTION

}//namespace operators

#endif // LOCALSEARCH_H

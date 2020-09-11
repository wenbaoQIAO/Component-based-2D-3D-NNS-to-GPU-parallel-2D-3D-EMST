#ifndef EVALUATION_H
#define EVALUATION_H
/*
 ***************************************************************************
 *
 * Author : H. Wang, J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"
#include "CellularMatrix.h"
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

#define ADAPTIVE_WEIGHT 0
#define MAX_COLOR_PARA  0.01
const int DEFAULT_WINDOW_RADIUS = 2;
const int blockDimX = 8;
const int blockDimY = 8;

// Smoothness term weight configuration
#define SMOOTH_WEIGHT_1     0
#define SMOOTH_WEIGHT_2     0
#define DIFF_THRESHOLD      0.01
#define SMOOTH_WEIGHT       5.0
#define COEF1               0.0
#define COEF2               1.0
#define AJUST 10.0

//#define CPU_DEBUG
//#define GPU_REDUCTION

#ifndef CUDA_CODE
#ifdef CPU_DEBUG
// For debug
extern Grid<int> gDebug;
#endif
#endif

#define SUBPIXEL_PRECISION 0
#define STEREO_COMP 1

using namespace std;
using namespace components;

namespace operators
{

#ifdef CUDA_CODE
//! KERNEL FUNCTION
//! Kernel call by object
//! Standard Kernel call with objects
template<class NodeObject>
__global__ void K_ObjecValReductionStep1(Grid<NodeObject> gObj)
{
    int _x = blockIdx.x * blockDim.x + threadIdx.x;
    int _y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ NodeObject cache[blockDimY][blockDimX];

    if (_x < gObj.getWidth() && _y < gObj.getHeight())
    {
        cache[threadIdx.y][threadIdx.x] = gObj[_y][_x];
    }
    else
    {
        NodeObject zero;
        cache[threadIdx.y][threadIdx.x] = zero;
    }

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
            cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }
    i = blockDim.y / 2;
    while (i != 0)
    {
        if (threadIdx.y < i)
            cache[threadIdx.y][0] += cache[threadIdx.y + i][0];
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        gObj[blockIdx.y][blockIdx.x] = cache[0][0];
    }

    END_KER_SCHED
}

template<class NodeObject>
__global__ void K_ObjecValReductionStep2(Grid<NodeObject> gObj)
{
    int _x = blockIdx.x * blockDim.x + threadIdx.x;
    int _y = blockIdx.y * blockDim.y + threadIdx.y;

    if (_x == 0 && _y == 0)
    {
        NodeObject energySum;
        energySum.copy_weights(gObj[0][0]);

        for (int j = 0; j < gridDim.y; j++)
            for (int i = 0; i < gridDim.x; i++)
                energySum += gObj[j][i];
        gObj[0][0] = energySum;
    }

    END_KER_SCHED
}
#endif

template <class NN>
KERNEL void K_E_init(NN nnr)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight())
    {
        for (size_t i = 0; i < 9; ++i) {
            nnr.objectivesMap[_y][_x].set(i, 0);
            nnr.objectivesMap[_y][_x].set_weights(i, 0);
        }
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_E_initializeWeights

template <class NN>
KERNEL void K_E_initializeWeights(NN nnr, AMObjNames obType, GLfloat w)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight())
    {
        if ((int)obType == 9)
        {
            for (size_t i = 0; i < 9; ++i)
                nnr.objectivesMap[_y][_x].set_weights(i, w);
        }
        else if ((int)obType >= 0 && (int)obType < 9)
            nnr.objectivesMap[_y][_x].set_weights((int)obType, w);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_E_initializeWeights

template <class NN>
KERNEL void K_E_initializeWeights(NN nnr, ActiveObj actObj)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight())
    {
        if (actObj.distr != 0)
            nnr.objectivesMap[_y][_x].set_weights(0, actObj.distr);
        if (actObj.length != 0)
            nnr.objectivesMap[_y][_x].set_weights(1, actObj.length);
        if (actObj.sqr_length != 0)
            nnr.objectivesMap[_y][_x].set_weights(2, actObj.sqr_length);
        if (actObj.cost != 0)
            nnr.objectivesMap[_y][_x].set_weights(3, actObj.cost);
        if (actObj.sqr_cost != 0)
            nnr.objectivesMap[_y][_x].set_weights(4, actObj.sqr_cost);
        if (actObj.cost_window != 0)
            nnr.objectivesMap[_y][_x].set_weights(5, actObj.cost_window);
        if (actObj.sqr_cost_window != 0)
            nnr.objectivesMap[_y][_x].set_weights(6, actObj.sqr_cost_window);
        if (actObj.smoothing != 0)
            nnr.objectivesMap[_y][_x].set_weights(7, actObj.smoothing);
        if (actObj.gd_error != 0)
            nnr.objectivesMap[_y][_x].set_weights(8, actObj.gd_error);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_E_initializeWeights

template <class NN>
KERNEL void K_E_copyWeights(NN nnr, NN nnrCopy)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight())
    {
        nnrCopy.objectivesMap[_y][_x].copy_weights(nnr.objectivesMap[_y][_x]);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_E_copyWeights

template <class NN>
KERNEL void K_E_initializePointWeight(NN nnr, PointCoord p, AMObjNames obType, GLfloat w)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight() && _x == p[0] && _y == p[1])
    {
        if ((int)obType == 9)
        {
            for (size_t i = 0; i < 9; ++i)
                nnr.objectivesMap[_y][_x].set_weights(i, w);
        }
        else if ((int)obType >= 0 && (int)obType < 9)
            nnr.objectivesMap[_y][_x].set_weights((int)obType, w);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_E_initializePointWeight

template <class Evaluation>
KERNEL void K_EG_evaluate(Evaluation eva, NN nnr, NN nnd, AMObjNames obType, size_t wr)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight())
    {
        PointCoord p(_x, _y);
        eva.evaluateNode(nnr, nnd, p, obType, wr);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_EG_evaluate

template <class Evaluation>
KERNEL void K_EG_evaluate(Evaluation eva, NN nnr, NN nnd, size_t wr)
{
    KER_SCHED(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight())

    if (_x < nnr.objectivesMap.getWidth() && _y < nnr.objectivesMap.getHeight())
    {
        PointCoord p(_x, _y);
        eva.evaluateNode(nnr, nnd, p, wr);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_EG_evaluate

template <class AMObjNames>
KERNEL void K_ECM_initializeWeights(Grid<AMObjectives> objMapCM, AMObjNames obType, GLfloat w)
{
    KER_SCHED(objMapCM.getWidth(), objMapCM.getHeight())

    if (_x < objMapCM.getWidth() && _y < objMapCM.getHeight())
    {
        if ((int)obType == 9)
        {
            for (size_t i = 0; i < 9; ++i)
                objMapCM[_y][_x].set_weights(i, w);
        }
        else if ((int)obType >= 0 && (int)obType < 9)
            objMapCM[_y][_x].set_weights((int)obType, w);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_ECM_initializeWeights

template <class ActiveObj>
KERNEL void K_ECM_initializeWeights(Grid<AMObjectives> objMapCM, ActiveObj actObj)
{
    KER_SCHED(objMapCM.getWidth(), objMapCM.getHeight())

    if (_x < objMapCM.getWidth() && _y < objMapCM.getHeight())
    {
        if (actObj.distr != 0)
            objMapCM[_y][_x].set_weights(0, actObj.distr);
        if (actObj.length != 0)
            objMapCM[_y][_x].set_weights(1, actObj.length);
        if (actObj.sqr_length != 0)
            objMapCM[_y][_x].set_weights(2, actObj.sqr_length);
        if (actObj.cost != 0)
            objMapCM[_y][_x].set_weights(3, actObj.cost);
        if (actObj.sqr_cost != 0)
            objMapCM[_y][_x].set_weights(4, actObj.sqr_cost);
        if (actObj.cost_window != 0)
            objMapCM[_y][_x].set_weights(5, actObj.cost_window);
        if (actObj.sqr_cost_window != 0)
            objMapCM[_y][_x].set_weights(6, actObj.sqr_cost_window);
        if (actObj.smoothing != 0)
            objMapCM[_y][_x].set_weights(7, actObj.smoothing);
        if (actObj.gd_error != 0)
            objMapCM[_y][_x].set_weights(8, actObj.gd_error);
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_ECM_initializeWeights

template <class Evaluation>
KERNEL void K_ECM_evaluate(Evaluation eva, NN nnr, NN nnd, Grid<AMObjectives> obMapCM, AMObjNames obType, size_t wr)
{
    KER_SCHED(eva.cmr.getWidth(), eva.cmr.getHeight())

    if (_x < eva.cmr.getWidth() && _y < eva.cmr.getHeight())
    {
        eva.evaluateCell(eva.cmr[_y][_x], nnr, nnd, obType, wr);
        obMapCM[_y][_x] = eva.getObjectives();
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_ECM_evaluate

template <class Evaluation>
KERNEL void K_ECM_evaluate(Evaluation eva, NN nnr, NN nnd, Grid<AMObjectives> obMapCM, size_t wr)
{
    KER_SCHED(eva.cmr.getWidth(), eva.cmr.getHeight())

    if (_x < eva.cmr.getWidth() && _y < eva.cmr.getHeight())
    {
        eva.evaluateCell(eva.cmr[_y][_x], nnr, nnd, wr);
        obMapCM[_y][_x] = eva.getObjectives();
    }

    END_KER_SCHED

    SYNCTHREADS
} // K_ECM_evaluate


// pstq: piecewise smooth prior with truncated quadratic
// pstl: piecewise smooth prior with truncated linear
// pcp: piecewise constant prior with Potts energy
// esl: everywhere smooth prior wirh linear energy
enum SmoothnessType { smooth_pstq, smooth_pstl, smooth_pcp, smooth_esl };

//! Internal parameters
struct MatchingCostParams {
    GLfloat alphaC;
    GLfloat maxColor;
    GLfloat alphaG;
    GLfloat maxGradient;
    GLfloat maxSmooth;
    SmoothnessType smoothType;

#if STEREO_COMP
    DEVICE_HOST inline MatchingCostParams() :
        alphaC(1.0), //alphaC(0.1),
        maxColor(255.0),
        alphaG(0.0),//alphaG(0.9),
        maxGradient(1000),
        maxSmooth (2.0),
        smoothType (smooth_pstl) {}
#else
    DEVICE_HOST inline MatchingCostParams() :
        alphaC(10.0), //alphaC(0.1),
        maxColor(1000),
        alphaG(1.0),//alphaG(0.9),
        maxGradient(1000),
        maxSmooth (40),
        smoothType (smooth_pstl) {}
#endif

    DEVICE_HOST inline MatchingCostParams(GLfloat _alphaC,
                                          GLfloat _maxColor,
                                          GLfloat _alphaG,
                                          GLfloat _maxGradient,
                                          GLfloat _maxSmooth,
                                          SmoothnessType _smoothType) :
        alphaC(_alphaC),
        maxColor(_maxColor),
        alphaG(_alphaG),
        maxGradient(_maxGradient),
        maxSmooth(_maxSmooth),
        smoothType(_smoothType) {}

    DEVICE_HOST inline void initialize(GLfloat _alphaC,
                                       GLfloat _maxColor,
                                       GLfloat _alphaG,
                                       GLfloat _maxGradient,
                                       GLfloat _maxSmooth,
                                       SmoothnessType _smoothType)
    {
        alphaC = _alphaC;
        maxColor = _maxColor;
        alphaG = _alphaG;
        maxGradient = _maxGradient;
        maxSmooth = _maxSmooth;
        smoothType = _smoothType;
    }
};

/*!
 * \brief Class EvaluationNode.
 * It is a mixte CPU/GPU class, then with
 * default copy constructor for allowing
 * parameter passing to GPU Kernel.
 **/

template <class NIter,
          class GridGT = MatMotion>
class EvaluationNode
{
protected:
    MatchingCostParams para;
    GridGT gt;

public:
    ActiveObj actObj;

    DEVICE_HOST inline EvaluationNode() {}

    DEVICE_HOST inline explicit EvaluationNode(
            GridGT _gt
            ) :
        gt(_gt) {
    }

    DEVICE_HOST inline explicit EvaluationNode(
            MatchingCostParams _para,
            GridGT _gt
            ) :
        para(_para),
        gt(_gt) {
    }

    DEVICE_HOST inline explicit EvaluationNode(
            MatchingCostParams _para,
            GridGT _gt,
            ActiveObj _actObj
            ) :
        para(_para),
        gt(_gt),
        actObj(_actObj) {
    }

    DEVICE_HOST void initialize(ActiveObj _actObj) {
        actObj = _actObj;
    }

    DEVICE_HOST void initialize(GridGT _gt) {
        gt = _gt;
    }

    template <class DistanceFunctor>
    DEVICE_HOST GLfloat evaluateNodeLengthBasic(NN& nn, PointCoord p) {
        DistanceFunctor distance;
        GLfloat ob = 0.0f;
        NIter iter(p, 0, 0);
        // right
        PointCoord pRight(p[0] + 1, p[1]);
        if (pRight[0] < nn.adaptiveMap.getWidth())
            ob += distance(p, pRight, nn, nn);
        // down
        PointCoord pDown(p[0], p[1] + 1);
        if (pDown[1] < nn.adaptiveMap.getHeight())
            ob += distance(p, pDown, nn, nn);
        if (iter.getSizeN() == 6) // for hexa only
        {
            // down right
            PointCoord pDownRight(p[0] + 1, p[1] + 1);
            if (pDownRight[0] < nn.adaptiveMap.getWidth() && pDownRight[1] < nn.adaptiveMap.getHeight())
                ob += distance(p, pDownRight, nn, nn);
        }
        return ob;
    }

    template <class DistanceFunctor>
    DEVICE_HOST GLfloat evaluateNodeLengthBasic(NN& nn, PointCoord p, PointEuclid newPos,
                                                GLfloat& newLen,
                                                GLfloat& newLenUp,
                                                GLfloat& newLenLeft,
                                                GLfloat& newLenUpLeft) {
        DistanceFunctor distance;
        NIter iter(p, 0, 0);

        // the considered pixel itself
        newLen = 0.0f;
        // right
        if (p[0] + 1 < nn.adaptiveMap.getWidth())
            newLen += distance(newPos, nn.adaptiveMap[p[1]][p[0]+1]);
        // down
        if (p[1] + 1 < nn.adaptiveMap.getHeight())
            newLen += distance(newPos, nn.adaptiveMap[p[1]+1][p[0]]);
        if (iter.getSizeN() == 6) // for hexa only
        {
            // down right
            if (p[0] + 1 < nn.adaptiveMap.getWidth() && p[1] + 1 < nn.adaptiveMap.getHeight())
                newLen += distance(newPos, nn.adaptiveMap[p[1]+1][p[0]+1]);
        }

        // the upper neighbor of the considered pixel
        newLenUp = 0.0f;
        PointCoord pTem(p[0], p[1] - 1);
        if (pTem[1] >= 0)
        {
            // right
            if (pTem[0] + 1 < nn.adaptiveMap.getWidth())
                newLenUp += distance(nn.adaptiveMap[pTem[1]][pTem[0]], nn.adaptiveMap[pTem[1]][pTem[0]+1]);
            // down
            newLenUp += distance(nn.adaptiveMap[pTem[1]][pTem[0]], newPos);
            if (iter.getSizeN() == 6) // for hexa only
            {
                // down right
                if (pTem[0] + 1 < nn.adaptiveMap.getWidth())
                    newLenUp += distance(nn.adaptiveMap[pTem[1]][pTem[0]], nn.adaptiveMap[pTem[1]+1][pTem[0]+1]);
            }
        }

        // the left neighbor of the considered pixel
        newLenLeft = 0.0f;
        pTem[0] = p[0] - 1;
        pTem[1] = p[1];
        if (pTem[0] >= 0)
        {
            // right
            newLenLeft += distance(nn.adaptiveMap[pTem[1]][pTem[0]], newPos);
            // down
            if (pTem[1] + 1 < nn.adaptiveMap.getHeight())
                newLenLeft += distance(nn.adaptiveMap[pTem[1]][pTem[0]], nn.adaptiveMap[pTem[1]+1][pTem[0]]);
            if (iter.getSizeN() == 6) // for hexa only
            {
                // down right
                if (pTem[1] + 1 < nn.adaptiveMap.getHeight())
                    newLenLeft += distance(nn.adaptiveMap[pTem[1]][pTem[0]], nn.adaptiveMap[pTem[1]+1][pTem[0]+1]);
            }
        }

        // the up left neighbor of the considered pixel
        newLenUpLeft = 0.0f;
        if (iter.getSizeN() == 6) {
            pTem[0] = p[0] - 1;
            pTem[1] = p[1] - 1;
            if (pTem[0] >= 0 && pTem[1] >= 0)
            {
                // right
                newLenUpLeft += distance(nn.adaptiveMap[pTem[1]][pTem[0]], nn.adaptiveMap[pTem[1]][pTem[0]+1]);
                // down
                newLenUpLeft += distance(nn.adaptiveMap[pTem[1]][pTem[0]], nn.adaptiveMap[pTem[1]+1][pTem[0]]);
                // down right
                newLenUpLeft += distance(nn.adaptiveMap[pTem[1]][pTem[0]], newPos);
            }
        }

        return (newLen + newLenUp + newLenLeft + newLenUpLeft);
    }

    DEVICE_HOST GLfloat evaluateNodeLength(NN& nn, PointCoord p) {
        nn.objectivesMap[p[1]][p[0]][(int)obj_length] = evaluateNodeLengthBasic<CM_DistanceEuclidean> (nn, p);
        return nn.objectivesMap[p[1]][p[0]][(int)obj_length];
    }

    DEVICE_HOST GLfloat evaluateNodeSquaredLength(NN& nn, PointCoord p) {
        nn.objectivesMap[p[1]][p[0]][(int)obj_sqr_length] = evaluateNodeLengthBasic<CM_DistanceSquaredEuclidean> (nn, p);
        return nn.objectivesMap[p[1]][p[0]][(int)obj_sqr_length];
    }

    DEVICE_HOST GLfloat evaluateNodeLength(NN& nn, PointCoord p, PointEuclid newPos,
                                           GLfloat& newLen, GLfloat& newLenUp, GLfloat& newLenLeft, GLfloat& newLenUpLeft) {
        return (evaluateNodeLengthBasic<DistanceEuclidean<PointEuclid> > (nn, p, newPos, newLen, newLenUp, newLenLeft, newLenUpLeft));
    }

    DEVICE_HOST GLfloat evaluateNodeSquaredLength(NN& nn, PointCoord p, PointEuclid newPos,
                                                  GLfloat& newLen, GLfloat& newLenUp, GLfloat& newLenLeft, GLfloat& newLenUpLeft) {
        return (evaluateNodeLengthBasic<DistanceSquaredEuclidean<PointEuclid> > (nn, p, newPos, newLen, newLenUp, newLenLeft, newLenUpLeft));
    }

    // length difference
    DEVICE_HOST GLfloat evaluateNodeSmooth2(NN& nnr, NN& nnd, PointCoord p) {
        GLfloat length_nnr = evaluateNodeLength(nnr, p);
        GLfloat length_nnd = evaluateNodeLength(nnd, p);
        nnr.objectivesMap[p[1]][p[0]][(int)obj_length] = length_nnr;
        nnr.objectivesMap[p[1]][p[0]][(int)obj_smoothing] = ABS(length_nnr - length_nnd);
        return nnr.objectivesMap[p[1]][p[0]][(int)obj_smoothing];
    }

    // Traditional 4-neighbor smoothness
    DEVICE_HOST GLfloat evaluateNodeSmooth(NN& nnr, NN& nnd, PointCoord p) {
#if STEREO_COMP
        components::DistanceManhattan<PointEuclid> distance;
#else
        components::DistanceEuclidean<PointEuclid> distance;
#endif
        GLfloat obSum = 0.0f;
        PointEuclid motion = nnr.adaptiveMap[p[1]][p[0]];
        motion -= nnd.adaptiveMap[p[1]][p[0]];
        // right
        PointCoord pRight(p[0] + 1, p[1]);
        if (pRight[0] < nnr.adaptiveMap.getWidth()) {
            GLfloat ob;
            PointEuclid motionRight = nnr.adaptiveMap[pRight[1]][pRight[0]];
            motionRight -= nnd.adaptiveMap[pRight[1]][pRight[0]];
            if (para.smoothType == smooth_pstq) {
                ob = (MIN((distance(motion, motionRight) * distance(motion, motionRight)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob = (MIN((distance(motion, motionRight)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (motion[0] != motionRight[0] || motion[1] != motionRight[1])
                    ob = 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob = distance(motion, motionRight);
            }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[pRight[1]][pRight[0]]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[pRight[1]][pRight[0]]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[p[1]][p[0]]));
#endif
            obSum += ob;
        }
        // down
        PointCoord pDown(p[0], p[1] + 1);
        if (pDown[1] < nnr.adaptiveMap.getHeight()) {
            GLfloat ob;
            PointEuclid motionDown = nnr.adaptiveMap[pDown[1]][pDown[0]];
            motionDown -= nnd.adaptiveMap[pDown[1]][pDown[0]];
            if (para.smoothType == smooth_pstq) {
                ob = (MIN((distance(motion, motionDown) * distance(motion, motionDown)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob = (MIN((distance(motion, motionDown)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (motion[0] != motionDown[0] || motion[1] != motionDown[1])
                    ob = 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob = distance(motion, motionDown);
            }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[pDown[1]][pDown[0]]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[pDown[1]][pDown[0]]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[p[1]][p[0]]));
#endif
            obSum += ob;
        }
        nnr.objectivesMap[p[1]][p[0]][(int)obj_smoothing] = obSum;
        return obSum;
    }

    DEVICE_HOST GLfloat evaluateNodeSmooth(NN& nnr, NN& nnd, PointCoord p,
                                           PointEuclid newPos,
                                           GLfloat& newSmo,
                                           GLfloat& newSmoUp,
                                           GLfloat& newSmoLeft) {
#if STEREO_COMP
        components::DistanceManhattan<PointEuclid> distance;
#else
        components::DistanceEuclidean<PointEuclid> distance;
#endif

        // the considered pixel itself
        newSmo = 0.0f;
        PointEuclid motion = newPos;
        motion -= nnd.adaptiveMap[p[1]][p[0]];
        // right
        if (p[0] + 1 < nnr.adaptiveMap.getWidth()) {
            GLfloat ob;
            PointEuclid motionRight = nnr.adaptiveMap[p[1]][p[0]+1];
            motionRight -= nnd.adaptiveMap[p[1]][p[0]+1];
            if (para.smoothType == smooth_pstq) {
                ob = (MIN((distance(motion, motionRight) * distance(motion, motionRight)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob = (MIN((distance(motion, motionRight)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (motion[0] != motionRight[0] || motion[1] != motionRight[1])
                    ob = 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob = distance(motion, motionRight);
            }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[p[1]][p[0]+1]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[p[1]][p[0]+1]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[p[1]][p[0]]));
#endif
            newSmo += ob;
        }
        // down
        if (p[1] + 1 < nnr.adaptiveMap.getHeight()) {
            GLfloat ob;
            PointEuclid motionDown = nnr.adaptiveMap[p[1]+1][p[0]];
            motionDown -= nnd.adaptiveMap[p[1]+1][p[0]];
            if (para.smoothType == smooth_pstq) {
                ob = (MIN((distance(motion, motionDown) * distance(motion, motionDown)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob = (MIN((distance(motion, motionDown)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (motion[0] != motionDown[0] || motion[1] != motionDown[1])
                    ob = 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob = distance(motion, motionDown);
            }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[p[1]+1][p[0]]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[p[1]+1][p[0]]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[p[1]][p[0]]));
#endif
            newSmo += ob;
        }

        // the upper neighbor of the considered pixel
        newSmoUp = 0.0f;
        PointCoord pTem(p[0], p[1] - 1);
        if (pTem[1] >= 0)
        {
            GLfloat ob;
            PointEuclid motionTem = nnr.adaptiveMap[pTem[1]][pTem[0]];;
            motionTem -= nnd.adaptiveMap[pTem[1]][pTem[0]];
            // right
            if (pTem[0] + 1 < nnr.adaptiveMap.getWidth()) {

                PointEuclid motionRight = nnr.adaptiveMap[pTem[1]][pTem[0]+1];
                motionRight -= nnd.adaptiveMap[pTem[1]][pTem[0]+1];
                if (para.smoothType == smooth_pstq) {
                    ob = (MIN((distance(motionTem, motionRight) * distance(motionTem, motionRight)), (para.maxSmooth * para.maxSmooth)));
                }
                else if (para.smoothType == smooth_pstl) {
                    ob = (MIN((distance(motionTem, motionRight)), (para.maxSmooth)));
                }
                else if (para.smoothType == smooth_pcp) {
                    if (motionTem[0] != motionRight[0] || motionTem[1] != motionRight[1])
                        ob = 1.0f;
                }
                else if (para.smoothType == smooth_esl) {
                    ob = distance(motionTem, motionRight);
                }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[pTem[1]][pTem[0]+1]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[pTem[1]][pTem[0]+1]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[pTem[1]][pTem[0]]));
#endif
            newSmoUp += ob;
            }
            // down
            if (para.smoothType == smooth_pstq) {
                ob = (MIN((distance(motionTem, motion) * distance(motionTem, motion)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob = (MIN((distance(motionTem, motion)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (motionTem[0] != motion[0] || motionTem[1] != motion[1])
                    ob = 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob = distance(motionTem, motion);
            }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[p[1]][p[0]]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[p[1]][p[0]]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[pTem[1]][pTem[0]]));
#endif
            newSmoUp += ob;
        }

        // the left neighbor of the considered pixel
        newSmoLeft = 0.0f;
        pTem[0] = p[0] - 1;
        pTem[1] = p[1];
        if (pTem[0] >= 0)
        {
            GLfloat ob;
            PointEuclid motionTem = nnr.adaptiveMap[pTem[1]][pTem[0]];;
            motionTem -= nnd.adaptiveMap[pTem[1]][pTem[0]];
            // right
            if (para.smoothType == smooth_pstq) {
                ob = (MIN((distance(motionTem, motion) * distance(motionTem, motion)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob = (MIN((distance(motionTem, motion)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (motionTem[0] != motion[0] || motionTem[1] != motion[1])
                    ob = 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob = distance(motionTem, motion);
            }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[p[1]][p[0]]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[p[1]][p[0]]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[pTem[1]][pTem[0]]));
#endif
            newSmoLeft += ob;

            // down
            if (pTem[1] + 1 < nnr.adaptiveMap.getHeight()) {

                PointEuclid motionDown = nnr.adaptiveMap[pTem[1]+1][pTem[0]];
                motionDown -= nnd.adaptiveMap[pTem[1]+1][pTem[0]];
                if (para.smoothType == smooth_pstq) {
                    ob = (MIN((distance(motionTem, motionDown) * distance(motionTem, motionDown)), (para.maxSmooth * para.maxSmooth)));
                }
                else if (para.smoothType == smooth_pstl) {
                    ob = (MIN((distance(motionTem, motionDown)), (para.maxSmooth)));
                }
                else if (para.smoothType == smooth_pcp) {
                    if (motionTem[0] != motionDown[0] || motionTem[1] != motionDown[1])
                        ob = 1.0f;
                }
                else if (para.smoothType == smooth_esl) {
                    ob = distance(motionTem, motionDown);
                }

#if SMOOTH_WEIGHT_1
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[pTem[1]+1][pTem[0]]);
            if (costTmpC < DIFF_THRESHOLD)
                ob *= SMOOTH_WEIGHT;
#endif
#if SMOOTH_WEIGHT_2
            DistanceEuclidean<Point3D> distance3D;
            GLfloat costTmpC = distance3D(nnr.colorMap[pTem[1]][pTem[0]], nnr.colorMap[pTem[1]+1][pTem[0]]);
            ob *= exp(- AJUST * (COEF1 * costTmpC + COEF2 * nnr.densityMap[pTem[1]][pTem[0]]));
#endif
            newSmoLeft += ob;

            }
        }

        return (newSmo + newSmoUp + newSmoLeft);
    }

    DEVICE_HOST GLfloat evaluateNodeSmooth(Motion pMotion1, Motion pMotion2) {

#if STEREO_COMP
        components::DistanceManhattan<PointEuclid> distance;
#else
        components::DistanceEuclidean<PointEuclid> distance;
#endif
        GLfloat ob = 0.0f;
            if (para.smoothType == smooth_pstq) {
                ob += (MIN((distance(pMotion1, pMotion2) * distance(pMotion1, pMotion2)), (para.maxSmooth * para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pstl) {
                ob += (MIN((distance(pMotion1, pMotion2)), (para.maxSmooth)));
            }
            else if (para.smoothType == smooth_pcp) {
                if (pMotion1[0] != pMotion2[0] || pMotion1[1] != pMotion2[1])
                    ob += 1.0f;
            }
            else if (para.smoothType == smooth_esl) {
                ob += distance(pMotion1, pMotion2);
            }
        return ob;
    }

#if STEREO_COMP
    DEVICE_HOST GLfloat evaluateNodeCost(NN& nnr, NN& nnd, PointCoord p) {

        // worst value for sumdiff below
        float worst_match = 3.0 * 255.0;
        // truncation threshold - NOTE: if squared, don't multiply by nColors (Eucl. dist.)
        float maxsumdiff = 3 * para.maxColor;
        // value for out-of-bounds matches
        float badcost = MIN(worst_match, maxsumdiff);

        DistanceManhattan<Point3D> distance3D;
//        DistanceManhattan<GLfloat> distance1D;
        PointEuclid newPos = nnr.adaptiveMap[p[1]][p[0]];

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = fabs(nnr.densityMap[p[1]][p[0]] - (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = fabs(nnr.densityMap[p[1]][p[0]] - nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        if (!(newPos[0] >=0 && newPos[0] < nnd.colorMap.getWidth()))
            costTmpC = badcost;
        costTmpC = MIN(costTmpC, maxsumdiff);
        costTmpG = MIN(costTmpG, para.maxGradient);

        nnr.objectivesMap[p[1]][p[0]][(int)obj_cost] = para.alphaC * costTmpC + para.alphaG * costTmpG;
        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    DEVICE_HOST GLfloat evaluateNodeCost(NN& nnr, NN& nnd, PointCoord p, PointEuclid newPos) {

        // worst value for sumdiff below
        float worst_match = 3.0 * 255.0;
        // truncation threshold - NOTE: if squared, don't multiply by nColors (Eucl. dist.)
        float maxsumdiff = 3 * para.maxColor;
        // value for out-of-bounds matches
        float badcost = MIN(worst_match, maxsumdiff);

        DistanceManhattan<Point3D> distance3D;
//        DistanceManhattan<GLfloat> distance1D;

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = fabs(nnr.densityMap[p[1]][p[0]] - (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = fabs(nnr.densityMap[p[1]][p[0]] - nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        if (!(newPos[0] >=0 && newPos[0] < nnd.colorMap.getWidth()))
            costTmpC = badcost;
        costTmpC = MIN(costTmpC, maxsumdiff);
        costTmpG = MIN(costTmpG, para.maxGradient);

        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    DEVICE_HOST GLfloat evaluateNodeCostMotion(NN& nnr, NN& nnd, PointCoord p, Motion pMotion) {

        // worst value for sumdiff below
        float worst_match = 3.0 * 255.0;
        // truncation threshold - NOTE: if squared, don't multiply by nColors (Eucl. dist.)
        float maxsumdiff = 3 * para.maxColor;
        // value for out-of-bounds matches
        float badcost = MIN(worst_match, maxsumdiff);

        DistanceManhattan<Point3D> distance3D;
//        DistanceManhattan<GLfloat> distance1D;
        PointEuclid newPos = nnd.adaptiveMap[p[1]][p[0]] + pMotion;

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = fabs(nnr.densityMap[p[1]][p[0]] - (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = fabs(nnr.densityMap[p[1]][p[0]] - nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        if (!(newPos[0] >=0 && newPos[0] < nnd.colorMap.getWidth()))
            costTmpC = badcost;
        costTmpC = MIN(costTmpC, maxsumdiff);
        costTmpG = MIN(costTmpG, para.maxGradient);

        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }
#else
    DEVICE_HOST GLfloat evaluateNodeCost(NN& nnr, NN& nnd, PointCoord p) {

        DistanceEuclidean<Point3D> distance3D;
        DistanceEuclidean<GLfloat> distance1D;
        PointEuclid newPos = nnr.adaptiveMap[p[1]][p[0]];

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        costTmpC = MIN(costTmpC, para.maxColor);
        costTmpG = MIN(costTmpG, para.maxGradient);

        nnr.objectivesMap[p[1]][p[0]][(int)obj_cost] = para.alphaC * costTmpC + para.alphaG * costTmpG;
        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    DEVICE_HOST GLfloat evaluateNodeCost(NN& nnr, NN& nnd, PointCoord p, PointEuclid newPos) {

        DistanceEuclidean<Point3D> distance3D;
        DistanceEuclidean<GLfloat> distance1D;

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        costTmpC = MIN(costTmpC, para.maxColor);
        costTmpG = MIN(costTmpG, para.maxGradient);

        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    DEVICE_HOST GLfloat evaluateNodeCostMotion(NN& nnr, NN& nnd, PointCoord p, Motion pMotion) {

        DistanceEuclidean<Point3D> distance3D;
        DistanceEuclidean<GLfloat> distance1D;
        PointEuclid newPos = nnd.adaptiveMap[p[1]][p[0]] + pMotion;

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        costTmpC = MIN(costTmpC, para.maxColor);
        costTmpG = MIN(costTmpG, para.maxGradient);

        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }
#endif // STEREO_COMP

    DEVICE_HOST GLfloat evaluateNodeSquaredCost(NN& nnr, NN& nnd, PointCoord p) {

        DistanceSquaredEuclidean<Point3D> distance3D;
        DistanceSquaredEuclidean<GLfloat> distance1D;
        PointEuclid newPos = nnr.adaptiveMap[p[1]][p[0]];

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        costTmpC = MIN(costTmpC, para.maxColor);
        costTmpG = MIN(costTmpG, para.maxGradient);

        nnr.objectivesMap[p[1]][p[0]][(int)obj_sqr_cost] = para.alphaC * costTmpC + para.alphaG * costTmpG;
        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    DEVICE_HOST GLfloat evaluateNodeSquaredCost(NN& nnr, NN& nnd, PointCoord p, PointEuclid newPos) {

        DistanceSquaredEuclidean<Point3D> distance3D;
        DistanceSquaredEuclidean<GLfloat> distance1D;

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        costTmpC = MIN(costTmpC, para.maxColor);
        costTmpG = MIN(costTmpG, para.maxGradient);

        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    DEVICE_HOST GLfloat evaluateNodeSquaredCostMotion(NN& nnr, NN& nnd, PointCoord p, Motion pMotion) {

        DistanceSquaredEuclidean<Point3D> distance3D;
        DistanceSquaredEuclidean<GLfloat> distance1D;
        PointEuclid newPos = nnd.adaptiveMap[p[1]][p[0]] + pMotion;

        // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
        GLfloat costTmpC = distance3D(nnr.colorMap[p[1]][p[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
        GLfloat costTmpG = distance1D(nnr.densityMap[p[1]][p[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
        costTmpC = MIN(costTmpC, para.maxColor);
        costTmpG = MIN(costTmpG, para.maxGradient);

        return (para.alphaC * costTmpC + para.alphaG * costTmpG);
    }

    //! mirror out-of-range position
    DEVICE_HOST void mirror(int w, int h, PointCoord& p)
    {
        if (p[0] < 0) p[0] = ABS(p[0] + 1);
        if (p[1] < 0) p[1] = ABS(p[1] + 1);
        if (p[0] >= w) p[0] = w * 2 - p[0] - 1;
        if (p[1] >= h) p[1] = h * 2 - p[1] - 1;
    }

    // Traditional window-based cost aggregation that assumes all the support pixels within the support region window
    // have the same motion value as the center pixel considered.
    template <class DistanceFunctor, class DistanceFunctor2>
    DEVICE_HOST GLfloat evaluateNodeCostBasicWindow(NN& nnr, NN& nnd, PointCoord p, size_t wr = DEFAULT_WINDOW_RADIUS) {
        GLfloat cost = 0.0f;
        DistanceFunctor distance;
        DistanceFunctor2 distance2;
        PointEuclid motion = nnr.adaptiveMap[p[1]][p[0]];
        motion -= nnd.adaptiveMap[p[1]][p[0]];
        NIter iter(p, 0, 0);
        iter.initialize(p, 0, wr);

#if ADAPTIVE_WEIGHT
        GLfloat weight_window = 0.0f;
#endif

        do {
            PointCoord pco = iter.getNodeIncr();
            mirror(nnr.adaptiveMap.getWidth(),
                   nnr.adaptiveMap.getHeight(),
                   pco);
            PointEuclid newPos = nnd.adaptiveMap[pco[1]][pco[0]] + motion;
            // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
            GLfloat costTmpC = distance(nnr.colorMap[pco[1]][pco[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
            GLfloat costTmpG = distance2(nnr.densityMap[pco[1]][pco[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
            GLfloat costTmpC = distance(nnr.colorMap[pco[1]][pco[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
            GLfloat costTmpG = distance2(nnr.densityMap[pco[1]][pco[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
            costTmpC = MIN(costTmpC, para.maxColor);
            costTmpG = MIN(costTmpG, para.maxGradient);

#if ADAPTIVE_WEIGHT
            DistanceEuclidean<Point3D> distance3D;
            DistanceEuclidean<Point2D> distance2D;
            GLfloat diffColor = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[pco[1]][pco[0]]);
            GLfloat diffSpatial = distance2D(nnd.adaptiveMap[p[1]][p[0]], nnd.adaptiveMap[pco[1]][pco[0]]);
            diffColor /= MAX_COLOR_PARA;
            diffSpatial /= (GLfloat)(wr);
            GLfloat weight = exp(-(diffColor + diffSpatial));
            cost += weight * (para.alphaC * costTmpC + para.alphaG * costTmpG);
            weight_window += weight;
#endif
            cost += para.alphaC * costTmpC + para.alphaG * costTmpG;

        }  while (iter.nextNodeIncr());

#if ADAPTIVE_WEIGHT
        cost /= weight_window;
#endif
        return cost;
    }

    template <class DistanceFunctor, class DistanceFunctor2>
    DEVICE_HOST GLfloat evaluateNodeCostBasicWindow(NN& nnr, NN& nnd, PointCoord p, Motion pMotion, size_t wr = DEFAULT_WINDOW_RADIUS) {
        GLfloat cost = 0.0f;
        DistanceFunctor distance;
        DistanceFunctor2 distance2;
        NIter iter(p, 0, 0);
        iter.initialize(p, 0, wr);

#if ADAPTIVE_WEIGHT
        GLfloat weight_window = 0.0f;
#endif

        do {
            PointCoord pco = iter.getNodeIncr();
            mirror(nnr.adaptiveMap.getWidth(),
                   nnr.adaptiveMap.getHeight(),
                   pco);
            PointEuclid newPos = nnd.adaptiveMap[pco[1]][pco[0]] + pMotion;
            // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
            GLfloat costTmpC = distance(nnr.colorMap[pco[1]][pco[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
            GLfloat costTmpG = distance2(nnr.densityMap[pco[1]][pco[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
            GLfloat costTmpC = distance(nnr.colorMap[pco[1]][pco[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
            GLfloat costTmpG = distance2(nnr.densityMap[pco[1]][pco[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
            costTmpC = MIN(costTmpC, para.maxColor);
            costTmpG = MIN(costTmpG, para.maxGradient);

#if ADAPTIVE_WEIGHT
            DistanceEuclidean<Point3D> distance3D;
            DistanceEuclidean<Point2D> distance2D;
            GLfloat diffColor = distance3D(nnr.colorMap[p[1]][p[0]], nnr.colorMap[pco[1]][pco[0]]);
            GLfloat diffSpatial = distance2D(nnd.adaptiveMap[p[1]][p[0]], nnd.adaptiveMap[pco[1]][pco[0]]);
            diffColor /= MAX_COLOR_PARA;
            diffSpatial /= (GLfloat)(wr);
            GLfloat weight = exp(-(diffColor + diffSpatial));
            cost += weight * (para.alphaC * costTmpC + para.alphaG * costTmpG);
            weight_window += weight;
#endif
            cost += para.alphaC * costTmpC + para.alphaG * costTmpG;

        }  while (iter.nextNodeIncr());

#if ADAPTIVE_WEIGHT
        cost /= weight_window;
#endif
        return cost;
    }

    // Real window-based cost aggregation. For the support pixels within the support region window,
    // their true/actual motion values are used.
    template <class DistanceFunctor, class DistanceFunctor2>
    DEVICE_HOST GLfloat evaluateNodeCostBasicWindowReal(NN& nnr, NN& nnd, PointCoord p, size_t wr = DEFAULT_WINDOW_RADIUS) {
#ifndef CUDA_CODE
#ifdef CPU_DEBUG
        //debug
        gDebug[p[1]][p[0]] += 1;
#endif
#endif
        GLfloat cost = 0.0f;
        DistanceFunctor distance;
        DistanceFunctor2 distance2;
        NIter iter(p, 0, 0);
        iter.initialize(p, 0, wr);
        do {
            PointCoord pco = iter.getNodeIncr();
            mirror(nnr.adaptiveMap.getWidth(),
                   nnr.adaptiveMap.getHeight(),
                   pco);
            PointEuclid newPos = nnr.adaptiveMap[pco[1]][pco[0]];
            // Here change fetchIntCoor() into fetchFloatCoor() for sub-pixel precision
#if SUBPIXEL_PRECISION
            GLfloat costTmpC = distance(nnr.colorMap[pco[1]][pco[0]], (Point3D)(nnd.colorMap.fetchFloatCoor(newPos[0], newPos[1])));
            GLfloat costTmpG = distance2(nnr.densityMap[pco[1]][pco[0]], (GLfloat)(nnd.densityMap.fetchFloatCoor(newPos[0], newPos[1])));
#else
            GLfloat costTmpC = distance(nnr.colorMap[pco[1]][pco[0]], nnd.colorMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
            GLfloat costTmpG = distance2(nnr.densityMap[pco[1]][pco[0]], nnd.densityMap.fetchIntCoor((int)newPos[0], (int)newPos[1]));
#endif
            costTmpC = MIN(costTmpC, para.maxColor);
            costTmpG = MIN(costTmpG, para.maxGradient);
            cost += para.alphaC * costTmpC + para.alphaG * costTmpG;
        }  while (iter.nextNodeIncr());
        return cost;
    }

    DEVICE_HOST GLfloat evaluateNodeCostWindow(NN& nnr, NN& nnd, PointCoord p, size_t wr = DEFAULT_WINDOW_RADIUS) {
        nnr.objectivesMap[p[1]][p[0]][(int)obj_cost_window] = evaluateNodeCostBasicWindow<DistanceEuclidean<Point3D>, DistanceEuclidean<GLfloat> >(nnr, nnd, p, wr);
//        nnr.objectivesMap[p[1]][p[0]][(int)obj_cost_window] = evaluateNodeCostBasicWindowReal<DistanceEuclidean<Point3D>, DistanceEuclidean<GLfloat> >(nnr, nnd, p, wr);
        return nnr.objectivesMap[p[1]][p[0]][(int)obj_cost_window];
    }

    DEVICE_HOST GLfloat evaluateNodeCostWindow(NN& nnr, NN& nnd, PointCoord p, Motion pMotion, size_t wr = DEFAULT_WINDOW_RADIUS) {
       return (evaluateNodeCostBasicWindow<DistanceEuclidean<Point3D>, DistanceEuclidean<GLfloat> >(nnr, nnd, p, pMotion, wr));
    }

    DEVICE_HOST GLfloat evaluateNodeSquaredCostWindow(NN& nnr, NN& nnd, PointCoord p, size_t wr = DEFAULT_WINDOW_RADIUS) {
        nnr.objectivesMap[p[1]][p[0]][(int)obj_sqr_cost_window] = evaluateNodeCostBasicWindow<DistanceSquaredEuclidean<Point3D>, DistanceSquaredEuclidean<GLfloat> >(nnr, nnd, p, wr);
//        nnr.objectivesMap[p[1]][p[0]][(int)obj_sqr_cost_window] = evaluateNodeCostBasicWindowReal<DistanceSquaredEuclidean<Point3D>, DistanceSquaredEuclidean<GLfloat> >(nnr, nnd, p, wr);
        return nnr.objectivesMap[p[1]][p[0]][(int)obj_sqr_cost_window];
    }

    DEVICE_HOST GLfloat evaluateNodeSquaredCostWindow(NN& nnr, NN& nnd, PointCoord p, Motion pMotion, size_t wr = DEFAULT_WINDOW_RADIUS) {
        return (evaluateNodeCostBasicWindow<DistanceSquaredEuclidean<Point3D>, DistanceSquaredEuclidean<GLfloat> >(nnr, nnd, p, pMotion, wr));
    }

    DEVICE_HOST GLfloat evaluateNodeGTerror(NN& nnr, NN& nnd, PointCoord p) {
        components::DistanceEuclidean<PointEuclid> distance;
        PointEuclid motion = nnr.adaptiveMap[p[1]][p[0]];
        motion -= nnd.adaptiveMap[p[1]][p[0]];
        nnr.objectivesMap[p[1]][p[0]][(int)obj_gd_error] = distance(motion, this->gt[p[1]][p[0]]);
        return nnr.objectivesMap[p[1]][p[0]][(int)obj_gd_error];
    }

    // To be filled ...
    DEVICE_HOST GLfloat evaluateNodeDistr(NN& nnr, NN& nnd, PointCoord p) {
        nnr.objectivesMap[p[1]][p[0]][(int)obj_distr] = 0.0f;
        return nnr.objectivesMap[p[1]][p[0]][(int)obj_distr];
    }

    DEVICE_HOST GLfloat evaluateNode(NN& nnr, NN& nnd, PointCoord p, AMObjNames obType, size_t wr = DEFAULT_WINDOW_RADIUS)
    {
        GLfloat obValue = 0.0f;
        switch (obType) {
        case obj_distr :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeDistr(nnr, nnd, p);
            break;
        case obj_length :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeLength(nnr, p);
            break;
        case obj_sqr_length :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeSquaredLength(nnr, p);
            break;
        case obj_cost :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeCost(nnr, nnd, p);
            break;
        case obj_sqr_cost :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeSquaredCost(nnr, nnd, p);
            break;
        case obj_cost_window :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeCostWindow(nnr, nnd, p, wr);
            break;
        case obj_sqr_cost_window :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeSquaredCostWindow(nnr, nnd, p, wr);
            break;
        case obj_smoothing :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeSmooth(nnr, nnd, p);
            break;
        case obj_gd_error :
            obValue = nnr.objectivesMap[p[1]][p[0]].get_weights((int)obType) * evaluateNodeGTerror(nnr, nnd, p);
            break;
        case 9 :
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_distr) * evaluateNodeDistr(nnr, nnd, p);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_length) * evaluateNodeLength(nnr, p);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_sqr_length) * evaluateNodeSquaredLength(nnr, p);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_cost) * evaluateNodeCost(nnr, nnd, p);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_sqr_cost) * evaluateNodeSquaredCost(nnr, nnd, p);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_cost_window) * evaluateNodeCostWindow(nnr, nnd, p, wr);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_sqr_cost_window) * evaluateNodeSquaredCostWindow(nnr, nnd, p, wr);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_smoothing) * evaluateNodeSmooth(nnr, nnd, p);
            obValue += nnr.objectivesMap[p[1]][p[0]].get_weights((int)obj_gd_error) * evaluateNodeGTerror(nnr, nnd, p);
            break;
        default :
            break;
        } // switch
        return obValue;
    }

    DEVICE_HOST GLfloat evaluateNode(NN& nnr, NN& nnd, PointCoord p, size_t wr = DEFAULT_WINDOW_RADIUS)
    {
        GLfloat obValue = 0.0f;

        if (actObj.distr != 0)
            obValue += actObj.distr * evaluateNodeDistr(nnr, nnd, p);
        if (actObj.length != 0)
            obValue += actObj.length * evaluateNodeLength(nnr, p);
        if (actObj.sqr_length != 0)
            obValue += actObj.sqr_length * evaluateNodeSquaredLength(nnr, p);
        if (actObj.cost != 0)
            obValue += actObj.cost * evaluateNodeCost(nnr, nnd, p);
        if (actObj.sqr_cost != 0)
            obValue += actObj.sqr_cost * evaluateNodeSquaredCost(nnr, nnd, p);
        if (actObj.cost_window != 0)
            obValue += actObj.cost_window * evaluateNodeCostWindow(nnr, nnd, p, wr);
        if (actObj.sqr_cost_window != 0)
            obValue += actObj.sqr_cost_window * evaluateNodeSquaredCostWindow(nnr, nnd, p, wr);
        if (actObj.smoothing != 0)
            obValue += actObj.smoothing * evaluateNodeSmooth(nnr, nnd, p);
        if (actObj.gd_error != 0)
            obValue += actObj.gd_error * evaluateNodeGTerror(nnr, nnd, p);

        return obValue;
    }

    GLOBAL inline void K_init(NN& nnr) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.objectivesMap.getWidth(),
                              nnr.objectivesMap.getHeight());

        K_E_init _KER_CALL_(b, t) (nnr);

    }

    GLOBAL inline void K_initializeWeights(NN& nnr, AMObjNames obType, GLfloat w = 1.0f) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.objectivesMap.getWidth(),
                              nnr.objectivesMap.getHeight());

        K_E_initializeWeights _KER_CALL_(b, t) (nnr, obType, w);

    }

    GLOBAL inline void K_initializeWeights(NN& nnr) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.objectivesMap.getWidth(),
                              nnr.objectivesMap.getHeight());

        K_E_initializeWeights _KER_CALL_(b, t) (nnr, actObj);

    }

    GLOBAL inline void K_copyWeights(NN& nnr, NN& nnrCopy) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.objectivesMap.getWidth(),
                              nnr.objectivesMap.getHeight());

        K_E_copyWeights _KER_CALL_(b, t) (nnr, nnrCopy);

    }

    GLOBAL inline void K_initializePointWeight(NN& nnr, PointCoord p, AMObjNames obType = (AMObjNames)9, GLfloat w = 1.0f) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.objectivesMap.getWidth(),
                              nnr.objectivesMap.getHeight());

        K_E_initializePointWeight _KER_CALL_(b, t) (nnr, p, obType, w);

    }

}; // EvaluationNode

/*!
 * \brief Class EvaluationGrid.
 * It is a mixte CPU/GPU class, then with
 * default copy constructor for allowing
 * parameter passing to GPU Kernel.
 **/

template <class NIter, class GridGT = MatMotion>
class EvaluationGrid : public EvaluationNode<NIter, GridGT>
{
protected:
    AMObjectives obSumGrid;

public:
    DEVICE_HOST inline EvaluationGrid() {}

    DEVICE_HOST inline explicit EvaluationGrid(GridGT _gt) :
        EvaluationNode<NIter, GridGT>(_gt) {}

    DEVICE_HOST inline explicit EvaluationGrid(
            MatchingCostParams _para,
            GridGT _gt
            ) :
        EvaluationNode<NIter, GridGT>(_para, _gt) {}

    DEVICE_HOST inline explicit EvaluationGrid(
            MatchingCostParams _para,
            GridGT _gt,
            ActiveObj _actObj
            ) :
        EvaluationNode<NIter, GridGT>(_para, _gt, _actObj) {}

    DEVICE_HOST void initialize(ActiveObj _actObj) {
        EvaluationNode<NIter, GridGT>::initialize(_actObj);
    }

    DEVICE_HOST void initialize(GridGT _gt) {
        EvaluationNode<NIter, GridGT>::initialize(_gt);
    }

    DEVICE_HOST AMObjectives getObjectives() {
        return obSumGrid;
    }

    DEVICE_HOST GLfloat getObjectives(AMObjNames obn) {
        return obSumGrid[(int)obn];
    }

    DEVICE_HOST void setObjectives(AMObjectives o) {
        obSumGrid = o;
    }

    DEVICE_HOST void setObjectives(AMObjNames obn, GLfloat o) {
        obSumGrid[(int)obn] = o;
    }

    DEVICE_HOST inline void initWeights(AMObjNames obType, GLfloat w = 1.0f) {

        if ((int)obType == 9)
        {
            for (size_t i = 0; i < 9; ++i)
                obSumGrid.set_weights(i, w);
        }
        else if ((int)obType >= 0 && (int)obType < 9)
            obSumGrid.set_weights((int)obType, w);
    }

    DEVICE_HOST inline void initWeights() {

        if (this->actObj.distr != 0)
            obSumGrid.set_weights(0, this->actObj.distr);
        if (this->actObj.length != 0)
            obSumGrid.set_weights(1, this->actObj.length);
        if (this->actObj.sqr_length != 0)
            obSumGrid.set_weights(2, this->actObj.sqr_length);
        if (this->actObj.cost != 0)
            obSumGrid.set_weights(3, this->actObj.cost);
        if (this->actObj.sqr_cost != 0)
            obSumGrid.set_weights(4, this->actObj.sqr_cost);
        if (this->actObj.cost_window != 0)
            obSumGrid.set_weights(5, this->actObj.cost_window);
        if (this->actObj.sqr_cost_window != 0)
            obSumGrid.set_weights(6, this->actObj.sqr_cost_window);
        if (this->actObj.smoothing != 0)
            obSumGrid.set_weights(7, this->actObj.smoothing);
        if (this->actObj.gd_error != 0)
            obSumGrid.set_weights(8, this->actObj.gd_error);
    }

    GLOBAL void K_evaluate(NN& nnr, NN& nnd, AMObjNames obType, size_t wr = DEFAULT_WINDOW_RADIUS) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.adaptiveMap.getWidth(),
                              nnr.adaptiveMap.getHeight());

        K_EG_evaluate _KER_CALL_(b, t) (*this, nnr, nnd, obType, wr);
    }

    GLOBAL void K_evaluate(NN& nnr, NN& nnd, size_t wr = DEFAULT_WINDOW_RADIUS) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nnr.adaptiveMap.getWidth(),
                              nnr.adaptiveMap.getHeight());

        K_EG_evaluate _KER_CALL_(b, t) (*this, nnr, nnd, wr);
    }

#ifdef CUDA_CODE
    GLOBAL GLfloat K_reductionForSum(NN& nnr, NN& nnd) {

#ifdef GPU_REDUCTION
        Grid<AMObjectives> objectivesMapTemp;
        objectivesMapTemp.gpuResize(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight());
        nnr.objectivesMap.gpuCopyDeviceToDevice(objectivesMapTemp);

        KER_CALL_THREAD_BLOCK(b, t,
                              blockDimY, blockDimX,
                              objectivesMapTemp.getWidth(),
                              objectivesMapTemp.getHeight());

        K_ObjecValReductionStep1<AMObjectives> _KER_CALL_(b, t) (objectivesMapTemp);
        K_ObjecValReductionStep2<AMObjectives> _KER_CALL_(b, t) (objectivesMapTemp);

        Grid<AMObjectives> objectivesMapTempCPU;
        objectivesMapTempCPU.resize(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight());
        objectivesMapTempCPU.gpuCopyDeviceToHost(objectivesMapTemp);
        obSumGrid = objectivesMapTempCPU[0][0];

        objectivesMapTemp.gpuFreeMem();
        objectivesMapTempCPU.freeMem();

        return obSumGrid.computeObjectif();
#else

        Grid<AMObjectives> objectivesMapTemp;
        objectivesMapTemp.resize(nnr.objectivesMap.getWidth(), nnr.objectivesMap.getHeight());
        objectivesMapTemp.gpuCopyDeviceToHost(nnr.objectivesMap);

        for (size_t i = 0; i < 9; ++i)
            setObjectives((AMObjNames)i, 0.0f);

        for (int j = 0; j < objectivesMapTemp.getHeight(); j++)
            for (int i = 0; i < objectivesMapTemp.getWidth(); i++)
                obSumGrid += objectivesMapTemp[j][i];

        objectivesMapTemp.freeMem();
        return obSumGrid.computeObjectif();
#endif
    }

#else
    GLOBAL GLfloat K_reductionForSum(NN& nnr, NN& nnd) {

        for (size_t i = 0; i < 9; ++i)
            setObjectives((AMObjNames)i, 0.0f);

        for (int j = 0; j < nnr.adaptiveMap.getHeight(); j++)
            for (int i = 0; i < nnr.adaptiveMap.getWidth(); i++)
                obSumGrid += nnr.objectivesMap[j][i];
        return obSumGrid.computeObjectif();
    }
#endif

}; // EvaluationGrid


/*!
 * \brief Class EvaluationGrid.
 * It is a mixte CPU/GPU class, then with
 * default copy constructor for allowing
 * parameter passing to GPU Kernel.
 **/
template <class NIter, class Cell, class CellularMatrix, class GridGT = MatMotion>
class EvaluationCellularMatrix : public EvaluationNode<NIter, GridGT>
{
protected:
    AMObjectives obCell;

public:
    CellularMatrix cmr;
    Grid<AMObjectives> objectivesMapCM;

public:

    DEVICE_HOST inline EvaluationCellularMatrix() {}

    DEVICE_HOST inline explicit EvaluationCellularMatrix(
            GridGT& _gt,
            CellularMatrix& _cmr) : EvaluationNode<NIter, GridGT>(_gt) {
    cmr = _cmr;
    objectivesMapCM.gpuResize(cmr.getWidth(), cmr.getHeight());
    }

    DEVICE_HOST inline explicit EvaluationCellularMatrix(
            MatchingCostParams& _para,
            GridGT& _gt,
            CellularMatrix& _cmr) : EvaluationNode<NIter, GridGT>(_para, _gt) {
    cmr = _cmr;
    objectivesMapCM.gpuResize(cmr.getWidth(), cmr.getHeight());
    }

    DEVICE_HOST inline void initCM(CellularMatrix& _cmr) {
        cmr = _cmr;
//        objectivesMapCM.gpuResize(cmr.getWidth(), cmr.getHeight());
    }

    DEVICE_HOST inline AMObjectives getObjectives() {
        return obCell;
    }

    DEVICE_HOST inline GLfloat getObjectives(AMObjNames obn) {
        return obCell[(int)obn];
    }

    DEVICE_HOST inline void setObjectives(AMObjectives o) {
        obCell = o;
    }

    DEVICE_HOST inline void setObjectives(AMObjNames obn, GLfloat o) {
        obCell[(int)obn] = o;
    }

    DEVICE_HOST inline void initWeights(AMObjNames obType, GLfloat w = 1.0f) {

        if ((int)obType == 9)
        {
            for (size_t i = 0; i < 9; ++i)
                obCell.set_weights(i, w);
        }
        else if ((int)obType >= 0 && (int)obType < 9)
            obCell.set_weights((int)obType, w);
    }

    DEVICE_HOST inline void initWeights() {

        if (this->actObj.distr != 0)
            obCell.set_weights(0, this->actObj.distr);
        if (this->actObj.length != 0)
            obCell.set_weights(1, this->actObj.length);
        if (this->actObj.sqr_length != 0)
            obCell.set_weights(2, this->actObj.sqr_length);
        if (this->actObj.cost != 0)
            obCell.set_weights(3, this->actObj.cost);
        if (this->actObj.sqr_cost != 0)
            obCell.set_weights(4, this->actObj.sqr_cost);
        if (this->actObj.cost_window != 0)
            obCell.set_weights(5, this->actObj.cost_window);
        if (this->actObj.sqr_cost_window != 0)
            obCell.set_weights(6, this->actObj.sqr_cost_window);
        if (this->actObj.smoothing != 0)
            obCell.set_weights(7, this->actObj.smoothing);
        if (this->actObj.gd_error != 0)
            obCell.set_weights(8, this->actObj.gd_error);
    }

    DEVICE_HOST GLfloat evaluateCellLength(Cell& cell, NN& nn) {
        cell.init();
        PointCoord p;
        GLfloat sumob = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nn.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nn.adaptiveMap.getHeight())
                sumob += this->evaluateNodeLength(nn, p);
        } while (cell.next());
        obCell[(int)obj_length] = sumob;
        return sumob;
    }

    DEVICE_HOST GLfloat evaluateCellSquaredLength(Cell& cell, NN& nn) {
        cell.init();
        PointCoord p;
        GLfloat sumob = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nn.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nn.adaptiveMap.getHeight())
                sumob += this->evaluateNodeSquaredLength(nn, p);
        } while (cell.next());
        obCell[(int)obj_sqr_length] = sumob;
        return sumob;
    }

    DEVICE_HOST GLfloat evaluateCellSmooth2(Cell& cell, NN& nnr, NN& nnd) {
        cell.init();
        PointCoord p;
        GLfloat sumob_nnr = 0.0f;
        GLfloat sumob_nnd = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight()) {
                sumob_nnr += this->evaluateNodeLength(nnr, p);
                sumob_nnd += this->evaluateNodeLength(nnd, p);
            }
        } while (cell.next());
        obCell[(int)obj_length] = sumob_nnr;
        obCell[(int)obj_smoothing] = ABS(sumob_nnr - sumob_nnd);
        return obCell[(int)obj_smoothing];
    }

    DEVICE_HOST GLfloat evaluateCellSmooth(Cell& cell, NN& nnr, NN& nnd) {
        cell.init();
        PointCoord p;
        GLfloat sumob = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight()) {
                sumob += this->evaluateNodeSmooth(nnr, nnd, p);
            }
        } while (cell.next());
        obCell[(int)obj_smoothing] = sumob;
        return sumob;
    }

    DEVICE_HOST GLfloat evaluateCellCost(Cell& cell, NN& nnr, NN& nnd) {

#ifndef CUDA_CODE
#ifdef CPU_DEBUG
        //debug
//        printf("cell info: pc = (%d, %d), PC = (%d, %d), radius = %d\n", cell.pc[0], cell.pc[1], cell.PC[0], cell.PC[1], (int)cell.radius);
#endif
#endif

        cell.init();
        PointCoord p;
        GLfloat sumCost = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight())
                sumCost += this->evaluateNodeCost(nnr, nnd, p);
        } while (cell.next());
        obCell[(int)obj_cost] = sumCost;
        return sumCost;
    }

    DEVICE_HOST GLfloat evaluateCellSquaredCost(Cell& cell, NN& nnr, NN& nnd) {
        cell.init();
        PointCoord p;
        GLfloat sumCost = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight())
                sumCost += this->evaluateNodeSquaredCost(nnr, nnd, p);
        } while (cell.next());
        obCell[(int)obj_sqr_cost] = sumCost;
        return sumCost;
    }

    DEVICE_HOST GLfloat evaluateCellCostWindow(Cell& cell, NN& nnr, NN& nnd,
                                               size_t wr = DEFAULT_WINDOW_RADIUS) {
        cell.init();
        PointCoord p;
        GLfloat sumCost = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight())
                sumCost += this->evaluateNodeCostWindow(nnr, nnd, p, wr);
        } while (cell.next());
        obCell[(int)obj_cost_window] = sumCost;
        return sumCost;
    }

    DEVICE_HOST GLfloat evaluateCellSquaredCostWindow(Cell& cell, NN& nnr, NN& nnd,
                                                      size_t wr = DEFAULT_WINDOW_RADIUS) {
        cell.init();
        PointCoord p;
        GLfloat sumCost = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight())
                sumCost += this->evaluateNodeSquaredCostWindow(nnr, nnd, p, wr);
        } while (cell.next());
        obCell[(int)obj_sqr_cost_window] = sumCost;
        return sumCost;
    }

    DEVICE_HOST GLfloat evaluateCellGTerror(Cell& cell, NN& nnr, NN& nnd) {
        cell.init();
        PointCoord p;
        GLfloat err = 0.0f;
        int count = 0;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight()) {
                err += this->evaluateNodeGTerror(nnr, nnd, p);
                count++;
            }
        } while (cell.next());
//        err /= (float)count;
        obCell[(int)obj_gd_error] = err;
        return err;
    }

    DEVICE_HOST GLfloat evaluateCellDistr(Cell& cell, NN& nnr, NN& nnd) {
        cell.init();
        PointCoord p;
        GLfloat _distr = 0.0f;
        do {
            cell.get(p);
            if (p[0] >= 0 && p[0] < nnr.adaptiveMap.getWidth()
                    && p[1] >= 0 && p[1] < nnr.adaptiveMap.getHeight()) {
                this->evaluateNodeDistr(nnr, nnd, p);
            }
        } while (cell.next());
        obCell[(int)obj_distr] = _distr;
        return _distr;
    }

    DEVICE_HOST GLfloat evaluateCell(Cell& cell, NN& nnr, NN& nnd, AMObjNames obType, size_t wr = DEFAULT_WINDOW_RADIUS) {
        GLfloat obValue = 0.0f;
        switch (obType) {
        case obj_distr :
            obValue = obCell.get_weights((int)obType) * evaluateCellDistr(cell, nnr, nnd);
            break;
        case obj_length :
            obValue = obCell.get_weights((int)obType) * evaluateCellLength(cell, nnr);
            break;
        case obj_sqr_length :
            obValue = obCell.get_weights((int)obType) * evaluateCellSquaredLength(cell, nnr);
            break;
        case obj_cost :
            obValue = obCell.get_weights((int)obType) * evaluateCellCost(cell, nnr, nnd);
            break;
        case obj_sqr_cost :
            obValue = obCell.get_weights((int)obType) * evaluateCellSquaredCost(cell, nnr, nnd);
            break;
        case obj_cost_window :
            obValue = obCell.get_weights((int)obType) * evaluateCellCostWindow(cell, nnr, nnd, wr);
            break;
        case obj_sqr_cost_window :
            obValue = obCell.get_weights((int)obType) * evaluateCellSquaredCostWindow(cell, nnr, nnd, wr);
            break;
        case obj_smoothing :
            obValue = obCell.get_weights((int)obType) * evaluateCellSmooth(cell, nnr, nnd);
            break;
        case obj_gd_error :
            obValue = obCell.get_weights((int)obType) * evaluateCellGTerror(cell, nnr, nnd);
            break;
        case 9 :
            obValue += obCell.get_weights((int)obj_distr) * evaluateCellDistr(cell, nnr, nnd);
            obValue += obCell.get_weights((int)obj_length) * evaluateCellLength(cell, nnr);
            obValue += obCell.get_weights((int)obj_sqr_length) * evaluateCellSquaredLength(cell, nnr);
            obValue += obCell.get_weights((int)obj_cost) * evaluateCellCost(cell, nnr, nnd);
            obValue += obCell.get_weights((int)obj_sqr_cost) * evaluateCellSquaredCost(cell, nnr, nnd);
            obValue += obCell.get_weights((int)obj_cost_window) * evaluateCellCostWindow(cell, nnr, nnd, wr);
            obValue += obCell.get_weights((int)obj_sqr_cost_window) * evaluateCellSquaredCostWindow(cell, nnr, nnd, wr);
            obValue += obCell.get_weights((int)obj_smoothing) * evaluateCellSmooth(cell, nnr, nnd);
            obValue += obCell.get_weights((int)obj_gd_error) * evaluateCellGTerror(cell, nnr, nnd);
            break;
        default :
            break;
        } // switch
        return obValue;
    }

    DEVICE_HOST GLfloat evaluateCell(Cell& cell, NN& nnr, NN& nnd, size_t wr = DEFAULT_WINDOW_RADIUS) {
        GLfloat obValue = 0.0f;

        if (this->actObj.distr != 0)
            obValue += this->actObj.distr * evaluateCellDistr(cell, nnr, nnd);
        if (this->actObj.length != 0)
            obValue += this->actObj.length * evaluateCellLength(cell, nnr);
        if (this->actObj.sqr_length != 0)
            obValue += this->actObj.sqr_length * evaluateCellSquaredLength(cell, nnr);
        if (this->actObj.cost != 0)
            obValue += this->actObj.cost * evaluateCellCost(cell, nnr, nnd);
        if (this->actObj.sqr_cost != 0)
            obValue += this->actObj.sqr_cost * evaluateCellSquaredCost(cell, nnr, nnd);
        if (this->actObj.cost_window != 0)
            obValue += this->actObj.cost_window * evaluateCellCostWindow(cell, nnr, nnd, wr);
        if (this->actObj.sqr_cost_window != 0)
            obValue += this->actObj.sqr_cost_window * evaluateCellSquaredCostWindow(cell, nnr, nnd, wr);
        if (this->actObj.smoothing != 0)
            obValue += this->actObj.smoothing * evaluateCellSmooth(cell, nnr, nnd);
        if (this->actObj.gd_error != 0)
            obValue += this->actObj.gd_error * evaluateCellGTerror(cell, nnr, nnd);

        return obValue;
    }

    GLOBAL inline void K_initializeWeightsCM(AMObjNames obType, GLfloat w = 1.0f) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              objectivesMapCM.getWidth(),
                              objectivesMapCM.getHeight());

        K_ECM_initializeWeights _KER_CALL_(b, t) (objectivesMapCM, obType, w);

    }

    GLOBAL inline void K_initializeWeightsCM() {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              objectivesMapCM.getWidth(),
                              objectivesMapCM.getHeight());

        K_ECM_initializeWeights _KER_CALL_(b, t) (objectivesMapCM, this->actObj);

    }

    GLOBAL void K_evaluate(NN& nnr, NN& nnd, AMObjNames obType, size_t wr = DEFAULT_WINDOW_RADIUS) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_ECM_evaluate _KER_CALL_(b, t) (*this, nnr, nnd, objectivesMapCM, obType, wr);
    }

    GLOBAL void K_evaluate(NN& nnr, NN& nnd, size_t wr = DEFAULT_WINDOW_RADIUS) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              cmr.getWidth(),
                              cmr.getHeight());

        K_ECM_evaluate _KER_CALL_(b, t) (*this, nnr, nnd, objectivesMapCM, wr);
    }

#ifdef CUDA_CODE
    GLOBAL GLfloat K_reductionForSum(NN& nnr, NN& nnd) {

#ifdef GPU_REDUCTION
        Grid<AMObjectives> objectivesMapCMTemp;
        objectivesMapCMTemp.gpuResize(objectivesMapCM.getWidth(), objectivesMapCM.getHeight());
        objectivesMapCM.gpuCopyDeviceToDevice(objectivesMapCMTemp);

        KER_CALL_THREAD_BLOCK(b, t,
                              blockDimY, blockDimX,
                              objectivesMapCMTemp.getWidth(),
                              objectivesMapCMTemp.getHeight());

        K_ObjecValReductionStep1<AMObjectives> _KER_CALL_(b, t) (objectivesMapCMTemp);
        K_ObjecValReductionStep2<AMObjectives> _KER_CALL_(b, t) (objectivesMapCMTemp);

        Grid<AMObjectives> objectivesMapCMTempCPU;
        objectivesMapCMTempCPU.resize(1,1);
        objectivesMapCMTempCPU.gpuCopyDeviceToHostFirst(objectivesMapCMTemp);
        obCell = objectivesMapCMTempCPU[0][0];

        objectivesMapCMTemp.gpuFreeMem();
        objectivesMapCMTempCPU.freeMem();

        return obCell.computeObjectif();
#else

        Grid<AMObjectives> objectivesMapCMTemp;
        objectivesMapCMTemp.resize(objectivesMapCM.getWidth(), objectivesMapCM.getHeight());
        objectivesMapCMTemp.gpuCopyDeviceToHost(objectivesMapCM);

        for (size_t i = 0; i < 9; ++i)
            setObjectives((AMObjNames)i, 0.0f);

        for (int j = 0; j < cmr.getHeight(); j++)
            for (int i = 0; i < cmr.getWidth(); i++)
                obCell += objectivesMapCMTemp[j][i];

        objectivesMapCMTemp.freeMem();

        return obCell.computeObjectif();

#endif

    }
#else
    GLOBAL GLfloat K_reductionForSum(NN& nnr, NN& nnd) {

        for (size_t i = 0; i < 9; ++i)
            setObjectives((AMObjNames)i, 0.0f);

        for (int j = 0; j < cmr.getHeight(); j++)
            for (int i = 0; i < cmr.getWidth(); i++)
                obCell += objectivesMapCM[j][i];
        return obCell.computeObjectif();
    }
#endif

}; // EvaluationCellularMatrix

} // namespace operators

#endif // EVALUATION_H

#ifndef SOM_OPERATOR_H
#define SOM_OPERATOR_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
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

#define CELL_REFRESH_RATE  1

using namespace std;
using namespace components;

namespace operators
{

template<class CellularMatrix, class Grid1, class Grid2>
void evaluateSPdensityCost(CellularMatrix &cmd, Grid1 &gSize, Grid2 &gWeight)
{
    // Edge density evaluation
    GLfloat ave = cmd.totalDensity / (float)(gSize.getWidth() * gSize.getHeight());
    GLfloat sum = 0.0f;
    Grid<GLfloat> gSizeTemp;
    Grid<GLfloat> gWeightTemp;
    int width = gSize.getWidth();
    int height = gSize.getHeight();
    gSizeTemp.resize(width, height);
    gWeightTemp.resize(width, height);
    gSizeTemp.gpuCopyDeviceToHost(gSize);
    gWeightTemp.gpuCopyDeviceToHost(gWeight);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            GLfloat den = gWeightTemp[y][x] * gSizeTemp[y][x];
            sum += ABS(den - ave);
        }
    gSizeTemp.freeMem();
    gWeightTemp.freeMem();
    cout << "The density cost is " << (sum / cmd.totalDensity) << endl;
}

template<class CellularMatrix, class Grid, class GetAdaptor>
void evaluateSPcolorCost(CellularMatrix &gCenter, Grid &gColor, Grid &gtColor, GetAdaptor getAdaptor)
{
    // Edge density evaluation
    CellularMatrix gCenterTemp;
    Grid gColorTemp;
    int w_gCenter = gCenter.getWidth();
    int h_gCenter = gCenter.getHeight();
    int w_gtColor = gtColor.getWidth();
    int h_gtColor = gtColor.getHeight();
    gCenterTemp.resize(w_gCenter, h_gCenter);
    gColorTemp.resize(w_gCenter, h_gCenter);
    gCenterTemp.gpuCopyDeviceToHost(gCenter);
    gColorTemp.gpuCopyDeviceToHost(gColor);
    float error(0.0f);
    DistanceEuclidean<Point3D> distance;

    for (int _y = 0; _y < h_gCenter; _y++)
        for (int _x = 0; _x < w_gCenter; _x++)
        {
            getAdaptor.init(gCenterTemp[_y][_x]);
            do {
                PointCoord PC(_x, _y);
                // Partition
                PointCoord ps;
                bool extracted = false;
                extracted = getAdaptor.get(gCenterTemp[_y][_x],
                                           ps);
                if (extracted) {
                    if (ps[0] >= 0 && ps[0] < w_gtColor && ps[1] >= 0 && ps[1] < h_gtColor)
                    error += distance(gColorTemp[_y][_x], gtColor[ps[1]][ps[0]]);
                }
            } while (getAdaptor.next(gCenterTemp[_y][_x]));
        }

    error /= (float)(w_gtColor * h_gtColor);
    gCenterTemp.freeMem();
    gColorTemp.freeMem();
    cout << "The color cost is " << error << endl;
}

template<class Grid, class Grid2>
void evaluateSPcolorCost(Grid &gColor, Grid &gtColor, Grid2 &gLabel)
{
    // Edge density evaluation
    int w_gColor = gColor.getWidth();
    int h_gColor = gColor.getHeight();
    int w_gtColor = gtColor.getWidth();
    int h_gtColor = gtColor.getHeight();
    int w_gLabel = gLabel.getWidth();
    int h_gLabel = gLabel.getHeight();
    Grid gColorTemp;
    gColorTemp.resize(w_gColor, h_gColor);
    gColorTemp.gpuCopyDeviceToHost(gColor);
    Grid2 gLabelTemp;
    gLabelTemp.resize(w_gLabel, h_gLabel);
    gLabelTemp.gpuCopyDeviceToHost(gLabel);
    float error(0.0f);
    int count(0);
    DistanceEuclidean<Point3D> distance;

    for (int _y = 0; _y < h_gLabel; _y++)
        for (int _x = 0; _x < w_gLabel; _x++)
        {
            PointCoord PC = gLabelTemp[_y][_x];
            if (PC[0] >= 0 && PC[0] < w_gColor && PC[1] >= 0 && PC[1] < h_gColor && _x < w_gtColor && _y < h_gtColor) {
                error += distance(gColorTemp[PC[1]][PC[0]], gtColor[_y][_x]);
                count++;
            }
        }

    if (count != 0)
        error /= (float)(count);
    gColorTemp.freeMem();
    gLabelTemp.freeMem();
    cout << "The color cost is " << error << endl;
}

template<class Grid>
void evaluateSPcolorCost(Grid &gColor, Grid &gtColor, float &__error__)
{
    // Edge density evaluation
    int w = gtColor.getWidth();
    int h = gtColor.getHeight();
    Grid gColorTemp;
    gColorTemp.resize(w, h);
    gColorTemp.gpuCopyDeviceToHost(gColor);
    float error(0.0f);
    DistanceEuclidean<Point3D> distance;

    for (int _y = 0; _y < h; _y++)
        for (int _x = 0; _x < w; _x++)
        {
            error += distance(gColorTemp[_y][_x], gtColor[_y][_x]);
        }

    error /= (float)(w * h);
    gColorTemp.freeMem();
    cout << "The color cost is " << error << endl;
    __error__ = error;
}

/*!
 * \brief K_SO_debug
 *
 */
template <class Operator>
KERNEL void K_SO_debug(Operator nn_cible)
{
    KER_SCHED(nn_cible.gGCenter.getWidth(), nn_cible.gGCenter.getHeight())

    if (_x == 0 && _y == 0)
    {
        printf("I am here in K_SO_debug ! \n");
        for (int y = 10; y < 15; y++)
        for (int x = 10; x < 11; x++)
        for (int i = 0; i < 5; i++)
        {
            printf("bCell.length = %d, gGCenter[%d][%d].bCell[%d] = %d, %d,  size = %d,  value = %f,%f,  gSize = %f, gWeight = %f, gColor = %f,%f,%f \n",
                   nn_cible.gGCenter[y][x].bCell.length,
                   y, x, i,
                   nn_cible.gGCenter[y][x].bCell[i][0],
                   nn_cible.gGCenter[y][x].bCell[i][1],
                   nn_cible.gGCenter[y][x].size,
                   nn_cible.gGCenter[y][x][0],
                   nn_cible.gGCenter[y][x][1],
                   nn_cible.gSize[y][x],
                   nn_cible.gWeight[y][x],
                   nn_cible.gColor[y][x][0], nn_cible.gColor[y][x][1], nn_cible.gColor[y][x][2]);
        }
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_debug

template <class Operator>
KERNEL void K_SO_debugNN(Operator nn_cible)
{
    KER_SCHED(nn_cible.adaptiveMap.getWidth(), nn_cible.adaptiveMap.getHeight())

    if (_x == 0 && _y == 0)
    {
        printf("I am here in K_SO_debugNN ! \n");
        for (int y = 0; y < 5; y++)
        for (int x = 0; x < 10; x++)
        {
            printf("adaptiveMap[%d][%d] = %f, %f,  colorMap = %f, %f, %f \n",
                   y, x, nn_cible.adaptiveMap[y][x][0], nn_cible.adaptiveMap[y][x][1],
                   nn_cible.colorMap[y][x][0], nn_cible.colorMap[y][x][1], nn_cible.colorMap[y][x][2]);
        }
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_debugNN

template <class Operator>
KERNEL void K_SO_debugNoBuffer(Operator nn_cible)
{
    KER_SCHED(nn_cible.gGCenter.getWidth(), nn_cible.gGCenter.getHeight())

    if (_x == 0 && _y == 0)
    {
        printf("I am here in K_SO_debugNoBuffer ! \n");
        for (int y = 0; y < 5; y++)
        for (int x = 0; x < 10; x++)
        {
            printf("gGCenter[%d][%d] = %f, %f,  gSize = %f, gWeight = %f, gColor = %f,%f,%f \n",
                   y, x,
                   nn_cible.gGCenter[y][x][0],
                   nn_cible.gGCenter[y][x][1],
                   nn_cible.gSize[y][x],
                   nn_cible.gWeight[y][x],
                   nn_cible.gColor[y][x][0], nn_cible.gColor[y][x][1], nn_cible.gColor[y][x][2]);
        }
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_debugNoBuffer

template <class Grid>
KERNEL void K_SO_debugGrid(Grid g)
{
    KER_SCHED(g.getWidth(), g.getHeight())

    if (_x == 0 && _y == 0)
    {
        printf("I am here in K_SO_debugGrid ! \n");
        for (int y = 50; y < 55; y++)
        for (int x = 50; x < 55; x++)
        {
            printf("g[%d][%d] = %d, %d \n", y, x, g[y][x][0], g[y][x][1]);
        }

        printf("g[%d][%d] = %d, %d \n", 0, 0, g[0][0][0], g[0][0][1]);
        printf("g[%d][%d] = %d, %d \n", 0, 10, g[0][10][0], g[0][10][1]);
        printf("g[%d][%d] = %d, %d \n", 10, 10, g[10][10][0], g[10][10][1]);
        printf("g[%d][%d] = %d, %d \n", 120, 120, g[120][120][0], g[120][120][1]);
        printf("g[%d][%d] = %d, %d \n", 230, 230, g[230][230][0], g[230][230][1]);
        printf("g[%d][%d] = %d, %d \n", 300, 100, g[300][100][0], g[300][100][1]);
        printf("g[%d][%d] = %d, %d \n", 40, 200, g[40][200][0], g[40][200][1]);
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_debugGrid





/*!
 * \brief K_SO_projector
 *
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNet, class Cell,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_SO_projector(CellularMatrix cm_source,
                           CellularMatrix2 cm_cible,
                           NN nn_source,
                           NeuralNet<Cell, GLfloat> nn_cible,
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
            bool extracted = false;
            extracted = getAdaptor.get(cm_source[_y][_x],
                                       nn_source,
                                       ps);

            if (extracted) {
                // Spiral search
                PointCoord minP;
                bool found = false;

                found = searchAdaptor.search(cm_cible,
                                             nn_source,
                                             (NN&)nn_cible,
                                             PC,
                                             ps,
                                             minP);

                if (found) {
                    operateAdaptor.operate(cm_cible[_y][_x],
                                           nn_source,
                                           nn_cible,
                                           ps,
                                           minP);
                }
            }

        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_projector

/*!
 * \brief K_SO_projectorBatch
 *
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNet, class Cell,
          class GetAdaptor,
          class SearchAdaptor
          >
KERNEL void K_SO_projectorBatch(CellularMatrix cm_source,
                           CellularMatrix2 cm_cible,
                           NN nn_source,
                           NeuralNet<Cell, GLfloat> nn_cible,
                           GetAdaptor getAdaptor,
                           SearchAdaptor searchAdaptor)
{
    KER_SCHED(cm_source.getWidth(), cm_source.getHeight())

    if (_x < cm_source.getWidth() && _y < cm_source.getHeight())
    {
        getAdaptor.init(cm_source[_y][_x]);
        do {
            PointCoord PC(_x, _y);

            // Partition
            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cm_source[_y][_x],
                                       nn_source,
                                       ps);

            if (extracted) {
                // Spiral search
                PointCoord minP;
                bool found = false;

                found = searchAdaptor.search(cm_cible,
                                             nn_source,
                                             (NN&)nn_cible,
                                             PC,
                                             ps,
                                             minP);
            }
        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_projectorBatch

//! HW 23/05/15 : overload for superpixel application
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNet, class Cell,
          class GetAdaptor,
          class SearchAdaptor
          >
KERNEL void K_SO_projectorBatch(CellularMatrix cm_source,
                           CellularMatrix2 cm_cible,
                           NN nn_source,
                           NeuralNet<Cell, GLfloat> nn_cible,
                           Grid<PointCoord> gLabel,
                           GetAdaptor getAdaptor,
                           SearchAdaptor searchAdaptor)
{
    KER_SCHED(cm_source.getWidth(), cm_source.getHeight())

    if (_x < cm_source.getWidth() && _y < cm_source.getHeight())
    {
        getAdaptor.init(cm_source[_y][_x]);
        do {
            PointCoord PC(_x, _y);

            // Partition
            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get(cm_source[_y][_x],
                                       nn_source,
                                       ps);

            if (extracted) {
                // Spiral search
                PointCoord minP;
                bool found = false;

                found = searchAdaptor.search(cm_cible,
                                             nn_source,
                                             (NN&)nn_cible,
                                             PC,
                                             ps,
                                             minP,
                                             gLabel);
            }
        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_projectorBatch


/*!
 * \brief K_SO_injector
 *
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNet, class Cell,
          class GetAdaptor,
          class OperateAdaptor
          >
KERNEL void K_SO_injector(CellularMatrix cm_source,
                           CellularMatrix2 cm_cible,
                           NeuralNet<Cell, GLfloat> nn_source,
                           NN nn_cible,
                           GetAdaptor getAdaptor,
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
            bool extracted = false;
            extracted = getAdaptor.get(cm_source[_y][_x],
                                       (NN&)nn_source,
                                       ps);

            if (extracted) {

                operateAdaptor.operate(cm_cible[_y][_x],
                                       (NN&)nn_source,
                                       nn_cible,
                                       PC,
                                       ps);
            }
        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_injector

//! HW 23/05/15 : overload for superpixel application
template <
          template<typename, typename> class NeuralNet, class Cell,
          class OperateAdaptor
          >
KERNEL void K_SO_injector(NeuralNet<Cell, GLfloat> nn_source,
                          NN nn_cible,
                          Grid<PointCoord> gLabel,
                          OperateAdaptor operateAdaptor)
{
    KER_SCHED(nn_cible.adaptiveMap.getWidth(), nn_cible.adaptiveMap.getHeight())

    if (_x < nn_cible.adaptiveMap.getWidth() && _y < nn_cible.adaptiveMap.getHeight())
    {
        PointCoord ps(_x,_y);
        PointCoord PC = gLabel[_y][_x];

        operateAdaptor.operate((NN&)nn_source,
                               nn_cible,
                               PC,
                               ps);
    }

    END_KER_SCHED

    SYNCTHREADS

}//K_SO_injector

template <class CellularMatrix,
          template<typename, typename> class NeuralNet, class Cell,
          class GetAdaptor,
          class OperateAdaptor
          >
KERNEL void K_SO_injectorDebug(CellularMatrix cm_source,
                           NeuralNet<Cell, GLfloat> nn_source,
                           NN nn_cible,
                           GetAdaptor getAdaptor,
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
            bool extracted = false;
            extracted = getAdaptor.get(cm_source[_y][_x],
                                       (NN&)nn_source,
                                       ps);

            if (extracted) {

                operateAdaptor.operate((NN&)nn_source,
                                       nn_cible,
                                       PC,
                                       ps);
            }
        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_injector

/*!
 * \brief K_SO_projector
 *
 */
template <template<typename, typename> class NeuralNet1, class Cell1,
          template<typename, typename> class NeuralNet2, class Cell2,
          class OperateAdaptor
          >
KERNEL void K_SO_trigger(NeuralNet1<Cell1, GLfloat> nn_source,
                         NeuralNet2<Cell2, GLfloat> nn_cible,
                         OperateAdaptor operateAdaptor)
{
    KER_SCHED(nn_source.adaptiveMap.getWidth(), nn_source.adaptiveMap.getHeight())

    if (_x < nn_source.adaptiveMap.getWidth() && _y < nn_source.adaptiveMap.getHeight())
    {
        PointCoord pCoord(_x,_y);

        //! HW 15/04/15 : add (NN&) type cast
        operateAdaptor.operate((NN&)nn_source,
                               nn_cible,
                               pCoord,
                               pCoord);
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_trigger

//! HW 13/04/15 : add K_SO_labelSet
/*!
 * \brief K_SO_labelSet
 *
 */
template < template<typename> class Grid1,
           class Cell1,
           template<typename> class Grid2,
           class Node2,
           class GetAdaptor
           >
KERNEL void K_SO_labelSet(Grid1<Cell1> cm_source,
                          Grid2<Node2> g_cible,
                          GetAdaptor getAdaptor)
{
    KER_SCHED(cm_source.getWidth(), cm_source.getHeight())

    if (_x < cm_source.getWidth() && _y < cm_source.getHeight())
    {
        PointCoord PC(_x, _y);
        getAdaptor.init((Cell1&)cm_source[_y][_x]);
        do {
            PointCoord ps;
            bool extracted = false;
            extracted = getAdaptor.get((Cell1&)cm_source[_y][_x], ps);
            if (extracted)
                g_cible[ps[1]][ps[0]] = PC;
        } while (getAdaptor.next((Cell1&)cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_labelSet

//! HW 22/05/15 : add K_SO_cleanCellsInGrid
/*!
 * \brief K_SO_cleanCellsInGrid
 *
 */
template <class Cell>
KERNEL void K_SO_cleanCellsInGrid(Grid<Cell> gCell)
{
    KER_SCHED(gCell.getWidth(), gCell.getHeight())

    if (_x < gCell.getWidth() && _y < gCell.getHeight())
    {
        gCell[_y][_x].clearCell();
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_SO_cleanCellsInGrid

//! Type of comportment for decreasing parameters
enum TypeWaveAlpha {
    TYPE_DOWN_PARAM_KOHONEN,
    TYPE_UP_PARAM_KOHONEN,
    TYPE_DOWN_WAVE_PARAM_KOHONEN,
    TYPE_UP_WAVE_PARAM_KOHONEN
};

//! Type of comportment for decreasing parameters
enum ModeCalcul {
    SO_ONLINE,
    SO_BATCH,
    SO_ONLINE_SEG,
    SO_BATCH_SEG,
    SO_BATCH_SEG_SAMPLING,
    SO_ONLINE_TSP
};

//! Type of comportment for decreasing parameters
struct TSomParams {
    GLfloat alphaInitial;
    GLfloat alphaFinal;

    GLfloat rInitial;
    GLfloat rFinal;

    int niter;
    size_t nGene;

    ModeCalcul modeCalcul;//online/batch

    bool buffered;//the matcher is buffered via the savgab

//! Wave
    TypeWaveAlpha typeWaveAlpha;

    DEVICE_HOST TSomParams() {
        typeWaveAlpha = TYPE_DOWN_PARAM_KOHONEN;
        alphaInitial = 1.0;
        alphaFinal = 0.1;
        rInitial = 10;
        rFinal = 0.5;
        niter = 1;
        nGene = 10;
        modeCalcul = SO_ONLINE;
        buffered = false;
    }

    /*!
     * \brief readParameters
     * \param name
     */
    DEVICE_HOST void readParameters(std::string const& name, ConfigParams& params) {
        params.readConfigParameter(name,"alphaInitial", alphaInitial);
        params.readConfigParameter(name,"alphaFinal", alphaFinal);
        params.readConfigParameter(name,"rInitial", rInitial);
        params.readConfigParameter(name,"rFinal", rFinal);
        params.readConfigParameter(name,"niter", niter);
        params.readConfigParameter(name,"nGene", nGene);
        params.readConfigParameter(name,"modeCalcul", (int&)modeCalcul);
        params.readConfigParameter(name,"typeWaveAlpha", (int&)typeWaveAlpha);
        params.readConfigParameter(name,"buffered", buffered);
    }//readParameters

};

//! Internal parameters
struct TExecSomParams {
    GLfloat alpha;
    GLfloat alphaCoeff;
    GLfloat radius;
    GLfloat radiusCoeff;
    size_t learningStep;
    //! Total iteration number
    size_t iterations;
};

/*!
 * \brief Class Som.
 * It is a mixte CPU/GPU class, then with
 * default copy constructor for allowing
 * parameter passing to GPU Kernel.
 **/
template <class CellularMatrixR,
          class CellularMatrixD,
          class CellR,
          class CellD,
          class NIter,
          class NIterDual,
          class ViewG
          >
class SomOperator
{
//protected:
public:
    //! Neural net that matches
    NN mr;
    //! Neural net that is matched
    NN md;
    //! Cellular matrix with neural net that matches
    CellularMatrixR cmr;
    //! Cellular matrix with neural net that is matched
    CellularMatrixD cmd;
    //! ViewGrid
    ViewG vgd;
    //! Distance functor
//    typename CellR::Dist dist;

    // Adaptator which needs initialize()
    GetRandomGridAdaptor<CellD> ga;
//    GetStdRandomGridAdaptor<CellD> ga;
//    GetRandomGridOppositeAdaptor<CellD> ga;
    GetStdRandomAdaptor<CellD> gsa;

    SearchSpiralComputeAvgAdaptor<
    CellularMatrixR,
    SpiralSearchCMIterator<CellR, NIterDual>,
    PointEuclid, NeuralNet
    > savga;

    SearchSpiralComputeAvgBufferAdaptor<
    CellularMatrixR,
    SpiralSearchCMIterator<CellR, NIterDual>,
    CellR, NeuralNet
    > savgab;

public:
    //! Standard Som parameters
    TSomParams somParams;
    //! Internal parameters
    TExecSomParams execParams;

public:
    DEVICE_HOST explicit SomOperator() {}

    //! \brief Constructeur par defaut.
    DEVICE_HOST explicit SomOperator(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            CellularMatrixD& cd,
            ViewG& vg,
            TSomParams& p
            ) :
        mr(nnr),
        md(nnd),
        cmr(cr),
        cmd(cd),
        vgd(vg),
        somParams(p) {}

    /*!
     * \brief initialize
     */
    GLOBAL void initialize(
            NN& nnr,
            NN& nnd,
            CellularMatrixR& cr,
            CellularMatrixD& cd,
            ViewG& vg,
            TSomParams& p
            ) {

        mr = nnr;
        md = nnd;
        cmr = cr;
        cmd = cd;
        vgd = vg;
        somParams = p;

        initialize();
    }

    /*!
     * \brief initialize
     */
    GLOBAL void initialize() {

        ga.initialize(cmd);
        gsa.initialize(cmd);
        savga.initialize(mr);
        if (somParams.buffered)
            savgab.initialize(mr);
        init();

    }//initialize()

    /*!
     * \brief free
     */
    GLOBAL void free() {

        savga.free();
        if (somParams.buffered)
            savgab.free();

    }//initialize()

    // Generation courante
    size_t gene;

    /*!
     * \brief init
     */
    void init() {
        gene = 0;
        ga.init(somParams.niter);
        gsa.init(somParams.niter);
        savga.init(mr);
        if (somParams.buffered)
        {
            savgab.init(mr);
        }
        setParams(somParams, execParams, somParams.nGene);
    }

    /*!
     * \brief run
     */
    GLOBAL void run() {
        for (int i = 0; i < somParams.nGene; i++) {
            bool ret = this->activate();
            if (!ret)
                break;
        }
    } // run()

#define EMST_PRINT_CM 0
    /*!
     * \brief activate
     * \return
     */
    GLOBAL bool activate() {
        bool ret = true;

        if (gene == somParams.nGene)
            return false;

        if (gene % CELL_REFRESH_RATE == 0) {

            K_refreshCells();
#if EMST_PRINT_CM
    cm_cpu.gpuCopyDeviceToHost(cm_gpu);
    cout << "CM gpuCopyDeviceToHost DONE" << endl;
    int numNode = 0;
    int maxSize = 0;
    IndexCM idxcm(0);
    cm_cpu.iterInit(idxcm);
    while (cm_cpu.iterNext(idxcm)) {
//        if (cm_cpu(idxcm).size > 0) {
//            cout << "cmd_cpu " <<  idxcm << endl;
//            cout << " " << cm_cpu(idxcm).size << endl;
//        }
        numNode += cm_cpu(idxcm).size;
        maxSize = (cm_cpu(idxcm).size > maxSize) ? cm_cpu(idxcm).size : maxSize;
    }
    cout << "=== check num of nodes inserted into cmd: " << numNode << " max cell size: "<< maxSize << endl;
#endif
        }

        if (somParams.modeCalcul == SO_ONLINE) {

            // Adaptators
            SearchSpiralAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > sa;
            OperateTriggerAdaptor<CellR, NIter> oa;

            // Set adaptator status
            ga.setGene(gene);
            oa.alpha = execParams.alpha;
            oa.radius = execParams.radius;

            // Project and trigger
            K_projector(cmd, cmr, md, mr, ga, sa, oa);

        } else if (somParams.modeCalcul == SO_BATCH) {

            // Refresh cells
            K_refreshCells();

            // Adaptators
            GetStdAdaptor<CellD> gsdta;
            OperateTriggerAdaptor<CellR, NIter> oa;

            // Set adaptator status
            oa.alpha = execParams.alpha;
            oa.radius = execParams.radius;
            savga.init(mr);

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gsdta, savga);

            // temp NN
            NN nn_gc;
            nn_gc.adaptiveMap = savga.gGCenter;

            // Trigger
            K_trigger(nn_gc, mr, oa);

        } else if (somParams.modeCalcul == SO_ONLINE_SEG) {

            // Adaptators
            SearchSpiralAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > sa;
            OperateTriggerAdaptorWithColor<CellR, NIter> oa;

            // Set adaptator status
            ga.setGene(gene);
            oa.alpha = execParams.alpha;
            oa.radius = execParams.radius;

            // Project and trigger
            K_projector(cmd, cmr, md, mr, ga, sa, oa);

        } else if (somParams.modeCalcul == SO_BATCH_SEG) {

            // Refresh cells
            K_refreshCells();

            // Adaptators
            GetStdAdaptor<CellD> gsdta;
            OperateTriggerAdaptorWithColor<CellR, NIter> oa;
            OperateInjectAdaptorWithColor<CellD> oia;

            // Set adaptator status
            oa.alpha = execParams.alpha;
            oa.radius = execParams.radius;
            savga.init(mr);

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gsdta, savga);

            // temp NN
            NN nn_gc;
            nn_gc.adaptiveMap = savga.gGCenter;
            nn_gc.colorMap = savga.gColor;
#ifdef WITH_DENSITY
            nn_gc.densityMap = savga.gWeight;
#endif

            // Trigger
            K_trigger(nn_gc, mr, oa);

        } else if (somParams.modeCalcul == SO_BATCH_SEG_SAMPLING) {

            // Refresh cells
            K_refreshCells();

            // Adaptators
            OperateTriggerAdaptorWithColor<CellR, NIter> oa;
            OperateInjectAdaptorWithColor<CellD> oia;

            // Set adaptator status
            gsa.setGene(gene);
            oa.alpha = execParams.alpha;
            oa.radius = execParams.radius;
            savga.init(mr);

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gsa, savga);

            // temp NN
            NN nn_gc;
            nn_gc.adaptiveMap = savga.gGCenter;
            nn_gc.colorMap = savga.gColor;

            // Trigger
            K_trigger(nn_gc, mr, oa);

        } else if (somParams.modeCalcul == SO_ONLINE_TSP) {

            // Adaptators
            SearchSpiralAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > sa;
            OperateTriggerTSPAdaptor<CellR, NIter1D> oa;

            // Set adaptator status
            ga.setGene(gene);
            oa.alpha = execParams.alpha;
            oa.radius = execParams.radius;

            // Project and trigger
            K_projector(cmd, cmr, md, mr, ga, sa, oa);

        }

        // Update Params
        updateParams(somParams, execParams);
        // Generation finished
        gene += 1;

        return ret;
    }//activate

    GLOBAL void K_refreshCells() {
        // Adaptators
        GetDivideAdaptor<CellR> gda(mr.adaptiveMap.getWidth(),
                                    mr.adaptiveMap.getHeight(),
                                    cmr.getWidth(),
                                    cmr.getHeight());
        SearchFindCellAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > sfa;
        OperateInsertAdaptor<NeuralNet, CellR> oia;

        // Insert to cmr
        NeuralNet<CellR, GLfloat> nn_cmr;
        nn_cmr.adaptiveMap.width = cmr.width;
        nn_cmr.adaptiveMap.height = cmr.height;
        nn_cmr.adaptiveMap.depth = cmr.depth;
        nn_cmr.adaptiveMap.others = cmr.others;
        nn_cmr.adaptiveMap.pitch = cmr.pitch;
        nn_cmr.adaptiveMap.extents = cmr.extents;
        nn_cmr.adaptiveMap.extents_pitch = cmr.extents_pitch;
        nn_cmr.adaptiveMap.extents_strides = cmr.extents_strides;
        nn_cmr.adaptiveMap.length_in_bytes = cmr.length_in_bytes;
        nn_cmr.adaptiveMap.length = cmr.length;
        nn_cmr.adaptiveMap._data = cmr._data;

        // Clear previous content
        cmr.K_clearCells();

        // Projection
        K_projector(cmr, cmr, mr, nn_cmr, gda, sfa, oia);

    }//K_refreshCells

    GLOBAL void K_injectVoronoiMatcherToMatched() {

        // Refresh cells
        K_refreshCells();

        // Adaptators
        GetStdAdaptor<CellD> gsdta;
        GetStdAdaptor<CellR> gsdtar;
        OperateInjectAdaptor<CellD> oia;
        OperateInjectAdaptorDebug<CellD> oiaDebug;

        if (somParams.buffered)
        {
            // Init search avg computation
            savgab.init(mr);

            // Insert to nn
            NeuralNet<CellR, GLfloat> nn_gc;
            nn_gc.adaptiveMap = savgab.gGCenter;
            nn_gc.colorMap = savgab.gColor;

            // Clean cell buffer before insertion
            K_cleanCellsInGrid(savgab.gGCenter);

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gsdta, savgab);

            // Clear md color map before Super Pixel injection
             md.colorMap.gpuMemSet(Point3D(1.0f,1.0f,1.0f));
          //   md.colorMap.gpuMemSet(Point3D(255,255,255));


            // Inject to mached
            K_injector(savgab.gGCenter, cmd, nn_gc, md, gsdtar, oia);

        }//BATCH_BUFFER
        else
        {
            // Grid of cluster center labels
            Grid<PointCoord> gpu_labelC;
            gpu_labelC.gpuResize(md.adaptiveMap.getWidth(),
                                 md.adaptiveMap.getHeight());
            // Init labels
            gpu_labelC.gpuResetValue(PointCoord(-1, -1));

            // Init search avg computation
            savga.init(mr);

            // Insert to nn
            NeuralNet<PointEuclid, GLfloat> nn_gc2;
            nn_gc2.adaptiveMap = savga.gGCenter;
            nn_gc2.colorMap = savga.gColor;

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gpu_labelC, gsdta, savga);

            // Clear md color map before Super Pixel injection
                 md.colorMap.gpuMemSet(Point3D(1.0f,1.0f,1.0f));
            //   md.colorMap.gpuMemSet(Point3D(255,255,255));

            // Inject to mached
            K_injector(nn_gc2, md, gpu_labelC, oiaDebug);
//            K_injectorDebug(cmd, md, md, gsdta, oiaDebug);

            // Free memory
            gpu_labelC.gpuFreeMem();

        }//BATCH_NO_BUFFER

    }//K_injectVoronoiMatcherToMatched

    GLOBAL void K_injectVoronoiAndDrawContoursAroundSegments(Grid<Point3D>& colorCopy) {

        // Refresh cells
        K_refreshCells();

        // Grid of cluster center labels
        Grid<PointCoord> gpu_labelC;
        gpu_labelC.gpuResize(md.adaptiveMap.getWidth(),
                             md.adaptiveMap.getHeight());
        // Init labels
        gpu_labelC.gpuResetValue(PointCoord(-1, -1));

        // Adaptators
        GetStdAdaptor<CellD> gsdta;
        GetStdAdaptor<CellR> gsdtar;

        if (somParams.buffered) {

            // Init search avg computation
            savgab.init(mr);

            // Clean cell buffer before insertion
            K_cleanCellsInGrid(savgab.gGCenter);

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gsdta, savgab);

            // Evaluation
            evaluateSPdensityCost(cmd, savgab.gSize, savgab.gWeight);
            evaluateSPcolorCost(savgab.gGCenter, savgab.gColor, colorCopy, gsdtar);

            // fill label grid
            K_labelSet(savgab.gGCenter, gpu_labelC, gsdtar);

        }//BATCH_BUFFER
        else
        {
            // Init search avg computation
            savga.init(mr);

            // Projection
            K_projectorBatch(cmd, cmr, md, mr, gpu_labelC, gsdta, savga);

            // Evaluation
            evaluateSPdensityCost(cmd, savga.gSize, savga.gWeight);
            evaluateSPcolorCost(savga.gColor, colorCopy, gpu_labelC);

        }//BATCH_NO_BUFFER

        // copy data from GPU to CPU, in order to draw contours on CPU side
        Grid<PointCoord> labelC;
        labelC.resize(md.adaptiveMap.getWidth(),
                      md.adaptiveMap.getHeight());
        labelC.gpuCopyDeviceToHost(gpu_labelC);

        // postprocessing of enforceLabelConnectivity
        Grid<PointCoord> nlabelC;
        nlabelC.resize(md.adaptiveMap.getWidth(),
                       md.adaptiveMap.getHeight());
        int numlabels;
        int minSize = 40;
//        int minSize = ((md.adaptiveMap.getWidth() * md.adaptiveMap.getWidth()) /
//                       (mr.adaptiveMap.getWidth() * mr.adaptiveMap.getWidth())) >> 2;
        enforceLabelConnectivity(labelC, nlabelC, numlabels, minSize);
        cout << "The number of superpixels is " << numlabels << endl;
        cout << "The minSize is " << minSize << endl;

        // draw contours on CPU side
        Grid<GLint> activeMapTemp;
        activeMapTemp.resize(md.activeMap.getWidth(), md.activeMap.getHeight());
        activeMapTemp.gpuCopyDeviceToHost(md.activeMap);
        drawContours(nlabelC, colorCopy, activeMapTemp);
        activeMapTemp.freeMem();

        // clean temp memories
        gpu_labelC.gpuFreeMem();
        labelC.freeMem();
        nlabelC.freeMem();

    }//K_injectVoronoiAndDrawContoursAroundSegments

    /*!
     * \brief K_projector
     *
     */
    template <class CellularMatrix,
              class CellularMatrix2,
              template<typename, typename> class NeuralNet, class Cell,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_projector(CellularMatrix& source,
                                   CellularMatrix2& cible,
                                   NN& nn_source,
                                   NeuralNet<Cell, GLfloat>& nn_cible,
                                   GetAdaptor& ga,
                                   SearchAdaptor& sa,
                                   OperateAdaptor& oa) {

        //! HW 01/04/15 : correction
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              source.getWidth(),
                              source.getHeight());
        K_SO_projector _KER_CALL_(b, t) (
                            source, cible, nn_source, nn_cible, ga, sa, oa);
    }

    /*!
     * \brief K_projectorBatch
     *
     */
    template <class CellularMatrix,
              class CellularMatrix2,
              template<typename, typename> class NeuralNet, class Cell,
              class GetAdaptor,
              class SearchAdaptor
              >
    GLOBAL inline void K_projectorBatch(CellularMatrix& source,
                                   CellularMatrix2& cible,
                                   NN& nn_source,
                                   NeuralNet<Cell, GLfloat>& nn_cible,
                                   GetAdaptor& ga,
                                   SearchAdaptor& sa) {

        //! HW 01/04/15 : correction
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              source.getWidth(),
                              source.getHeight());
        K_SO_projectorBatch _KER_CALL_(b, t) (
                            source, cible, nn_source, nn_cible, ga, sa);
    }

    //! HW 23/05/15 : overload for superpixel application
    template <class CellularMatrix,
              class CellularMatrix2,
              template<typename, typename> class NeuralNet, class Cell,
              class GetAdaptor,
              class SearchAdaptor
              >
    GLOBAL inline void K_projectorBatch(CellularMatrix& source,
                                   CellularMatrix2& cible,
                                   NN& nn_source,
                                   NeuralNet<Cell, GLfloat>& nn_cible,
                                   Grid<PointCoord>& gLabel,
                                   GetAdaptor& ga,
                                   SearchAdaptor& sa) {

        //! HW 01/04/15 : correction
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              source.getWidth(),
                              source.getHeight());
        K_SO_projectorBatch _KER_CALL_(b, t) (
                            source, cible, nn_source, nn_cible, gLabel, ga, sa);
    }

    /*!
     * \brief K_injector
     *
     */
    template <class CellularMatrix,
              class CellularMatrix2,
              template<typename, typename> class NeuralNet, class Cell,
              class GetAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_injector(CellularMatrix& source,
                                   CellularMatrix2& cible,
                                   NeuralNet<Cell, GLfloat>& nn_source,
                                   NN& nn_cible,
                                   GetAdaptor& ga,
                                   OperateAdaptor& oa) {

        //! HW 01/04/15 : correction
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              source.getWidth(),
                              source.getHeight());
        K_SO_injector _KER_CALL_(b, t) (
                            source, cible, nn_source, nn_cible, ga, oa);
    }

    //! HW 23/05/15 : overload for superpixel application
    template <
              template<typename, typename> class NeuralNet, class Cell,
              class OperateAdaptor
              >
    GLOBAL inline void K_injector(NeuralNet<Cell, GLfloat>& nn_source,
                                  NN& nn_cible,
                                  Grid<PointCoord>& gLabel,
                                  OperateAdaptor& oa) {

        //! HW 01/04/15 : correction
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_cible.adaptiveMap.getWidth(),
                              nn_cible.adaptiveMap.getHeight());
        K_SO_injector _KER_CALL_(b, t) (nn_source, nn_cible, gLabel, oa);
    }

    //! HW 25/05/15 : add for debug
    template <class CellularMatrix,
              template<typename, typename> class NeuralNet, class Cell,
              class GetAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_injectorDebug(CellularMatrix& source,
                                  NeuralNet<Cell, GLfloat>& nn_source,
                                  NN& nn_cible,
                                  GetAdaptor& ga,
                                  OperateAdaptor& oa) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              source.getWidth(),
                              source.getHeight());
        K_SO_injectorDebug _KER_CALL_(b, t) (
                            source, nn_source, nn_cible, ga, oa);
    }

    /*!
     * \brief K_trigger
     *
     */
    template <template<typename, typename> class NeuralNet1, class Cell1,
              template<typename, typename> class NeuralNet2, class Cell2,
              class OperateAdaptor
              >
    GLOBAL inline void K_trigger(NeuralNet1<Cell1, GLfloat>& nn_source,
                                 NeuralNet2<Cell2, GLfloat>& nn_cible,
                                 OperateAdaptor& oa) {

        //! HW 01/04/15 : correction
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_source.adaptiveMap.getWidth(),
                              nn_source.adaptiveMap.getHeight());
        K_SO_trigger _KER_CALL_(b, t) (nn_source,
                                       nn_cible,
                                       oa);
    }

    //! HW 13/04/15 : add K_labelSet
    /*!
     * \brief K_labelSet
     *
     */
//    template <class CellularMatrix,
//              class Grid,
//              class GetAdaptor>
//    GLOBAL inline void K_labelSet(CellularMatrix& source,
//                                  Grid& cible,
//                                  GetAdaptor& ga) {
    template < template<typename> class Grid1,
               class Cell1,
               template<typename> class Grid2,
               class Node2,
               class GetAdaptor
               >
    GLOBAL inline void K_labelSet(Grid1<Cell1>& source,
                                  Grid2<Node2>& cible,
                                  GetAdaptor& ga) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              source.getWidth(),
                              source.getHeight());
        K_SO_labelSet _KER_CALL_(b, t) (source, cible, ga);
    }

    //! HW 22/05/15 : add K_cleanCellsInGrid
    /*!
     * \brief K_cleanCellsInGrid
     *
     */
    GLOBAL inline void K_cleanCellsInGrid(Grid<CellR>& gCell) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              gCell.getWidth(),
                              gCell.getHeight());
        K_SO_cleanCellsInGrid _KER_CALL_(b, t) (gCell);
    }


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
    GLOBAL inline void K_debugNN(NN& nn_cible) {

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



    /*!
     * \brief setParams
     */
    DEVICE_HOST void setParams(TSomParams& somParams, TExecSomParams& execParams, size_t iterations) {

        execParams.iterations = iterations;
        execParams.learningStep = 0;

        switch (somParams.typeWaveAlpha) {
        case TYPE_DOWN_PARAM_KOHONEN:
            execParams.radius = somParams.rInitial;
            execParams.alpha = somParams.alphaInitial;
            break;
        case TYPE_UP_PARAM_KOHONEN:
            execParams.radius = somParams.rFinal;
            execParams.alpha = somParams.alphaFinal;
            break;
        case TYPE_UP_WAVE_PARAM_KOHONEN:
            execParams.radius = somParams.rFinal;
            execParams.alpha = somParams.alphaFinal;
            break;
        case TYPE_DOWN_WAVE_PARAM_KOHONEN:
            execParams.radius = somParams.rInitial;
            execParams.alpha = somParams.alphaInitial;
            break;
        default:
            execParams.radius = somParams.rInitial;
            execParams.alpha = somParams.alphaInitial;
            break;
        }
        if (somParams.rFinal != somParams.rInitial) {
            execParams.radiusCoeff = exp(log(somParams.rFinal / somParams.rInitial)
                             / (GLfloat) execParams.iterations);
        } else {
            execParams.radiusCoeff = 1;
        }
        if (somParams.alphaFinal != somParams.alphaInitial) {
            execParams.alphaCoeff = exp(log(somParams.alphaFinal / somParams.alphaInitial)
                             / (GLfloat) execParams.iterations);
        } else {
            execParams.alphaCoeff = 1;
        }
    }//setParams

    /*!
     * \brief updateParams
     */
    DEVICE_HOST void updateParams(TSomParams& somParams, TExecSomParams& execParams) {
        execParams.learningStep++;
        if (execParams.learningStep >= execParams.iterations) {
            //setParams(somParams, execParams, execParams.iterations);
        } else {
            switch (somParams.typeWaveAlpha) {
            case TYPE_DOWN_PARAM_KOHONEN:
                execParams.alpha = execParams.alpha * execParams.alphaCoeff;
                execParams.radius = execParams.radius * execParams.radiusCoeff;
                break;
            case TYPE_UP_PARAM_KOHONEN:
                execParams.alpha = std::min(execParams.alpha * (2 - execParams.alphaCoeff), somParams.alphaInitial);
                execParams.radius = std::min(execParams.radius * (2 - execParams.radiusCoeff), somParams.rInitial);
                break;
            case TYPE_UP_WAVE_PARAM_KOHONEN:
                if (execParams.learningStep > execParams.iterations / 2) {
                    execParams.alpha = execParams.alpha * execParams.alphaCoeff;
                    execParams.radius = execParams.radius * execParams.radiusCoeff;
                } else {
                    execParams.alpha = std::min(execParams.alpha * (2 - execParams.alphaCoeff), somParams.alphaInitial);
                    execParams.radius = std::min(execParams.radius * (2 - execParams.radiusCoeff), somParams.rInitial);
                }
                break;
            case TYPE_DOWN_WAVE_PARAM_KOHONEN:
                if (execParams.learningStep > execParams.iterations / 2) {
                    execParams.alpha = std::min(execParams.alpha * (2 - execParams.alphaCoeff), somParams.alphaInitial);
                    execParams.radius = std::min(execParams.radius * (2 - execParams.radiusCoeff), somParams.rInitial);
                } else {
                    execParams.alpha = execParams.alpha * execParams.alphaCoeff;
                    execParams.radius = execParams.radius * execParams.radiusCoeff;
                }
                break;
            default:
                execParams.alpha = execParams.alpha * execParams.alphaCoeff;
                execParams.radius = execParams.radius * execParams.radiusCoeff;
                break;
            }
        }
    }//setParams

    //! HW 21/04/15 : add enforceLabelConnectivity
    //!	EnforceLabelConnectivity:
    //!   1. finding an adjacent label for each new component at the start
    //!	  2. if a certain component is too small, assigning the previously found
    //!	     adjacent label to this component, and not incrementing the label.
    //! Note that this function only needs to be executed once as postprocessing
    //! and it is better to be implemented on CPU side.
    //! ======================================================================
    //! Code courtesy from:
    //! SLIC.cpp: implementation of the SLIC class. Radhakrishna Achanta 2012
    //! ======================================================================

    /*!
     * \brief enforceLabelConnectivity
     *
     */
    HOST void enforceLabelConnectivity(
            Grid<PointCoord>&	labels,//input labels that need to be corrected to remove stray labels
            Grid<PointCoord>&	nlabels,//new labels
            int&		        numlabels,//the number of labels changes in the end if segments are removed
            const int&          K) //the number of superpixels desired by the user
    {
        const int dx4[4] = {-1,  0,  1,  0};
        const int dy4[4] = { 0, -1,  0,  1};
        int width = labels.getWidth();
        int height = labels.getHeight();
        const int sz = width*height;
        nlabels.resetValue(PointCoord(-1));
        int label(0);
        int* xvec = new int[sz];
        int* yvec = new int[sz];
        int adjlabel(0);//adjacent label
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                if (0 > nlabels[j][k][0])
                {
                    nlabels[j][k] = PointCoord(label, label);
                    //--------------------
                    // Start a new segment
                    //--------------------
                    xvec[0] = k;
                    yvec[0] = j;
                    //-------------------------------------------------------
                    // Quickly find an adjacent label for use later if needed
                    //-------------------------------------------------------
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if ((x >= 0 && x < width) && (y >= 0 && y < height))
                        {
                            if (nlabels[y][x][0] >= 0)
                                adjlabel = nlabels[y][x][0];
                        }
                    }

                    int count(1);
                    for (int c = 0; c < count; c++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            int x = xvec[c] + dx4[n];
                            int y = yvec[c] + dy4[n];

                            if ((x >= 0 && x < width) && (y >= 0 && y < height))
                            {
                                if ((0 > nlabels[y][x][0]) && (labels[j][k][0] == labels[y][x][0] && labels[j][k][1] == labels[y][x][1]))
                                {
                                    xvec[count] = x;
                                    yvec[count] = y;
                                    nlabels[y][x] = PointCoord(label, label);
                                    count++;
                                }
                            }

                        }
                    }
                    //-------------------------------------------------------
                    // If segment size is less then a limit, assign an
                    // adjacent label found before, and decrement label count.
                    //-------------------------------------------------------
                    if (count <= K)
                    {
                        for (int c = 0; c < count; c++)
                        {
                            nlabels[yvec[c]][xvec[c]] = PointCoord(adjlabel, adjlabel);
                        }
                        label--;
                    }
                    label++;
                }//if (0 > nlabels[j][k][0])
            }
        }
        numlabels = label;
        if(xvec) delete [] xvec;
        if(yvec) delete [] yvec;
    }//enforceLabelConnectivity


    //! HW 13/04/15 : add drawContours
    //! Note that this function only needs to be executed once and
    //! it is better to be implemented on CPU side.
    //! ======================================================================
    //! Code courtesy from:
    //! SLIC.cpp: implementation of the SLIC class. Radhakrishna Achanta 2012
    //! ======================================================================
    /*!
     * \brief drawContours
     *
     */
    template <class Grid1, class Grid2, class Grid3>
    HOST inline void drawContours(Grid1& labels, Grid2& colorMap, Grid3& edgeMap)
    {
        const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
        const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
        int width = labels.getWidth();
        int height = labels.getHeight();
        int sz = width * height;
        vector<bool> istaken(sz, false);
        vector<int> contourx(sz);
        vector<int> contoury(sz);
        int mainindex(0);
        int hitEdge(0);
        int cind(0);
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                int np(0);
                for (int i = 0; i < 8; i++)
                {
                    int x = k + dx8[i];
                    int y = j + dy8[i];
                    if ((x >= 0 && x < width) && (y >= 0 && y < height))
                    {
                        int index = y * width + x;
                        if (false == istaken[index])//comment this to obtain internal contours
                        {
                            if (!(labels[j][k][0] == labels[y][x][0] && labels[j][k][1] == labels[y][x][1]))
                                np++;
                        }
                    }
                }
                if (np > 1)
                {
                    contourx[cind] = k;
                    contoury[cind] = j;
                    istaken[mainindex] = true;
                    cind++;

                    if (edgeMap[j][k])
                        hitEdge++;
                }
                mainindex++;
            }
        }
        int numboundpix = cind; //int(contourx.size());

        cout << "cmd.edgeCM = " << cmd.edgeNumCM << ", hitNum = " << hitEdge << ", hitRate = " <<
                ((double)(hitEdge) / (double)(cmd.edgeNumCM)) << endl;

        for (int j = 0; j < numboundpix; j++)
        {
              colorMap[contoury[j]][contourx[j]] = Point3D(1.0f,1.0f,1.0f);
           // colorMap[contoury[j]][contourx[j]] = Point3D(255,255,255);
            for (int n = 0; n < 8; n++)
            {
                int x = contourx[j] + dx8[n];
                int y = contoury[j] + dy8[n];
                if ((x >= 0 && x < width) && (y >= 0 && y < height))
                {
                    int ind = y * width + x;
                    if (!istaken[ind])
                          colorMap[y][x] = Point3D(0.0f,0.0f,0.0f);
                   //     colorMap[y][x] = Point3D(255,255,255);

                }
            }
        }
    }//drawContours

};//Class SomOperator

}//namespace operators

#endif // SOM_OPERATOR_H

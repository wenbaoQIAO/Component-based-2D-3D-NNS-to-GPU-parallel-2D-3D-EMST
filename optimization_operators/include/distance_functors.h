#ifndef DISTANCE_FUNCTORS_H
#define DISTANCE_FUNCTORS_H
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
#ifdef CUDA_CODE
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

using namespace std;
using namespace components;

namespace operators
{

struct CM_ConditionTrue
{
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return true;
    }
};

/*!
 * \brief The DistanceEuclidean struct
 * Basic functor for Euclidean distance
 */

struct CM_DistanceEuclidean
{
    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pp1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<PointEuclid>()(pp1, pp2);
    }

    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NNP3D& nn1, NNP3D& nn2)
    {
        Point3D pp1 = nn1.adaptiveMap[p1[1]][p1[0]];
        Point3D pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<Point3D>()(pp1, pp2);
    }

    DEVICE_HOST inline GLfloat operator()(PointEuclid& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<PointEuclid>()(p1, pp2);
    }

    DEVICE_HOST inline GLfloat operator()(Point3D& p1, PointCoord& p2, NNP3D& nn1, NNP3D& nn2)
    {
        Point3D pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<Point3D>()(p1, pp2);
    }
};

struct CM_DistanceSquaredEuclidean
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pp1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceSquaredEuclidean<PointEuclid>()(pp1, pp2);
    }

    //! HW 25.04.15 : functor override
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointEuclid& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceSquaredEuclidean<PointEuclid>()(p1, pp2);
    }
};

struct CM_ConditionNotFixed
{
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return (!nn1.fixedMap[p1[1]][p1[0]] && !nn2.fixedMap[p2[1]][p2[0]]);
    }
};

struct CM_ConditionActive
{
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return (!nn1.activeMap[p1[1]][p1[0]] && !nn2.activeMap[p2[1]][p2[0]]);
    }

    //! HW 25.04.15 : functor override
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointEuclid& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return (!nn2.activeMap[p2[1]][p2[0]]);
    }
};

struct CM_ColorDistanceEuclidean
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        Point3D pp1 = nn1.colorMap[p1[1]][p1[0]];
        Point3D pp2 = nn2.colorMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<Point3D>()(pp1, pp2);
    }
};

struct CM_ColorDistanceSquaredEuclidean
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        Point3D pp1 = nn1.colorMap[p1[1]][p1[0]];
        Point3D pp2 = nn2.colorMap[p2[1]][p2[0]];
        return components::DistanceSquaredEuclidean<Point3D>()(pp1, pp2);
    }
};

//! HW 08/04/15 : Super-pixel segmentation parameters
//! Normally, sigmaC is set to the maximum color difference of any two pixels within the expected super-pixel;
//! and sigmaS is set to the maximum spatial distance of any two pixels within the expected super-pixel.
struct SuperPixelParams {
    GLfloat sigmaC;
    GLfloat sigmaS;
    DEVICE_HOST SuperPixelParams() {}
    DEVICE_HOST SuperPixelParams(GLfloat sc, GLfloat ss) : sigmaC(sc), sigmaS(ss) {}
};

#define SIGMA_C (0.5f)
#define SIGMA_S 40.0f

struct CM_SuperPixelDistance
{
    SuperPixelParams para;

    DEVICE_HOST CM_SuperPixelDistance()
    {
        para = SuperPixelParams(SIGMA_C, SIGMA_S);
    }

    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointCoord& p, PointCoord& pKC, NN& nn)
    {
        Point3D pColor = nn.colorMap[p[1]][p[0]];
        Point3D pColorKC = nn.colorMap[pKC[1]][pKC[0]];
        PointEuclid pSpati = nn.adaptiveMap[p[1]][p[0]];
        PointEuclid pSpatiKC = nn.adaptiveMap[pKC[1]][pKC[0]];
        float distColor = components::DistanceEuclidean<Point3D>()(pColor, pColorKC);
        float distSpati = components::DistanceEuclidean<PointEuclid>()(pSpati, pSpatiKC);
        distColor /= para.sigmaC;
        distSpati /= para.sigmaS;
        return (distColor + distSpati);
    }

    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        Point3D pColor1 = nn1.colorMap[p1[1]][p1[0]];
        Point3D pColor2 = nn2.colorMap[p2[1]][p2[0]];
        PointEuclid pSpati1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pSpati2 = nn2.adaptiveMap[p2[1]][p2[0]];
        float distColor = components::DistanceEuclidean<Point3D>()(pColor1, pColor2);
        float distSpati = components::DistanceEuclidean<PointEuclid>()(pSpati1, pSpati2);
//        cout << "distColor 1 = " << distColor << ", distSpati 1 = " << distSpati;
        distColor /= para.sigmaC;
        distSpati /= para.sigmaS;
//        cout << "distColor 2 = " << distColor << ", distSpati 2 = " << distSpati << endl;
        return (distColor + distSpati);
    }
};

struct CM_SuperPixelSquaredDistance
{
    SuperPixelParams para;

    DEVICE_HOST CM_SuperPixelSquaredDistance()
    {
        para = SuperPixelParams(SIGMA_C, SIGMA_S);
    }

    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointCoord& p, PointCoord& pKC, NN& nn)
    {
        Point3D pColor = nn.colorMap[p[1]][p[0]];
        Point3D pColorKC = nn.colorMap[pKC[1]][pKC[0]];
        PointEuclid pSpati = nn.adaptiveMap[p[1]][p[0]];
        PointEuclid pSpatiKC = nn.adaptiveMap[pKC[1]][pKC[0]];
        float distColor = components::DistanceSquaredEuclidean<Point3D>()(pColor, pColorKC);
        float distSpati = components::DistanceSquaredEuclidean<PointEuclid>()(pSpati, pSpatiKC);
        distColor /= (para.sigmaC * para.sigmaC);
        distSpati /= (para.sigmaS * para.sigmaS);
        return (distColor + distSpati);
    }

    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        Point3D pColor1 = nn1.colorMap[p1[1]][p1[0]];
        Point3D pColor2 = nn2.colorMap[p2[1]][p2[0]];
        PointEuclid pSpati1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pSpati2 = nn2.adaptiveMap[p2[1]][p2[0]];
        float distColor = components::DistanceSquaredEuclidean<Point3D>()(pColor1, pColor2);
        float distSpati = components::DistanceSquaredEuclidean<PointEuclid>()(pSpati1, pSpati2);
        distColor /= (para.sigmaC * para.sigmaC);
        distSpati /= (para.sigmaS * para.sigmaS);
        return (distColor + distSpati);
    }
};

//! HW 24/04/15 : The following functions are added for the expperiments of superpixel paper.
struct DistanceWeights {
    GLfloat spatial;
    GLfloat color;
    GLfloat gradient;
    DEVICE_HOST DistanceWeights() {}
    DEVICE_HOST DistanceWeights(GLfloat s, GLfloat c, GLfloat g) : spatial(s), color(c), gradient(g) {}
};

#define W_SPA_CONST 1.0
#define W_COL_CONST 0.0
#define W_GRA_CONST 0.0
#define W_SPA_IMPRO (1.0/300.0)
//#define W_SPA_IMPRO (1.0/1200.0)
//#define W_SPA_IMPRO (1.0/2700.0)
//#define W_SPA_IMPRO (1.0/4800.0)
//#define W_SPA_IMPRO 0.0
#define W_COL_IMPRO (1.0/3.0)
#define W_GRA_IMPRO 0.0

struct CM_SuperPixelDistanceConstruction
{
    DistanceWeights para;

    DEVICE_HOST CM_SuperPixelDistanceConstruction()
    {
        para = DistanceWeights(W_SPA_CONST, W_COL_CONST, W_GRA_CONST);
    }

    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pSpati1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pSpati2 = nn2.adaptiveMap[p2[1]][p2[0]];
        Point3D pColor1 = nn1.colorMap[p1[1]][p1[0]];
        Point3D pColor2 = nn2.colorMap[p2[1]][p2[0]];
        GLfloat pGrad1 = nn1.densityMap[p1[1]][p1[0]];
        GLfloat pGrad2 = nn2.densityMap[p2[1]][p2[0]];
        float distSpati = components::DistanceEuclidean<PointEuclid>()(pSpati1, pSpati2);
        float distColor = components::DistanceEuclidean<Point3D>()(pColor1, pColor2);
        float distGrad = ABS((pGrad1-pGrad2));
        return (para.spatial * distSpati + para.color * distColor + para.gradient * distGrad);
    }
};
struct CM_SuperPixelDistanceImprovement
{
    DistanceWeights para;

    DEVICE_HOST CM_SuperPixelDistanceImprovement()
    {
        para = DistanceWeights(W_SPA_IMPRO, W_COL_IMPRO, W_GRA_IMPRO);
    }

    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pSpati1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pSpati2 = nn2.adaptiveMap[p2[1]][p2[0]];
        Point3D pColor1 = nn1.colorMap[p1[1]][p1[0]];
        Point3D pColor2 = nn2.colorMap[p2[1]][p2[0]];
        GLfloat pGrad1 = nn1.densityMap[p1[1]][p1[0]];
        GLfloat pGrad2 = nn2.densityMap[p2[1]][p2[0]];
        float distSpati = components::DistanceEuclidean<PointEuclid>()(pSpati1, pSpati2);
        float distColor = components::DistanceEuclidean<Point3D>()(pColor1, pColor2);
        float distGrad = ABS((pGrad1-pGrad2));
//        cout << (para.spatial * distSpati + para.color * distColor + para.gradient * distGrad) << endl;
        return (para.spatial * distSpati + para.color * distColor + para.gradient * distGrad);
    }
};

}//namespace operators

#endif // DISTANCE_FUNCTORS_H

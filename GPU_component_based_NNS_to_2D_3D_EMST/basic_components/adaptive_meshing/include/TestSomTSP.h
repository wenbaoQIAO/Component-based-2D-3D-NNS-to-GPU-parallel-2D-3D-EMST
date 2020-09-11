#ifndef TESTSOMTSP_H
#define TESTSOMTSP_H
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

#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "ViewGrid.h"
#include "Cell.h"
#include "SpiralSearch.h"
#include "Objectives.h"
#include "Trace.h"
#include "ConfigParams.h"
#include "CellularMatrix.h"
#include "SomOperator.h"
#include "ImageRW.h"
#include "Converter.h"

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
using namespace operators;

namespace meshing
{

/*!
 * \brief The TestSom class
 */
class TestSomTSP {
    char* fileData;
    char* fileSolution;
    char* fileStats;
    ConfigParams params;
    Trace trace;

public:
    // Types
    typedef ViewGridHexa ViewG;
    typedef NIterHexa NIter;
    typedef NIterHexaDual NIterDual;

    typedef CellB<CM_DistanceEuclidean,
                    CM_ConditionTrue,
                    NIter, ViewG> CB;
    typedef CellSpS<CM_DistanceEuclidean,
                    CM_ConditionTrue,
                    NIter, ViewG> CSpS;

    typedef CellularMatrix<CSpS, ViewG> CMSpS;
    typedef CellularMatrix<CB, ViewG> CMB;

    typedef SomOperator<CMB, CMSpS, CB, CSpS, NIter, NIterDual, ViewG> Som;

    // Data
    NN md;
    NN mr;
    NN md_gpu;
    NN mr_gpu;
    ViewG vgd;
    CMSpS cmd;
    CMB cmr;

    Som som;

    TestSomTSP(char* fileData, char* fileSolution, char* fileStats, ConfigParams params) :
        fileData(fileData),
        fileSolution(fileSolution),
        fileStats(fileStats),
        params(params), vgd()
    {}

    void initialize() {
        // Data
        cout << "debut test Som ..." << endl;

        // Initialize NN netwoks CPU/GPU
        // Image read/write
        IRW irw;
        // Matched
        irw.read(fileData, md);
        size_t _w = md.colorMap.getWidth();
        size_t _h = md.colorMap.getHeight();
        md_gpu.gpuResize(_w, _h);
        md.adaptiveMap.resize(_w, _h);

        //! JCC : create GPU filters and operators
//        for (int y = 0; y < md.densityMap.getHeight(); ++y)
//            for (int x = 0; x < md.densityMap.getWidth(); ++x)
//            md.densityMap[y][x] = md.densityMap[y][x] <= MIN_MESH_DISPARITY ? 0 : md.densityMap[y][x];
        md.densityMap *= md.densityMap;
        md.densityMap *= md.densityMap;
        md_gpu.adaptiveMap.gpuResize(_w, _h);
        md.gpuCopyHostToDevice(md_gpu);

        // Matcher
        //!JCC irw.read(fileData, mr);
        int _Rr = params.levelRadiusMatcher ;
        PointCoord pcr(_w / 2, _h / 2);
        ViewG vgdr = ViewG(pcr, _w, _h, _Rr);
        size_t _wr = vgdr.getWidthBase();
        size_t _hr = vgdr.getHeightBase();

        //!JCC 190415 : test
//        _wr = _wr * _hr;
//        _hr = 1;

        cout << "matcher " << vgdr.getWidthBase() << " "
                << vgdr.getHeightBase() << endl;

        mr.resize(_wr, _hr);
        mr_gpu.gpuResize(_wr, _hr);
        //mr.adaptiveMap.resize(_wr, _hr);
        //mr.fixedMap.resize(_wr, _hr);
        mr.fixedMap = false;
        //mr.densityMap.resize(_w, _h);
        //mr.densityMap = 1;
        mr_gpu.adaptiveMap.gpuResize(_wr, _hr);
        mr_gpu.fixedMap.gpuResize(_wr, _hr);
        mr_gpu.densityMap.gpuResize(_wr, _hr);
        mr.gpuCopyHostToDevice(mr_gpu);

        // ViewGrid
        PointCoord pc(_w / 2, _h / 2);
        int _R = params.levelRadius;
        vgd = ViewG(pc, _w, _h, _R);

        cout << "vgd dual " << vgd.getWidthDual() << " "
                << vgd.getHeightDual() << endl;

        // Cellular matrix matched
        cmd.setViewG(vgd);
        cmd.gpuResize(vgd.getWidthDual(), vgd.getHeightDual());
        cmd.K_initialize(vgd);
        cmd.K_cellDensityComputation(md_gpu);
        cmd.K_initializeRegularIntoPlane<LOW_LEVEL>(vgd, md_gpu.adaptiveMap);

        // Cellular matrix matcher
        cmr.setViewG(vgd);
        cmr.gpuResize(vgd.getWidthDual(), vgd.getHeightDual());
        cmr.K_initialize(vgd);
        //cmr.K_cellDensityComputation(mr_gpu);
        cmr.K_initializeRegularIntoPlane<BASE>(vgdr, mr_gpu.adaptiveMap);

        // Trace object
        trace.initialize(fileStats);

        int testSom = 0;
        params.readConfigParameter("test_som","testSom", testSom);

        TSomParams paramSom;
        if (testSom == 0) {
            paramSom.readParameters("som_op_0", params);
        }
        else if (testSom == 1) {
            paramSom.readParameters("som_op_batch", params);
        }
        else if (testSom == 2) {
            paramSom.readParameters("som_op_seg", params);
        }
        else if (testSom == 3) {
            paramSom.readParameters("som_op_batch_seg", params);
        }
        else if (testSom == 4) {
            paramSom.readParameters("som_op_batch_seg_sampling", params);
        }
        else if (testSom == 5) {
            paramSom.readParameters("som_op_tsp", params);
        }

        cout << "Initialize" << endl;
        som.initialize(mr_gpu, md_gpu, cmr, cmd, vgd, paramSom);

    }//initialize()

    void activate() {
        som.activate();
    }

    void run() {

        // Kmean run
        cout << "Som start" << endl;
        som.run();
        cout << "Som done" << endl;
        // Get from device
        mr.gpuCopyDeviceToHost(mr_gpu);

        // Objectives
        AMObjectives objs;

        class Evaluation {
        public:
            void evaluate(AMObjectives& objs, Som& som) {
                Grid<AMObjectives> go;// = som.getDistrGrid();
                BOp bop;
                //bop.K_sumReduction(go, objs);
            }
        };

        Evaluation eval;
        eval.evaluate(objs, som);

        // Trace Writting
        trace.setObjs(objs);
        trace.writeStatistics((int)som.execParams.learningStep, cout);
        trace.writeStatistics((int)som.execParams.learningStep);
        trace.closeStatistics();

        // Save solution
        mr.write(fileSolution);
        md.write(fileData);

        //! HW 13/04/15 : draw superpixel contours
        int drawSuperpixelContours = 0;
        params.readConfigParameter("test_som","drawSuperpixelContours", drawSuperpixelContours);
        if (drawSuperpixelContours) {
            // Copy color map
            Grid<Point3D> colorCopy;
            colorCopy.resize(md.colorMap.getWidth(), md.colorMap.getHeight());
            colorCopy.gpuCopyDeviceToHost(md_gpu.colorMap);
            // Draw superpixel contours
            som.K_injectVoronoiAndDrawContoursAroundSegments(colorCopy);
            // Save superpixel image
            IRW irw;
            string str = fileData;
            int pos = irw.getPos(str);
            string s = str.substr(0, pos);
            s.insert(0, "SP");
            s.append(".colors");
            std::ofstream fo;
            fo.open(s.c_str(), ofstream::out);
            if (fo)
                fo << colorCopy;
            irw.write(s, colorCopy);
            colorCopy.freeMem();
        }

        int projectSuperPixelVoronoi = 0;
        params.readConfigParameter("test_som","projectSuperPixelVoronoi", projectSuperPixelVoronoi);
        if (projectSuperPixelVoronoi) {
            // Save SuperPixel map
            som.K_injectVoronoiMatcherToMatched();
            // Get from device
            md.gpuCopyDeviceToHost(md_gpu);
            // Save image
            IRW irw;
            string str = fileData;
            int pos = irw.getPos(str);
            string s = str.substr(0, pos);
            s.insert(0, "B");
            s.append(".colors");
            std::ofstream fo;
            fo.open(s.c_str(), ofstream::out);
            if (fo)
                fo << md.colorMap;
            irw.write(s, md);
        }
        PointCoord pc = vgd.getCenter();
        cout << pc[0] << " " << pc[1] << endl;
        PointCoord PC = vgd.F(vgd.getCenterBase());
        cout << PC[0] << " " << PC[1] << endl;
        PointCoord PCD = vgd.FDual(vgd.getCenterDual());
        cout << PCD[0] << " " << PCD[1] << endl;

        cout << vgd.FEuclid(pc)[0] << " " << vgd.FEuclid(pc)[1] << endl;
        cout << vgd.FEuclid(PC)[0] << " " << vgd.FEuclid(PC)[1] << endl;
        cout << vgd.FEuclid(PCD)[0] << " " << vgd.FEuclid(PCD)[1] << endl;
        cout << "Test Som done" << endl;
    }
};

}//namespace meshing

#endif // TESTSOMTSP_H

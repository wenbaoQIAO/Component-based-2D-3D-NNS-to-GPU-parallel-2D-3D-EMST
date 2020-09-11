#ifndef NEURALNET_H
#define NEURALNET_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang, W. Qiao
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
//#include <boost/geometry/geometries/point.hpp>

#ifdef CUDA_CODE
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

#include "macros_cuda.h"
#include "Node.h"
#include "Objectives.h"
#include "GridOfNodes.h"

#define NN_COMPLETE_RW 0 // 0 is false, != 0 is true

using namespace std;

#define NN_SFX_ADAPTIVE_MAP_IMAGE  ".points"
#define NN_SFX_COLOR_MAP_IMAGE  ".gridcolors"
#define NN_SFX_DENSITY_MAP_IMAGE  ".data"

#define NN_SFX_EVALUATION_COST_FILE  ".evaCost"
#define NN_SFX_EVALUATION_LENGTH_FILE  ".evaLength"
#define NN_SFX_EVALUATION_SMOOTH_FILE  ".evaSmooth"
#define NN_SFX_EVALUATION_GTERROR_FILE  ".evaGTerr"
#define NN_SFX_OPTICAL_FLOW_FILE  ".flo"
#define NN_SFX_DISPARITY_FILE  ".pfm"
#define TAG_FLOAT 202021.25

struct CError : public exception
{
CError(const char* msg)                 { strcpy(message, msg); }
CError(const char* fmt, int d)          { sprintf(message, fmt, d); }
CError(const char* fmt, float f)        { sprintf(message, fmt, f); }
CError(const char* fmt, const char *s)  { sprintf(message, fmt, s); }
CError(const char* fmt, const char *s,
        int d)                          { sprintf(message, fmt, s, d); }
char message[1024];         // longest allowable message
};

namespace components
{

/*!
 * \brief The NeuralNet struct
 * JCC 06/03/15 : A neural networks is a set of grids
 * that represent its different attributes
 * (position, color, active, fixed ...) where
 * we must at least find :
 * - a grid of 2D/3D Points that represent locations in euclidean plane
 * - a grid of density value.
 *
 * Example :
 * - Adaptive mesh : it is a NeuralNet<Point2D, GLfloat>
 * that moves onto 2D/3D space, with a density value
 * associated to each node.
 * - Density mesh : it is a NeuralNet<Point2D, GLfloat>
 * that represents a density distribution, where each
 * a density value is associated to each node in the plane.
 * It is generally represented as an image pixel map.
 * - Ring Mesh : it is a NeuralNet<Point2D, GLfloat>
 * organized as a ring (w=2*N) or a set of ring.
 * - City mesh : it is a set of cities implemented
 * as a grid, where each city has a location in the plane,
 * and a density value associated to it.
 *
 * Then, the Goal of Adaptive Meshing or Grid Matching Problem is to
 * realize "the" best matching of a Neural Network (or mesh) onto another
 * Neural Network (or mesh).
 *
 * Examples :
 * - Mesh onto Density Mesh (meshing pb)
 * - Ring onto City Mesh (TSP pb)
 * - Clustering k-mean : Mesh at given level to a low level mesh
 * - Image Left Mesh onto Image Right Mesh (stereo)
 * - Image First Mesh onto Second Image Mesh (flow2D)
 * - Match between four Image Meshes (flow3D)
 */
template <typename Point,
          typename Value = GLfloat>
struct NeuralNet {

    NeuralNet() {}

    //! Differential objectives
    //! that evaluate adaptation or matching
    //! k-mean distortion, length, cost, smoothing, gdtruth ...
    Grid<AMObjectives> objectivesMap;

    //! \brief High level cluster centers grid.
    //! Can be a mesh or a ring,
    //! or any grid of nodes that can move into
    //! a space of any dimension (1,2,3 usual)
    Grid<Point> adaptiveMap;

    //! Pattern of activation/fixation
    Grid<GLint> activeMap;
    Grid<GLint> fixedMap;

    //! Colored grid and gray values
    Grid<Point3D> colorMap;
    Grid<GLint> grayValueMap;

    //! \brief Low level density map for point extraction.
    //! Can be a density distribution or a grid of cities,
    //! or any grid of nodes that return a scalar intensity value
    //! or at least have a relation order <= and >= comparators
    //! for roulette wheel extraction,
    //! and relative addition + operator.
    Grid<Value> densityMap;//level 1 density map

    DEVICE_HOST explicit NeuralNet(int w, int h) :
        objectivesMap(w, h),
        adaptiveMap(w, h),
        activeMap(w, h),
        fixedMap(w, h),
        colorMap(w, h),
        grayValueMap(w, h),
        densityMap(w, h) { }

    /*! @name Globales functions specific for controling the GPU.
     * \brief Memory allocation and communication. Useful
     * for mixte utilisation.
     * @{
     */

    void clone(NeuralNet& nn) {
        objectivesMap.clone(nn.objectivesMap);
        adaptiveMap.clone(nn.adaptiveMap);
        activeMap.clone(nn.activeMap);
        fixedMap.clone(nn.fixedMap);
        colorMap.clone(nn.colorMap);
        grayValueMap.clone(nn.grayValueMap);
        densityMap.clone(nn.densityMap);
    }

    void gpuClone(NeuralNet& nn) {
        objectivesMap.gpuClone(nn.objectivesMap);
        adaptiveMap.gpuClone(nn.adaptiveMap);
        activeMap.gpuClone(nn.activeMap);
        fixedMap.gpuClone(nn.fixedMap);
        colorMap.gpuClone(nn.colorMap);
        grayValueMap.gpuClone(nn.grayValueMap);
        densityMap.gpuClone(nn.densityMap);
    }

    void setIdentical(NeuralNet& nn) {
        objectivesMap.setIdentical(nn.objectivesMap);
        adaptiveMap.setIdentical(nn.adaptiveMap);
        activeMap.setIdentical(nn.activeMap);
        fixedMap.setIdentical(nn.fixedMap);
        colorMap.setIdentical(nn.colorMap);
        grayValueMap.setIdentical(nn.grayValueMap);
        densityMap.setIdentical(nn.densityMap);
    }

    void gpuSetIdentical(NeuralNet& nn) {
        objectivesMap.gpuSetIdentical(nn.objectivesMap);
        adaptiveMap.gpuSetIdentical(nn.adaptiveMap);
        activeMap.gpuSetIdentical(nn.activeMap);
        fixedMap.gpuSetIdentical(nn.fixedMap);
        colorMap.gpuSetIdentical(nn.colorMap);
        grayValueMap.gpuSetIdentical(nn.grayValueMap);
        densityMap.gpuSetIdentical(nn.densityMap);
    }

    //! For CPU side
    void allocMem() {
        objectivesMap.allocMem();
        adaptiveMap.allocMem();
        activeMap.allocMem();
        fixedMap.allocMem();
        colorMap.allocMem();
        grayValueMap.allocMem();
        densityMap.allocMem();
    }

    void freeMem() {
        objectivesMap.freeMem();
        adaptiveMap.freeMem();
        activeMap.freeMem();
        fixedMap.freeMem();
        colorMap.freeMem();
        grayValueMap.freeMem();
        densityMap.freeMem();
    }

    void resize(int w, int h) {
        objectivesMap.resize(w, h);
        adaptiveMap.resize(w, h);
        activeMap.resize(w, h);
        fixedMap.resize(w, h);
        colorMap.resize(w, h);
        grayValueMap.resize(w, h);
        densityMap.resize(w, h);
    }

    //! For GPU side
    void gpuAllocMem() {
        objectivesMap.gpuAllocMem();
        adaptiveMap.gpuAllocMem();
        activeMap.gpuAllocMem();
        fixedMap.gpuAllocMem();
        colorMap.gpuAllocMem();
        grayValueMap.gpuAllocMem();
        densityMap.gpuAllocMem();
    }

    void gpuFreeMem() {
        objectivesMap.gpuFreeMem();
        adaptiveMap.gpuFreeMem();
        activeMap.gpuFreeMem();
        fixedMap.gpuFreeMem();
        colorMap.gpuFreeMem();
        grayValueMap.gpuFreeMem();
        densityMap.gpuFreeMem();
    }

    void gpuResize(int w, int h) {
        objectivesMap.gpuResize(w, h);
        adaptiveMap.gpuResize(w, h);
        activeMap.gpuResize(w, h);
        fixedMap.gpuResize(w, h);
        colorMap.gpuResize(w, h);
        grayValueMap.gpuResize(w, h);
        densityMap.gpuResize(w, h);
    }

    //! HOST to DEVICE
    void gpuCopyHostToDevice(NeuralNet & gpuNeuralNet) {
        objectivesMap.gpuCopyHostToDevice(gpuNeuralNet.objectivesMap);
        adaptiveMap.gpuCopyHostToDevice(gpuNeuralNet.adaptiveMap);
        activeMap.gpuCopyHostToDevice(gpuNeuralNet.activeMap);
        fixedMap.gpuCopyHostToDevice(gpuNeuralNet.fixedMap);
        colorMap.gpuCopyHostToDevice(gpuNeuralNet.colorMap);
        grayValueMap.gpuCopyHostToDevice(gpuNeuralNet.grayValueMap);
        densityMap.gpuCopyHostToDevice(gpuNeuralNet.densityMap);
    }

    //! DEVICE TO HOST
    void gpuCopyDeviceToHost(NeuralNet & gpuNeuralNet) {
        objectivesMap.gpuCopyDeviceToHost(gpuNeuralNet.objectivesMap);
        adaptiveMap.gpuCopyDeviceToHost(gpuNeuralNet.adaptiveMap);
        activeMap.gpuCopyDeviceToHost(gpuNeuralNet.activeMap);
        fixedMap.gpuCopyDeviceToHost(gpuNeuralNet.fixedMap);
        colorMap.gpuCopyDeviceToHost(gpuNeuralNet.colorMap);
        grayValueMap.gpuCopyDeviceToHost(gpuNeuralNet.grayValueMap);
        densityMap.gpuCopyDeviceToHost(gpuNeuralNet.densityMap);
    }

    //! DEVICE TO DEVICE
    void gpuCopyDeviceToDevice(NeuralNet & gpuNeuralNet) {
        objectivesMap.gpuCopyDeviceToDevice(gpuNeuralNet.objectivesMap);
        adaptiveMap.gpuCopyDeviceToDevice(gpuNeuralNet.adaptiveMap);
        activeMap.gpuCopyDeviceToDevice(gpuNeuralNet.activeMap);
        fixedMap.gpuCopyDeviceToDevice(gpuNeuralNet.fixedMap);
        colorMap.gpuCopyDeviceToDevice(gpuNeuralNet.colorMap);
        grayValueMap.gpuCopyDeviceToDevice(gpuNeuralNet.grayValueMap);
        densityMap.gpuCopyDeviceToDevice(gpuNeuralNet.densityMap);
    }
    //! @}

    int getPos(string str){

        std::size_t pos = str.find('.',  0);
        if (pos ==  std::string::npos)
            pos = str.length();

        //! no-matter aaa_xx.xx or aaa.xx or aaa_xx,  aaa.xx_xx
        //! we can always get the first three aaa and the same pos =  = 3
        std::size_t  posTiret = str.find('_',  0);

            if(pos > posTiret)
                pos = posTiret; //! ensure to get the aaa in any name format aaa_xxx or aaa_xxx.lll

        return pos;
    }

    void read(string str) {

        int pos = getPos(str);
        ifstream fi;
        //! read adaptiveMap
        string str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_ADAPTIVE_MAP_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read " << str_sub << endl; }
        fi >> adaptiveMap;
        fi.close();

        //! read colorMap
        str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_COLOR_MAP_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> colorMap;
        fi.close();

        //! read densityMap
        str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_DENSITY_MAP_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> densityMap;
        fi.close();

#if NN_COMPLETE_RW
        //! read objectivesMap
        str_sub = str.substr(0, pos);
        str_sub.append(".objectivesMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> objectivesMap;
        fi.close();

        //! read fixedMap
        str_sub = str.substr(0, pos);
        str_sub.append(".fixedMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> fixedMap;
        fi.close();

        //! read grayValueMap;
        str_sub = str.substr(0, pos);
        str_sub.append(".grayValueMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> grayValueMap;
        fi.close();

        //! read activeMap;
        str_sub = str.substr(0, pos);
        str_sub.append(".activeMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> activeMap;
        fi.close();
#endif

    }

    void write(string str) {

        int pos=getPos(str);
        string str_sub;

        ofstream fo;
        //! write adaptiveMap;
        if(adaptiveMap.width != 0 && adaptiveMap.height != 0) {
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_ADAPTIVE_MAP_IMAGE);
            fo.open(str_sub.c_str());
            if (!fo) {
                std::cout << "erreur  write " << str_sub << endl; }
            fo << adaptiveMap;
            fo.close();
        }

        //! write densityMap
        if(densityMap.width != 0 && densityMap.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_DENSITY_MAP_IMAGE);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << densityMap;
            fo.close();
        }

        //! read fixedMap
        if(fixedMap.width!=0 || fixedMap.height!=0){
            str_sub = str.substr(0, pos);

            str_sub.append(".fixedMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << fixedMap;
            fo.close();
        }

        //! write colorMap
        if(colorMap.width != 0 && colorMap.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_COLOR_MAP_IMAGE);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write " << str_sub << endl; }
            fo << colorMap;
            fo.close();
        }

#if NN_COMPLETE_RW
        //! read grayValueMap;
        if(grayValueMap.width!=0 || grayValueMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".grayValueMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << grayValueMap;
            fo.close();
        }

        //! read objectivesMap
        if(objectivesMap.width!=0 || objectivesMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".objectivesMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << objectivesMap;
            fo.close();
        }

        //! read activeMap;
        if(activeMap.width!=0 || activeMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".activeMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << activeMap;
            fo.close();
        }


#endif
    } //write

    //! HW 15.06.15 : overload for evaluation map writing
    void write(string str, AMObjNames evaType) {

        //! write objMap
        if (objectivesMap.width != 0 && objectivesMap.height != 0) {
            int pos=getPos(str);
            string str_sub;
            ofstream fo;
            int width = objectivesMap.width;
            int height = objectivesMap.height;
            Grid<GLfloat> objMapVal;
            objMapVal.resize(width, height);
            GLfloat minObjValue, maxObjValue;

            if (evaType == obj_cost) {
                minObjValue = 99999999.9;
                maxObjValue = 0.0;
                for (int _y = 0; _y < height; _y++)
                {
                    for (int _x = 0; _x < width; _x++)
                    {
//                        if (objectivesMap[_y][_x][(int)evaType] == 0) printf("(%d, %d)\n",_x,_y);

                        if (objectivesMap[_y][_x][(int)evaType] < minObjValue)
                            minObjValue = objectivesMap[_y][_x][(int)evaType];
                        if (objectivesMap[_y][_x][(int)evaType] > maxObjValue)
                            maxObjValue = objectivesMap[_y][_x][(int)evaType];

                        objMapVal[_y][_x] = objectivesMap[_y][_x][(int)evaType];
                    }
                }
                cout << "Statistic: minCostValue = " << minObjValue << ", maxCostValue = " << maxObjValue << endl;
                str_sub = str.substr(0, pos);
                str_sub.append(NN_SFX_EVALUATION_COST_FILE);
                fo.open(str_sub.c_str() );
                if (!fo) {
                    std::cout << "erreur write "<< str_sub << endl; }
                fo << objMapVal;
            }

            if (evaType == obj_length) {
                minObjValue = 99999999.9;
                maxObjValue = 0.0;
                for (int _y = 0; _y < height; _y++)
                {
                    for (int _x = 0; _x < width; _x++)
                    {
                        if (_y != (height - 1) && _x != (width - 1)) {
                            if (objectivesMap[_y][_x][(int)evaType] < minObjValue)
                                minObjValue = objectivesMap[_y][_x][(int)evaType];
                            if (objectivesMap[_y][_x][(int)evaType] > maxObjValue)
                                maxObjValue = objectivesMap[_y][_x][(int)evaType];
                        }
                        objMapVal[_y][_x] = objectivesMap[_y][_x][(int)evaType];
                    }
                }
                cout << "Statistic: minLengthValue = " << minObjValue << ", maxLengthValue = " << maxObjValue << endl;
                str_sub = str.substr(0, pos);
                str_sub.append(NN_SFX_EVALUATION_LENGTH_FILE);
                fo.open(str_sub.c_str() );
                if (!fo) {
                    std::cout << "erreur write "<< str_sub << endl; }
                fo << objMapVal;
            }

            if (evaType == obj_smoothing) {
                minObjValue = 99999999.9;
                maxObjValue = 0.0;
                for (int _y = 0; _y < height; _y++)
                {
                    for (int _x = 0; _x < width; _x++)
                    {
                        if (objectivesMap[_y][_x][(int)evaType] < minObjValue)
                            minObjValue = objectivesMap[_y][_x][(int)evaType];
                        if (objectivesMap[_y][_x][(int)evaType] > maxObjValue)
                            maxObjValue = objectivesMap[_y][_x][(int)evaType];

                        objMapVal[_y][_x] = objectivesMap[_y][_x][(int)evaType];
                    }
                }
                cout << "Statistic: minSmoothValue = " << minObjValue << ", maxSmoothValue = " << maxObjValue << endl;
                str_sub = str.substr(0, pos);
                str_sub.append(NN_SFX_EVALUATION_SMOOTH_FILE);
                fo.open(str_sub.c_str() );
                if (!fo) {
                    std::cout << "erreur write "<< str_sub << endl; }
                fo << objMapVal;
            }

            if (evaType == obj_gd_error) {
                minObjValue = 99999999.9;
                maxObjValue = 0.0;
                for (int _y = 0; _y < height; _y++)
                {
                    for (int _x = 0; _x < width; _x++)
                    {
                        if (objectivesMap[_y][_x][(int)evaType] < minObjValue)
                            minObjValue = objectivesMap[_y][_x][(int)evaType];
                        if (objectivesMap[_y][_x][(int)evaType] > maxObjValue)
                            maxObjValue = objectivesMap[_y][_x][(int)evaType];

                        objMapVal[_y][_x] = objectivesMap[_y][_x][(int)evaType];
                    }
                }
                cout << "Statistic: minGTerrValue = " << minObjValue << ", maxGTerrValue = " << maxObjValue << endl;
                str_sub = str.substr(0, pos);
                str_sub.append(NN_SFX_EVALUATION_GTERROR_FILE);
                fo.open(str_sub.c_str() );
                if (!fo) {
                    std::cout << "erreur write "<< str_sub << endl; }
                fo << objMapVal;
            }

            fo.close();
            objMapVal.freeMem();
        }
    } //write

    //! HW 16.05.15 : overload for flow map writing
    void write(string str, Grid<Point>& adaptiveMapOri) {

//        write(str);

        int pos=getPos(str);
        string str_sub;

        //! write flowMap
        if(adaptiveMap.width != 0 && adaptiveMap.height != 0){
            int width = adaptiveMap.width;
            int height = adaptiveMap.height;
            float *h_u_GT  = new float [width * height];
            float *h_v_GT  = new float [width * height];
            for (int _y = 0; _y < height; _y++)
            {
                for (int _x = 0; _x < width; _x++)
                {
                    h_u_GT[_x + _y * width] = adaptiveMap[_y][_x][0] - adaptiveMapOri[_y][_x][0];
                    h_v_GT[_x + _y * width] = adaptiveMap[_y][_x][1] - adaptiveMapOri[_y][_x][1];
                }
            }
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_OPTICAL_FLOW_FILE);
            writeFloFile(str_sub.c_str(), width, height, h_u_GT, h_v_GT);
            delete [] h_u_GT;
            delete [] h_v_GT;
        }
    } //write

    //! HW 12.08.15 : overload for disparity writing
    void writeDisparity(string str, Grid<Point>& adaptiveMapOri) {

        int pos=getPos(str);
        string str_sub;

        //! write flowMap
        if(adaptiveMap.width != 0 && adaptiveMap.height != 0){
            int width = adaptiveMap.width;
            int height = adaptiveMap.height;
            float *disparity  = new float [width * height];
            for (int _y = 0; _y < height; _y++)
            {
                for (int _x = 0; _x < width; _x++)
                {
                    disparity[_x + _y * width] = (adaptiveMap[_y][_x][0] - adaptiveMapOri[_y][_x][0]); // to be verified 120815
//                    if (disparity[_x + _y * width] == 0.0f)
//                        disparity[_x + _y * width] = INFINITY;
                }
            }
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_DISPARITY_FILE);
            writeDisparityFile(disparity, str_sub.c_str());
            delete [] disparity;
        }
    } //write

    //! HW 14.05.15 : add ReadFlowFile function
    //! ======================================================================
    //! Code courtesy from: http://vision.middlebury.edu/flow/
    //! ======================================================================
    // read a flow file into 2-band image
    void readFlowFile(float* u, float* v, const char* filename)
    {
        if (filename == NULL)
        throw CError("ReadFlowFile: empty filename");

        const char *dot = strrchr(filename, '.');
        if (strcmp(dot, ".flo") != 0)
        throw CError("ReadFlowFile (%s): extension .flo expected", filename);

        FILE *stream = fopen(filename, "rb");
        if (stream == 0)
            throw CError("ReadFlowFile: could not open %s", filename);

        int width, height;
        float tag;

        if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
        (int)fread(&width, sizeof(int), 1, stream) != 1 ||
        (int)fread(&height, sizeof(int), 1, stream) != 1)
        throw CError("ReadFlowFile: problem reading file %s", filename);

        if (tag != TAG_FLOAT) // simple test for correct endian-ness
        throw CError("ReadFlowFile(%s): wrong tag (possibly due to big-endian machine?)", filename);

        // another sanity check to see that integers were read correctly (99999 should do the trick...)
        if (width < 1 || width > 99999)
        throw CError("ReadFlowFile(%s): illegal width %d", filename, width);

        if (height < 1 || height > 99999)
        throw CError("ReadFlowFile(%s): illegal height %d", filename, height);

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if ((int)fread(u++, sizeof(float), 1, stream) != 1)
                    throw CError("ReadFlowFile(%s): file is too short", filename);
                if ((int)fread(v++, sizeof(float), 1, stream) != 1)
                    throw CError("ReadFlowFile(%s): file is too short", filename);
            }
        }
        if (fgetc(stream) != EOF)
        throw CError("ReadFlowFile(%s): file is too long", filename);

        fclose(stream);
    } //readFlowFile

    //! HW 14.05.15 : add WriteFloFile
    //! /////////////////////////////////////////////////////////////////////////////
    //! \brief save optical flow in format described on vision.middlebury.edu/flow
    //! \param[in] name output file name
    //! \param[in] w    optical flow field width
    //! \param[in] h    optical flow field height
    //! \param[in] s    optical flow field row stride
    //! \param[in] u    horizontal displacement
    //! \param[in] v    vertical displacement
    //! ////////////////////////////////////////////////////////////////////////////
    void writeFloFile(const char *name, int w, int h, const float *u, const float *v)
    {
        FILE *stream;
        stream = fopen(name, "wb");

        if (stream == 0)
        {
            printf("Could not save flow to \"%s\"\n", name);
            return;
        }

        float data = TAG_FLOAT;
        fwrite(&data, sizeof(float), 1, stream);
        fwrite(&w, sizeof(w), 1, stream);
        fwrite(&h, sizeof(h), 1, stream);

        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                const int pos = j + i * w;
                fwrite(u + pos, sizeof(float), 1, stream);
                fwrite(v + pos, sizeof(float), 1, stream);
            }
        }

        fclose(stream);
    } //writeFloFile

    //! HW 12.08.15 : add readDisparityFile and writeDisparityFile
    //! ======================================================================
    //! Code courtesy from: http://vision.middlebury.edu/flow/
    //! ======================================================================
    void skip_comment(FILE *fp)
    {
        // skip comment lines in the headers of pnm files

        char c;
        while ((c=getc(fp)) == '#')
            while (getc(fp) != '\n') ;
        ungetc(c, fp);
    }
    void skip_space(FILE *fp)
    {
        // skip white space in the headers or pnm files

        char c;
        do {
            c = getc(fp);
        } while (c == '\n' || c == ' ' || c == '\t' || c == '\r');
        ungetc(c, fp);
    }
    void read_header(FILE *fp, const char *imtype, char c1, char c2,
                     int *width, int *height, int *nbands, int thirdArg)
    {
        // read the header of a pnmfile and initialize width and height

        char c;

        if (getc(fp) != c1 || getc(fp) != c2)
            throw CError("ReadFilePGM: wrong magic code for %s file", imtype);
        skip_space(fp);
        skip_comment(fp);
        skip_space(fp);
        fscanf(fp, "%d", width);
        skip_space(fp);
        fscanf(fp, "%d", height);
        if (thirdArg) {
            skip_space(fp);
            fscanf(fp, "%d", nbands);
        }
        // skip SINGLE newline character after reading image height (or third arg)
        c = getc(fp);
        if (c == '\r')      // <cr> in some files before newline
            c = getc(fp);
        if (c != '\n') {
            if (c == ' ' || c == '\t' || c == '\r')
                throw CError("newline expected in file after image height");
            else
                throw CError("whitespace expected in file after image height");
      }
    }
    // check whether machine is little endian
    int littleendian()
    {
        int intval = 1;
        uchar *uval = (uchar *)&intval;
        return uval[0] == 1;
    }

    //======================= read disparity ==========================
    // 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
    // 3-band not yet supported
    void readDisparityFile(float* disparity, const char* filename)
    {
        // Open the file and read the header
        FILE *fp = fopen(filename, "rb");
        if (fp == 0)
            throw CError("ReadFilePFM: could not open %s", filename);

        int width, height, nBands;
        read_header(fp, "PFM", 'P', 'f', &width, &height, &nBands, 0);

        //! HW 12.08.15
        if (!(width == adaptiveMap.getWidth() && height == adaptiveMap.getHeight()))
            throw CError("ReadFilePFM: the size of %s is not the same as the NN.", filename);

        skip_space(fp);

        float scalef;
        fscanf(fp, "%f", &scalef);  // scale factor (if negative, little endian)

        // skip SINGLE newline character after reading third arg
        char c = getc(fp);
        if (c == '\r')      // <cr> in some files before newline
            c = getc(fp);
        if (c != '\n') {
            if (c == ' ' || c == '\t' || c == '\r')
                throw CError("newline expected in file after scale factor");
            else
                throw CError("whitespace expected in file after scale factor");
        }

        int littleEndianFile = (scalef < 0);
        int littleEndianMachine = littleendian();
        int needSwap = (littleEndianFile != littleEndianMachine);
        //printf("endian file = %d, endian machine = %d, need swap = %d\n",
        //       littleEndianFile, littleEndianMachine, needSwap);

        for (int y = height-1; y >= 0; y--) { // PFM stores rows top-to-bottom!!!!
            int n = width;
            float* ptr = (float *) (disparity + y * n);
            if ((int)fread(ptr, sizeof(float), n, fp) != n)
                throw CError("ReadFilePFM(%s): file is too short", filename);

            if (needSwap) { // if endianness doesn't agree, swap bytes
                uchar* ptr = (uchar *) (disparity + y * n);
                int x = 0;
                uchar tmp = 0;
                while (x < n) {
                    tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
                    tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
                    ptr += 4;
                    x++;
                }
            }
        }
        if (fclose(fp))
            throw CError("ReadFilePGM(%s): error closing file", filename);
    }

    //======================= write disparity =========================
    // 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
    // 3-band not yet supported
    void writeDisparityFile(const float *disparity, const char* filename, float scalefactor=1/255.0)
    {
        int width = adaptiveMap.getWidth();
        int height = adaptiveMap.getHeight();

        // Open the file
        FILE *stream = fopen(filename, "wb");
        if (stream == 0)
            throw CError("WriteFilePFM: could not open %s", filename);

        // sign of scalefact indicates endianness, see pfms specs
        if (littleendian())
        scalefactor = -scalefactor;

        // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
        fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

        int n = width;
        // write rows -- pfm stores rows in inverse order!
        for (int y = height-1; y >= 0; y--) {
        float* ptr = (float *) (disparity + y * n);
        if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
            throw CError("WriteFilePFM(%s): file is too short", filename);
        }

        // close file
        if (fclose(stream))
            throw CError("WriteFilePFM(%s): error closing file", filename);
    }


    //    Grid<Point> getAdaptiveMap() const;
    //    void setAdaptiveMap(const Grid<Point> &value);

    Grid<Point> getAdaptiveMap() const
    {
        return adaptiveMap;
    }

    void setAdaptiveMap(const Grid<Point> &value)
    {
        adaptiveMap = value;
    }

};

typedef NeuralNet<Point2D, GLfloat> NN;
typedef NeuralNet<Point3D, GLfloat> NNP3D;

}//namespace components
#endif // NEURALNET_H

#ifndef CONVERTER_H
#define CONVERTER_H
/*
 ***************************************************************************
 *
 * Author : W.B Qiao, J.C. Creput
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>

#include "macros_cuda.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "Node.h"
#include "ImageRW.h"
//#include "ViewGrid.h"
#include "ConfigParams.h"

using namespace std;
using namespace components;


#define SCALE_FACTOR    4.0      // depend of the disparity map
#define BASE_LINE       0.16   // in meters, 3D coordinates are also in meters
#define FOCAL_DISTANCE  374.0   // in pixels (3740 indicated in middlebury database)
#define BACKGROUND_DISPARITY 80.0/SCALE_FACTOR // disparity 0 is replaced by BACKGROUND_DISPARITY
#define MIN_MESH_DISPARITY 50
#define USERDISPARITYMAP 0

// teddy
//scaleFactor = 4.0
//baseLine = 0.16
//focalDistance = 374.0
//#focalDistance = 935.0
//#focalDistance = 3740.0
//disparityRange = 60
//backgroundDisparity = 68
//minMeshDisparity = 68


typedef Grid<Point3D> grid3DColor;
typedef Grid<Point3D> gridEuclideanPoints;

namespace components
{
template<
        class Point,
        class Value
        >
class Converter
{

public:

    typedef NeuralNet<Point, Value> nnInput;

    //!QWB: 0415 change 2D adaptiveMap into 3D adaptiveMap
    void densEu(Grid<Point> &nn2dAdap, Grid<Value> &dens, gridEuclideanPoints &eu3D, ConfigParams& param, bool fixedDisparity){
        int useDisparityMap;
        double minMeshDisparity;// minimum disparity for meshing
        double scaleFactor;// depend of the disparity map
        double baseLine;// in meters, 3D coordinates are also in meters (=0.16 in middlebury database)
        double focalDistance;// in pixels (3740 indicated in middlebury database)
        double backgroundDisparity;// disparity 0 is replaced by BACKGROUND_DISPARITY/scaleFactor

        param.readConfigParameter("input","useDisparityMap", useDisparityMap);
        param.readConfigParameter("param_2","scaleFactor", scaleFactor  );
        param.readConfigParameter("param_2","baseLine", baseLine  );
        param.readConfigParameter("param_2","focalDistance", focalDistance  );
        param.readConfigParameter("param_2","backgroundDisparity", backgroundDisparity );
        param.readConfigParameter("param_2","minMeshDisparity", minMeshDisparity  );

        int gridWidth = nn2dAdap.width;
        int gridHeight = nn2dAdap.height;
        GLfloat disparity=0;
        GLfloat _x, _y;
        eu3D.resize(gridWidth, gridHeight);
        for (int _h = 0; _h < gridHeight; _h++)
        {
            for (int _w = 0; _w < gridWidth; _w++)
            {
                _x = nn2dAdap[_h][_w][0];
                _y = nn2dAdap[_h][_w][1];

                int __x = (int) _x;
                if (_x >= __x + 0.5)
                    __x = __x + 1;
                int __y = (int) _y;
                if (_y >= __y + 0.5)
                    __y = __y + 1;

                if (fixedDisparity == 0)
                {
                    if (__x < 0)
                        __x = 0;
                    if (__x >= dens.width)
                        __x = dens.width - 1;
                    if (__y < 0)
                        __y = 0;
                    if (__y >= dens.height)
                        __y = dens.height - 1;
                }

                //! QWB : modif, fixedDisparity means using the 2D
                if (!useDisparityMap || fixedDisparity)
                {

                    disparity = minMeshDisparity;
                }
                else
                {
                    if (dens.get(__x, __y) == 0 )
                    {
                        disparity = backgroundDisparity/scaleFactor;
                    }
                    else
                    {
                        disparity = dens.get(__x, __y) / scaleFactor;
                    }
                }
                eu3D[_h][_w][2] = -focalDistance * baseLine / disparity;
                eu3D[_h][_w][0] = (_x - gridWidth/2) * baseLine / disparity;
                eu3D[_h][_w][1] = -(_y - gridWidth/2) * baseLine / disparity;
            }
        }
    }


    //!QWB: 0415 find the correct color for new colorMap according adaptiveMap
        void readColor(Grid<Point> &adaptiveMap, grid3DColor &inputColorMap, grid3DColor &outputColorMap, bool noColorInput){

        int gridWidth = adaptiveMap.width;
        int gridHeight = adaptiveMap.height;
        GLfloat _x, _y;


        outputColorMap.resize(gridWidth, gridHeight);
        for (int _h = 0; _h < gridHeight; _h++)
        {
            for (int _w = 0; _w < gridWidth; _w++)
            {
                _x = adaptiveMap[_h][_w][0];
                _y = adaptiveMap[_h][_w][1];

                int __x = (int) _x;
                if (_x >= __x + 0.5)
                    __x = __x + 1;
                int __y = (int) _y;
                if (_y >= __y + 0.5)
                    __y = __y + 1;

                bool debord = false;
                if (noColorInput == 1)
                    debord = true;

                //! JCC 130315 : modif
                if (__x < 0) {
                    debord = true;
                    __x = 0;
                }
                if (__x >= inputColorMap.getWidth()) {
                    debord = true;
                    __x = inputColorMap.getWidth() - 1;
                }
                if (__y < 0) {
                    debord = true;
                    __y = 0;
                }
                if (__y >= inputColorMap.getHeight()) {
                    debord = true;
                    __y = inputColorMap.getHeight() - 1;
                }

                //! JCC 130315 : modif
                if (debord){
                    outputColorMap[_h][_w][0] = 0;//255;
                    outputColorMap[_h][_w][1] = 0;//255;
                    outputColorMap[_h][_w][2] = 0;//255;
                }
                else {

                    outputColorMap[_h][_w][0] = inputColorMap[__y][__x][0];
                    outputColorMap[_h][_w][1] = inputColorMap[__y][__x][1];
                    outputColorMap[_h][_w][2] = inputColorMap[__y][__x][2];
                }
            }
        }

    }

    //! qiao add: read nnGrid2dpts.colorMap and its own densityMap to produce Euclidean outPut
    void convertGridItself(nnInput& nn, NNP3D& nno, ConfigParams& param, bool fixedDisparity){

        if (nn.adaptiveMap.width != 0 && nn.adaptiveMap.height != 0){
            int gridWidth = nn.adaptiveMap.width;
            int gridHeight = nn.adaptiveMap.height;

            if (nn.adaptiveMap.width != nn.densityMap.width)
                std::cout << "Converter by itself : nnGrid densityMap_Width != AdaptiveMap_Width " << endl;

            //! fill outputEuclidean3Dpoints with the nn2dpts.densityMap
            if(nn.densityMap.width == 0)
            {
                cout << "Converter by itself : not define densityMap, display 2D" << endl;
                nn.densityMap.resize(gridWidth, gridHeight);
            }

            densEu(nn.adaptiveMap, nn.densityMap, nno.adaptiveMap,  param, fixedDisparity);

            //! fill the outputColorMap with nn2dpts.colorMap
            bool noColorInput = 0;
            if (nn.colorMap.width == 0){
                cout << "converter by itself: not define colorMap, dispaly white " << endl;
                noColorInput = 1;
            }
            readColor(nn.adaptiveMap, nn.colorMap, nno.colorMap, noColorInput);

            //! fill other Map of nno
            nno.densityMap.resize(gridWidth, gridHeight);
        }
        else
            std::cout << "Converter: by itself, nnGrid has no 2dpts" << endl;
    }

    //! qiao add
    void convertImageItself(nnInput& nnI, NNP3D& nnIo, ConfigParams& param, bool fixedDisparity){
        int useDisparityMap;
        double minMeshDisparity;// minimum disparity for meshing
        double scaleFactor;// depend of the disparity map
        double baseLine;// in meters, 3D coordinates are also in meters (=0.16 in middlebury database)
        double focalDistance;// in pixels (3740 indicated in middlebury database)
        double backgroundDisparity;// disparity 0 is replaced by BACKGROUND_DISPARITY/scaleFactor

        param.readConfigParameter("input","useDisparityMap", useDisparityMap); // qiao: useDisparityMap means display 3D or not in config file
        param.readConfigParameter("param_2","scaleFactor", scaleFactor  );
        param.readConfigParameter("param_2","baseLine", baseLine  );
        param.readConfigParameter("param_2","focalDistance", focalDistance  );
        param.readConfigParameter("param_2","backgroundDisparity", backgroundDisparity );
        param.readConfigParameter("param_2","minMeshDisparity", minMeshDisparity  );

        int width = nnI.colorMap.width;
        int height = nnI.colorMap.height;

        //! output: nnIo.adaptiveMap has 3D Euclidean coordinates
        GLfloat disparity=0;
        nnIo.adaptiveMap.resize(width, height);
        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                if (!useDisparityMap || fixedDisparity)
                {
                    disparity = minMeshDisparity;

                }
                else
                {
                    if (nnI.densityMap.get(_x, _y) == 0)
                    {
                        disparity = backgroundDisparity / scaleFactor;
                    }
                    else
                    {
                        disparity = nnI.densityMap.get(_x, _y)/ scaleFactor;
                    }
                }
                nnIo.adaptiveMap[_y][_x][2] = -focalDistance * baseLine / disparity;
                nnIo.adaptiveMap[_y][_x][0] = (_x - width/2) * baseLine / disparity;
                nnIo.adaptiveMap[_y][_x][1] = -(_y - height/2) * baseLine / disparity;
            }
        }

        //! fill other Map of nnIo
        nnIo.densityMap.resize(width, height);// qiao: the viewer does not need densityMap to display
        nnIo.colorMap = nnI.colorMap;
    }



    //! qiao add: convert <point2D>nn to <point3D>nno with the disparity image and color image
    void convertFromImage(nnInput& nnI, nnInput& nn, NNP3D& nno, ConfigParams& param, bool fixedDisparity){

        //! convert 2dGrid to 3d
        if (nn.adaptiveMap.width != 0 && nn.adaptiveMap.height != 0){
            int gridWidth = nn.adaptiveMap.width;
            int gridHeight = nn.adaptiveMap.height;

            //! Convert nn.2dpts to  nno.adaptiveMap(3d) that has the 3D Euclidean coordinates
            if (nnI.densityMap.width == 0)
                cout << "imageRW error: gtNN does not exist densityMap, dispalay 2D" << endl;
            densEu(nn.adaptiveMap, nnI.densityMap, nno.adaptiveMap,  param, fixedDisparity);

            //! fill the outputColorMap with nnImage.colorMap
            bool noColorInput = 0;
            if (nnI.colorMap.width == 0){
                cout << "imageRW error: gtNN does not exist colorMap, , dispaly white" << endl;
                noColorInput = 1;
            }

            readColor(nn.adaptiveMap, nnI.colorMap, nno.colorMap, noColorInput);

            //! fill other Map of nno
            nno.densityMap.resize(gridWidth, gridHeight);

        }
    }


};


}//! namespace components

#endif //! CONVERTER.H

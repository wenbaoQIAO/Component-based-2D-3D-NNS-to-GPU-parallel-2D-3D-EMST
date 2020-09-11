#ifndef CONVERTERLINKNET_H
#define CONVERTERLINKNET_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao
 * Creation date : December. 2016
 * Converter adapted to NeuralNetLink data structure.
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>
#include "Converter.h"
#include "NeuralNetEMST.h"

using namespace std;
using namespace components;


typedef Grid<Point3D> grid3DColor;
typedef Grid<Point3D> gridEuclideanPoints;

namespace components
{
template<
        class Point,
        class Value
        >
class ConverterNetLink : public Converter<Point, Value>
{

public:

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

    //! 090516 QWB add: convert nnLinks to <point3D>nno with the gt disparity image and color image
    void convertNnLinkPointsFromImage(nnInput& nnI,
                                      NetLinkPointCoord& nn,
                                      MatNetLinks& nno,
                                      ConfigParams& param,
                                      bool fixedDisparity){
        //! convert 2dGrid to 3d
        if (nn.adaptiveMap.width != 0 && nn.adaptiveMap.height != 0){

            //! Convert nn.2dpts to nno.adaptiveMap(3d) that has the 3D Euclidean coordinates
            if (nnI.densityMap.width == 0)
                cout << "imageRW error: gtNN does not exist densityMap, dispalay 2D" << endl;

            nno.fixedMap = nn.fixedMap;
            densEuLinks(nn, nnI.densityMap, nno,  param, fixedDisparity);

            //! fill the outputColorMap with nnImage.colorMap
            bool noColorInput = 0;
            if (nnI.colorMap.width == 0){
                cout << "imageRW error: gtNN does not exist colorMap, , dispaly white" << endl;
                noColorInput = 1;
            }

            //! fill other Map of nno
            //            nno.densityMap.resize(gridWidth, gridHeight);

        }
    }
    //! 090516 qiao add: convert nnLinks to <point3D>nno with the gt disparity image and color image
    void convertNnLinkColorFromImage(nnInput& nnI,
                                     NetLinkPointCoord& nn,
                                     MatNetLinks& nno,
                                     ConfigParams& param,
                                     bool ColorInput){
        //! convert 2dGrid to 3d
        if (nn.adaptiveMap.width != 0 && nn.adaptiveMap.height != 0){

            //! Convert nn.2dpts to  nno.adaptiveMap(3d) that has the 3D Euclidean coordinates
            if (nnI.densityMap.width == 0)
                cout << "imageRW error: gtNN does not exist densityMap, dispalay 2D" << endl;

            nno.fixedMap = nn.fixedMap; // need to adapt to the new version

            //! fill the outputColorMap with nnImage.colorMap
            if (nnI.colorMap.width == 0){
                cout << "imageRW error: gtNN does not exist colorMap, , dispaly white" << endl;
                ColorInput = 0;
            }

            readLinkColor(nn, nnI.colorMap, nno,  ColorInput);

            //! fill other Map of nno
            //            nno.densityMap.resize(gridWidth, gridHeight);

        }
    }

    //! QWB 040816 add for tsp visilization
    void convertNnLinkPointsFromItself(NetLinkPointCoord& nn,
                                       MatNetLinks& nno,
                                       ConfigParams& param,
                                       bool fixedDisparity,
                                       bool colorInput){

         if (nn.adaptiveMap.width != 0 && nn.adaptiveMap.height != 0){

             if (nn.densityMap.width != nn.adaptiveMap.width || nn.densityMap.height != nn.adaptiveMap.height)
                 cout << "netLinks densityMap != adaptiveMap " << endl;
             else {
                 densEu(nn.adaptiveMap, nn.densityMap, nno.adaptiveMap, param, fixedDisparity);
             }

             if (nn.colorMap.width != nn.adaptiveMap.width || nn.colorMap.height != nn.adaptiveMap.height)
                 cout << "netLinks colorMap != adaptiveMap " << endl;
             else {
                 readColor(nn.adaptiveMap, nn.colorMap, nno.colorMap, colorInput);

             }
         }

    }


    //! QWB 0618 add for 3D visilization
    void convertNnLinkPointsFromItself(MatNetLinks& nn,
                                       MatNetLinks& nno,
                                       ConfigParams& param,
                                       bool fixedDisparity,
                                       bool colorInput){

         if (nn.adaptiveMap.width != 0 && nn.adaptiveMap.height != 0){

             if (nn.densityMap.width != nn.adaptiveMap.width || nn.densityMap.height != nn.adaptiveMap.height)
                 cout << "netLinks densityMap != adaptiveMap " << endl;
             else {
                 densEu(nn.adaptiveMap, nn.densityMap, nno.adaptiveMap, param, fixedDisparity);
             }

             if (nn.colorMap.width != nn.adaptiveMap.width || nn.colorMap.height != nn.adaptiveMap.height)
                 cout << "netLinks colorMap != adaptiveMap " << endl;
             else {
                 readColor(nn.adaptiveMap, nn.colorMap, nno.colorMap, colorInput);

             }
         }

    }


};

}//! namespace components

#endif //! CONVERTER.H

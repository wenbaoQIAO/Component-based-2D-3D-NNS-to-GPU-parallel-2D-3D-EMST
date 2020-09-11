#ifndef IMAGERW_H
#define IMAGERW_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, W. Qiao, H. Wang
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>

#include <QtGui/QImage>

#include "macros_cuda.h"
#include "GridOfNodes.h"
#include "Node.h"
#include "NeuralNet.h"

#define NORMALIZATION   0

#define IRW_ADAPTIVE    1
#define IRW_DENSITY     1
#define IRW_GRAY        0
#define IRW_FLOW        0
#define IRW_DISPARITY   0

#define SFX_ADAPTIVE_MAP_IMAGE  "_adaptiveMap.pgm"
#define SFX_COLOR_MAP_IMAGE  ".png"
#define SFX_DENSITY_MAP_IMAGE  "_groundtruth.pgm"

#define SFX_COLOR_MAP_IMAGE_FRAME10  "_frame10.png"
#define SFX_COLOR_MAP_IMAGE_FRAME11  "_frame11.png"
#define SFX_GRAY_MAP_IMAGE10  "_gray_frame10.png"
#define SFX_GRAY_MAP_IMAGE11  "_gray_frame11.png"
#define SFX_FLOW_MAP  "_flow10.flo"

//#define SFX_STEREO_COLOR_MAP_IMAGE0  "_im0.ppm"
//#define SFX_STEREO_COLOR_MAP_IMAGE1  "_im1.ppm"
#define SFX_STEREO_COLOR_MAP_IMAGE0  "_im0.png"
#define SFX_STEREO_COLOR_MAP_IMAGE1  "_im1.png"
#define SFX_DISPARITY_MAP  "_disp0GT.pfm"

//! HW 040216 : used to load Middlebury_stereo_datasets
//! left image : view1.png
//! right image : view2.png
#define LOAD_MIDDLEBURY_STEREO_DATASETS 1

using namespace std;
using namespace components;

namespace components
{

template <class Point,
          class Value>
class ImageRW
{
public:

    int getPos(string str){

        std::size_t pos = str.find('.',  0);
        if (pos == std::string::npos)
            pos = str.length();

        //! no-matter aaa_xx.xx or aaa.xx or aaa_xx,  aaa.xx_xx
        //! we can always get the first three aaa and the same pos =  = 3
        std::size_t posTiret = str.find('_',  0);

        if(pos > posTiret)
            pos = posTiret; //! ensure to get the aaa in any name format aaa_xxx or aaa_xxx.lll

        return pos;
    }

    void read(string str, NeuralNet<Point, Value>& nn) {

        //! Input
        QImage colorMapImage; //! input_color_image
        QImage densityMapImage; //! input_densityMapImage

        int pos = getPos(str);
        string str_sub;

#if IRW_ADAPTIVE
        QImage adaptiveMapImage; //! input_adaptiveMap

        //! load the adaptiveMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_ADAPTIVE_MAP_IMAGE);
        adaptiveMapImage.load(str_sub.c_str());
        cout << "base_name_adaptive= " << str_sub << endl;

#endif

        //! load colorMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE);
        colorMapImage.load(str_sub.c_str());

#if IRW_DENSITY
        //! Load densityMap(groundtruth image)
        str_sub = str.substr(0,  pos);
        str_sub.append(SFX_DENSITY_MAP_IMAGE);
        densityMapImage.load(str_sub.c_str());
#endif

        int width = colorMapImage.width(); //! width of input_color_image
        int height = colorMapImage.height(); //! height of input_color_image
cout << "width = " << width << " height = " << height << endl;
        //! fill the colorMap of nn
        nn.colorMap.resize(width,  height);
        for(int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                //! fill colorMap
                nn.colorMap[_y][_x][0] = qRed(colorMapImage.pixel(_x, _y));
                nn.colorMap[_y][_x][1] = qGreen(colorMapImage.pixel(_x, _y));
                nn.colorMap[_y][_x][2] = qBlue(colorMapImage.pixel(_x, _y));
#if NORMALIZATION
                //! HW 24/04/15 : normalization
                nn.colorMap[_y][_x][0] /= 255.0f;
                nn.colorMap[_y][_x][1] /= 255.0f;
                nn.colorMap[_y][_x][2] /= 255.0f;
#endif
            }
        }

        //! fill the density_map of nn from the groundtruth.image
        int _dispValue = 0;
        nn.densityMap.resize(width, height);
#if IRW_DENSITY
        for( int _y = 0; _y < height; _y++ )
        {
            for ( int _x = 0; _x < width; _x++ )
            {
                QRgb gray = densityMapImage.pixel( _x,  _y );
                _dispValue = qGray( gray );
                nn.densityMap.set( _x, _y, _dispValue);
            }
        }
#endif
    }//read

    //! HW 140515 : overload for optical flow application
    void read(string str, NeuralNet<Point, Value>& nnr, NeuralNet<Point, Value>& nnd, bool right = false) {

        //! Input
        QImage colorMapImageR; //! input color image
        QImage grayMapImageR; //! input gray image
        QImage colorMapImageD; //! input color image
        QImage grayMapImageD; //! input gray image

        int pos = getPos(str);
        string str_sub;

        //! load colorMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE_FRAME10);
        if (!right)
            colorMapImageR.load(str_sub.c_str());
        else
            colorMapImageD.load(str_sub.c_str());

        str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE_FRAME11);
        if (!right)
            colorMapImageD.load(str_sub.c_str());
        else
            colorMapImageR.load(str_sub.c_str());

        int width = colorMapImageR.width(); //! width of input_color_image
        int height = colorMapImageR.height(); //! height of input_color_image
        cout << "width = " << width << " height = " << height << endl;

        //! fill the colorMap of nnr and nnd
        nnr.colorMap.resize(width,  height);
        nnd.colorMap.resize(width,  height);
        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                //! fill colorMap
                nnr.colorMap[_y][_x][0] = qRed(colorMapImageR.pixel(_x, _y));
                nnr.colorMap[_y][_x][1] = qGreen(colorMapImageR.pixel(_x, _y));
                nnr.colorMap[_y][_x][2] = qBlue(colorMapImageR.pixel(_x, _y));
                nnd.colorMap[_y][_x][0] = qRed(colorMapImageD.pixel(_x, _y));
                nnd.colorMap[_y][_x][1] = qGreen(colorMapImageD.pixel(_x, _y));
                nnd.colorMap[_y][_x][2] = qBlue(colorMapImageD.pixel(_x, _y));
#if NORMALIZATION
                //! HW 24/04/15 : normalization
                nnr.colorMap[_y][_x] = nnr.colorMap[_y][_x] / 255.0f;
                nnd.colorMap[_y][_x] = nnd.colorMap[_y][_x] / 255.0f;
#endif
            }
        }

#if IRW_GRAY
        //! Load densityMap(groundtruth image)
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_GRAY_MAP_IMAGE10);
        if (!right)
            grayMapImageR.load(str_sub.c_str());
        else
            grayMapImageD.load(str_sub.c_str());

        str_sub = str.substr(0, pos);
        str_sub.append(SFX_GRAY_MAP_IMAGE11);
        if (!right)
            grayMapImageD.load(str_sub.c_str());
        else
            grayMapImageR.load(str_sub.c_str());

        //! fill the grayValueMap of nnr and nnd
        int _grayValue = 0;
        nnr.grayValueMap.resize(width, height);
        nnd.grayValueMap.resize(width, height);
        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                QRgb gray = grayMapImageR.pixel(_x,  _y);
                _grayValue = qGray(gray);
                nnr.grayValueMap.set(_x, _y, _grayValue);
                gray = grayMapImageD.pixel(_x,  _y);
                _grayValue = qGray(gray);
                nnd.grayValueMap.set(_x, _y, _grayValue);
            }
        }
#endif

#if IRW_FLOW

        float *h_u_GT  = new float [width * height];
        float *h_v_GT  = new float [width * height];

        //! load the adaptiveMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_FLOW_MAP);
        nnr.readFlowFile(h_u_GT, h_v_GT, str_sub.c_str());
        nnr.adaptiveMap.resize(width,  height);
        nnr.activeMap.resize(width,  height);

        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                float xNew = h_u_GT[_x + _y * width];// + _x;
                float yNew = h_v_GT[_x + _y * width];// + _y;

//                if (_x >= 0 && _x < 20 && _y >= 0 && _y < 20)
//                    printf("(%d,%d) = (%f,%f)\n", _x, _y, xNew, yNew);

                // Here the activeMap of the NN is employed to mark if the pixel is occluded or not
                nnr.activeMap[_y][_x] = false;
//                if (xNew == INFINITY || yNew == INFINITY) {// in the GT file, the disparity of occluded pixel is INFINITY
                if (xNew > 99999.0f || yNew > 99999.0f || xNew < -99999.0f || yNew < -99999.0f) {
                    xNew = 0.0f;
                    yNew = 0.0f;
                    nnr.activeMap[_y][_x] = true; // occluded
                }

                if (!right)
                    nnr.adaptiveMap.set(_x, _y, Point2D(xNew, yNew));
                else
                    nnr.adaptiveMap.set(_x, _y, Point2D(-xNew, -yNew));
            }
        }

        delete [] h_u_GT;
        delete [] h_v_GT;
#endif

    }//read

    //! HW 120815 : overload for stereo matching application
    void readStereo(string str, NeuralNet<Point, Value>& nnr, NeuralNet<Point, Value>& nnd, bool right = false) {

        //! Input
        QImage colorMapImageR; //! input color image
        QImage colorMapImageD; //! input color image

        int pos = getPos(str);
        string str_sub;

#if LOAD_MIDDLEBURY_STEREO_DATASETS

        if (!right)
            colorMapImageR.load("view1.png");
        else
            colorMapImageD.load("view1.png");

        if (!right)
            colorMapImageD.load("view5.png");
        else
            colorMapImageR.load("view5.png");

#else

        //! load colorMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_STEREO_COLOR_MAP_IMAGE0);
        if (!right)
            colorMapImageR.load(str_sub.c_str());
        else
            colorMapImageD.load(str_sub.c_str());

        str_sub = str.substr(0, pos);
        str_sub.append(SFX_STEREO_COLOR_MAP_IMAGE1);
        if (!right)
            colorMapImageD.load(str_sub.c_str());
        else
            colorMapImageR.load(str_sub.c_str());

#endif

        int width = colorMapImageR.width(); //! width of input_color_image
        int height = colorMapImageR.height(); //! height of input_color_image
        cout << "width = " << width << " height = " << height << endl;

        //! fill the colorMap of nnr and nnd
        nnr.colorMap.resize(width,  height);
        nnd.colorMap.resize(width,  height);
        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                //! fill colorMap
                nnr.colorMap[_y][_x][0] = qRed(colorMapImageR.pixel(_x, _y));
                nnr.colorMap[_y][_x][1] = qGreen(colorMapImageR.pixel(_x, _y));
                nnr.colorMap[_y][_x][2] = qBlue(colorMapImageR.pixel(_x, _y));
                nnd.colorMap[_y][_x][0] = qRed(colorMapImageD.pixel(_x, _y));
                nnd.colorMap[_y][_x][1] = qGreen(colorMapImageD.pixel(_x, _y));
                nnd.colorMap[_y][_x][2] = qBlue(colorMapImageD.pixel(_x, _y));
#if NORMALIZATION
                //! HW 24/04/15 : normalization
                nnr.colorMap[_y][_x] = nnr.colorMap[_y][_x] / 255.0f;
                nnd.colorMap[_y][_x] = nnd.colorMap[_y][_x] / 255.0f;
#endif
            }
        }

#if IRW_DISPARITY

        float *disparity_GT  = new float [width * height];
        nnr.adaptiveMap.resize(width,  height);
        nnr.activeMap.resize(width,  height);

        //! load the adaptiveMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_DISPARITY_MAP);
        nnr.readDisparityFile(disparity_GT, str_sub.c_str());

        float min = 99999.0f;
        float max = -99999.0f;

        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                // Here the activeMap of the NN is employed to mark if the pixel is occluded or not
                nnr.activeMap[_y][_x] = false;

                float xNew = disparity_GT[_x + _y * width];// + _x;

                if (xNew < min)
                    min = xNew;
//                if (xNew == INFINITY) // in the GT file, the disparity of occluded pixel is INFINITY
                if (xNew > 99999.0f) {
                    xNew = 0.0f;
                    nnr.activeMap[_y][_x] = true; // occluded
                }
                if (xNew > max)
                    max = xNew;

                if (!right)
                    nnr.adaptiveMap.set(_x, _y, Point2D(-xNew, 0.0f));
                else
                    nnr.adaptiveMap.set(_x, _y, Point2D(xNew, 0.0f));
            }
        }

        printf("min = %f \n", min);
        printf("max = %f \n", max);

        delete [] disparity_GT;
#endif

    }//read
    
    //! write the NN_colorMap to image
    void write(string str, NeuralNet<Point, Value>& nn) {

        //! Convert grid to qimages
        int width = nn.colorMap.width;
        int height = nn.colorMap.height;
        QImage reImage(width, height, QImage::Format_RGB32);

        for (int _y = 0; _y < height;_y++)
        {
            for (int _x = 0; _x < width;_x++)
            {
#if NORMALIZATION
                //! HW 24/04/15 : normalization
                nn.colorMap[_y][_x][0] *= 255.0f;
                nn.colorMap[_y][_x][1] *= 255.0f;
                nn.colorMap[_y][_x][2] *= 255.0f;
#endif
                int r = nn.colorMap[_y][_x][0];
                int g = nn.colorMap[_y][_x][1];
                int b = nn.colorMap[_y][_x][2];

                QRgb pixel_color = qRgb(r, g, b);

                reImage.setPixel(_x, _y, pixel_color);
            }
        }

        int pos = getPos(str);
        string str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE);
        reImage.save(str_sub.c_str());

        //! HW 210915 : add function to write densityMap as gray image
#if IRW_DENSITY

        for (int _y = 0; _y < height;_y++)
        {
            for (int _x = 0; _x < width;_x++)
            {
                int r = nn.densityMap[_y][_x];
                int g = r;
                int b = r;
                QRgb pixel_color = qRgb(r, g, b);
                reImage.setPixel(_x, _y, pixel_color);
            }
        }

        string str_sub2 = str.substr(0, pos);
        str_sub2.append("densityMap.png");
        reImage.save(str_sub2.c_str());
#endif

    }

    //! HW 13/04/15 : method overloading for directly writing color grid
    void write(string str, Grid<Point3D>& gColor) {

        //! Convert grid to qimages
        int width = gColor.getWidth();
        int height = gColor.getHeight();
        QImage reImage(width, height, QImage::Format_RGB32);

        for (int _y = 0; _y < height;_y++)
        {
            for (int _x = 0; _x < width;_x++)
            {                
#if NORMALIZATION
                //! HW 24/04/15 : normalization
                gColor[_y][_x][0] *= 255.0f;
                gColor[_y][_x][1] *= 255.0f;
                gColor[_y][_x][2] *= 255.0f;
#endif
                int r = gColor[_y][_x][0];
                int g = gColor[_y][_x][1];
                int b = gColor[_y][_x][2];

                QRgb pixel_color = qRgb(r, g, b);

                reImage.setPixel(_x, _y, pixel_color);
            }
        }

        int pos = getPos(str);
        string str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE);
        reImage.save(str_sub.c_str());
    }

    //! HW 210915 : add function to write disparity values as gray image
    void writeDisparityAsGrayImage(string str,
                                   Grid<Point>& adaptiveMap,
                                   Grid<Point>& adaptiveMapOri,
                                   int outscale) {

        if(adaptiveMap.width != 0 && adaptiveMap.height != 0){
            int width = adaptiveMap.width;
            int height = adaptiveMap.height;
            QImage reImage(width, height, QImage::Format_RGB32);

            for (int _y = 0; _y < height; _y++)
            {
                for (int _x = 0; _x < width; _x++)
                {
                    int r = abs(adaptiveMap[_y][_x][0] - adaptiveMapOri[_y][_x][0]);
                    float val = (float)r * (float)outscale;
                    r = (int)val;
                    int g = r;
                    int b = r;
                    QRgb pixel_color = qRgb(r, g, b);
                    reImage.setPixel(_x, _y, pixel_color);
                }
            }

//            int pos = getPos(str);
//            string str_sub = str.substr(0, pos);
//            str_sub.append("_disparity_gray.png");
//            reImage.save(str_sub.c_str());

            str.append("_disparity_gray.png");
            reImage.save(str.c_str());
        }
    }

};

typedef ImageRW<Point2D, GLfloat> IRW;

} // namespace components

#endif // IMAGERW_H

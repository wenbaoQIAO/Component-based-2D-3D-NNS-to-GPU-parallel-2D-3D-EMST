#ifndef INPUTRW_H
#define INPUTRW_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao
 * Creation date : December. 2016
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

#define SFX_TSP_FILE  ".tsp"
#define SFX_3D_FILE  ".3d"


//! qiao add for LR-check
#define SFX_STEREO_GT_MAP_L "disp0GT.pfm"
#define SFX_STEREO_GT_MAP_R "disp1GT.pfm"

using namespace std;
using namespace components;

namespace components
{

template <class Point,
          class Value>
class InputRW
{
public:

    //! wb.Q add March 2015
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

    bool readFile(string file, NeuralNet<Point, Value>& nn, Point& pMin, Point& pMax ) {

        int pos = getPos(file);
        string str_sub;

        str_sub = file.substr(pos, strlen(SFX_3D_FILE));

        if (str_sub.compare(SFX_3D_FILE) == 0)
            read3D(file, nn, pMin, pMax);
        else
            readTSP(file, nn, pMin, pMax);

        return false;
    }

    bool readTSP(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        return readTSP(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool readTSP(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax ) {
        pMin[2] = 0;
        pMax[2] = 0;
        return readTSP(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool read3D(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        read3DPoints(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
        return false;
    }

    bool read3D(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax ) {
        read3DPoints(file, nn, pMin[0], pMin[1], pMin[2], pMax[0], pMax[1], pMax[2]);
        return false;
    }

    //! wb.Q 062018 add to read 3D point from files.
    bool read3DPoints(string str_file, NeuralNet<Point, Value>& nn,
                      GLfloatP& min_x, GLfloatP& min_y, GLfloatP& min_z,
                      GLfloatP& max_x, GLfloatP& max_y, GLfloatP& max_z)
    {
        char buf[256];

        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3D_FILE);

        cout << "read 3d points from: " << str_sub << endl;

        // count total size of input points
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture 3D file: " << str_sub << endl;
        }
        else
        {
            int size = 0;
            while(fi.getline(buf, 256)){
                size ++;
            }
            fi.close();
            cout << "qiao size " << size << endl;

            if (size)
                nn.resize(size, 1);
            else
                cout << "error read cities: num = 0." << endl;

            fi.open(str_sub.c_str());

            float cx, cy, cz;
            float temp;
            max_x = -INFINITY;
            max_y = -INFINITY;
            max_z = -INFINITY;
            min_x = +INFINITY;
            min_y = +INFINITY;
            min_z = +INFINITY;

            // Lecture de la matrice de pixels ligne par ligne
            for (int i = 0; i < size; i++)
            {

                fi >> cx;
                fi >> cy;
                fi >> cz;
                fi >> temp;

                Point city(cx, cy, cz);

                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cz >= max_z)
                    max_z = cz;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
                if (cz < min_z)
                    min_z = cz;
            }

            fi.close();
        }


        return true;
    }

    //! wb.Q 062018 add to read 3D point from files.
    bool read3DPoints(string str_file, NeuralNet<Point, Value>& nn,
                      GLfloatP& min_x, GLfloatP& min_y,
                      GLfloatP& max_x, GLfloatP& max_y)
    {
        char buf[256];

        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3D_FILE);

        cout << "read 3d points from: " << str_sub << endl;

        // count total size of input points
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture 3D file: " << str_sub << endl;
        }
        else
        {
            int size = 0;
            while(fi.getline(buf, 256)){
                size ++;
            }
            fi.close();
            cout << "qiao size " << size << endl;

            if (size)
                nn.resize(size, 1);
            else
                cout << "error read cities: num = 0." << endl;

            fi.open(str_sub.c_str());

            float cx, cy;
            float temp;
            max_x = -INFINITY;
            max_y = -INFINITY;
            min_x = +INFINITY;
            min_y = +INFINITY;

            // Lecture de la matrice de pixels ligne par ligne
            for (int i = 0; i < size; i++)
            {

                fi >> cx;
                fi >> cy;
                fi >> temp;
                fi >> temp;

                Point city(cx, cy);

                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
            }

            fi.close();
        }

        return true;
    }

    //! wb.Q 200716 add to read tsp file into a (N * 1) NN (md).
    bool readTSP(string str_file, NeuralNet<Point, Value>& nn, GLfloatP& min_x, GLfloatP& min_y, GLfloatP& max_x, GLfloatP& max_y){

        char buf[256];
        char *ptr;

        char str[256];
        char str_2[10];

        char name[50];
        int size = 0;

        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_TSP_FILE);

        cout << "read tsp from: " << str_sub << endl;

        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture tsp file: " << SFX_TSP_FILE << endl;
        }
        else
        {
            fi.getline(buf, 256);
            if (!fi)
                return(0);
            //                return(fi);// qiao: windows can not transfer correctly from ifstream to bool

            if ( (ptr = strstr(buf, "NAME")) != NULL )
            {
                sscanf(ptr, "%s%s%s", str, str_2, &name);
            }
            else
                return(0);
            //                return(fi);

            for (int l = 1; l <= 3; l++)
            {
                fi.getline(buf, 256);
                if (!fi)
                    return(0);
            }

            if ( (ptr = strstr(buf, "DIMENSION")) != NULL )
            {
                sscanf(ptr, "%s%s%d", str, str_2, &size);
            }
            else
                return(0);

            if (size)
                nn.resize(size, 1);
            else
                cout << "error read cities: num = 0." << endl;

            for (int l = 1; l <= 2; l++)
            {
                fi.getline(buf, 256);
                if (!fi)
                    return(0);
            }

            int num = 0;
            float cx, cy;

            max_x = -INFINITY;
            max_y = -INFINITY;
            min_x = +INFINITY;
            min_y = +INFINITY;


            // Lecture de la matrice de pixels ligne par ligne
            for (int i = 0; i < size; i++)
            {

                fi >> num;
                fi >> cx;
                fi >> cy;

                Point city(cx, cy);

                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
            }

        }
        fi.close();

    }

};

typedef InputRW<Point2D, GLfloat> EMSTRW;

} // namespace components

#endif // IMAGERW_H

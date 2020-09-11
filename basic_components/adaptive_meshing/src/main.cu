#include <iostream>
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NIter.h"
#include "ViewGrid.h"

#include "TestCellular.h"
#include "TestSom.h"
//#include "TestSomSuperpixel.h"
#include "TestSomTSP.h"
//#include "TestOpticalFlow.h"
//#include "TestStereo.h"

using namespace std;
using namespace components;
//using namespace operators;
using namespace meshing;

#define TEST_CODE  0
#define SECTION_PARAMETRES  0

int main(int argc, char *argv[])
{
    char* fileData;
    char* fileSolution;
    char* fileStats;
    char* fileConfig;

    /*
     * Lecture des fichiers d'entree
     */
    if (argc <= 1)
    {
        fileData = "input.data";
        fileSolution = "output.data";
        fileStats = "output.stats";
        fileConfig = "config.cfg";
    }
    else
    if (argc == 2)
    {
        fileData = argv[1];
        fileSolution = "output.data";
        fileStats = "output.stats";
        fileConfig = "config.cfg";
    }
    else
    if (argc == 3)
    {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = "output.stats";
        fileConfig = "config.cfg";
    }
    else
    if (argc == 4)
    {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = "config.cfg";
    }
    else
    if (argc >= 5)
    {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = argv[4];
    }
    //cout << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
    ConfigParams params(fileConfig);
    params.readConfigParameters();

    /*
     * Modification eventuelle des parametres
     */
//!JCC 030415 : I found "#if!"
#if SECTION_PARAMETRES

    //[global_param]

    //# choix du mode de fonctionnement 0:evaluation, 1:som
    params.functionModeChoice = 2;

#endif //SECTION_PARAMETRES

    if (params.functionModeChoice == 0) {
        cout << "TEST NITER" << endl;
        TestNiter<GLfloat, 21, 21, 0, 6, NIterHexa > t0(0);
        t0.run();
        TestNiter<GLfloat, 21, 21, 2, 6, NIterHexa > t1(0);
        t1.run();
        TestNiter<GLfloat, 21, 21, 0, 6, NIterTetra > t2(0);
        t2.run();
        TestNiter<GLfloat, 21, 21, 2, 6, NIterTetra > t3(0);
        t3.run();
        TestNiter<GLfloat, 21, 21, 0, 6, NIterQuad > t4(0);
        t4.run();
        TestNiter<GLfloat, 21, 21, 2, 6, NIterQuad > t5(0);
        t5.run();
//        TestNiter<Point2D, 230, 190, 0, 100, NIterHexa > t4(Point2D(5,5));
//        t4.run();
        cout << "Fin de test " << params.functionModeChoice << '\n';
    }
    else if (params.functionModeChoice == 1) {
        cout << "TEST VIEW GRID" << endl;
        const size_t _X = 434, _Y = 383;
        const int R = 24;
        Grid<Point2D> carte(_X, _Y);
        for (int y = 0; y < _Y; y++)
        {
            for (int x = 0; x < _X; x++)
            {
                carte[y][x].set(0, x);
                carte[y][x].set(1, y);
            }
        }
        PointCoord initPoint(carte.getWidth() / 2, carte.getHeight() / 2 + 1);
        TestViewGrid <ViewGridHexa, Grid<Point2D>, _X, _Y> t1(carte, initPoint, fileSolution);
        t1.run();
        cout << "Fin de test " << params.functionModeChoice << endl;
    }
    else if (params.functionModeChoice == 2) {
        cout << "TEST CELLULAR MATRIX" << endl;
        TestCellular t(fileData, fileSolution, fileStats, params);
        t.initialize();
        t.run();
        cout << "Fin de test " << params.functionModeChoice << endl;
    }
    else if (params.functionModeChoice == 3) {
        cout << "TEST SOM" << endl;
        TestSom t(fileData, fileSolution, fileStats, params);
        t.initialize();
        t.run();
        cout << "Fin de test " << params.functionModeChoice << endl;
    }
//    else if (params.functionModeChoice == 4) {
//        cout << "TEST SOM SP" << endl;
//        TestSomSuperpixel t(fileData, fileSolution, fileStats, params);
////        t.initialize();
//        t.run();
//        cout << "Fin de test " << params.functionModeChoice << endl;
//    }
    else if (params.functionModeChoice == 5) {
        cout << "TEST SOM TSP" << endl;
        TestSomTSP t(fileData, fileSolution, fileStats, params);
        t.initialize();
        t.run();
        cout << "Fin de test " << params.functionModeChoice << endl;
    }
//    else if (params.functionModeChoice == 6) {
//        cout << "TEST Optical Flow" << endl;
//        TestOpticalFlow t(fileData, fileSolution, fileStats, params);
//#if LR_CHECK_AND_POST_PROCESS
//        t.initialize(true);
//        t.run(true);
//        t.initialize();
//        t.run();
//        t.post_process();
//#else
//        t.initialize();
//        t.run();
//#endif
//        cout << "Fin de test " << params.functionModeChoice << endl;
//    }
//    else if (params.functionModeChoice == 7) {
//        cout << "TEST Stereo" << endl;
//        TestStereo t(fileData, fileSolution, fileStats, params);
//#if LR_CHECK_AND_POST_PROCESS
//        t.initialize(true);
//        t.run(true);
//        t.initialize();
//        t.run();
//        t.post_process();
//#else
//        t.initialize();
//        t.run();
//#endif
//        cout << "Fin de test " << params.functionModeChoice << endl;
//    }
    return 0;
}//main


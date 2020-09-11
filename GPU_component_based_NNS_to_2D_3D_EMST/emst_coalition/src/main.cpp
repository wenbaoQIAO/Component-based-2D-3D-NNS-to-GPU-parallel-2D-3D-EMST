#include <iostream>
#include "config/ConfigParamsCF.h"
#include "CalculateurEMST.h"
#include "NeuralNet.h"

#include "random_generator_cf.h"
//#include "Multiout.h"

using namespace std;

#define TEST_3D_GRID        1

using namespace components;

NN mrH;
Grid3D grid3D;
//extern Solution solH;

int main(int argc, char *argv[])
{
    char* fileData;
    char* fileSolution;
    char* fileStats;
    char* fileConfig;

    /*
     * Lecture des fichiers d'entree
     */
    if (argc <= 1) {
        fileData = (char*) "input.svg";
        fileSolution = (char*) "output.svg";
        fileStats = (char*) "output.stats";
        fileConfig = (char*) "config.cfg";
    } else if (argc == 2) {
        fileData = argv[1];
        fileSolution = (char*) "output.svg";
        fileStats = (char*) "output.stats";
        fileConfig = (char*) "config.cfg";
    } else if (argc == 3) {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = (char*) "output.stats";
        fileConfig = (char*) "config.cfg";
    } else if (argc == 4) {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = (char*) "config.cfg";
    } else if (argc >= 5) {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = argv[4];
    }
    cout << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
    config::ConfigParamsCF* params = new config::ConfigParamsCF(fileConfig);
    params->readConfigParameters();

    /*
     * Modification eventuelle des parametres
     */
#if TEST_3D_GRID
    grid3D.resize(6,222,180);
    // Test Grille
//    ifstream fi;
//    fi.open("essaiH.txt");
//    if (fi) {
//        fi >> grid3D;
//        fi.close();
//    }
//    else
//        cout << "pb reading file" << endl;

    grid3D = Point3D(2,2,2);//.gpuResetValue(Point3D(1,1,1));
    ofstream fo;
    fo.open("essaiH.txt");
    if (fo) {
        fo << grid3D;
        fo.close();
    }
    else
        cout << "pb file" << endl;
#endif
    CalculateurEMST::initialize(fileData, fileSolution, fileStats, params);
    CalculateurEMST::run();

    return 0;
}


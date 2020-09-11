#include <iostream>
#include "config/ConfigParamsCF.h"
#include "Calculateur.h"
#include "NeuralNet.h"

//#include "Solution.h"

#include "random_generator_cf.h"
//#include "Multiout.h"

using namespace std;

// Deux exemples d'appel de calcul d'imbrication via fichiers d'entree
// ou via fichier avec surcharge par programme des parammetres avant le lancement
#define EXEMPLE_1   0
#define EXEMPLE_2   1

#define SECTION_PARAMETRES  0

using namespace components;
//using namespace nnH;

NN mrH;
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
    //lout << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
    config::ConfigParamsCF* params = new config::ConfigParamsCF(fileConfig);
    params->readConfigParameters();

//    ConfigParams paramsOF(fileConfig);
//    paramsOF.readConfigParameters();
    /*
     * Modification eventuelle des parametres
     */

    // Lancement de l'imbricateur
    mrH.colorMap.resize(10,10);
    mrH.colorMap = Point3D(1,1,1);//.gpuResetValue(Point3D(1,1,1));
    ofstream fo;
    fo.open("essaiH.txt");
    if (fo) {
        fo << mrH.colorMap;
        fo.close();
    }
    else
        cout << "pb file" << endl;

//    solH.initialize(fileData, fileSolution, fileStats);
//    //solH.readSolution();
//    solH.initStatisticsFile();

    Calculateur::initialize(fileData, fileSolution, fileStats, params);
    Calculateur::run();

    return 0;
}


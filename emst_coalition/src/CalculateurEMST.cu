#include "CalculateurEMST.h"

#include "SolutionEMST.h"
#include "LocalSearch.h"
#include "AgentMetaSolver.h"
#include "random_generator_cf.h"
//#include "Multiout.h"

using namespace std;

#define TEST_CODE   0
#define TEST_ALEAT_DOUBLE   1
#define TEST_3D_GRID        0

typedef SolutionEMST<2, 2> Solution;

typedef SolutionEMST<3, 3> Solution3D;

typedef SolutionEMST<3, 2> Solution2DPoint5;

Solution* sol = NULL;
Solution* sol1 = NULL;
LocalSearch<Solution>* lS = NULL;
AgentMetaSolver<Solution>* Gm = NULL;

Solution3D* sol3D = NULL;

Solution2DPoint5* sol2DPoint5 = NULL;

#if TEST_3D_GRID
extern Grid3D grid3D;
extern NN mrH;
//Solution solH;
#endif

Grid<Point3D>* CalculateurEMST::getAdaptiveMap() {
    return sol2DPoint5->getAdpativeMap();
}

Grid<Point3D>* CalculateurEMST::getAdaptiveMap3() {
    return sol3D->getAdpativeMap();
}

Grid<BufferLinkPointCoord>* CalculateurEMST::getLinks() {
    return sol2DPoint5->getLinks();
}

Grid<BufferLinkPointCoord>* CalculateurEMST::getLinks3() {
    return sol3D->getLinks();
}

void CalculateurEMST::initialize(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params)
{
    g_ConfigParameters = params;

    // Initialise le générateur de nombres aléatoires
    if (!g_ConfigParameters->useSeed) {
        g_ConfigParameters->seedValue = random_cf::aleat_get_time();
        cout << "SEED VALUE " << g_ConfigParameters->seedValue << endl;
    }
    random_cf::aleat_initialize(g_ConfigParameters->seedValue);
#if TEST_ALEAT_DOUBLE
    for (int i = 0; i < 10 ; ++i) {
        cout << random_cf::aleat_double(0, 1) << " ";
    }
    cout << endl;
#endif
#if TEST_3D_GRID
    //NN mrHgpu;
    Grid3D grid3D_gpu;
    //mrH.colorMap.resize(10,10,19);
    grid3D_gpu.gpuResize(6,222,180);
    //mrH.gpuCopyHostToDevice(mrHgpu);
    grid3D_gpu.gpuResetValue(Point3D(5,6,7));
    grid3D.gpuCopyDeviceToHost(grid3D_gpu);
    ofstream fo;
    fo.open("essaiD.txt");
    if (fo) {
        fo << grid3D;
        fo.close();
    }
    else
        cout << "pb file" << endl;
#endif
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "INIT EVALUATE" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile();
        break;

    case LOCAL_SEARCH:
        cout << "INIT LOCAL SEARCH" << endl;

        lS = new LocalSearch<Solution>();
        lS->initialize(fileData, fileSolution, fileStats);
        break;

    case GENETIC_METHOD:
    {
        cout << "INIT GENETIC METHOD" << endl;

        Gm = new AgentMetaSolver<Solution>();
        Gm->initialize(fileData, fileSolution, fileStats);
    }
        break;

    case CONSTRUCTION:
    {
        cout << "CONSTRUCTION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile();
    }
        break;

    case RUN:
    {
        cout << "RUN 4 initialize" << endl;

        sol1 = new Solution();
        sol1->initialize(fileData, fileSolution, fileStats);
        sol1->readSolution();
        sol1->initStatisticsFile();

        //sol = new Solution();
        //sol1->clone(sol);

        // Avant construction
        //sol->evaluate();

        //sol->writeStatisticsToFile(-1);
        //sol->writeHeaderStatistics(lout);
        //sol->writeStatistics(-1, lout);
    }
        break;

    case RUN_3D:
    {
        cout << "RUN 3D initialize" << endl;

        sol3D = new Solution3D();
        sol3D->initialize(fileData, fileSolution, fileStats);
        sol3D->readSolution();
        sol3D->initStatisticsFile();
    }
        break;
    case RUN_2DPOINT5:
    {
        cout << "RUN 2D initialize" << endl;

        sol2DPoint5 = new Solution2DPoint5();
        sol2DPoint5->initialize(fileData, fileSolution, fileStats);
        sol2DPoint5->readSolution();
        sol2DPoint5->initStatisticsFile();
    }
        break;
    default:
        cout << "UNSUPPORTED FUNCTIONMODE=" << params->functionModeChoice << " !! " << endl;
        break;
    }

}//initialize

void CalculateurEMST::run()
{
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);

        sol->writeSolution();

        delete sol;
        break;

    case LOCAL_SEARCH:
        cout << "LOCAL SEARCH" << endl;

        lS->run();

        delete lS;
        break;

    case GENETIC_METHOD:
        cout << "GENETIC METHOD" << endl;

        Gm->run();

        cout << "END GENETIC METHOD" << endl;

        delete Gm;
        break;

    case CONSTRUCTION:
         cout << "CONSTRUCTION" << endl;

        sol->constructSolutionSeq();
        // Après construction
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN:
        cout << "RUN 4 run " << endl;

        sol1->run();
        // Après run
        sol1->evaluate();
        //sol->writeStatisticsToFile(-1);
        //sol->setIdentical(sol1);
        sol1->writeStatisticsToFile();
        sol1->writeHeaderStatistics(cout);
        sol1->writeStatistics(cout);
        sol1->writeSolution();
        delete sol;
        break;

    case RUN_3D:
        cout << "RUN 3D run" << endl;

        sol3D->run();
        // Après run
        sol3D->evaluate();// wq delete this evaluation to do running time statistic again 24November18
        sol3D->writeStatisticsToFile();
        sol3D->writeHeaderStatistics(cout);
        sol3D->writeStatistics(cout);
        sol3D->writeSolution();// wq delete this evaluation to do running time statistic again 24November18
        delete sol3D;
        break;

    case RUN_2DPOINT5:
        cout << "RUN 2D run" << endl;

        sol2DPoint5->run();
        // Après run
        sol2DPoint5->evaluate();// wq delete this evaluation to do running time statistic again 24November18
        sol2DPoint5->writeStatisticsToFile();
        sol2DPoint5->writeHeaderStatistics(cout);
        sol2DPoint5->writeStatistics(cout);
        sol2DPoint5->writeSolution();// wq delete this evaluation to do running time statistic again 24November18
        delete sol2DPoint5;
        break;

    default:
        cout << "UNSUPPORTED FUNCTIONMODE=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

}//run

bool CalculateurEMST::activate()
{
    bool ret = true;
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);

        sol->writeSolution();

        delete sol;
        break;

    case LOCAL_SEARCH:
        cout << "LOCAL SEARCH" << endl;

        lS->run();

        delete lS;
        break;

    case GENETIC_METHOD:
        cout << "GENETIC METHOD" << endl;

        Gm->run();

        cout << "END GENETIC METHOD" << endl;

        delete Gm;
        break;

    case CONSTRUCTION:
         cout << "CONSTRUCTION" << endl;

        sol->constructSolutionSeq();
        // Après construction
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN:
        cout << "RUN 4 activate " << endl;

        sol1->run();
        // Après run
        sol1->evaluate();
        //sol->writeStatisticsToFile(-1);
        //sol->setIdentical(sol1);
        sol1->writeStatisticsToFile();
        sol1->writeHeaderStatistics(cout);
        sol1->writeStatistics(cout);
        sol1->writeSolution();
        delete sol;
        break;

    case RUN_3D:
        cout << "RUN 3D activate" << endl;

        sol3D->activate();
        // Après run
        sol3D->evaluate();
        sol3D->writeStatisticsToFile();
        sol3D->writeHeaderStatistics(cout);
        sol3D->writeStatistics(cout);
        sol3D->writeSolution();
        delete sol3D;
        break;

    case RUN_2DPOINT5:
        cout << "RUN 2D activate" << endl;

        ret = sol2DPoint5->activate();
        // Après run
        sol2DPoint5->evaluate();
        sol2DPoint5->writeStatisticsToFile();
        sol2DPoint5->writeHeaderStatistics(cout);
        sol2DPoint5->writeStatistics(cout);
        sol2DPoint5->writeSolution();

        break;

    default:
        cout << "UNSUPPORTED FUNCTIONMODE=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

    return ret;

}//activate

#ifndef SEPARATE_COMPILATION
#include "..\src\SolutionEMST.cu"
#include "..\src\SolutionEMSTRW.cu"
#include "..\src\SolutionEMSTOperators.cu"
#endif

#include "Calculateur.h"

#include "Solution.h"
#include "LocalSearch.h"
#include "AgentMetaSolver.h"
#include "random_generator_cf.h"
#include "Multiout.h"

using namespace std;

#define TEST_CODE   0

Solution* sol = NULL;
LocalSearch<Solution>* lS = NULL;
AgentMetaSolver<Solution>* Gm = NULL;

void Calculateur::calcul(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params)
{
    g_ConfigParameters = params;

    lout << "VERSION CALCULATEUR : " << VERSION_CALCULATEUR << endl;

    // Initialise le générateur de nombres aléatoires
    if (!g_ConfigParameters->useSeed) {
        g_ConfigParameters->seedValue = random::aleat_get_time();
    }
    aleat_initialize(g_ConfigParameters->seedValue);

    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
        case EVAL_ONLY:
        {
            lout << "EVALUATE" << endl;

            Solution* sol;
            sol = new Solution();
            sol->initialize(fileData, fileSolution, fileStats);
            sol->readSolution();
            sol->initStatisticsFile();

            sol->initEvaluate();
            sol->evaluate();

            sol->writeStatisticsToFile(-1);
            sol->writeHeaderStatistics(lout);
            sol->writeStatistics(-1, lout);

            sol->writeSolution();

            delete sol;
        }
        break;

        case LOCAL_SEARCH:
        {
            lout << "LOCAL SEARCH" << endl;

            LocalSearch<Solution>* lS = NULL;
            lS = new LocalSearch<Solution>();
            lS->initialize(fileData, fileSolution, fileStats);
            lS->run();

            delete lS;
        }
        break;

        case GENETIC_METHOD:
        {
            cout << "GENETIC METHOD" << endl;

            AgentMetaSolver<Solution>* Gm = NULL;
            Gm = new AgentMetaSolver<Solution>();
            Gm->initialize(fileData, fileSolution, fileStats);
            Gm->run();

            cout << "END GENETIC METHOD" << endl;

            delete Gm;
        }
        break;

        case CONSTRUCTION:
        {
            lout << "CONSTRUCTION" << endl;

            Solution* sol;

            sol = new Solution();
            sol->initialize(fileData, fileSolution, fileStats);
            sol->readSolution();
            sol->initStatisticsFile();

            // Avant construction
            sol->evaluate();

            sol->writeStatisticsToFile(-1);
            sol->writeHeaderStatistics(lout);
            sol->writeStatistics(-1, lout);

            for (int i = 0; i < 10; ++i) {
                sol->constructSolutionSeq();
                char tmp[255];
                strncpy(tmp, fileSolution, strlen(fileSolution));
                char* pos = tmp;
                if ((pos = strstr(tmp, ".sol")) != NULL)
                {
                    *pos = '\0';
                }
                ostringstream os;
                os << fileSolution << "__" << i << ".sol";
                sol->writeSolution(os.str().c_str());

                if (i == 0) {
                    // Après construction
                    sol->evaluate();
                    sol->writeStatisticsToFile(-1);
                    sol->writeHeaderStatistics(lout);
                    sol->writeStatistics(-1, lout);
                }
            }

            delete sol;
        }
        break;

        default:
            lout << "UNSUPPORTED FUNCTIONMODE=" << params->functionModeChoice << " !! " << endl;
            break;
    }

}//calcul

extern NN mrH;
//Solution solH;

void Calculateur::initialize(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params)
{
    g_ConfigParameters = params;

    lout << "VERSION CALCULATEUR : " << VERSION_CALCULATEUR << endl;

    // Initialise le générateur de nombres aléatoires
    if (!g_ConfigParameters->useSeed) {
        g_ConfigParameters->seedValue = random::aleat_get_time();
    }
    aleat_initialize(g_ConfigParameters->seedValue);

    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        lout << "INIT EVALUATE" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile();
        break;

    case LOCAL_SEARCH:
        lout << "INIT LOCAL SEARCH" << endl;

        lS = new LocalSearch<Solution>();
        lS->initialize(fileData, fileSolution, fileStats);
        break;

    case GENETIC_METHOD:
    {
        cout << "INIT GENETIC METHOD" << endl;

        NN mrHgpu;
        mrH.resize(10,10);
        mrHgpu.gpuResize(10,10);
        //mrH.gpuCopyHostToDevice(mrHgpu);
        mrHgpu.colorMap.gpuResetValue(Point3D(2,2,2));
        mrH.colorMap.gpuCopyDeviceToHost(mrHgpu.colorMap);
        ofstream fo;
        fo.open("essaiD.txt");
        if (fo) {
            fo << mrH.colorMap;
            fo.close();
        }
        else
            cout << "pb file" << endl;
        Gm = new AgentMetaSolver<Solution>();
        Gm->initialize(fileData, fileSolution, fileStats);
    }
        break;

    case CONSTRUCTION:
        lout << "CONSTRUCTION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile();

        // Avant construction
        sol->evaluate();

        sol->writeStatisticsToFile(-1);
        sol->writeHeaderStatistics(lout);
        sol->writeStatistics(-1, lout);

        break;

    case RUN:
        lout << "RUN" << endl;

        Solution* sol1;
        sol1 = new Solution();
        sol1->initialize(fileData, fileSolution, fileStats);
        sol1->readSolution();
        sol1->initStatisticsFile();

        sol = new Solution();
        sol1->clone(sol);

        // Avant construction
        sol->evaluate();

        sol->writeStatisticsToFile(-1);
        sol->writeHeaderStatistics(lout);
        sol->writeStatistics(-1, lout);

        break;

    default:
        lout << "UNSUPPORTED FUNCTIONMODE=" << params->functionModeChoice << " !! " << endl;
        break;
    }

}//initialize

void Calculateur::run()
{
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        lout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile(-1);
        sol->writeHeaderStatistics(lout);
        sol->writeStatistics(-1, lout);

        sol->writeSolution();

        delete sol;
        break;

    case LOCAL_SEARCH:
        lout << "LOCAL SEARCH" << endl;

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
         lout << "CONSTRUCTION" << endl;

        sol->constructSolutionSeq();
        // Après construction
        sol->evaluate();
        sol->writeStatisticsToFile(-1);
        sol->writeHeaderStatistics(lout);
        sol->writeStatistics(-1, lout);
        delete sol;
        break;

    case RUN:
        lout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile(-1);
        sol->writeHeaderStatistics(lout);
        sol->writeStatistics(-1, lout);
        delete sol;
        break;

    default:
        lout << "UNSUPPORTED FUNCTIONMODE=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

}//run

#ifndef SEPARATE_COMPILATION
#include "..\src\Solution.cu"
#include "..\src\SolutionRW.cu"
#include "..\src\SolutionOperators.cu"
#endif

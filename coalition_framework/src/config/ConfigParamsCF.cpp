#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "config/ConfigParamsCF.h"

//#include "config/chameleon.h"
#include "geometry_prop.h"
//#include "Multiout.h"

config::ConfigParamsCF* g_ConfigParameters = NULL;

namespace config
{

ConfigParamsCF::ConfigParamsCF()
{}

ConfigParamsCF::ConfigParamsCF(char* file) : ConfigParams(file)
{

    if (file == NULL)
        configFile = "config.cfg";

    initDefaultParameters();
}

void ConfigParamsCF::initDefaultParameters()
{
    // Parametres generaux
    ConfigParams::initDefaultParameters();
    
    functionModeChoice = 2;

    useThreads = false;

    useSeed = false;
    seedValue = 0;

    consoleoutput = true;
    fileoutput = false;

    constructFromScratchParam = true;

    traceActive = true;
    traceReportBest = true;
    traceSaveSolutionFile = false;

    bruitConstruction_x = 2000;// en mm
    bruitConstruction_y = 1000;
    aleatMove_x = 1000;
    aleatMove_y = 500;

    // Recherche locale

    // 0: marche aleatoire, 1: LS first improvement, 2: LS best improvement
    localSearchType = 1;

    neighborhoodSize = 1000;
    nbOfConstructAndRepairs = 10;
    nbOfInternalConstructs = 100;
    nbOfInternalRepairs = 20000;

    // Genetic method

    memeticAlgorithm = true;

    gaPercentageOfBestIndividuals = 0.2;
    gaPercentageOfElitistIndividual = 0.1;

    probaMutateGA2 = 0.001;
    probaMutateMA = 0.2;

    double probaOperators_val[8] ={0.01, 0.05, 0.05, 0.0003, 0.005, 0.6, 0.1, 0.01};
    probaOperators.clear();
    for (int i = 0; i<sizeof(probaOperators_val)/sizeof(double); i++)
    {
        probaOperators.push_back(probaOperators_val[i]);
    }

    // "genetic_method"
    populationSize = 10;
    generationNumber = 50;
    MAneighborhoodSize = 100;
    MAnbOfInternalConstructs = 50;
    MAnbOfInternalRepairs = 2000;
    firstAgentUseInit = false;

    gaPercentageOfBestIndividuals = 0.2;
    gaPercentageOfElitistIndividual = 0.1;

    probaMutateGA2 = 0.001;
    probaMutateMA = 0.2;

    // Generateur automatique d'instances

    nInstances = 100;

    // Intensite des mouvements de structure deploie
    structIntensityRandom = true;

    // Intensite des mouvements de structure
    // en phase d'amélioration.
    // type 0 : mvt constant, 1 mvt en proportion (les valeurs ci-dessous deviennent des %)
    randomIntensityType = 1;
    useRecuitRandomStructure = false;
    useGaussianRandomIntensity = false;

    debugUseProfiler = false;
}//initDefaultParameters

void ConfigParamsCF::readConfigParameters()
{
    ConfigParams::readConfigParameters();
    ConfigFile cf(configFile);

    consoleoutput = (bool)cf.Value("global_param", "consoleoutput", consoleoutput);
    fileoutput = (bool)cf.Value("global_param", "fileoutput", fileoutput);

    if (fileoutput)
    {
        // Supprimer le contenu du fichier de trace
        std::ofstream ofs;
        ofs.open("lout.trace", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    // Global Parameters
    functionModeChoice = (int)cf.Value("coalition_global_param", "functionModeChoice", functionModeChoice);
    cout << "coalition_global_param functionModeChoice " << functionModeChoice << endl;
    problemType = (int)cf.Value("coalition_global_param", "problemType", problemType);

    useThreads = (bool)cf.Value("global_param", "utiliseThreads", useThreads);
    if (useThreads)
    {
        cout << "Use Threads..." << endl;
    }

    traceActive = (bool)cf.Value("global_param", "traceActive", traceActive);

    useSeed = (bool)cf.Value("global_param", "useSeed", useSeed);
    seedValue = (int)cf.Value("global_param", "seedValue", seedValue);

    bruitConstruction_x = (int)cf.Value("global_param", "bruitConstruction_x", bruitConstruction_x);
    bruitConstruction_y = (int)cf.Value("global_param", "bruitConstruction_y", bruitConstruction_y);

    aleatMove_x = (int)cf.Value("global_param", "aleatMove_x", aleatMove_x);
    aleatMove_y = (int)cf.Value("global_param", "aleatMove_y", aleatMove_y);

    // Lit les probabilités associées aux opérateurs
    s_probaOperators = (string)cf.Value("global_param", "probaOperators", s_probaOperators);
    probaOperators.clear();
    boost::char_separator<char> sep(" ,");
    boost::tokenizer<boost::char_separator<char> > tok(s_probaOperators, sep);
    std::transform(tok.begin(), tok.end(), std::back_inserter(probaOperators), ToDouble());
    /*
    lout << "Operator probabilities: ";
    for (std::vector<double>::iterator it = probaOperators.begin(); it != probaOperators.end(); ++it)
    {
        lout << (*it) << ", ";
    }
    lout << endl;
    */

    // Local Search
    constructFromScratchParam = (bool)cf.Value("local_search", "constructFromScratchParam", constructFromScratchParam);
    localSearchType = (int)cf.Value("local_search", "localSearchType", localSearchType);

    neighborhoodSize = (int)cf.Value("local_search", "neighborhoodSize", neighborhoodSize);
    nbOfConstructAndRepairs = (int)cf.Value("local_search", "nbOfConstructAndRepairs", nbOfConstructAndRepairs);
    nbOfInternalConstructs = (int)cf.Value("local_search", "nbOfInternalConstructs", nbOfInternalConstructs);
    nbOfInternalRepairs = (int)cf.Value("local_search", "nbOfInternalRepairs", nbOfInternalRepairs);

    // Genetic Method
    memeticAlgorithm = (bool)cf.Value("genetic_method", "memeticAlgorithm", memeticAlgorithm);

    populationSize = (int)cf.Value("genetic_method", "populationSize", populationSize);
    cout << "generationNumber " << generationNumber << endl;
    generationNumber = (int)cf.Value("genetic_method", "generationNumber", generationNumber);
    cout << "generationNumber 2 " << generationNumber << endl;

    MAneighborhoodSize = (int)cf.Value("genetic_method", "MAneighborhoodSize", MAneighborhoodSize);
    MAnbOfInternalConstructs = (int)cf.Value("genetic_method", "MAnbOfInternalConstructs", MAnbOfInternalConstructs);
    MAnbOfInternalRepairs = (int)cf.Value("genetic_method", "MAnbOfInternalRepairs", MAnbOfInternalRepairs);
    firstAgentUseInit = (bool)cf.Value("genetic_method", "firstAgentUseInit", firstAgentUseInit);

    // Generateur d'instances
    nInstances = (int)cf.Value("instance_generator", "nInstances", nInstances);

    structIntensityRandom = (bool)cf.Value("structure", "structIntensityRandom", structIntensityRandom);

    // Intensite des mouvements de structure
    // en phase d'amélioration.
    // type 0 : mvt constant, 1 mvt en proportion
    randomIntensityType = (int)cf.Value("structure", "randomIntensityType", randomIntensityType);
    useRecuitRandomStructure = (bool)cf.Value("structure", "useRecuitRandomStructure", useRecuitRandomStructure);
    useGaussianRandomIntensity = (bool)cf.Value("structure", "useGaussianRandomIntensity", useGaussianRandomIntensity);

    // Selection des activations autorisees
    debugUseProfiler = (bool)cf.Value("debug", "debugUseProfiler", debugUseProfiler);

}//readConfigParameters

}//namespace config

#ifndef SOLUTION_H
#define SOLUTION_H
/*
 ***************************************************************************
 *
 * Auteurs : J.C. Creput, A. Mansour et F. Lauri
 * Date creation : mars 2014
 * Date derniere modification : septembre 2016
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <time.h>
//#include <sys/time.h>

using namespace std;

//#include <QtCore>
//#include <QDomDocument>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include "config/ConfigParamsCF.h"

#ifdef CUDA_CODE
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_35_atomic_functions.h>
#endif

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "ViewGrid.h"
#include "Cell.h"
#include "SpiralSearch.h"
#include "Objectives.h"
//#include "Trace.h"
#include "ConfigParams.h"
#include "CellularMatrix.h"
#include "SomOperator.h"
#include "ImageRW.h"
#include "Converter.h"
#include "filters.h"
#include "SpiralSearch.h"
#include "Evaluation.h"

#include "ViewGrid.h"
//#include "ViewGridHexa.h"
#include "NIter.h"
//#include "NIterHexa.h"

using namespace std;
using namespace components;
using namespace operators;

/*!
 * \brief Classe principale qui definit la structure
 * d'une solution du problème.
 *
 * Elle contient :
 * - Variables du problemes
 * - Objectifs du probleme et procedures d'evaluation
 * - Operations de manipulation d'une solution
 * - Operateurs utiles pour les algorithmes d'optimisation
 * - Templates des differents composants : il s'agit des composants
 * de la solution.
 * - Procedures de lecture/ecriture
 * - Utilitaires divers
 **/
class Solution
{
#pragma region Membres prives

    //! Fichier svg d'entree contenant une instance du probleme
    char* fileData;
    //! Fichier svg de sortie contenant une solution du probleme
    char* fileSolution;
    //! Fichier de sortie avec valeurs de criteres et objectifs de la solution
    char* fileStats;
    //! Flux de sortie ouvert pour statistiques
    static std::ofstream* OutputStream;
    //! Calcul duree d'execution
    time_t t0;
    //! Calcul duree d'execution
    time_t tf;
    //! Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
    double x0;
    //! Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
    double xf;
    // cuda timer
#ifdef CUDA_CODE
    cudaEvent_t start, stop;
#endif

    /*!
     * \brief Compteur d'instances pour gestion mémoire d'objets partagés.
     *
     * Attention est initialisée lors de la definition
     * i.e. int Solution::cptInstance = 0;
     * Certaines allocations mémoire ne sont réalisées qu'une seule fois
     * et seront partagées entre instances (pointeurs).
     */
    static int cptInstance;
#pragma endregion

public:
#pragma region Objectifs et criteres du probleme
    /*! @name Objectifs et criteres du probleme
     * @{
     */
     //! Valeur fonction objectif globale
    double global_objectif;
    //! @}
#pragma endregion

#pragma region OPTICAL FLOW
    /*! @name Data
     * @{
     */
//JCC    Trace trace;

public:
    // Types
    typedef ViewGridQuad ViewG;
    typedef NIterQuad NIter;
    typedef NIterQuadDual NIterDual;

    typedef CellB<CM_ColorDistanceEuclidean,
                    CM_ConditionTrue,
                    NIter, ViewG> CB;
    typedef CellSpS<CM_ColorDistanceEuclidean,
                    CM_ConditionTrue,
                    NIter, ViewG> CSpS;

    typedef CellularMatrix<CSpS, ViewG> CMSpS;
    typedef CellularMatrix<CB, ViewG> CMB;

    // Data
    NN md;
    NN mr;

    //! @}
#pragma endregion

#pragma region Manipulation d instance
    //! \brief Constructeur par defaut.
    //! Il cree explicitement les objets partages a toutes les
    //! instances
    explicit Solution()
    {
        cptInstance += 1;
        if (cptInstance == 1)
        {
            CreateCommonData();
        }
    }
    //! \brief Destructeur.
    //! Il detruit explicitement les objets partages
    //! s'il s'agit de la derniere instance existante.
    ~Solution()
    {
        cptInstance -= 1;
        if (cptInstance == 0)
        {
            freeCommonData();
        }
    }
    void CreateCommonData()
    {
        OutputStream = new ofstream;
    }
    //! \brief Ne peut etre appelee qu'une seule fois lors de la
    //! destruction de la derniere instance
    void freeCommonData()
    {
        delete OutputStream;
    }

    //! Initialisations
    void initialize(char* data, char* sol, char* stats);
    //! Initialisations
    void initialize();
    void initialize(NN& md, NN& mr);
    //! Operation de copie
    void setIdentical(Solution* imb);
    //! Operation de copie
    void clone(Solution* imb);
#pragma endregion

#pragma region Evaluation globale

    /*!
     * @name Evaluation globale
     * \brief Fonctions d'evaluation d'une solution.
     * @{
     */
     //! \brief Valeurs par defaut des objectif
    void initEvaluate();
    //! \brief Evaluation complete d'une solution
    double evaluate();
    //! \brief Fonction objectif agregative globale
    double computeObjectif();
    //! \brief Comparaison de 2 solutions
    bool isBest(Solution* imb);
    //! \brief Test d'admissibilite de la solution
    bool isSolution();

    //! @}
#pragma endregion

#pragma region Operateurs
    /**
     * @name Operateurs
     * \brief Operations de transformation (manipulation, deplacement)
     * d'une solution. Implementations dans le fichier SolutionOperateurs.cpp.
     * @{
     */
    //! \brief Cette methode est appelee au debut du processus d'optimisation
    //! une seule fois en principe.
    void initConstruct();
    /** \brief Construction Sequentielle de depart.
     */
    void constructSolutionSeq();

    //! \brief La methode "generateNeighbor()" constitue l'operateur principal de
    //! la recherche locale.
    bool generateNeighbor();

    //! Application d'un des opérateurs
    bool applyOperator(int i);
    //! Nombre d'opérateurs pouvant être appliqués
    int nbrOperators() const;

    //! \brief evalPartielleComposantX
    //! \return Contribution d'un composant x
    //!
    double evalPartielleComposantX() { return 0; }

    //! \brief Operateurs de base (mouvement, swap)
    bool operator_1();
    bool operator_2();

    //! \brief Run et activate
    void run() {
        for (int i = 0; i < g_ConfigParameters->generationNumber; ++i) {
            activate();
            writeStatisticsToFile(i);
            writeStatistics(i, cout);
        }
    }

    void activate() {
        operator_1();
    }

    //! @}
#pragma endregion

#pragma region file parsing
    /**
     * @name Lecture/ecriture
     * \brief Operateurs de lecture/ecriture.
     *
     * @{
     */
     //! Lecture d'une instance du probleme
    void readPbInstance()
    {
        readSolution();
    }
    //! Ecriture d'une instance du probleme
    void writePbInstance()
    {
        writeSolution();
    }
    //! Ecriture d'une instance du probleme
    void writePbInstance(char* fileData)
    {
        writeSolution(fileData);
    }
    //! Lecture 'une solution du probleme
    void readSolution();
    //! Lecture d'une solution du probleme a partir du fichier specifie en parametre
    void readSolution(const char* fileName);
    //! Ecriture d'une solution du probleme
    void writeSolution();
    //! Ecriture d'une solution du probleme a partir du fichier specifie en parametre
    void writeSolution(const char* fileName);
    //! @}
#pragma endregion

#pragma region Statistics
protected:
    //! Ouverture du fichier de sortie (texte) pour statistiques
    void openStatisticsFile();
    //! Fermeture du fichier de sortie (texte) pour statistiques
    void closeStatisticsFile();
public:

    //! Initialisation du fichier de sortie (texte) pour statistiques
    void initStatisticsFile();
    //! Initialisation de l'entete du fichier de sortie (texte) pour statistiques
    void writeHeaderStatistics(std::ostream& o);
    //! Ecriture des valeurs de criteres/objectifs dans le fichier (texte) de statistiques
    void writeStatisticsToFile(int iteration);
    //! Ecriture des valeurs de criteres/objectifs dans le fichier de statistiques dans un flux
    void writeStatistics(int iteration, std::ostream& o);
#pragma endregion

};


#endif // SOLUTION_H

#ifndef SolutionEMST_H
#define SolutionEMST_H
/*
 ***************************************************************************
 * Author : Wenbao Qiao, J.C. Créput
 * Creation date : Septembre. 2016
 *
 * Note: This application aims at building parallel Euclidean minimum spanning tree without predefining N*N edge list.
 * Main idea of this parallel EMST method is that it combines Borůvka algorithm with Elias' nearest neighbor spiral search approaches.
 * If you use these codes, please reference our published papers.
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <iomanip>

#include <time.h>
//#include <sys/time.h>

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "ViewGrid.h"
#include "Cell.h"
#include "SpiralSearch.h"
#include "Objectives.h"
#include "basic_operations.h"
//#include "Trace.h"
#include "ConfigParams.h"
#include "CellularMatrix.h"
//#include "SomOperator.h" // wb.Q 201908 temporarily delete SomOperator.h for EMST due to conflication
#include "ImageRW.h"
#include "Converter.h"
#include "filters.h"
//#include "cudaerrorcheck.h"

//! WB.Q reference to EMST components
#include "NeuralNetEMST.h"
//#include "TraceEMST.h"
#include "InputRW.h"
//#include "macros_cuda_EMST.h"
#include "NodeEMST.h"
#include "EMSTOperators.h"

using namespace std;
using namespace components;
using namespace operators;

#define EMST_SPIRAL_SEARCH_2D 0 // Old 2D spiral search or new KD spiral search with slab technique
#define EMST_SPIRAL_SEARCH_KD_SLAB_2 1

/*!
 * \brief Classe principale qui definit la structure
 * d'une solution du probleme.
 *
 * Elle contient :
 * - Variables du problemes
 * - Objectifs du probleme et procedures d'evaluation
 * - Operations de manipulation d'une SolutionEMST
 * - Operateurs utiles pour les algorithmes d'optimisation
 * - Templates des differents composants : il s'agit des composants
 * de la SolutionEMST.
 * - Procedures de lecture/ecriture
 * - Utilitaires divers
 **/
template<std::size_t DimP, std::size_t DimCM>
class SolutionEMST
{
#pragma region Membres prives

    //! Fichier svg d'entree contenant une instance du probleme
    char* fileData;
    //! Fichier svg de sortie contenant une SolutionEMST du probleme
    char* fileSolution;
    //! Fichier de sortie avec valeurs de criteres et objectifs de la SolutionEMST
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
#ifdef CUDA_CODE
    float time_next_closest;
    float time_find_pair;
    //wb.Q add accumulative time for these two steps
    float time_connect_union;
    float time_flatten;
    float time_terminate;
#endif

    /*!
     * \brief Compteur d'instances pour gestion mémoire d'objets partagés.
     *
     * Attention est initialisée lors de la definition
     * i.e. int SolutionEMST::cptInstance = 0;
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
    double global_objectif = 0;
    //! @}
#pragma endregion

#pragma region OPTICAL FLOW
    /*! @name Data
     * @{
     */

public:
    // Constante
    static const unsigned int DimG = 2;
    // Types
    typedef Index<DimCM> IndexCM;
    typedef IndexCM ExtentsCM; // extends are dimensions lengths
    typedef Index<DimG> IndexG;
    typedef PointE<DimP> PointEuclidean;
    typedef ViewGridQuadMD<DimP,DimCM> ViewG;// QWB only Quad is adapted to cpu only version
    typedef NIterQuad NIter;
    typedef NIterQuadDual NIterDual;

    typedef CellBMD<CM_DistanceEuclidean,
    CM_ConditionTrue,
    NIter, ViewG, Buffer, PointCoord> CB;

    typedef CellularMatrixMD<CB, ViewG, DimCM> CMB;

    typedef InputRW<PointEuclidean, GLfloatP> EMST_RW;
    typedef NeuralNetEMST<BufferLinkPointCoord, PointEuclidean> NetLink;

    typedef EMSTOperators<CMB, CMB, NetLink, CB, CB, NIter, NIterDual, ViewG, BufferLinkPointCoord> BoruvkaOp;

    // Data
    NetLink md_links_cpu; // read input data into an one-dimension NetLink, replace origginal city, wb.Q
    Grid<PointEuclidean> adaptiveMapOriginal; // read input data into an one-dimension NetLink, replace origginal city, wb.Q

    NetLink mr_links_cpu;
    NetLink mr_links_gpu;

    Grid<PointEuclidean>* getAdpativeMap() {
        return &mr_links_cpu.adaptiveMap;
    }

    Grid<BufferLinkPointCoord>* getLinks() {
        return &mr_links_cpu.networkLinks;
    }

    Grid<GLfloatP> distanceMap;
    Grid<GLfloatP> distanceMap_cpu;

    // Working buffer
    Grid<GLfloatP> minDistMap;
    Grid<GLfloatP> minDistMap_cpu;

    //Grid<GLint> evtMap;
    //Grid<GLint> evtMap_cpu;

    Grid<GLint> stateMap;
    Grid<GLint> stateMap_cpu;

    // Spiral search nodes
#if EMST_SPIRAL_SEARCH_KD_SLAB_2
    Grid<NodeComputeOctantMD<IndexCM,IndexG,DimCM> > spiralSearchMap;
    Grid<NodeComputeOctantMD<IndexCM,IndexG,DimCM> > spiralSearchMap_cpu;
#else
#if EMST_SPIRAL_SEARCH_2D
    Grid<NodeSpiralSearch<IndexCM,IndexG>> spiralSearchMap;
    Grid<NodeSpiralSearch<IndexCM,IndexG>> spiralSearchMap_cpu;
#else
    Grid<NodeSpiralSearchMD<IndexCM,IndexG,DimCM> > spiralSearchMap;
    Grid<NodeSpiralSearchMD<IndexCM,IndexG,DimCM> > spiralSearchMap_cpu;
#endif
#endif

    // Evaluation values
    TspResultInfo traceParallelMST;

    // Working buffers
    ViewG vgd;
    CMB cm_gpu; // for tsp
    CMB cm_cpu;

    // EMST operator
    BoruvkaOp boruvkaOp;

    // size numbers
    //float max_x, max_y, min_x, min_y;
    //int w, h; // width and height of input size
    PointEuclidean pMin, pMax;

    int radiusSearchCells;
    int iteration;

    //! @}
#pragma endregion

#pragma region Manipulation d instance
    //! \brief Constructeur par defaut.
    //! Il cree explicitement les objets partages a toutes les
    //! instances
    explicit SolutionEMST()
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
    ~SolutionEMST()
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
    void initialize(NetLink& md_links, PointEuclidean& pMin, PointEuclidean& pMax);
    //! Operation de copie
    void setIdentical(SolutionEMST* imb);
    //! Operation de copie
    void clone(SolutionEMST* imb);
#pragma endregion

#pragma region Evaluation globale

    /*!
     * @name Evaluation globale
     * \brief Fonctions d'evaluation d'une SolutionEMST.
     * @{
     */
     //! \brief Valeurs par defaut des objectif
    void initEvaluate();
    //! \brief Evaluation complete d'une SolutionEMST
    double evaluate();
    //! \brief Fonction objectif agregative globale
    double computeObjectif();
    //! \brief Comparaison de 2 SolutionEMSTs
    bool isBest(SolutionEMST* imb);
    //! \brief Test d'admissibilite de la SolutionEMST
    bool isSolution();

    //! @}
#pragma endregion

#pragma region Operateurs
    /**
     * @name Operateurs
     * \brief Operations de transformation
     * d'une SolutionEMST. Implementations dans le fichier SolutionEMSTOperateurs.cpp.
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

    void run();

    bool activate();
    bool activate_gpu();

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
    //! Lecture d'une SolutionEMST du probleme
    void readSolution();
    //! Lecture d'une SolutionEMST du probleme a partir du fichier specifie en parametre
    void readSolution(const char* fileName);
    //! Ecriture d'une SolutionEMST du probleme
    void writeSolution();
    //! Ecriture d'une SolutionEMST du probleme a partir du fichier specifie en parametre
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
    //! Ecriture des valeurs de criteres/objectifs dans le fichier (texte) de statistiques
    void writeStatisticsToFile();
    //! Ecriture des valeurs de criteres/objectifs dans le fichier de statistiques dans un flux
    void writeStatistics(std::ostream& o);
#pragma endregion

};

#endif // SolutionEMST_H

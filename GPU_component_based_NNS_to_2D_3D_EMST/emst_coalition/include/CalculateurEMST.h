#ifndef CALCULATEUR_EMST_H
#define CALCULATEUR_EMST_H

/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2018
 *
 ***************************************************************************
 */
#include "config/ConfigParamsCF.h"
#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "BufferLink.h"

using namespace components;

/*! \brief Fonction principale de calcul.
 * Classe principale de lancement d'un calcul d'optimisation.
 */
class CalculateurEMST
{
public:
    //! \brief Choix du mode de fonctionnement
    /**
    * 0:evaluation, 1:local search,
    * 2:genetic algorithm, 3:construction initiale seule,
    * 4:generation automatique d'instances
    */
    enum WorkingMode
    {
        EVAL_ONLY = 0,
        LOCAL_SEARCH = 1,
        GENETIC_METHOD = 2,
        CONSTRUCTION = 3,
        RUN = 4,
        RUN_3D = 5,
        RUN_2DPOINT5 = 6
    };

    static Grid<Point3D>* getAdaptiveMap();
    static Grid<Point3D>* getAdaptiveMap3();
    static Grid<BufferLinkPointCoord>* getLinks();
    static Grid<BufferLinkPointCoord>* getLinks3();

    /*! \brief Calcul avec surcharge des paramètres de configuration par programme appelant.
     * \param fileData nom du fichier d'entree, contient la donnee d'entree du probleme
     * \param fileSolution nom du fichier de sortie, contient la solution du probleme
     * \param fileStats nom du fichier de sortie contenant les evaluations de la solution
     * suivant les differents criteres du probleme
     * \param params structure contenant les parametres de configuation de l'algorithme
     * d'imbrication
     */
    static void initialize(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params);
    static void run();
    static bool activate();
};

#endif // CALCULATEUR_H

#ifndef CALCULATEUR_H
#define CALCULATEUR_H

/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput, A. Mansouri
 * Date creation : mars 2014
 *
 ***************************************************************************
 */
#include "config/ConfigParamsCF.h"

/*! \brief Fonction principale de calcul.
 * Classe principale de lancement d'un calcul d'optimisation.
 */
class Calculateur
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
    };

    /*! \brief Calcul d'imbrication avec surcharge des paramètres
     * de configuration par programme appelant.
     * \param fileData nom du fichier d'entree, contient la donnee d'entree du probleme
     * \param fileSolution nom du fichier de sortie, contient la solution du probleme
     * \param fileStats nom du fichier de sortie contenant les evaluations de la solution
     * suivant les differents criteres du probleme
     * \param params structure contenant les parametres de configuation de l'algorithme
     * d'imbrication
     */
    static void calcul(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params);
    static void initialize(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params);
    static void run();
};

#endif // CALCULATEUR_H

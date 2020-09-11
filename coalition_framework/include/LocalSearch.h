#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H
/*
 ***************************************************************************
 *
 * Auteurs : J.C. Creput, A. Mansour et F. Lauri
 * Date creation : mars 2014
 * Date derniere modification : juin 2016
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>

#include "config/ConfigParamsCF.h"

 //TODO,BM cette classe semble contenir des mthodes redondantes avec la classe solution (ex: fonctions statistiques).
 // La classe LocalSearch ne semble utilise qu'en mode de fonctionnement LOCAL_SEARCH
template <typename Solution>
class LocalSearch
{
    template <typename Solution1>
    friend class Agent;

    //! \name Fichiers
    //! \{
    char* fileData;
    char* fileSolution;
    char* fileStats;
    std::ofstream* OutputStream;
    //! \}

    //! \name Solutions appartenant a chaque agent
    //! \{

    //! Solutions d'imbrication variables
    Solution* imbCurrent;
    Solution* imbBestConstruct;
    Solution* imbBestImprove;
    // Meilleure solution
    Solution* imbBest;
    Solution* imbBestBest;

    // Solution initiale memorisee au debut de la recherche locale
    Solution* imbMemo;
    //! \}

public:
    LocalSearch(Solution* solution = NULL);

    /*!
     * @name Controles
     * \brief Fonctions de controle et d'execution.
     * @{
     */
     //!
     //! \brief initialize
     //! \param data input
    //! \param sol ouput
    //! \param stats stats
    //!
    void initialize(char* data, char* sol, char* stats);
    //!
    //! \brief initialize
    //!
    void initialize();

    //! Encapsulation
    void setCurrent(Solution* imb)
    {
        imbCurrent = imb;
    }

    //! Init avant execution
    void init();
    //! Boucle principale d'execution
    void run();
    //! @}

   
    /*!
     * @name Constructions
     * \brief Construction de base et reiteration de constructions.
     * @{
     */
     //!
    void constructSolution();
    void iteratedConstruct();
    //! @}

    /*!
     * @name Recherches locales
     * \brief Seule la recherche locale (localSearch) est efficace
     * sur le probleme d'imbrication
     * @{
     */
     //!
     //! \brief iteratedRepair
     //!
    void iteratedRepair();
    //!
    //! \brief localSearch (first improvement)
    //! \param is_FI first improvement (best improvement pas efficace ici)
    //!
    void localSearch(bool is_FI);
    //! @}

    /*!
     * @name Fonction principale de recherche locale iterative.
     *
     * @{
     */
     //!
     //! \brief iteratedConstructAndImprove
     //! \param best la solution
     //! \param bestConstruct la meilleure solution construite
     //! \param bestImprove la meilleure solution amelioree
     //! \param imbMemo solution memoire temporaire
     //!
    void iteratedConstructAndImprove();
    //! @}

};


#include "LocalSearch.inl"


#endif // LOCAL_SEARCH_H

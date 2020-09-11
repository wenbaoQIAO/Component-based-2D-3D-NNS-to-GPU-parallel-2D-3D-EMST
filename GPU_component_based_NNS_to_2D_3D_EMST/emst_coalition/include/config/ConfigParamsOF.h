#ifndef CONFIG_PARAMS_CF_H
#define CONFIG_PARAMS_CF_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2014
 *
 ***************************************************************************
 */
#include <string>
#include <vector>
#include "lib_global.h"
#include "ConfigParams.h"

using namespace std;

//! Version courante
#define VERSION_CALCULATEUR "alpha 2.1"

enum
{
    MutateMA_Probability,

    NbrProbabilities
};

namespace config
{

class LIBSHARED_EXPORT ConfigParamsCF : public ConfigParams
{
#pragma region Membres prives
private:
    //! Nom du fichier de configuration
    //! string configFile;
#pragma endregion

public:
#pragma region Constructeur et methodes

    ConfigParamsCF();
    ConfigParamsCF(char* file);
    //! Defaults de tous les parametres
    void initDefaultParameters();
    //! Fonction de lecture et surcharge des paramètres depuis le disque
    void readConfigParameters();

#pragma endregion

#pragma region Parametres globaux
    // Global Parameters

    /** Profile type du probleme
     * 0 aucun
     * 1
     * 2
     */
    int problemType;

    //! \brief Choix du mode de fonctionnement 0:evaluation, 1:local search,
    //! 2:genetic algorithm, 3:construction initiale seule,
    //! 4:generation automatique d'instances
    int functionModeChoice;

    //! \brief Choix de l'utilisation de threads
    bool useThreads;

    //! Choix d'utiliser une graîne pour la génération aléatoire. Si false, utilise le temps courant
    bool useSeed;
    //! Valeur de la graîne pour la génération aléatoire (suppose useSeed==true)
    int seedValue;

    //! Sortie "standard" dans la console (cummulable avec fileoutput)
    bool consoleoutput;
    //! Sortie "standard" vers un fichier (cummulable avec consoleoutput)
    bool fileoutput;

#pragma endregion

#pragma region Parametres sur la construction

    //! \brief Mouvement aleatoire lors de la construction,
    //! position en x sur matrice 2xM + bruit en x
    int bruitConstruction_x;
    //! position en y sur matrice 2xM + bruit en y
    int bruitConstruction_y;
#pragma endregion

#pragma region Parametres sur la phase d amelioration
    //! \brief Mouvement aleatoire lors de la phase d'amélioration
    //! lors d'une operation de voisinage (neighborhood move) au sein des
    //! recherches locales,
    //! position en x : position + aleatMove (mm)
    int aleatMove_x;
    //! position en y : position + aleatMove (mm)
    int aleatMove_y;
#pragma endregion

#pragma region Parametres Local Search
    // Local Search

    //! \brief Faux s'il s'agit de partir de l'input comme solution pre-construite,
    //! vrai si on construit a partir d'une initialisation aleatoire
    bool constructFromScratchParam;

    //! \brief Type de recherche locale, 0: marche aleatoire, 
    //! 1: first improvement, 2: best improvement
    int localSearchType;//0, 1, 2

    //! \brief Parametre de la recherche locale (si utilisee seule).
    //! Taille voisinage autour de la solution courante
    int neighborhoodSize;
    //! \brief Parametre de la recherche locale (si utilisee seule).
    //! Nombre totale de reiterations construction/amelioration
    int nbOfConstructAndRepairs;
    //! \brief Parametre de la recherche locale (si utilisee seule).
    //! Nombre d'essais de construction de départ
    int nbOfInternalConstructs;
    //! \brief Parametre de la recherche locale (si utilisee seule).
    //! Nombre d'iterations maximal de recherche locale
    int nbOfInternalRepairs;
#pragma endregion

#pragma region Parametres Genetic Method
    // Genetic Method

    //! Algo memetique (vrai) ou génétique (faux)
    bool memeticAlgorithm;

    double gaPercentageOfBestIndividuals;
    double gaPercentageOfElitistIndividual;

    //! Probabilities
    double probaMutateGA2;
    double probaMutateMA;

    std::string s_probaOperators;
    std::vector<double> probaOperators;

    //! Taille population
    int populationSize;
    //! Nombre de generations
    int generationNumber;

    //! \brief Parametre de la recherche locale surcharge pour l'algorithme memetique.
    //! Taille voisinage autour de la solution courante
    int MAneighborhoodSize;
    //! \brief Parametre de la recherche locale surcharge pour l'algorithme memetique.
    //! Nombre d'essais de construction de départ
    int MAnbOfInternalConstructs;
    //! \brief Parametre de la recherche locale surcharge pour l'algorithme memetique.
    //! Nombre d'iterations maximal de recherche locale
    int MAnbOfInternalRepairs;
    // Utilise la solution chargée comme point de départ pour le premier agent créé.
    bool firstAgentUseInit;
#pragma endregion

#pragma region Parametres Generateur automatique d instances (pas utilise ?)
    // Generateur automatique d'instances

    //! Nombre d'instances generees
    int nInstances;

    // Gestion des palettes vides

#pragma endregion

#pragma region Parametres Intensite des mouvements de structure en phase de construction
    /** Intensite des mouvements de structure en phase initiale de construction.
     */
     //! Random between 0 and max % intensity, or max intensity if false
    bool structIntensityRandom;
#pragma endregion

#pragma region Intensite des mouvements de structure en phase aleatoire d amelioration
    /** Intensite des mouvements de structure en phase aléatoire d'amélioration.
     */
     //! type 0 : mvt constant, 1 mvt en proportion
     //! Sélectionne si l'amplitude de la perturbation est constante entre 0 and max mm/deg ou en % entre 0 and le max
    int randomIntensityType;
    //! Utilise un recuit simulé pour atténuer les mouvements aléatoires des véhicules et des structures en phase d'amélioration
    bool useRecuitRandomStructure;
    //! Utilise un bruit gaussien pour générer l'intensité des mouvements (probas à fixer dans le SVG?)
    bool useGaussianRandomIntensity;

#pragma endregion

    //! Sortie de profilage (dans "profile.stats")
    bool debugUseProfiler;
    //! Affichage trace d'exection statistique
    bool traceActive;
    //! Trace d'exection statistique avec la meilleure solution rencontree, ou la solution courante
    bool traceReportBest;
    //! Trace d'exection avec sauvegarde de la solution courante
    bool traceSaveSolutionFile;

};

}//namespace config

extern config::ConfigParamsCF* g_ConfigParameters;

#endif // CONFIG_PARAMS_CF_H

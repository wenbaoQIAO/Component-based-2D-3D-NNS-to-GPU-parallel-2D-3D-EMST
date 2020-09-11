#ifndef AGENTMETASOLVER_H
#define AGENTMETASOLVER_H
/*
 ***************************************************************************
 *
 * Auteurs : J.C. Creput, F. Lauri, A. Mansouri
 * Date creation : mars 2014
 * Date dernière modification : mars 2018
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>

#include "config/ConfigParamsCF.h"
#include "LocalSearch.h"

#include "Threads.h"


 //! Agent
template <typename Solution>
#ifdef USE_CPP11THREADS
class Agent
#else
class Agent : public QThread
#endif
{
public:
    Solution* m_Solution;
    LocalSearch<Solution>* lS;
    Solution* m_BestConstruct;
    Solution* m_BestImprove;

private:
    double m_Fitness;

public:
    Agent();
    Agent(Solution* solution);
    ~Agent();

    double fitness() const;

    /*! @name Fonctions standards memetiques et genetiques
     * \brief Les fonctions memetiques sont postfixees par MA.
     */
     //! @{
    void initialize(Solution* solution);
    void clone(Agent<Solution>* i);
    void setIdentical(Agent<Solution>* indiv);

    void activate();
    void run();

    void evaluate();
    void generateGA();
    void mutateGA_1();
    void mutateGA_2();
    void generateMA();
    void mutateMA();
    void localSearch();
    //! @}
};


// ================================================
// CLASS AGENTMETASOLVER
// ================================================
template <typename Solution>
class AgentMetaSolver
{
    int m_PopulationSize;
    Agent<Solution>** m_Individuals;

    int m_BestIndividualNumber;         //! numero du meilleur indiv. rencontre
    int m_BestIndividualGeneration;     //! a la generation
    double m_BestFit;                   //! meilleure fitness rencontree
    Agent<Solution>* m_BestIndividual;  //! le meilleur

    int m_BestIndividualNumberEverEncountered;          //! numero du meilleur indiv. rencontre
    int m_BestIndividualGenerationEverEncountered;      //! a la generation
    double m_BestFitEverEncountered;                    //! meilleure fitness rencontree
    Agent<Solution>* m_BestIndividualEverEncountered;   //! le meileur

public:
    AgentMetaSolver();
    ~AgentMetaSolver();

    void initialize(char* data, char* sol, char* stats);
    void run();

protected:
    void findBestInInitialPop();
    //! Sélectionne la meilleure solution et si celle-ci réponds à tous les critères de validation, renvoie true
    bool findBestInCurrentPopAndSelect();

    int getBestIndividual();
    int getWorstIndividual();

    bool activate();

    void select(int size);
    void selectElit(int size);

    void destroy();

    void clear();
};


#include "AgentMetaSolver.inl"


#endif

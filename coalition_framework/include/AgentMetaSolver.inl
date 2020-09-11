#include "AgentMetaSolver.h"

#include "random_generator_cf.h"

#include "Threads.h"
//#include "Multiout.h"


#define AFFICH_ITE                  1

#define OPTIMIZE_LONG_RUN           0
#define WRITE_TIMESTAMPED_INTERMEDIATE 0

#if DEBUG
#define VERBOSE 1
#else
#define VERBOSE 0
#endif

extern int generation;

extern SynchronizationPoint sp;
extern bool inLoop;
extern bool onInit;
extern bool onFirstAgentInit;

#ifdef USE_CPP11THREADS
extern Thread* g_Threads;
#endif


template <typename Solution>
Agent<Solution>::Agent()
{}

template <typename Solution>
Agent<Solution>::Agent(Solution* solution)
{
    initialize(solution);
}

template <typename Solution>
Agent<Solution>::~Agent()
{
    if (m_Solution)
    {
        delete m_Solution;
    }

    if (m_BestImprove)
    {
        delete m_BestImprove;
    }

    delete lS;
}


template <typename Solution>
double
Agent<Solution>::fitness() const
{
    return m_Fitness;
}


template <typename Solution>
void
Agent<Solution>::initialize(Solution* solution)
{
    m_Solution = new Solution();

    lS = new LocalSearch<Solution>(solution);
    lS->setCurrent(m_Solution);

    if (solution)
    {
        solution->clone(m_Solution);

        m_BestImprove = new Solution();
        solution->clone(m_BestImprove);
    }
}


template <typename Solution>
void
Agent<Solution>::clone(Agent<Solution>* indiv)
{
    m_Solution->clone(indiv->m_Solution);
    indiv->m_Fitness = fitness();
}

template <typename Solution>
void
Agent<Solution>::setIdentical(Agent<Solution>* indiv)
{
    m_Solution->setIdentical(indiv->m_Solution);
    indiv->m_Fitness = fitness();
}


template <typename Solution>
void Agent<Solution>::activate()
{
    if (g_ConfigParameters->memeticAlgorithm)
    {
        // Si initialisation, génère un agent par construction
        if (onInit)
        {
            // FIXME,BM: voir si cela ne ralenti pas la construction pour rien...
            // Le premier agent utilise la meilleure solution comme point de départ
            // et non une construction
            if (onFirstAgentInit)
            {
                onFirstAgentInit = false;
                // Recherche locale
                localSearch();
            }
            else
            {
                generateMA();
            }
        }
        else
        {
            // Saut de diversification si generation suivante à construction
            if (generation > 0)
            {
                // Fait aleatoirement une construction (saut de diversification)
                mutateMA();
            }
            // Recherche locale
            localSearch();
        }
    }
    else
    {
        if (onInit)
        {
            if (onFirstAgentInit)
            {
                onFirstAgentInit = false;
                // Recherche
                mutateGA_1();
            }
            else
            {
                generateGA();
            }
        }
        else
        {
            if (generation > 0)
            {
                mutateGA_2();
            }

            mutateGA_1();
        }
    }
    // Evaluation de la solution
    evaluate();
#if VERBOSE
    lout << "AgentMS.fitness= " << this->m_Fitness << endl;
#endif
}


template <typename Solution>
void
Agent<Solution>::run()
{
    Mutex mu;

    do
    {
        activate();
        sp.dec();
        sp.wait(&mu);
    }
    while (inLoop);
}


template <typename Solution>
void
Agent<Solution>::evaluate()
{
    m_Fitness = -m_Solution->evaluate();
}


template <typename Solution>
void
Agent<Solution>::generateGA()
{
    m_Solution->initConstruct();
    m_Solution->constructSolutionSeq();
}

template <typename Solution>
void
Agent<Solution>::mutateGA_1()
{
    m_Solution->generateNeighbor();
}

template <typename Solution>
void
Agent<Solution>::mutateGA_2()
{
    if (random_cf::occurs(g_ConfigParameters->probaMutateGA2))
    {
        m_Solution->constructSolutionSeq();
    }
}


//***************************************************************************************


template <typename Solution>
void
Agent<Solution>::generateMA()
{
#if VERBOSE
    lout << "Generate MA ... (construction)" << endl;
#endif
    m_Solution->initConstruct();
    if (g_ConfigParameters->constructFromScratchParam)
    {
        lS->imbBestConstruct->initEvaluate();
    }
    lS->iteratedConstruct();
}


template <typename Solution>
void
Agent<Solution>::mutateMA()
{
    if (random_cf::occurs(g_ConfigParameters->probaMutateMA))
    {
#if VERBOSE
        lout << "Mutate MA ... (saut diversification)" << endl;
#endif
        m_Solution->setIdentical(lS->imbBestConstruct);
        // Saut de diversification
        lS->imbBestConstruct->initEvaluate();
        lS->iteratedConstruct();

        localSearch();
    }
}


template <typename Solution>
void Agent<Solution>::localSearch()
{
    m_Solution->setIdentical(lS->imbBestImprove);
    lS->imbBestImprove->evaluate();
    m_Solution->setIdentical(lS->imbCurrent);
    lS->imbCurrent->evaluate();
    lS->localSearch(true);
}


// ================================================
// CLASS AGENTMETASOLVER
// ================================================


template <typename Solution>
AgentMetaSolver<Solution>::AgentMetaSolver() :
    m_Individuals(NULL)
{}

template <typename Solution>
AgentMetaSolver<Solution>::~AgentMetaSolver()
{
    clear();
}


template <typename Solution>
void AgentMetaSolver<Solution>::initialize(char* data, char* sol, char* stats)
{
    //surcharge local search
    g_ConfigParameters->neighborhoodSize = g_ConfigParameters->MAneighborhoodSize;
    g_ConfigParameters->nbOfInternalConstructs = g_ConfigParameters->MAnbOfInternalConstructs;
    g_ConfigParameters->nbOfInternalRepairs = g_ConfigParameters->MAnbOfInternalRepairs;

    // Chargement de la solution d'entrée
    Solution* main_solution;
    main_solution = new Solution();
    main_solution->initialize(data, sol, stats);
    main_solution->readPbInstance();
    main_solution->initStatisticsFile();

    // Evaluation de la solution chargée
    main_solution->evaluate();
    // Sortie sur fichier (avant première génération)
    main_solution->writeStatisticsToFile(-1);
    // Sortie sur console avant première génération)
    main_solution->writeHeaderStatistics(cout);
    main_solution->writeStatistics(-1, cout);

    m_BestIndividual = new Agent<Solution>(main_solution);
    m_BestIndividualEverEncountered = new Agent<Solution>(main_solution);

    m_PopulationSize = g_ConfigParameters->populationSize;

    clear();

    m_Individuals = new Agent<Solution>*[(unsigned)m_PopulationSize];

#ifdef USE_CPP11THREADS
    g_Threads = new Thread[popSize];
#endif

    for (int i = 0; i < m_PopulationSize; i++)
    {
        m_Individuals[i] = new Agent<Solution>(main_solution);

        m_BestIndividual->clone(m_Individuals[i]);
    }
}//initialize


template <typename Solution>
int AgentMetaSolver<Solution>::getBestIndividual()
{
    double oldFitness;
    int chosen = 0;

    oldFitness = m_Individuals[0]->fitness();
    chosen = 0;
    for (int i = 1; i < m_PopulationSize; i++)
    {
        //<FIXME,BM> La ligne de code commentée et proposée par Fabrice Laury 
        // devrait théoriquement faire la même chose. Nos tests ont montré que
        // la convergence n'était plus assurée, nous avons donc retabli le code
        // original. A creuser...
        if (m_Individuals[i]->m_Solution->isBest(m_Individuals[0]->m_Solution))
            //if (m_Individuals[i]->fitness() > oldFitness)
        {
            oldFitness = m_Individuals[i]->fitness();
            chosen = i;

#if WRITE_TIMESTAMPED_INTERMEDIATE
            string name = "tmp";
            QTime now = QTime::currentTime();
            name.append(now.toString().replace(":", "_").toStdString());
            name.append(".svg");
            m_Individuals[i]->m_Solution->writeSolution(name.c_str());
#endif
        }
    }

    return chosen;
}

template <typename Solution>
int AgentMetaSolver<Solution>::getWorstIndividual()
{
    double oldFitness;
    int chosen = 0;

    oldFitness = m_Individuals[0]->fitness();
    chosen = 0;
    for (int i = 1; i < m_PopulationSize; i++)
    {
        if (m_Individuals[i]->fitness() < oldFitness)
        {
            oldFitness = m_Individuals[i]->fitness();
            chosen = i;
        }
    }

    return chosen;
}


template <typename Solution>
bool AgentMetaSolver<Solution>::activate()
{
    if (g_ConfigParameters->useThreads)
    {
        for (int i = 0; i < m_PopulationSize; i++)
        {
#ifdef USE_CPP11THREADS
            g_Threads[i] = Thread(&Agent::run, indPool[i]);
#else
            m_Individuals[i]->start();
#endif
        }
    }
    else
    {
        for (int i = 0; i < m_PopulationSize; i++)
        {
#if VERBOSE
            lout << "Population " << i << "/" << m_PopulationSize << endl;
#endif
            // Activation & evaluation
            m_Individuals[i]->activate();
#if OPTIMIZE_LONG_RUN
#else
            if (m_Individuals[i]->m_Solution->isSolution())
            {
                break;
            }
#endif
        }
    }

    return true;
}


template <typename Solution>
void AgentMetaSolver<Solution>::destroy()
{
#ifdef USE_CPP11THREADS
    for (uint k = 0; k < popSize; ++k)
    {
        g_Threads[k].join();
    }
#else
    for (int i = 0; i < m_PopulationSize; i++)
    {
        m_Individuals[i]->wait();
    }
#endif
}


/*
 * Selections
 */
template <typename Solution>
void AgentMetaSolver<Solution>::select(int size)
{
    double oldFitness;
    int chosen = 0;

    // Mettre les "size" best au debut
    for (int k = 0; k < size; k++)
    {
        oldFitness = m_Individuals[k]->fitness();
        chosen = k;
        for (int i = k+1; i < m_PopulationSize; i++)
        {
            if (m_Individuals[i]->fitness() > oldFitness)
            {
                oldFitness = m_Individuals[i]->fitness();
                chosen = i;
            }
        }
        Agent<Solution>* tmp = m_Individuals[k];
        m_Individuals[k] = m_Individuals[chosen];
        m_Individuals[chosen] = tmp;
    }
    // Mettre les "size" worst à la fin
    for (int k = 0; k < size; k++)
    {
        oldFitness = m_Individuals[m_PopulationSize-1-k]->fitness();
        chosen = m_PopulationSize-1-k;
        for (int i = m_PopulationSize-1-k-1; i >= size; i--)
        {
            if (m_Individuals[i]->fitness() < oldFitness)
            {
                oldFitness = m_Individuals[i]->fitness();
                chosen = i;
            }
        }
        Agent<Solution>* tmp = m_Individuals[m_PopulationSize-1-k];
        m_Individuals[m_PopulationSize-1-k] = m_Individuals[chosen];
        m_Individuals[chosen] = tmp;
    }
    // Remplacer les contenus des "size" worst par les contenus des "size" bests
    for (int k = 0; k < size; k++)
    {
        m_Individuals[k]->setIdentical(m_Individuals[m_PopulationSize-size+k]);
    }
}//select()

template <typename Solution>
void AgentMetaSolver<Solution>::selectElit(int size)
{
    if (size)
    {
        double oldFitness;
        int chosen = 0;

        // Mettre les "size" worst à la fin
        for (int k = 0; k < size; k++)
        {
            oldFitness = m_Individuals[m_PopulationSize-1-k]->fitness();
            chosen = m_PopulationSize-1-k;
            for (int i = m_PopulationSize-1-k-1; i >= 0; i--)
            {
                if (m_Individuals[i]->fitness() < oldFitness)
                {
                    oldFitness = m_Individuals[i]->fitness();
                    chosen = i;
                }
            }
            Agent<Solution>* tmp = m_Individuals[m_PopulationSize-1-k];
            m_Individuals[m_PopulationSize-1-k] = m_Individuals[chosen];
            m_Individuals[chosen] = tmp;
        }
        // Remplacer les contenus des "size" worst par le best
        for (int k = 0; k < size; k++)
        {
            m_BestIndividual->setIdentical(m_Individuals[m_PopulationSize-size+k]);
        }
    }
}//select()


template <typename Solution>
void AgentMetaSolver<Solution>::clear()
{
    if (m_Individuals)
    {
        delete[] m_Individuals;
    }
#ifdef USE_CPP11THREADS
    delete[] g_Threads;
#endif
}


template <typename Solution>
void AgentMetaSolver<Solution>::findBestInInitialPop()
{
    m_BestIndividualNumber = getBestIndividual();
    m_BestIndividualGeneration = -1;
    m_Individuals[m_BestIndividualNumber]->setIdentical(m_BestIndividual);
    m_BestFit = m_BestIndividual->fitness();

    m_BestIndividualNumberEverEncountered = m_BestIndividualNumber;
    m_BestIndividualGenerationEverEncountered = m_BestIndividualGeneration;
    m_Individuals[m_BestIndividualNumber]->setIdentical(m_BestIndividualEverEncountered);
    m_BestFitEverEncountered = m_BestIndividualEverEncountered->fitness();

    cout 	<< "\n"
        << "Best individual in Pop: " << "\n"
        << "id number: "
        << m_BestIndividualNumber << "\t"
        << "fitness: "
        << m_BestFit << "\t"
        << endl;
}


template <typename Solution>
bool AgentMetaSolver<Solution>::findBestInCurrentPopAndSelect()
{
    int best_ind_number, worst_ind_number;
    double best_fit, worst_fit;

    // save best encountered individual and take current best in population
    best_ind_number = getBestIndividual();
    best_fit = m_Individuals[best_ind_number]->fitness();

    // worst individual
    worst_ind_number = getWorstIndividual();
    worst_fit = m_Individuals[worst_ind_number]->fitness();

    //if (bestFit < best_fit) pas utilse (pb de precision en flotant)
    if (m_Individuals[best_ind_number]->m_Solution->isBest(m_BestIndividual->m_Solution))
    {
        m_BestIndividualNumber = best_ind_number;
        m_BestIndividualGeneration = generation;
        m_BestFit = best_fit;
        m_Individuals[best_ind_number]->setIdentical(m_BestIndividual);
    }

    //if (bestFit < best_fit) pas utilse (pb de precision en flotant)
    if (m_BestIndividual->m_Solution->isBest(m_BestIndividualEverEncountered->m_Solution))
    {
        m_BestIndividualNumberEverEncountered = m_BestIndividualNumber;
        m_BestIndividualGenerationEverEncountered = m_BestIndividualGeneration;
        m_BestFitEverEncountered = m_BestFit;
        m_BestIndividual->setIdentical(m_BestIndividualEverEncountered);

        // Sortie sur fichier des statistiques pour ce nouveau candidat retenu
        if (g_ConfigParameters->traceActive)
        {
            m_BestIndividualEverEncountered->m_Solution->writeStatisticsToFile(generation);
        }
        // Sortie SVG de la meilleure solution courante
        m_BestIndividualEverEncountered->m_Solution->writeSolution();
    }

#if OPTIMIZE_LONG_RUN
#else
    // Si c'est une solution qui vérifie toutes les contraintes alors renvoie "true"
    if (m_BestIndividualEverEncountered->m_Solution->isSolution())
    {
        return true;
    }
#endif

    // repport
    if ((generation % AFFICH_ITE) == 0)
    {
        cout << "\n"
            << "Generation: " << "\t"
            << generation << "\n"
            << "Best individual in Pop: " << "\n"
            << "id number: "
            << best_ind_number << "\t"
            << "fitness: "
            << best_fit << "\t"
            << endl;

        cout << "Worst individual in Pop: " << "\n";
        cout << "id number: "
            << worst_ind_number << "\t"
            << "fitness: "
            << worst_fit << "\t"
            << endl;

        cout << "Best encountered: " << "\n";
        cout << "id number: "
            << m_BestIndividualNumber << "\t"
            << "fitness: "
            << m_BestFit << "\t"
            << "generation: "
            << m_BestIndividualGeneration << "\t"
            << endl;

        cout << "Best Ever Encountered: " << "\n";
        cout << "id number: "
            << m_BestIndividualNumberEverEncountered << "\t"
            << "fitness: "
            << m_BestFitEverEncountered << "\t"
            << "generation: "
            << m_BestIndividualGenerationEverEncountered << "\t"
            << endl;

        m_BestIndividualEverEncountered->m_Solution->writeStatistics(generation, cout);
    }

    // Selections
    select((int)ceil((double)m_PopulationSize * g_ConfigParameters->gaPercentageOfBestIndividuals));
    selectElit((int)ceil((double)m_PopulationSize * g_ConfigParameters->gaPercentageOfElitistIndividual));

    return false;
}


// Run one generation
template <typename Solution>
void AgentMetaSolver<Solution>::run()
{
    if (g_ConfigParameters->useThreads)
    {
        bool optimal_found = false;

        inLoop = false;
        if (g_ConfigParameters->firstAgentUseInit)
        {
            onFirstAgentInit = true;
        }
        onInit = true;

        sp.init(g_ConfigParameters->populationSize);
        sp.reset();

        activate();

        cout << "**************************" << endl;
        cout << "**  Generate initial pop" << endl;
        cout << "**************************" << endl;

        cout << "GENERATION NUMBER : " << g_ConfigParameters->generationNumber << endl;
        cout << "POPULATION SIZE : " << g_ConfigParameters->populationSize << endl;

        sp.synchronize();
        findBestInInitialPop();
        generation = 0;
        inLoop = true;
        onInit = false;
        sp.notifyAll();

        for (; generation < g_ConfigParameters->generationNumber; ++generation)
        {
#if OPTIMIZE_LONG_RUN
#else
            if (m_BestIndividualEverEncountered->m_Solution->isSolution())
            {
                optimal_found = true;
                break;
            }
#endif

            sp.synchronize();
            if (optimal_found = findBestInCurrentPopAndSelect())
            {
                break;
            }
            sp.notifyAll();
        }//for

        if (!optimal_found)
        {
            sp.synchronize();
        }
        inLoop = false;
        sp.notifyAll();

        destroy();
    }
    else
    {
        cout << "**************************" << endl;
        cout << "**  Generate initial pop" << endl;
        cout << "**************************" << endl;

        onInit = true;
        if (g_ConfigParameters->firstAgentUseInit)
        {
            onFirstAgentInit = true;
        }
        activate();
        findBestInInitialPop();

        onInit = false;
        // Boucle sur le nombre de génération à réaliser au maximum
        for (generation = 0; generation < g_ConfigParameters->generationNumber; ++generation)
        {
#if OPTIMIZE_LONG_RUN
#else
            if (m_BestIndividualEverEncountered->m_Solution->isSolution())
            {
                break;
            }
#endif

            activate();

            // Sélectionne la meilleure solution et si celle-ci réponds à tous
            // les critères de validation, arrête l'exécution
            if (findBestInCurrentPopAndSelect())
            {
                break;
            }
}//for
    }

    m_BestIndividualEverEncountered->m_Solution->writeSolution();
    m_BestIndividualEverEncountered->m_Solution->writeStatistics(generation, cout);
    m_BestIndividualEverEncountered->m_Solution->writeStatisticsToFile(generation);
}

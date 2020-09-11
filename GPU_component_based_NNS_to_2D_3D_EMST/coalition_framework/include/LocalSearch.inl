//#include "Solution.h"
#include "LocalSearch.h"

#include "random_generator.h"

#define OPTIMIZE_LONG_RUN           0

#define REPAIR_DOUBLE_BEST          0
#define LS_DOUBLE_BEST              0

#if DEBUG
#define VERBOSE 1
#else
#define VERBOSE 0
#endif

template <typename T> std::string tostr(const T& t)
{
    std::ostringstream os; os<<t; return os.str();
}


template <typename Solution>
LocalSearch<Solution>::LocalSearch(Solution* solution)
{
    imbMemo = new Solution();
    imbBestImprove = new Solution();
    imbBestConstruct = new Solution();
    if (solution)
    {
        solution->clone(imbMemo);
        solution->clone(imbBestImprove);
        solution->clone(imbBestConstruct);
    }
}


template <typename Solution>
void LocalSearch<Solution>::initialize(char* data, char* sol, char* stats)
{
    fileData = data;
    fileSolution = sol;
    fileStats = stats;

    initialize();
}

template <typename Solution>
void LocalSearch<Solution>::initialize()
{
#if VERBOSE
    lout << "Initialize" << endl;
#endif

    imbCurrent = new Solution();
    imbCurrent->initialize(fileData, fileSolution, fileStats);
    imbCurrent->readPbInstance();
    imbCurrent->initStatisticsFile();

    imbCurrent->evaluate();
    imbCurrent->writeStatisticsToFile(-1);

    imbCurrent->writeHeaderStatistics(cout);
    imbCurrent->writeStatistics(-1, cout);

    imbBestConstruct = new Solution();
    imbBestImprove = new Solution();
    imbBest = new Solution();
    imbBestBest = new Solution();

    imbCurrent->clone(imbMemo);
    imbCurrent->clone(imbBestConstruct);
    imbCurrent->clone(imbBestImprove);
    imbCurrent->clone(imbBest);
    imbCurrent->clone(imbBestBest);

    init();
}

template <typename Solution>
void LocalSearch<Solution>::init()
{
    if (g_ConfigParameters->constructFromScratchParam)
    {
        imbCurrent->initConstruct();
        constructSolution();
        imbCurrent->evaluate();
    }

    imbCurrent->setIdentical(imbMemo);
    imbCurrent->setIdentical(imbBestConstruct);
    imbCurrent->setIdentical(imbBestImprove);
    imbCurrent->setIdentical(imbBest);
    imbCurrent->setIdentical(imbBestBest);
}

template <typename Solution>
void LocalSearch<Solution>::run()
{
    double x0 = clock();
    iteratedConstructAndImprove();

    imbBest->writeHeaderStatistics(cout);
    imbBest->writeStatistics(-2, cout);
    cout << "duree : " << (clock() - x0)/CLOCKS_PER_SEC << endl;
    imbBest->writeSolution();
}

template <typename Solution>
void LocalSearch<Solution>::constructSolution()
{
    imbCurrent->constructSolutionSeq();
}//constructSolution()

template <typename Solution>
void LocalSearch<Solution>::iteratedConstruct()
{
#if VERBOSE
    lout << "Iterated Contruct" << endl;
#endif
    Solution* best = imbBestConstruct;

    int nb_iter = g_ConfigParameters->MAnbOfInternalConstructs;
    int cptIter = 0;
    // Construction itrative d'une solution par perturbations successives sur
    // une solution de base
    for (cptIter = 0; cptIter < nb_iter; cptIter++)
    {
#if VERBOSE
        lout << "const.iter " << cptIter << "/" << nb_iter << endl;
#endif
        //TODO,BM: Evaluer l'intrt de faire une copie la meilleure en cours:
        //best->setIdentical(imbCurrent);

        // perturbation incrementale de construction sur la solution courante
        this->constructSolution();
        // Evaluation
        imbCurrent->evaluate();
#if VERBOSE
        imbCurrent->computeObjectif();
        lout << "const.eval= " << imbCurrent->global_objectif << endl;
#endif
        // Selection
        if (imbCurrent->isBest(best))
        {
            imbCurrent->setIdentical(best);
        }

#if OPTIMIZE_LONG_RUN
#else
        // Si on trouve djA une solution, alors arrete la construction
        if (best->isSolution())
        {
            break;
        }
#endif
    }//for

    // sauvegarde la meilleure solution dans la solution courante
    best->setIdentical(imbCurrent);
}//IteratedConstruct

// Recherches locales
template <typename Solution>
void LocalSearch<Solution>::iteratedRepair()
{
    Solution* best = imbBestImprove;
    int nb_iter = g_ConfigParameters->nbOfInternalRepairs;
#if VERBOSE
    lout << "Iterated Repair" << endl;
#endif

#if REPAIR_DOUBLE_BEST
    best->setIdentical(imbBestBest);
#endif
    int cptIter = 0;
    for (cptIter = 0; cptIter < nb_iter; cptIter++)
    {
#if VERBOSE
        lout << "repair.iter " << cptIter << "/" << nb_iter << endl;
#endif
        imbCurrent->generateNeighbor();
#if VERBOSE
        imbCurrent->computeObjectif();
        lout << "repair.eval= " << imbCurrent->global_objectif << endl;
#endif
        if (imbCurrent->isBest(best))
        {
            imbCurrent->setIdentical(best);
#if REPAIR_DOUBLE_BEST
            if (best->isBest(imbBestBest))
                best->setIdentical(imbBestBest);
#endif
        }
        if ((cptIter % 5) == 0)
            best->setIdentical(imbCurrent);
#if REPAIR_DOUBLE_BEST
        if ((cptIter % 1000) == 0)
            imbBestBest->setIdentical(imbCurrent);
#endif
#if OPTIMIZE_LONG_RUN
#else
        if (best->isSolution())
        {
            break;
        }
#endif
    }//for
#if REPAIR_DOUBLE_BEST
    if (params->traceActive)
    {
        writeStatistics(cptIter, lout, imbBestBest);
        imbBestBest->writeStatistics(cptIter);
        string s(fileSolution);
        ostringstream os; os<<"_"<<cptIter;
        s.append(os.str());
        s.append(".improve.trace.svg");
        imbBestBest->writeSolution((char*)s.c_str());
    }
    imbBestBest->setIdentical(best);
#else
#endif
}//iteratedRepair


template <typename Solution>
void LocalSearch<Solution>::localSearch(bool is_FI)
{
    Solution* best = imbBestImprove;
#if VERBOSE
    lout << "Local Search" << endl;
#endif

    int nb_iter = g_ConfigParameters->MAnbOfInternalRepairs;
#if LS_DOUBLE_BEST
    best->setIdentical(imbBestBest);
#endif
    int neighbor_size = g_ConfigParameters->neighborhoodSize;
    int cptIter = 0;
    int cptIIter = 0;
    int improvementFound = true;

    while (improvementFound && cptIIter < nb_iter)
    {
#if VERBOSE
        lout << "local.maxiter " << cptIIter << "/" << nb_iter << endl;
#endif
        cptIter = 0;
        improvementFound = false;

        // Memorise la solution courante dans solution initiale
        imbCurrent->setIdentical(imbMemo);

        while (cptIter < neighbor_size && (is_FI ? !improvementFound : true))
        {
            cptIter++;
            cptIIter++;
#if VERBOSE
            lout << "local.iter " << cptIter << "/" << neighbor_size << endl;
#endif
            // Applique au moins un operateur. Les oprateurs font eux-mmes
            // la mise  jour des sous-objectifs en fonction des modifications
            // qu'ils apportent  la solution
            while (!imbCurrent->generateNeighbor());
#if VERBOSE
            imbCurrent->computeObjectif();
            lout << "local.eval= " << imbCurrent->global_objectif << endl;
#endif

            // Selection si meilleur que le 'best'
            if (imbCurrent->isBest(best))
            {
                // Ecrase le 'best' avec la nouvelle solution
                imbCurrent->setIdentical(best);
                improvementFound = true;
#if LS_DOUBLE_BEST
                if (best->isBestLS_3(imbBestBest))
                    best->setIdentical(imbBestBest);
#endif
                if (best->isSolution())
                {
                    break;
                }
            }
            // reprendre le pivot = solution initiale
            imbMemo->setIdentical(imbCurrent);
        }//while

        // Copie best dans la solution courante
        best->setIdentical(imbCurrent);
#if OPTIMIZE_LONG_RUN
#else
        if (best->isSolution())
        {
            break;
        }
#endif
    }//while
#if LS_DOUBLE_BEST
    if (params->traceActive)
    {
        writeStatistics(cptIIter, lout, imbBestBest);
        imbBestBest->writeStatistics(cptIIter);
        string s(fileSolution);
        ostringstream os; os<<"_"<<(cptIIter/nb_iter) * nb_iter;
        s.append(os.str());
        s.append(".improve.trace.svg");
        imbBestBest->writeSolution((char*)s.c_str());
    }
    imbBestBest->setIdentical(best);
#else
#endif

    best->setIdentical(imbCurrent);
}//local_search

template <typename Solution>
void LocalSearch<Solution>::iteratedConstructAndImprove()
{
    cout << "Iterated Contruct and Improve" << endl;

    Solution* best = imbBest;
    Solution* bestImprove = imbBestImprove;

    int cptIter = 0;
    for (cptIter = 0; cptIter < g_ConfigParameters->nbOfConstructAndRepairs; cptIter++)
    {
        // Construction
        if (g_ConfigParameters->nbOfInternalConstructs > 0)
        {
            if (g_ConfigParameters->constructFromScratchParam)
            {
                imbCurrent->constructSolutionSeq();
                imbCurrent->evaluate();
            }
            this->iteratedConstruct();
            bestImprove->setIdentical(imbCurrent);
        }
        else
        {
            imbCurrent->setIdentical(bestImprove);
        }

#if OPTIMIZE_LONG_RUN
#else
        if (bestImprove->isSolution())
        {
            bestImprove->setIdentical(best);
        }
        else
        {
#endif
            if (g_ConfigParameters->nbOfInternalRepairs > 0)
            {
                // Improvement
                switch (g_ConfigParameters->localSearchType)
                {
                    case 0:
                        this->iteratedRepair();
                        break;
                    case 1:
                        this->localSearch(true);
                        break;
                    case 2:
                        this->localSearch(false);
                        break;
                    default:
                        break;
                }
            }
#if OPTIMIZE_LONG_RUN
#else
        }
#endif
        bestImprove->evaluate();
        if (bestImprove->isBest(best))
        {
            bestImprove->setIdentical(best);
            best->writeSolution();
        }

        best->writeStatistics(cptIter, cout);
#if OPTIMIZE_LONG_RUN
#else
        if (best->isSolution())
        {
            break;
        }
#endif
    }//for
    best->writeStatistics(cptIter, cout);
    best->writeStatisticsToFile(cptIter);
}//iteratedConstructAndImprove


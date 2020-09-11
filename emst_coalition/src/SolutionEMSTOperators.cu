#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"
#include "SolutionEMST.h"

//#include <windows.h>
//#include <stdio.h>

/** Operateurs de changement de SolutionEMST courante.
 *
 */
#define FULL_GPU 1
#define FULL_GPU_FIND_MIN1 1
#define FULL_GPU_FIND_MIN2 1
#define FULL_GPU_CGU 1
#define FULL_GPU_FLATTENING 1

#define EMST_DETECT_CYCLE  0
#define EMST_FIND_MIN_PAIR_LIST 1// Distributed broadcast or distributed linked list

template<std::size_t DimP, std::size_t DimCM>
void SolutionEMST<DimP, DimCM>::initConstruct()
{
}//initConstruct

/** Construction Sequentielle
 */
template<std::size_t DimP, std::size_t DimCM>
void SolutionEMST<DimP, DimCM>::constructSolutionSeq()
{
    cout << "CONSTRUCTION SEQUENTIELLE ..." << endl;
    int nNodes = mr_links_cpu.adaptiveMap.getWidth();

    int iteration = 0;// maximum iterations

    int radiusSearchCells = 0;
    g_ConfigParameters->readConfigParameter("test_2opt", "radiusSearchCells", radiusSearchCells);

    float gpuTimingKernels = 0;
    float mstTotalTimeFrequen = 0;

    cout << "CONSTRUCTION done" << endl;
}

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionEMST<DimP, DimCM>::operator_1() {
    bool ret = true;


    global_objectif = computeObjectif();

    return ret;
}//operator_1

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionEMST<DimP, DimCM>::operator_2() {
    bool noUsed = true;

    return noUsed;
}//operator_1

template<std::size_t DimP, std::size_t DimCM>
bool SolutionEMST<DimP, DimCM>::generateNeighbor()
{
    int no_op = 0;
    double totalCapacity = 0;
    while (no_op < g_ConfigParameters->probaOperators.size()) {
        totalCapacity += g_ConfigParameters->probaOperators[no_op];
        no_op += 1;
    }

    // Tirage aleatoire par "roulette"
    double d = random_cf::aleat_double(0, totalCapacity);
    cout << "RANDOM VALUE OPERATOR !!!!!!!!!!!!!!!! " << d << endl;
     // Determiner no d'operateur
    no_op = -1;
    double t_sise = 0;
    int size = g_ConfigParameters->probaOperators.size();
    for (int k = 0; k < size; k++) {
        t_sise += g_ConfigParameters->probaOperators[k];
        //cout << "probaOperators " << g_ConfigParameters->probaOperators[k] << endl;
        if (d < t_sise) {
            no_op = k;
            break;
        }
    }
    if (no_op == -1)
        cout << "PB TIRAGE OPERATEUR !!! " << g_ConfigParameters->probaOperators.size() << endl;
    else
        cout << "Choix operator : " << no_op << endl;

    // Appliquer l'operateur ...
    if (applyOperator(no_op))
    {
        this->computeObjectif();
        if (this->global_objectif < 0)
        {
            cout << "ERROR!!! OPERATEUR num." << no_op << " A DONNE OBJECTIF NEGATIF : " << this->global_objectif << endl;
        }
    }
    return true;
}//generateNeighbor

template<std::size_t DimP, std::size_t DimCM>
bool SolutionEMST<DimP, DimCM>::applyOperator(int i)
{
    bool ret = false;
    switch (i)
    {
        case 0:
            break ;
        case 1:
            break ;
        case 2:
            break ;
        case 3:
            break ;
        case 4:
            break ;
        case 5:
            break ;
        case 6:
            break ;
        case 7:
            break ;
    }
    ret = operator_1();
    return ret;
}

template<std::size_t DimP, std::size_t DimCM>
int SolutionEMST<DimP, DimCM>::nbrOperators() const
{
    return g_ConfigParameters->probaOperators.size();
}

//! \brief Run et activate
template<std::size_t DimP, std::size_t DimCM>
void SolutionEMST<DimP, DimCM>::run() {
    while (activate()) {
        if (g_ConfigParameters->traceActive) {
            evaluate();
            writeStatisticsToFile(iteration);
        }
    }
}

template<std::size_t DimP, std::size_t DimCM>
bool SolutionEMST<DimP, DimCM>::activate() {
    bool ret = true;

    /*********************************************************
    * COMPACT GRAPH
    * wb.Q: refresh each vertex's component root identifier using union-find algorithm.
    * input: disjointSetMap[0][i] = C; C is a constant value from 0 to N-1. In the first iteration, C==i.
    * output: disjointSetMap[0][i] new root identifier for each vertex
    * *******************************************************
    */
    // Time
    float timeGpu;
    cudaEvent_t pcFreq;
    cudaEvent_t pcFreq2;
    cudaEventCreate(&pcFreq);
    cudaEventCreate(&pcFreq2);
    cudaEventRecord(pcFreq, 0);

    boruvkaOp.K_flatten_DST(mr_links_gpu.disjointSetMap,
                            mr_links_gpu.nodeParentMap);

    // End time
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    cout << "Time flatten ST: " << timeGpu << endl;
    time_flatten += timeGpu;
    traceParallelMST.timeFlatten = timeGpu;
    traceParallelMST.timeCumulativeFlatten = time_flatten;

    /**********************************************************************
    * TEST TERMINAISON
    **********************************************************************
    */

    // Time
    cudaEventRecord(pcFreq, 0);

    mr_links_cpu.disjointSetMap.gpuCopyDeviceToHost(mr_links_gpu.disjointSetMap);

    int finish = boruvkaOp.testTermination(mr_links_cpu.disjointSetMap);

    //qwb test
//    (ofstream&)cout << mr_links_cpu.disjointSetMap << endl;

    // End time
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    cout << "Time Test Termination: " << timeGpu << endl;
    time_terminate += timeGpu;
    traceParallelMST.timeTestTermination = timeGpu;
    traceParallelMST.timeCumulativeTermination = time_terminate;

    if (finish == 0 || iteration == 100) {
        cout << "ACTIVATE FINISHED !!!" << endl;
        // wq delete these copies to do running time statistic again 24November18
//        distanceMap_cpu.gpuCopyDeviceToHost(distanceMap);
//        mr_links_cpu.nodeParentMap.gpuCopyDeviceToHost(mr_links_gpu.nodeParentMap);
//        mr_links_cpu.gpuCopyDeviceToHost(mr_links_gpu);
//        mr_links_cpu.nVisitedMap.gpuCopyDeviceToHost(mr_links_gpu.nVisitedMap);
//        mr_links_cpu.evtMap.gpuCopyDeviceToHost(mr_links_gpu.evtMap);
//        std::ofstream fo;
//        fo.open("ESSAI.nodeParentMap");
//        if (fo)
//            fo << mr_links_cpu.nodeParentMap;
//        std::ofstream fo2;
//        fo2.open("ESSAI.distanceMap");
//        if (fo2)
//            fo2 << distanceMap_cpu;
        return ret = false;
    }

    iteration ++;
    cout << "iteration " << iteration << endl;

    /**********************************************************************
    * FIND MIN 1
    * wb.Q: Find each vertex's minimum outgoing length and closest outgoing point
    * input: adaptiveMap stores the new coordinate of one vertex;
    *        disjointSetMap refreshed in last COMPACT GRAPH step;
    *        refresh distanceMap[0][i] to be INFINITY at each iteration;
    *        refresh correspondenceMap to be (-1, -1) at each iteration;
    * output: distanceMap, stores the minimum outgoing length for each vertex;
    *         correspondenceMap, stores the cloest outgoing point for each vertex.
    **********************************************************************
    */

    // Time
    cudaEventRecord(pcFreq, 0);

#if NEMST_TEST_PREVIOUS
#else
    distanceMap.gpuResetValue(HUGE_VAL);
    PointCoord pInitial(-1);
    mr_links_gpu.correspondenceMap.gpuResetValue(pInitial);
#endif
#if EMST_SPIRAL_SEARCH_KD_SLAB_2
    boruvkaOp.K_computeOctants(cm_gpu,
                                     mr_links_gpu.disjointSetMap,
                                     mr_links_gpu.adaptiveMap,
                                     distanceMap,
                                     mr_links_gpu.correspondenceMap,
                                     spiralSearchMap);
#endif
    boruvkaOp.K_findNextClosestPoint(cm_gpu,
                                     mr_links_gpu.disjointSetMap,
                                     mr_links_gpu.adaptiveMap,
                                     distanceMap,
                                     mr_links_gpu.correspondenceMap,
                                     spiralSearchMap);

    // End time
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    time_next_closest += timeGpu;
    cout << "Time Find Next Closest: " << timeGpu << endl;
    traceParallelMST.timeFindNextClosest = timeGpu;

    /**********************************************************************
    * FIND MIN 2
    * wb.Q: traverse each component using BFS graph search, find the minimum outgoing edge for each component.
    * input: results of FIND MIN 1: distanceMap(length), correspondenceMap(BCP point);
    *        refresh fixedMap() to be 0 at each iteration;
    * output: fixedMap[0][i]: one component's winner node possessing the shortest ougoing edge.
    **********************************************************************
    */

    // Time cpu
    cudaEventRecord(pcFreq, 0);

#if EMST_FIND_MIN_PAIR_LIST
    mr_links_gpu.nodeParentMap.gpuResetValue(-1);
    boruvkaOp.K_createComponentList(mr_links_gpu.nodeParentMap,
                            mr_links_gpu.disjointSetMap,
                            mr_links_gpu.correspondenceMap
                            );

    mr_links_gpu.fixedMap.gpuResetValue(0);
    boruvkaOp.K_findMinPair(mr_links_gpu.nodeParentMap,
                            mr_links_gpu.disjointSetMap,
                            mr_links_gpu.fixedMap,
                            mr_links_gpu.correspondenceMap,
                            distanceMap
                            );
#else
    // Initialize working buffers for DB
    mr_links_gpu.evtMap.gpuResetValue(0);
    mr_links_gpu.nVisitedMap.gpuResetValue(0x7FFFFFFF);
    mr_links_gpu.nodeParentMap.gpuResetValue(-1);
    mr_links_gpu.nodeWinMap.gpuResetValue(IndexG(-1));
    mr_links_gpu.nodeDestMap.gpuResetValue(IndexG(-1));
    minDistMap.gpuResetValue(HUGE_VAL);
    mr_links_gpu.fixedMap.gpuResetValue(0);

    boruvkaOp.K_findMinInComponentDB(
                mr_links_gpu.networkLinks,
                mr_links_gpu.disjointSetMap,
                mr_links_gpu.fixedMap,
                mr_links_gpu.correspondenceMap,
                distanceMap,
                mr_links_gpu.evtMap,
                mr_links_gpu.nVisitedMap,
                mr_links_gpu.nodeParentMap,
                mr_links_gpu.nodeWinMap,
                mr_links_gpu.nodeDestMap,
                minDistMap,
                stateMap,
                stateMap_cpu
                );
#endif

    // End time cpu
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    time_find_pair += timeGpu;
    cout << "Time Find Min Pair in Component: " << timeGpu << endl;
    traceParallelMST.timeFindMinPair = timeGpu;

    // Cumulative time
    cout << "Cumulative Time Find Next Closest: " << time_next_closest << endl;
    cout << "Cumulative Time Find Min Pair in Component: " << time_find_pair << endl;
    traceParallelMST.timeCumulativeFindNextClosest = time_next_closest;
    traceParallelMST.timeCumulativeFindMinPair = time_find_pair;

    /************************************************************
    * Connect Component and Union
    ***********************************************************/

   //qwb test
//    mr_links_cpu.disjointSetMap.gpuCopyDeviceToHost(mr_links_gpu.disjointSetMap);
//    mr_links_cpu.fixedMap.gpuCopyDeviceToHost(mr_links_gpu.fixedMap);
//    cout << ">>>>before connect uniono " << endl;
//    (ofstream&)cout << mr_links_cpu.disjointSetMap << endl;
//    cout << ">>> before connect union winner map " << endl;
//    (ofstream&)cout << mr_links_cpu.fixedMap << endl;

    // Time cpu
    cudaEventRecord(pcFreq, 0);

    boruvkaOp.K_connectComponentAndUnion(
                mr_links_gpu.networkLinks,
                mr_links_gpu.disjointSetMap,
                mr_links_gpu.fixedMap,
                mr_links_gpu.correspondenceMap,
                mr_links_gpu.nodeParentMap
                );

    // End time cpu
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    cout << "Time Connect Graph Union: " << timeGpu << endl;
    time_connect_union += timeGpu;
    traceParallelMST.timeConnectGraphUnion = timeGpu;
    traceParallelMST.timeCumulativeConnetUnion = time_connect_union;

#if EMST_DETECT_CYCLE
    // Initialize working buffers for DB
    mr_links_gpu.evtMap.gpuResetValue(0);
    mr_links_gpu.nVisitedMap.gpuResetValue(0x7FFFFFFF);
    mr_links_gpu.nodeParentMap.gpuResetValue(-1);

    stateMap.gpuResetValue(-2);

    boruvkaOp.K_diffusateDetectCycleDB(
                mr_links_gpu.networkLinks,
                mr_links_gpu.disjointSetMap,
                mr_links_gpu.evtMap,
                mr_links_gpu.nVisitedMap,
                mr_links_gpu.nodeParentMap,
                stateMap,
                stateMap_cpu
                );
    if (iteration == 2) {
        cout << "ACTIVATE FINISHED 2 !!!" << endl;
        distanceMap_cpu.gpuCopyDeviceToHost(distanceMap);
        mr_links_cpu.nodeParentMap.gpuCopyDeviceToHost(mr_links_gpu.nodeParentMap);
        mr_links_cpu.evtMap.gpuCopyDeviceToHost(mr_links_gpu.evtMap);
        mr_links_cpu.gpuCopyDeviceToHost(mr_links_gpu);
        std::ofstream fo;
        fo.open("ESSAI.nodeParentMap");
        if (fo)
            fo << mr_links_cpu.nodeParentMap;
        std::ofstream fo2;
        fo2.open("ESSAI.distanceMap");
        if (fo2)
            fo2 << distanceMap_cpu;
        std::ofstream fo3;
        fo3.open("ESSAI.evtMap");
        if (fo3)
            fo3 << mr_links_cpu.evtMap;
        return ret = false;
    }
#endif
    // Copy Spanning forest for IHM drawing.
    // wq delete this for time statistic
//    cudaDeviceSynchronize();
    mr_links_cpu.networkLinks.gpuCopyDeviceToHost(mr_links_gpu.networkLinks);
//    cudaDeviceSynchronize();

//    Sleep(10000);

    return ret;
}


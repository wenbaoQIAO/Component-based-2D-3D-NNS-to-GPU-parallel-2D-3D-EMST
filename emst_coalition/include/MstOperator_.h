#ifndef MST_OPERATOR_H
#define MST_OPERATOR_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao
 * Creation date : Sep. 2016
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>

#include <vector>
#include <iterator>

#include "macros_cuda.h"
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "basic_operations.h"
#include "Cell.h"
#include "adaptator_basics.h"
#include "CellularMatrix.h"
#include "distance_functors.h"
#include "cudaerrorcheck.h"

#include "ViewGrid.h"
#ifdef TOPOLOGIE_HEXA
#include "ViewGridHexa.h"
#endif
#include "NIter.h"

//! reference EMST components
#include "NeuralNetEMST.h"
#include "SpiralSearchNNL.h"
#include "adaptator_EMST.h"
#include "macros_cuda_EMST.h"

//! WB.Q add macro for divide-conquer tsp that allow division work only on CPU side
#define cpuDivision 0
#define CpuCompact 1 // cpu compact while gpu findMin2
#define BLOCKDIMX 1024
#define CPUMERGINGGRAPH 0

using namespace std;
using namespace components;

namespace operators
{

//!wb.Q add swap
DEVICE_HOST void inline swapClass4(BufferLinkPointCoord** tabOPointer, BufferLinkPointCoord** tabDPointer)
{

    BufferLinkPointCoord* tabDPointerTemp = *tabOPointer;
    * tabOPointer = *tabDPointer;
    *tabDPointer = tabDPointerTemp;
}


//!wb.Q add swap
DEVICE_HOST void inline swapClass4(BufferLinkPointCoordVector** tabOPointer, BufferLinkPointCoordVector** tabDPointer)
{

    BufferLinkPointCoordVector* tabDPointerTemp = *tabOPointer;
    * tabOPointer = *tabDPointer;
    *tabDPointer = tabDPointerTemp;
}


////!wb.Q operator initial component ID of array, not for 2D array
//struct OperateInitialComponentID{

//    DEVICE_HOST OperateInitialComponentID(){}


//    template <class NetLinkPointCoord>
//    DEVICE_HOST void operate(NetLinkPointCoord &nn_source, PointCoord ps)
//    {
//        nn_source.grayValueMap[ps[1]][ps[0]] = ps[0];

//    }
//};


////!WB.Q operator compact graph using disjoint set data structure
//struct OperateCompactDisjointSetCpu{

//    OperateCompactDisjointSetCpu(){}

//    //! find one father
//    int find(Grid<GLint> &idGpuMap, int x)
//    {
//        GLint r = x;
//        while(r != idGpuMap[0][r])
//        {
//            r = idGpuMap[0][r]; // r is the root
//        }
//        return r;
//    }

//    //! wb.Q access to root and union two components
//    template <typename NetLinkPointCoord>
//    void operate(NetLinkPointCoord &nn_source, PointCoord ps){

//        // find root
//        int r = find(nn_source.grayValueMap, ps[0]);
//        // compact id
//        GLint i = ps[0], j;
//        while(i != r)
//        {
//            j = nn_source.grayValueMap[0][i]; // template value of current id
//            nn_source.grayValueMap[0][i] = r;

//            i = j;
//        }
//    }
//};

//!wb.Q operator compact disjoint set data structure
struct OperateCompactDisjointSet{

    DEVICE_HOST OperateCompactDisjointSet(){}

    DEVICE_HOST int find(Grid<GLint> &idGpuMap, int x)
    {
        GLint r = x;
        while(r != idGpuMap[0][r])
        {
            r = idGpuMap[0][r]; // r is the root
        }

        return r;
    }

#ifdef CUDA_CODE
    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord &nn_source, PointCoord ps){

        // find root
        int r = find(nn_source.grayValueMap, ps[0]);
        nn_source.activeMap[0][r] = 1; // mark rout as active
        // compact id
        GLint i = ps[0], j;
        while(i != r)
        {
            j = nn_source.grayValueMap[0][i];
            atomicExch(&(nn_source.grayValueMap[0][i]), r);

            i = j;
        }
    }
#else

    DEVICE_HOST void operate(NetLinkPointCoord &nn_source, PointCoord ps){

        // find root
        int r = find(nn_source.grayValueMap, ps[0]);
        nn_source.activeMap[0][r] = 1; // mark rout as active
        // compact id
        GLint i = ps[0], j;
        while(i != r)
        {
            j = nn_source.grayValueMap[0][i]; // template value of current id
            nn_source.grayValueMap[0][i] = r;

            i = j;
        }

    }

#endif
};


//! WB.Q add for EMST
DEVICE int minimum(int ps, int fameNode){

    if(ps < fameNode)
        return ps;
    else
        return fameNode;
}

//! WB.Q add for EMST
DEVICE int maximum(int ps, int fameNode){

    if(ps > fameNode)
        return ps;
    else
        return fameNode;
}



//! wb.Q add find one vertex possessing minimum outgoing distance
template <class NetLinkPointCoord>
DEVICE void findShortestLexicoEdgeBetweenComponentsMinimum(NetLinkPointCoord& nn_source,
                                                           PointCoord& ps_,
                                                           superVertexDataMinimum& neighborClosestComponent)
{

    PointCoord fammeNode = nn_source.correspondenceMap[ps_[1]][ps_[0]];

    if(fammeNode[0] != -1){

        int fammeNodeID = nn_source.grayValueMap[fammeNode[1]][fammeNode[0]];
        double distance = nn_source.densityMap[ps_[1]][ps_[0]];

        int minimumId = minimum(ps_[0], fammeNode[0]);
        int minimumId2 = maximum(ps_[0], fammeNode[0]);

        if(fammeNodeID == INFINITY)
            printf("attention ,  do not find neighbor node for ps_, %d, %d ", ps_[0], ps_[1]);

        if(distance < neighborClosestComponent.minCoupleDistance){
            neighborClosestComponent.minCoupleDistance = distance;
            neighborClosestComponent.couple1 = ps_;
            neighborClosestComponent.id2 = fammeNodeID;// QWB compare big ID
            neighborClosestComponent.id1 = minimumId; // QWB compare small id of curent ps
            neighborClosestComponent.minimumId2 = minimumId2;
        }
        else if(distance == neighborClosestComponent.minCoupleDistance){

            if(
                    // (fammeNodeID == neighborClosestComponent.id2)&&
                minimumId < neighborClosestComponent.id1 ){

                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.couple1 = ps_;
                neighborClosestComponent.id2 = fammeNodeID;
                neighborClosestComponent.id1 = minimumId;
                neighborClosestComponent.minimumId2 = minimumId2;

            }
            else if(
                    // (fammeNodeID == neighborClosestComponent.id2)&&
                    minimumId == neighborClosestComponent.id1
                    && minimumId2 < neighborClosestComponent.minimumId2 ){

                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.couple1 = ps_;
                neighborClosestComponent.id2 = fammeNodeID;
                neighborClosestComponent.id1 = minimumId;
                neighborClosestComponent.minimumId2 = minimumId2;
            }
        }
    }
}



/*!
         * \brief 0817 WB.Q add findMin2 improvement
         */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_MST_findMin2_improve(NeuralNetLinks<BufferDimension, Point> nn_source)
{

    int _x = blockIdx.x * blockDim.x + threadIdx.x;

    if (_x < nn_source.adaptiveMap.width && nn_source.correspondenceMap[0][_x][0] != -1
            && nn_source.activeMap[0][_x])
    {

        PointCoord ps(_x, 0);

        superVertexDataMinimum neiClosCompo; // for each component in superVertices, find all its different neighborId

        {
            BufferLinkPointCoordVector nodeAlreadyTraversed; // change to arrays, do not need count, just int
            nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

            BufferLinkPointCoordVector tabO;
            tabO.init(MAX_VECTOR_SIZE);
            BufferLinkPointCoordVector tabD;
            tabD.init(MAX_VECTOR_SIZE);

            tabO.insert(ps);

            BufferLinkPointCoordVector* tabO_ = &tabO;
            BufferLinkPointCoordVector* tabD_ = &tabD;
            BufferLinkPointCoordVector* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

            bool traversedComp = 0;

            while((*tabO_).numLinks > 0 && !traversedComp){

                for (int i = 0; i < (*tabO_).numLinks; i++){

                    PointCoord pCoord = (*tabO_).bCell[i];

                    findShortestLexicoEdgeBetweenComponentsMinimum(nn_source, pCoord, neiClosCompo);


                    Point2D pInitial(-1, -1);
                    int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                    for (int pLink = 0; pLink < nLinks; pLink++){

                        Point2D pLinkOfNode;
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        if(pLinkOfNode != pInitial)
                        {
                            PointCoord pCoordLink(-1, -1);
                            pCoordLink[0] = (int)pLinkOfNode[0];
                            pCoordLink[1] = (int)pLinkOfNode[1];

                            // compare if the current pCoord is already be teached
                            bool traversed = 0;
                            for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
                                PointCoord pLinkTemp(-1, -1);
                                pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
                                if (pCoordLink == pLinkTemp)
                                    traversed = 1;
                            }
                            if (!traversed)
                                (*tabD_).insert(pCoordLink);

                        }
                    }

                }


                if((*nodeAlreadyTraversed_).numLinks > MAX_VECTOR_SIZE)
                    printf("size findmin2 %d, %d, %d \n", (*nodeAlreadyTraversed_).numLinks, (*tabO_).numLinks, (*tabD_).numLinks);

                (*nodeAlreadyTraversed_).clearLinks();


                swapClass4(&nodeAlreadyTraversed_, &tabO_);
                swapClass4(&tabO_, &tabD_);


            }// while
        }



        // build mst, only the ps == winner node, then export the winner couple to do reconnection step
        if(neiClosCompo.minCoupleDistance != INFINITY
                && neiClosCompo.minCoupleDistance != 999999
                //                && neiClosCompo.couple1 == ps // findMin2 improve
                )
        {
            nn_source.fixedMap[0][neiClosCompo.couple1[0]] = 1; //  mark the winner node for current component, used in step4

        }

    }
}





////! kernel template for operation on array of component id
//template < template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
//           class OperateAdaptor
//           >
//KERNEL void K_Mst_OperatorOnComponentID(NeuralNetLinks<BufferDimension, Point> nn_source,
//                                        OperateAdaptor operateAdaptor)
//{
//    KER_SCHED(nn_source.adaptiveMap.getWidth(), nn_source.adaptiveMap.getHeight())
//            if(_x < nn_source.adaptiveMap.getWidth() && _y < nn_source.adaptiveMap.getHeight())
//    {
//        PointCoord ps(_x, _y);

//        operateAdaptor.operate(nn_source, ps);
//    }

//    END_KER_SCHED

//            SYNCTHREADS
//}//K_Mst_OperatorOnComponentID


/*!
 * \brief 190916 QWB: add K_MST_projector_nodeSpiralSearch
 * \to spiral search every node's neighborhood nodes
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_MST_projector_links_nodeSpiralSearch(CellularMatrix cm_source,
                                                   CellularMatrix2 cm_cible,
                                                   NeuralNetLinks<BufferDimension, Point> nn_source,
                                                   NeuralNetLinks<BufferDimension, Point> nn_cible,
                                                   GetAdaptor getAdaptor,
                                                   SearchAdaptor searchAdaptor,
                                                   OperateAdaptor operateAdaptor,
                                                   int radiusSearchCells)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {

        PointCoord ps(_x, _y);
        PointCoord cellCoor;

        if(!nn_source.activeMap[ps[1]][ps[0]])
        {

            // find cell coordinate of ps
            PointCoord PC;// no use
            bool extracted = false;
            extracted = getAdaptor.search(cm_source,
                                          nn_source,
                                          nn_cible,
                                          PC,
                                          ps,
                                          cellCoor);

            if(extracted){

                // Spiral search centering with current cellCoor
                PointCoord minP;
                bool found = false;

                found = searchAdaptor.search(cm_source,
                                             nn_source,
                                             nn_cible,
                                             cellCoor,
                                             ps,
                                             minP,
                                             radiusSearchCells);

            }

        }

    }

    END_KER_SCHED

            SYNCTHREADS
}//END


/*!
 * \brief 190916 QWB: add K_MST_projector_nodeSpiralSearch
 * \to spiral search every node's neighborhood nodes, overload for cpu
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
void K_MST_projector_links_nodeSpiralSearch_cpu(CellularMatrix cm_source,
                                                CellularMatrix2 cm_cible,
                                                NeuralNetLinks<BufferDimension, Point> nn_source,
                                                NeuralNetLinks<BufferDimension, Point> nn_cible,
                                                GetAdaptor getAdaptor,
                                                SearchAdaptor searchAdaptor,
                                                OperateAdaptor operateAdaptor,
                                                int radiusSearchCells)
{

    for (int _y = 0; _y < (nn_source.adaptiveMap.height); ++_y) {
        for (int _x = 0; _x < (nn_source.adaptiveMap.width); ++_x) {


            PointCoord ps(_x, _y);
            PointCoord cellCoor;

            if(!nn_source.activeMap[ps[1]][ps[0]])
            {

                cout << " ps sss " << ps[0] << endl;

                // find cell coordinate of ps
                PointCoord PC;// no use
                bool extracted = false;
                extracted = getAdaptor.search(cm_source,
                                              nn_source,
                                              nn_cible,
                                              PC,
                                              ps,
                                              cellCoor);

                if(extracted){

                    // Spiral search centering with current cellCoor
                    PointCoord minP;
                    bool found = false;

                    found = searchAdaptor.search(cm_source,
                                                 nn_source,
                                                 nn_cible,
                                                 cellCoor,
                                                 ps,
                                                 minP,
                                                 radiusSearchCells);

                }

            }

        }
    }
}//END




/*!
 * \brief 260916 QWB: add to mark superVertice
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_markSuperVertice(CellularMatrix cm_source,
                                   NeuralNetLinks<BufferDimension, Point> nn_source,
                                   OperateAdaptor operateAdaptor)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height){

        PointCoord ps(_x, _y);

        if(nn_source.activeMap[ps[1]][ps[0]])// WB.Q final winner node marked at beginning and at [connect graph]
        {
            operateAdaptor.operate(nn_source, ps);
        }
    }

    END_KER_SCHED
            SYNCTHREADS


}//


/*!
 * \brief 031216 QWB: add to mark superVertice
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_markSuperVertice_final(CellularMatrix cm_source,
                                         NeuralNetLinks<BufferDimension, Point> nn_source,
                                         OperateAdaptor operateAdaptor,
                                         bool *dev_finishMst)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height){

        PointCoord ps(_x, _y);

        if(nn_source.fixedMap[ps[1]][ps[0]])// WB.Q final winner node marked at beginning and at [connect graph]
        {

            operateAdaptor.operate(nn_source, ps, dev_finishMst);
        }
    }

    END_KER_SCHED
            SYNCTHREADS


}//






/*!
 * \brief 031216 QWB: add to compact graph2
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_CompactGraph2_final(NeuralNetLinks<BufferDimension, Point> nn_source,
                                      OperateAdaptor operateAdaptor)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height){

        PointCoord ps(_x, _y);

        if(nn_source.activeMap[ps[1]][ps[0]])// WB.Q final winner node marked at beginning and at [connect graph]
        {
            //            printf("[2]begin active ps %d, %d \n", ps[0], ps[1]);
            operateAdaptor.operate(nn_source, ps);
        }
    }

    END_KER_SCHED
            SYNCTHREADS


}//



/*!
 * \brief 260916 QWB: add to mark superVertice on CPU side
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
void K_MST_markSuperVertice_cpu(CellularMatrix cm_source,
                                NeuralNetLinks<BufferDimension, Point> nn_source,
                                OperateAdaptor operateAdaptor)
{
    for (int _y = 0; _y < (nn_source.adaptiveMap.height); ++_y) {
        for (int _x = 0; _x < (nn_source.adaptiveMap.width); ++_x) {

            PointCoord ps(_x, _y);

            if(nn_source.activeMap[ps[1]][ps[0]]) // WB.Q activeMap is the final winner node found at step4
                operateAdaptor.operate(nn_source, ps);

        }
    }


}//


/*!
 * \brief 141016 QWB: add to operate shortest outging edge
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_MST_operateShortestOutgingEdges(CellularMatrix cm_source,
                                              NeuralNetLinks<BufferDimension, Point> nn_source,
                                              SearchAdaptor searchAdaptor,
                                              OperateAdaptor operateAdaptor,
                                              Grid<BufferDimension> netWorkLinkCopy)
{

    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {

        PointCoord ps(_x, _y);

        bool found = false;
        PointCoord couple1(-1, -1); //! WB.Q do not change this initial value, later important use in reconnection step
        PointCoord couple2(-1, -1);

        if(nn_source.correspondenceMap[_y][_x] != couple1){

            found = searchAdaptor.search(nn_source,
                                         ps, couple1, couple2);

            //! WB.Q do the connection step seperately , because device_host and device, found is necessary, 091116 try to do this step on cpu side
            //        if(found){
            //            operateAdaptor.operate(nn_source, ps, couple1, couple2, netWorkLinkCopy);
            //        }
        }

    }

    END_KER_SCHED
            SYNCTHREADS
}//


/*!
         * \brief 141016 QWB: add to operate shortest outging edge
         */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_operateConnectGraphRootMergingGraph(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                      OperateAdaptor operateAdaptor,
                                                      Grid<BufferLinkPointCoord> rootMergingGraphGpu)
{

    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height && nn_source.fixedMap[_y][_x])
    {
        PointCoord ps(_x, _y);

        operateAdaptor.operate(nn_source, ps, rootMergingGraphGpu);
    }

    END_KER_SCHED
            SYNCTHREADS
}//


/*!
 * \brief 031216 QWB: add to mark superVertice
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_markSuperVertice_Disjoint(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                 Grid<BufferLinkPointCoord> rootMergingGraphGpu,
                                                 OperateAdaptor operateAdaptor,
                                                 float *dev_finishMst)
{
    int _x = blockIdx.x * blockDim.x + threadIdx.x;

    if (_x < nn_source.adaptiveMap.width && nn_source.fixedMap[0][_x]){


        PointCoord ps(_x, 0);

        operateAdaptor.operate(nn_source, rootMergingGraphGpu, ps, dev_finishMst);

    }

    SYNCTHREADS

}//


/*!
 * \brief 031216 QWB: add to compact graph2
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_CompactGraph2_finalDisjoint(NeuralNetLinks<BufferDimension, Point> nn_source,
                                              Grid<BufferLinkPointCoord> rootMergingGraph,
                                              OperateAdaptor operateAdaptor)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height && nn_source.activeMap[_y][_x]){

        PointCoord ps(_x, _y);
        operateAdaptor.operate(nn_source, rootMergingGraph, ps);
    }

    END_KER_SCHED
            SYNCTHREADS
}//




/*!
 * \brief 141016 QWB: add to operate shortest outging edge
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_operateConnectGraph(NeuralNetLinks<BufferDimension, Point> nn_source,
                                      OperateAdaptor operateAdaptor)
{

    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height && nn_source.fixedMap[_y][_x])
    {
        PointCoord ps(_x, _y);

        operateAdaptor.operate(nn_source, ps);
    }

    END_KER_SCHED
            SYNCTHREADS
}//



/*!
 * \brief 011116 QWB: add to operate shortest outging edge on cpu side
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class SearchAdaptor,
          class OperateAdaptor
          >
void K_MST_operateShortestOutgingEdges_cpu(CellularMatrix cm_source,
                                           NeuralNetLinks<BufferDimension, Point> nn_source,
                                           SearchAdaptor searchAdaptor,
                                           OperateAdaptor operateAdaptor,
                                           Grid<BufferDimension> netWorkLinkCopy)
{

    for (int _y = 0; _y < (nn_source.adaptiveMap.height); ++_y) {
        for (int _x = 0; _x < (nn_source.adaptiveMap.width); ++_x) {

            PointCoord ps(_x, _y);

            if(nn_source.fixedMap[ps[1]][ps[0]]){
                operateAdaptor.operate(nn_source, ps, netWorkLinkCopy);
            }
        }
    }
}




/*!
 * \brief 191016 QWB: add double links to winner outgoing node
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
KERNEL void K_MST_addDoubleLinkToWinnerNode(CellularMatrix cm_source,
                                            NeuralNetLinks<BufferDimension, Point> nn_source,
                                            OperateAdaptor operateAdaptor,
                                            Grid<BufferDimension> netWorkLinkCopy)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)


            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {

        PointCoord ps(_x, _y);


        // if this node is winner node
        if(nn_source.fixedMap[ps[1]][ps[0]]){

            operateAdaptor.operate(nn_source, ps, netWorkLinkCopy);
        }


    }

    END_KER_SCHED
            SYNCTHREADS
}//



/*!
 * \brief 191016 QWB: add double links to winner outgoing node
 */
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
void K_MST_addDoubleLinkToWinnerNode_cpu(CellularMatrix cm_source,
                                         NeuralNetLinks<BufferDimension, Point> nn_source,
                                         OperateAdaptor operateAdaptor,
                                         Grid<BufferDimension> netWorkLinkCopy)
{
    for (int _y = 0; _y < (nn_source.adaptiveMap.height); ++_y) {
        for (int _x = 0; _x < (nn_source.adaptiveMap.width); ++_x) {

            PointCoord ps(_x, _y);

            // if this node is winner node
            if(nn_source.fixedMap[ps[1]][ps[0]]){

                cout << "ps step4 " << ps[0] << endl;
                operateAdaptor.operate(nn_source, ps, netWorkLinkCopy);
            }
        }
    }
}//




/*!
 * \brief 041016 QWB: add K_SO_linkNetGetSearchOperate
 * \ to get cell coordiate for every node, spiral search its closest nodes, and operate on these closes nodes
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class GetAdaptor,
          class SearchAdaptor
          >
KERNEL void K_MST_linkNetGetSearch(CellularMatrix cm_source,
                                   CellularMatrix2 cm_cible,
                                   NeuralNetLinks<BufferDimension, Point> nn_source,
                                   NeuralNetLinks<BufferDimension, Point> nn_cible,
                                   GetAdaptor getAdaptor,
                                   SearchAdaptor searchAdaptor,
                                   int radiusSearchCells)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {
        PointCoord ps(_x, _y);
        PointCoord cellCoor;

        // find cell coordinate of ps
        PointCoord PC;
        bool extracted = false;
        extracted = getAdaptor.search(cm_cible,
                                      nn_source,
                                      nn_cible,
                                      PC,
                                      ps,
                                      cellCoor);


        if(extracted){

            // Spiral search centering with current cellCoor
            PointCoord minP;
            bool found = false;

            found = searchAdaptor.search(cm_cible,
                                         nn_source,
                                         nn_cible,
                                         cellCoor,
                                         ps,
                                         minP,
                                         radiusSearchCells);
        }



    }

    END_KER_SCHED

            SYNCTHREADS
}//K_SO_projector_links

/*!
 * \brief 041016 QWB: add K_SO_linkNetGetSearchOperate
 * \ to get cell coordiate for every node, spiral search its closest nodes, and operate on these closes nodes
 */
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_MST_linkNetGetSearchOperate(CellularMatrix cm_source,
                                          CellularMatrix2 cm_cible,
                                          NeuralNetLinks<BufferDimension, Point> nn_source,
                                          NeuralNetLinks<BufferDimension, Point> nn_cible,
                                          GetAdaptor getAdaptor,
                                          SearchAdaptor searchAdaptor,
                                          OperateAdaptor operateAdaptor,
                                          int radiusSearchCells)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {
        PointCoord ps(_x, _y);
        PointCoord cellCoor;

        // find cell coordinate of ps
        PointCoord PC;// no use
        bool extracted = false;
        extracted = getAdaptor.search(cm_cible,
                                      nn_source,
                                      nn_cible,
                                      PC,
                                      ps,
                                      cellCoor);


        if(extracted){

            // Spiral search centering with current cellCoor
            PointCoord minP;
            bool found = false;

            found = searchAdaptor.search(cm_cible,
                                         nn_source,
                                         nn_cible,
                                         cellCoor,
                                         ps,
                                         minP,
                                         radiusSearchCells);


            if(found){
                operateAdaptor.operate(cm_cible[cellCoor[1]][cellCoor[0]],
                        nn_source,
                        nn_source,
                        ps,
                        minP);

            }



        }



    }

    END_KER_SCHED

            SYNCTHREADS
}//K_SO_projector_links



//!############################### kernel call global side #######################################################
//! 131016 QWB: add to mark superVertices on GPU side
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_markSuperVertices(CellularMatrix& source,
                                       NeuralNetLinks<BufferDimension, Point>& nn_source,
                                       class OperateAdaptor& oa) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);

    K_MST_markSuperVertice _KER_CALL_(b, t)(source, nn_source, oa);
}



//! 031216 QWB: add to mark superVertices on GPU side
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_markSuperVertices_final(CellularMatrix& source,
                                             NeuralNetLinks<BufferDimension, Point>& nn_source,
                                             class OperateAdaptor& oa,
                                             bool *dev_finishMst) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);

    K_MST_markSuperVertice_final _KER_CALL_(b, t)(source, nn_source, oa, dev_finishMst);
}

//! 031216 QWB: add to mark superVertices on GPU side
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_CompactGraph2_final(NeuralNetLinks<BufferDimension, Point>& nn_source,
                                         class OperateAdaptor& oa) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);

    K_MST_CompactGraph2_final _KER_CALL_(b, t)(nn_source, oa);
}


//! 131016 QWB: add to mark superVertices on CPU side
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
inline void K_markSuperVertices_cpu(CellularMatrix& source,
                                    NeuralNetLinks<BufferDimension, Point>& nn_source,
                                    class OperateAdaptor& oa) {

    K_MST_markSuperVertice_cpu(source, nn_source, oa);
}


//! 011116 QWB: add to operate shortest outgoing edge for each superVertices on CPU side
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class SearchAdaptor,
          class OperateAdaptor
          >
inline void K_operateShortestOutgingEdge_cpu(CellularMatrix& source,
                                             NeuralNetLinks<BufferDimension, Point>& nn_source,
                                             class SearchAdaptor& sa,
                                             class OperateAdaptor& oa,
                                             Grid<BufferDimension>& netWorkLinkCopy) {

    K_MST_operateShortestOutgingEdges_cpu(source, nn_source, sa, oa, netWorkLinkCopy);
}



//! 141016 QWB: add to operate shortest outgoing edge for each superVertices on GPU side
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class SearchAdaptor,
          class OperateAdaptor
          >
GLOBAL inline void K_operateShortestOutgingEdge(CellularMatrix& source,
                                                NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                class SearchAdaptor& sa,
                                                class OperateAdaptor& oa,
                                                Grid<BufferDimension>& netWorkLinkCopy) {


    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);

    K_MST_operateShortestOutgingEdges _KER_CALL_(b, t)(source, nn_source, sa, oa, netWorkLinkCopy);
}



//! 031216 QWB: add single link to link-network
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_operateConnectGraph(NeuralNetLinks<BufferDimension, Point>& nn_source,
                                         class OperateAdaptor& oa) {


    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);

    K_MST_operateConnectGraph _KER_CALL_(b, t)(nn_source, oa);
}

//! 031216 QWB: add single link to link-network, construct root merging graph
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_operateConnectGraphWithRootMergingG(NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                         class OperateAdaptor& oa,
                                                         Grid<BufferLinkPointCoord>& rootMergingGraphGpu)
{

    KER_CALL_THREAD_BLOCK_1D(b, t,
                             BLOCKSIZE, 1,
                             nn_source.adaptiveMap.width,
                             nn_source.adaptiveMap.height);

    K_MST_operateConnectGraphRootMergingGraph _KER_CALL_(b, t)(nn_source, oa, rootMergingGraphGpu);
}


//! 030317 wenbao Qiao: add to mark superVertices on GPU side to mark final winner root
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_markSuperVertices_Disjoint(CellularMatrix& source,
                                                     NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                     Grid<BufferLinkPointCoord>& rootMergingGraphGpu,
                                                     class OperateAdaptor& oa,
                                                     float *dev_finishMst)
{
    //! wb.Q set register size larger
    cudaFuncSetCacheConfig(K_MST_markSuperVertice_Disjoint<NeuralNetLinks, BufferDimension, Point, OperateAdaptor>, cudaFuncCachePreferL1);




    KER_CALL_THREAD_BLOCK_1D(b, t,
                             BLOCKSIZE, 1,
                             nn_source.adaptiveMap.width,
                             nn_source.adaptiveMap.height);

        K_MST_markSuperVertice_Disjoint _KER_CALL_(b, t)(nn_source, rootMergingGraphGpu, oa, dev_finishMst);


}


//! 031216 wb.Q: add to refresh root id on disjoint set
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_CompactGraph2_finalDisjoint(NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                 Grid<BufferLinkPointCoord>& rootMergingGraph,
                                                 class OperateAdaptor& oa) {


    // wb.q set register size larger
    cudaFuncSetCacheConfig(K_MST_CompactGraph2_finalDisjoint<NeuralNetLinks, BufferDimension, Point, OperateAdaptor>, cudaFuncCachePreferL1);



    KER_CALL_THREAD_BLOCK_1D(b, t,
                             BLOCKSIZE, 1,
                             nn_source.adaptiveMap.width,
                             nn_source.adaptiveMap.height);


    K_MST_CompactGraph2_finalDisjoint _KER_CALL_(b, t)(nn_source, rootMergingGraph, oa);


    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("compact2 Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("compact2 Async kernel error: %s\n", cudaGetErrorString(errAsync));
}


//! 191016 QWB: add double link to the winner outgoing node for mst
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
GLOBAL inline void K_addDoubleLinkToWinnerNode(CellularMatrix& source,
                                               NeuralNetLinks<BufferDimension, Point>& nn_source,
                                               class OperateAdaptor& oa,
                                               Grid<BufferDimension>& netWorkLinkCopy) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);

    K_MST_addDoubleLinkToWinnerNode _KER_CALL_(b, t)(source, nn_source, oa, netWorkLinkCopy);
}



//! 191016 QWB: add double link to the winner outgoing node for mst
template <class CellularMatrix,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class OperateAdaptor
          >
inline void K_addDoubleLinkToWinnerNode_cpu(CellularMatrix& source,
                                            NeuralNetLinks<BufferDimension, Point>& nn_source,
                                            class OperateAdaptor& oa,
                                            Grid<BufferDimension>& netWorkLinkCopy) {

    K_MST_addDoubleLinkToWinnerNode_cpu(source, nn_source, oa, netWorkLinkCopy);
}






////! 131016 QWB: add to use one thread one node for get cell coordinate, search neighbor and operate
//template <class CellularMatrix,
//          class CellularMatrix2,
//          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
//          class GetAdaptor,
//          class SearchAdaptor
//          >
//GLOBAL inline void K_linkNetGetSearch(CellularMatrix& source,
//                                      CellularMatrix2& cible,
//                                      NeuralNetLinks<BufferDimension, Point>& nn_source,
//                                      NeuralNetLinks<BufferDimension, Point>& nn_cible,
//                                      class GetAdaptor& ga,
//                                      class SearchAdaptor& sa,
//                                      int radiusSearchCells) {

//    KER_CALL_THREAD_BLOCK(b, t,
//                          4, 4,
//                          nn_source.adaptiveMap.width,
//                          nn_source.adaptiveMap.height);
//    K_MST_linkNetGetSearch _KER_CALL_(b, t) (
//                source, cible, nn_source, nn_cible, ga, sa, radiusSearchCells);
//}


//! 131016 QWB: add to use one thread one node for get cell coordinate, search neighbor and operate
template <class CellularMatrix,
          class CellularMatrix2,
          template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
GLOBAL inline void K_linkNetGetSearchOperate(CellularMatrix& source,
                                             CellularMatrix2& cible,
                                             NeuralNetLinks<BufferDimension, Point>& nn_source,
                                             NeuralNetLinks<BufferDimension, Point>& nn_cible,
                                             class GetAdaptor& ga,
                                             class SearchAdaptor& sa,
                                             class OperateAdaptor& oa) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn_source.adaptiveMap.width,
                          nn_source.adaptiveMap.height);
    K_MST_linkNetGetSearchOperate _KER_CALL_(b, t) (
                source, cible, nn_source, nn_cible, ga, sa, oa);
}

//! WB.Q 1216 add to generate random searcher and searched on GPU side
KERNEL void K_MST_generateUniformSearchedSearched(curandState *state,
                                                  Grid<bool > gRand)
{
    KER_SCHED(gRand.width, gRand.height)

            if (_x < gRand.width && _y < gRand.height) {

        int cid = _x ;
        curandState localState = state[cid];
        float random;

#ifdef CUDA_CODE
        random = curand_uniform(&localState);
#endif
        random = (random * 1000 );

        //        for(int i = 0; i < MAX_RAND_BUFFER_SIZE; i++) {
        //            /* It return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded */
        //            random = curand_uniform(&localState);
        //            random = (random * 1000 );
        //        }
        printf("random Gpu %f \n", random );
        if(random > 0.3)
            gRand[_y][_x] = 1;

    }
    END_KER_SCHED
}
//! WB.Q 1216 add to generate random searcher and searched on GPU side
GLOBAL void K_generateUniformSearchedSearched(curandState *state,
                                              Grid<bool>& gRand){
    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          gRand.width,
                          gRand.height);
    K_MST_generateUniformSearchedSearched  _KER_CALL_(b, t) (state, gRand);

}

//! 140817 QWB: add to find min2 improvement, work correctly with step division
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_findMin2Improve(NeuralNetLinks<BufferDimension, Point>& nn_source) {

    // wb.q set register size larger
    cudaFuncSetCacheConfig(K_MST_findMin2_improve<NeuralNetLinks, BufferDimension, Point>, cudaFuncCachePreferL1);



    KER_CALL_THREAD_BLOCK_1D(b, t,
                             BLOCKSIZE, 1,
                             nn_source.adaptiveMap.width,
                             nn_source.adaptiveMap.height);

    K_MST_findMin2_improve _KER_CALL_(b, t)(nn_source);


    //! wb.Q check cuda errors
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("findmin2 Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("findmin2 Async kernel error: %s\n", cudaGetErrorString(errAsync));
}





//!######################################################################################





//!######################################################################################


/*!
 * \brief WB.Q Class Mst.
 * It is a mixte CPU/GPU class, then with
 * default copy constructor for allowing
 * parameter passing to GPU Kernel.
 **/
template <class CellularMatrixR,
          class CellularMatrixD,
          class NetLinkPointCoord,
          class CellR,
          class CellD,
          class NIter,
          class NIterDual,
          class ViewG,
          class BufferDimension
          >
class MstBoruvkaOperator
{
    //protected:
public:
    //! Link neural network matched
    NetLinkPointCoord mdLinks;
    //! Cellular matrix with neural net that is matched
    CellularMatrixD cmd;
    //! ViewGrid
    ViewG vgd;


public:
    DEVICE_HOST explicit MstBoruvkaOperator() {}

    //! \brief Constructeur par defaut.
    DEVICE_HOST explicit MstBoruvkaOperator(
            NetLinkPointCoord& nnLd,
            CellularMatrixD& cd,
            ViewG& vg
            ) :
        mdLinks(nnLd),
        cmd(cd),
        vgd(vg) {}

    /*!
     * \brief initialize
     */
    GLOBAL void initialize(
            NetLinkPointCoord& nnLd,
            CellularMatrixD& cd,
            ViewG& vg
            ) {
        mdLinks = nnLd;
        cmd = cd;
        vgd = vg;
    }

    GLOBAL void gpuResetValue(NetLinkPointCoord& nnLGpu){

        nnLGpu.densityMap.gpuResetValue(999999);// on cpu, infinity turn out to be 1
        nnLGpu.grayValueMap.gpuResetValue(999999);
        nnLGpu.activeMap.gpuResetValue(1);// QWB 241016 only for mst, specially initialize activeMap = 1, to mark the thread running shortest outgoing
        nnLGpu.fixedMap.gpuResetValue(1);// QWB 241016, only for mst, specially initialize fixedmap = 1 to mark initial winner nodes
        nnLGpu.minRadiusMap.gpuResetValue(0);
        nnLGpu.sizeOfComponentMap.gpuResetValue(1);
        PointCoord pInitial(-1, -1); // necessary to initialized to -1,-1
        nnLGpu.correspondenceMap.gpuResetValue(pInitial);
    }

    GLOBAL void cpuResetValue(NetLinkPointCoord& nnLCpu){

        nnLCpu.densityMap.resetValue(999999);
        nnLCpu.grayValueMap.resetValue(999999);
        nnLCpu.activeMap.resetValue(1); // QWB 241016 only for mst, specially initialization
        nnLCpu.fixedMap.resetValue(1);// QWB 241016, only for mst, specially initialize fixedmap = 1 to mark initial winner nodes
        nnLCpu.minRadiusMap.resetValue(0);
        nnLCpu.sizeOfComponentMap.resetValue(1);
        PointCoord pInitial(-1, -1); // necessary to be initialized with -1,-1
        nnLCpu.correspondenceMap.resetValue(pInitial);
    }


//    /*!
//     * \brief
//             QWB 22/11/16 : prototype version: build EMST from by combing Elias' nearest neighbor search with Borůvka's algorithm.
//     *
//     */
//    GLOBAL void buildBoruvkaEMST_Elias_version1(int radiusSearchCells){

//        size_t _w = mdLinks.adaptiveMap.width;
//        size_t _h = mdLinks.adaptiveMap.height;

//        this->gpuResetValue(mdLinks); // fixedMap = 1
//        NetLinkPointCoord cityCopy;
//        cityCopy.resize(_w, _h);
//        this->cpuResetValue(cityCopy); // activeMap = 1

//        int iteration = 0;// maximum iterations


//        bool finishMst = 0;
//#ifdef CUDA_CODE
//        bool *dev_finishMst;
//        cudaMalloc((void**)&dev_finishMst, sizeof(bool));
//        cudaMemcpy(dev_finishMst, &finishMst, sizeof(bool), cudaMemcpyHostToDevice);
//#else
//        bool *dev_finishMst = &finishMst;
//#endif
//        while(iteration < 10)
//        {


//            mdLinks.sizeOfComponentMap.gpuResetValue(0);
//            operateMarkMstSuperVertices_final<CellR> oaMarkSuperVertice;
//            operateMarkMstSuperVertices_finalPoint<CellR> oaMarkSuperVerticePoint;

//            K_markSuperVertices_final(cmd, mdLinks, oaMarkSuperVerticePoint, dev_finishMst);

//#if  CUDA_CODE
//            cudaMemcpy(&finishMst, dev_finishMst, sizeof(bool), cudaMemcpyDeviceToHost);
//#else
//            finishMst = *dev_finishMst;
//#endif
//            cout << "finishMst = " << finishMst << endl;



//            operateCompactGraph2_final<CellR> oaCompactGraph2;
//            K_CompactGraph2_final(mdLinks, oaCompactGraph2);



//            cityCopy.grayValueMap.gpuCopyDeviceToHost(mdLinks.grayValueMap);
//            int finish = testGridNum<int>(cityCopy.grayValueMap);
//            if (!finish)
//                break;



//            mdLinks.densityMap.gpuResetValue(999999); // WB.Q necessary, refresh distance at each iteration
//            PointCoord pInitial(-1, -1);
//            mdLinks.correspondenceMap.gpuResetValue(pInitial);

//            SearchFindCellAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > gsa; // qiao note: mr is base level
//            SearchSpiralAdaptorLinksWithRadius<CellularMatrixR, SpiraSearchEMST<CellR, NIterDual> > searchNodeWithDiffId;

//            K_linkNetGetSearch(cmd, cmd,
//                               mdLinks, mdLinks,
//                               gsa, searchNodeWithDiffId,
//                               radiusSearchCells);



//            mdLinks.fixedMap.gpuResetValue(0);// mark the winner node for current component
//            mdLinks.activeMap.gpuResetValue(0);

//            searchMstShortestOutgoingEdge_final<CellR> searchNeighborComponent;
//            // searchMstShortestOutgoingEdge<CellR> searchNeighborComponent; // wrong gpu performing

//            Grid<BufferLinkPointCoord> netWorkLinkTempGpu;
//            netWorkLinkTempGpu.gpuResize(_w, _h); // no use

//            OperateConnectMstSuperVertices<CellR, CM_DistanceEuclidean> connectNeighComponentsGpuNoUse; //WB.Q 11/16 no use

//            K_operateShortestOutgingEdge(cmd, mdLinks,
//                                         searchNeighborComponent,
//                                         connectNeighComponentsGpuNoUse,
//                                         netWorkLinkTempGpu);


//            OperateConnectMstSuperVertices_final<CellR, CM_DistanceEuclidean> connectNeighComponentsGpu; //WB.Q 11/16 no use
//            K_operateConnectGraph(mdLinks, connectNeighComponentsGpu);



//            OperateAddDoubleLinkToWinnerNode_final<CellR, CM_DistanceEuclidean> connectNeighComponents2Gpu;
//            K_operateConnectGraph(mdLinks, connectNeighComponents2Gpu);



//        }// while mst

//#ifdef CUDA_CODE
//        cudaFree(dev_finishMst);
//#endif

//        cout << "***End paralle mst after " << iteration << " th iteration. " << endl;

//    }// end buildBoruvkaMST prototype version

//    /*!
//             * \brief
//                     Wenbao Qiao 19/11/17 : Build Euclidean MST from by combing Elias' nearest neighbor search with Borůvka's algorithm.

//             *
//             */
//    GLOBAL void buildBoruvkaEMST_Elias_version2(int radiusSearchCells, int& iteration,
//                                                      float& gpuTimingKernels, float& mstTotalTime){

//        size_t _w = mdLinks.adaptiveMap.width;
//        size_t _h = mdLinks.adaptiveMap.height;

//        iteration = 0;
//        gpuTimingKernels = 0;

//        this->gpuResetValue(mdLinks);
//        NetLinkPointCoord cityCopy;
//        cityCopy.resize(_w, _h);
//        this->cpuResetValue(cityCopy);


//        bool finishMst = 0;// WB.Q necessary
//        bool *dev_finishMst = &finishMst;



//        OperateInitialComponentID operatorInitalIDArray;
//        templateOperationOnArray(mdLinks, operatorInitalIDArray);
//        cityCopy.grayValueMap.gpuCopyDeviceToHost(mdLinks.grayValueMap);// cy to cpu side, for second iteration

//        while(iteration < 20)
//        {


//            OperateCompactDisjointSetCpu cpuCompact;


//            __int64 CounterStart = 0;
//            double pcFreq = 0.0;
//            StartCounter(pcFreq, CounterStart);

//            for(int i = 0; i < cityCopy.adaptiveMap.width; i++)
//            {
//                PointCoord ps(i, 0);
//                cpuCompact.operate(cityCopy, ps);
//            }
//            // end time cpu
//            double timeCpuCompact = GetCounter(pcFreq, CounterStart);
//            cout << "cpu time compact mst: " << timeCpuCompact << endl;
//            cityCopy.grayValueMap.gpuCopyHostToDevice(mdLinks.grayValueMap);


//            finishMst = *dev_finishMst;

//            if(finishMst){
//                cout << "Finish criterion satisfied : end algorithm. " << endl;
//                break;
//            }
//            // for [compact graph *]
//            int finish = testGridNum<int>(cityCopy.grayValueMap);
//            if(finish == 0)
//                break;


//            iteration ++;
//            cout << " begin " << iteration << " 's iteration ." << endl;



//            SearchFindCellAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > gsa; // qiao note: mr is base level
//            SearchSpiralAdaptorLinksWithRadius<CellularMatrixR, SpiraSearchEMST<CellR, NIterDual> > searchNodeWithDiffId;


//            mdLinks.densityMap.gpuResetValue(infinity);
//            PointCoord pInitial(-1, -1);
//            mdLinks.correspondenceMap.gpuResetValue(pInitial);


//            CounterStart = 0;
//            pcFreq = 0.0;
//            StartCounter(pcFreq, CounterStart);

//            K_linkNetGetSearch(cmd, cmd,
//                               mdLinks, mdLinks,
//                               gsa, searchNodeWithDiffId,
//                               radiusSearchCells);


//            double cpuTimeFindMin1 = GetCounter(pcFreq, CounterStart);
//            cout << "cpu time find min " << cpuTimeFindMin1 << endl;



//            cityCopy.densityMap.gpuCopyDeviceToHost(mdLinks.densityMap);
//            cityCopy.correspondenceMap.gpuCopyDeviceToHost(mdLinks.correspondenceMap);
//            cityCopy.minRadiusMap.gpuCopyDeviceToHost(mdLinks.minRadiusMap);


//            cityCopy.fixedMap.resetValue(0);
//            cityCopy.activeMap.resetValue(0);

//            mdLinks.fixedMap.gpuResetValue(0);

//            //!timing runing time on CPU
//            CounterStart = 0;
//            pcFreq = 0.0;
//            StartCounter(pcFreq, CounterStart);

//            findMin2_cpu(cityCopy);

//            connectGraph_cpuDisjoint(cityCopy);

//            // end time cpu
//            double timeCpuFindMin2Connect = GetCounter(pcFreq, CounterStart);
//            cout << "cpu time findMin2 Connect mst: " << timeCpuFindMin2Connect << endl;
//            cityCopy.grayValueMap.gpuCopyHostToDevice(mdLinks.grayValueMap);


//            mstTotalTime += timeCpuCompact + cpuTimeFindMin1 + timeCpuFindMin2Connect;

//        }// while mst

//        cityCopy.networkLinks.gpuCopyHostToDevice(mdLinks.networkLinks);


//    }// end buildBoruvkaMST cpu pure version



//    /*!
//            * \brief
//            * Wenbao Qiao 19/11/17 :  Build EMST from by combing Elias' nearest neighbor search with Borůvka's algorithm.
//            * Optimized version with disjoint set data structure.
//            *
//            */
//    GLOBAL void buildBoruvkaEMST_Elias_version3(int radiusSearchCells, int& iteration,
//                                                           float& gpuTimingKernels, float& mstTotalTime){

//        size_t _w = mdLinks.adaptiveMap.width;
//        size_t _h = mdLinks.adaptiveMap.height;

//        iteration = 0;
//        gpuTimingKernels = 0;

//        this->gpuResetValue(mdLinks); // fixedMap = 1
//        NetLinkPointCoord cityCopy;
//        cityCopy.resize(_w, _h);
//        this->cpuResetValue(cityCopy); // activeMap = 1


//        bool finishMst = 0;// WB.Q necessary
//        bool *dev_finishMst = &finishMst;



//        OperateInitialComponentID operatorInitalIDArray;
//        templateOperationOnArray(mdLinks, operatorInitalIDArray);
//        cityCopy.grayValueMap.gpuCopyDeviceToHost(mdLinks.grayValueMap);// cy to cpu side, for second iteration

//        while(iteration < 20)
//        {


//            OperateCompactDisjointSetCpu cpuCompact;


//            __int64 CounterStart = 0;
//            double pcFreq = 0.0;
//            StartCounter(pcFreq, CounterStart);

//            for(int i = 0; i < cityCopy.adaptiveMap.width; i++)
//            {
//                PointCoord ps(i, 0);
//                cpuCompact.operate(cityCopy, ps);
//            }

//            double timeCpuCompact = GetCounter(pcFreq, CounterStart);
//            cityCopy.grayValueMap.gpuCopyHostToDevice(mdLinks.grayValueMap);


//            finishMst = *dev_finishMst;

//            if(finishMst){
//                cout << "Finish criterion satisfied : end algorithm. " << endl;
//                break;
//            }

//            int finish = testGridNum<int>(cityCopy.grayValueMap);
//            if(finish == 0)
//                break;



//            iteration ++;
//            cout << " begin " << iteration << " 's iteration ." << endl;



//            SearchFindCellAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > gsa; // qiao note: mr is base level
//            SearchSpiralAdaptorLinksWithRadius<CellularMatrixR, SpiraSearchEMST<CellR, NIterDual> > searchNodeWithDiffId;
//            PointCoord pInitial(-1, -1);
//            // time for findMin1
//            float elapsedTimeFindMin1 = 0;
//            cudaEvent_t start, stop;
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord(start, 0);

//            mdLinks.densityMap.gpuResetValue(infinity); // WB.Q necessary, refresh distance at each iteration
//            mdLinks.correspondenceMap.gpuResetValue(pInitial);

//            K_linkNetGetSearch(cmd, cmd,
//                               mdLinks, mdLinks,
//                               gsa, searchNodeWithDiffId,
//                               radiusSearchCells);

//            cudaEventRecord(stop, 0);
//            cudaEventSynchronize(stop);
//            cudaEventElapsedTime(&elapsedTimeFindMin1, start, stop);
//            cudaEventDestroy(start);
//            cudaEventDestroy(stop);
//            cout << "cuda time find min1 " << elapsedTimeFindMin1 << endl;



//            float elapsedTimeMemcpyD2H = 0;
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord(start, 0);


//            cityCopy.densityMap.gpuCopyDeviceToHost(mdLinks.densityMap);
//            cityCopy.correspondenceMap.gpuCopyDeviceToHost(mdLinks.correspondenceMap);
//            cityCopy.minRadiusMap.gpuCopyDeviceToHost(mdLinks.minRadiusMap);

//            cudaEventRecord(stop, 0);
//            cudaEventSynchronize(stop);
//            cudaEventElapsedTime(&elapsedTimeMemcpyD2H, start, stop);
//            cudaEventDestroy(start);
//            cudaEventDestroy(stop);
//            cout << "cuda time elapsedTimeMemcpyD2H " << elapsedTimeMemcpyD2H << endl;
//            cityCopy.fixedMap.resetValue(0);// mark the winner node for current component
//            cityCopy.activeMap.resetValue(0);


//            mdLinks.fixedMap.gpuResetValue(0);// mark winner node of current component


//            CounterStart = 0;
//            pcFreq = 0.0;
//            StartCounter(pcFreq, CounterStart);



//            findMin2_cpu(cityCopy);



//            connectGraph_cpuDisjoint(cityCopy);

//            // end time cpu
//            double timeCpuFindMin2Connect = GetCounter(pcFreq, CounterStart);
//            cout << "cpu time findMin2 Connect mst: " << timeCpuFindMin2Connect << endl;
//            cityCopy.grayValueMap.gpuCopyHostToDevice(mdLinks.grayValueMap);// for gpu refresh disjoint set



//            gpuTimingKernels += elapsedTimeFindMin1 + elapsedTimeMemcpyD2H;
//            mstTotalTime += timeCpuCompact + elapsedTimeFindMin1 + elapsedTimeMemcpyD2H + timeCpuFindMin2Connect;

//        }// while mst

//        cityCopy.networkLinks.gpuCopyHostToDevice(mdLinks.networkLinks);
//        errorCheckCudaThreadSynchronize();

//    }


//    /*!
//             * \brief
//                Wenbao Qiao 19/10/17 WenbaoQiao: Build Euclidean MST from by combing Elias' nearest neighbor search with Borůvka's algorithm.

//             *
//             */
//    GLOBAL void buildBoruvkaEMST_Elias_version4(int radiusSearchCells, int& iteration,
//                                                        float& gpuTimeTotal, float& mstTotalTime
//                                                        ){


//        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//        gpuTimeTotal = 0;
//        mstTotalTime = 0;
//        iteration  = 0;

//        size_t _w = mdLinks.adaptiveMap.width;
//        size_t _h = mdLinks.adaptiveMap.height;

//        this->gpuResetValue(mdLinks); // fixedMap = 1
//        NetLinkPointCoord cityCopy;
//        cityCopy.resize(_w, _h);
//        this->cpuResetValue(cityCopy); // activeMap = 1


//        float * h_finishMst = (float*)malloc(sizeof(float));
//        float *dev_finishMst;
//        cudaMalloc(&dev_finishMst, sizeof(float));
//        bool finishMst = 0;




//        OperateInitialComponentID operatorInitalIDArray;
//        templateOperationOnArray(mdLinks, operatorInitalIDArray);
//        cityCopy.grayValueMap.gpuCopyDeviceToHost(mdLinks.grayValueMap);

//        while(iteration < 20)
//        {
//            //! wb.Q 191017 refresh id from each node on gpu side for findMin2
//            mdLinks.activeMap.gpuResetValue(0);
//            OperateCompactDisjointSet operatorCompactDisjointSet;
//            // time for compact
//            float elapsedTimeCompactDisjoint = 0;
//            cudaEvent_t startCompact, stopCompact;
//            cudaEventCreate(&startCompact);
//            cudaEventCreate(&stopCompact);
//            cudaEventRecord(startCompact, 0);

//            templateOperationOnArray(mdLinks, operatorCompactDisjointSet);

//            cudaEventRecord(stopCompact, 0);
//            cudaEventSynchronize(stopCompact);
//            cudaEventElapsedTime(&elapsedTimeCompactDisjoint, startCompact, stopCompact);
//            cudaEventDestroy(startCompact);
//            cudaEventDestroy(stopCompact);
//            cout << "cuda time compact " << elapsedTimeCompactDisjoint << endl;
//            cityCopy.grayValueMap.gpuCopyDeviceToHost(mdLinks.grayValueMap);



//            if(finishMst){
//                cout << "Finish criterion satisfied : end algorithm. " << endl;
//                break;
//            }
//            // for [compact graph *]
//            int finish = testGridNum<int>(cityCopy.grayValueMap);
//            if(finish == 0)
//                break;

//            iteration ++;
//            cout << " begin " << iteration << " 's iteration ." << endl;


//            float elapsedTimeFindMinConnect = 0;
//            cudaEvent_t startFindMinConnect, stopFindMinConnect;
//            cudaEventCreate(&startFindMinConnect);
//            cudaEventCreate(&stopFindMinConnect);
//            cudaEventRecord(startFindMinConnect, 0);



//            PointCoord pInitial(-1, -1);
//            mdLinks.densityMap.gpuResetValue(infinity);
//            mdLinks.correspondenceMap.gpuResetValue(pInitial);

//            SearchFindCellAdaptor<CellularMatrixR, SpiralSearchCMIterator<CellR, NIterDual> > gsa;
//            SearchSpiralAdaptorLinksWithRadius<CellularMatrixR, SpiraSearchEMST<CellR, NIterDual> > searchNodeWithDiffId;

//            float elapsedTimeFindMin1 = 0;
//            cudaEvent_t start, stop;
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord(start, 0);

//            K_linkNetGetSearch(cmd, cmd,
//                               mdLinks, mdLinks,
//                               gsa, searchNodeWithDiffId,
//                               radiusSearchCells);

//            cudaEventRecord(stop, 0);
//            cudaEventSynchronize(stop);
//            cudaEventElapsedTime(&elapsedTimeFindMin1, start, stop);
//            cudaEventDestroy(start);
//            cudaEventDestroy(stop);
//            cout << "cuda time find min1 " << elapsedTimeFindMin1 << endl;


//            mdLinks.fixedMap.gpuResetValue(0);


//            float elapsedTimeFindMin2 = 0;
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord(start, 0);

//            K_findMin2Improve(mdLinks);

//            cudaEventRecord(stop, 0);
//            cudaEventSynchronize(stop);
//            cudaEventElapsedTime(&elapsedTimeFindMin2, start, stop);
//            cudaEventDestroy(start);
//            cudaEventDestroy(stop);
//            cout << "cuda time find min2 " << elapsedTimeFindMin2 << endl;


//            errorCheckCudaThreadSynchronize();
//            gpuErrchk( cudaPeekAtLastError() );
//            gpuErrchk( cudaDeviceSynchronize() );
//            cout << "end findMin2" << endl;

//            float elapsedTimeConnect = 0;
//            double timeCpuCompact = 0;

//#if CPUMERGINGGRAPH
//            //! [Connect Graph 1] CPU side
//            cityCopy.fixedMap.gpuCopyDeviceToHost(mdLinks.fixedMap);
//            cityCopy.correspondenceMap.gpuCopyDeviceToHost(mdLinks.correspondenceMap);
//            cityCopy.networkLinks.gpuCopyDeviceToHost(mdLinks.networkLinks);

//            //!timing runing time on CPU
//            __int64 CounterStart = 0;
//            double pcFreq = 0.0;
//            StartCounter(pcFreq, CounterStart);

//            connectGraph_cpuDisjoint(cityCopy);

//            timeCpuCompact = GetCounter(pcFreq, CounterStart);
//            cout << "cpu time connect graph: " << timeCpuCompact << endl;
//            //for stoping the algorithm
//            cityCopy.grayValueMap.gpuCopyHostToDevice(mdLinks.grayValueMap);// because connect on cpu side but compact on gpu side
//            cityCopy.networkLinks.gpuCopyHostToDevice(mdLinks.networkLinks);

//#else


//            Grid<BufferLinkPointCoord> rootMergingGraphCpu;
//            rootMergingGraphCpu.resize(_w, _h);
//            Grid<BufferLinkPointCoord> rootMergingGraphGpu;
//            rootMergingGraphGpu.gpuResize(_w, _h);
//            rootMergingGraphCpu.gpuCopyHostToDevice(rootMergingGraphGpu);
//            mdLinks.activeMap.gpuResetValue(0);// mark final winner root


//            OperateAddDoubleLinkToWinnerNode_disjoint<CellR, CM_DistanceEuclidean> connectNeighComponents2Gpu;

//            operateMarkMstSuperVertices_finalPointDisjoint<CellR> oaMarkSuperVerticePoint; // corectly , begin with each winner node


//            operateCompactGraph2_final2Disjoint<CellR> oaCompactGraph2; //


//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord(start, 0);

//            K_operateConnectGraphWithRootMergingG(mdLinks, connectNeighComponents2Gpu, rootMergingGraphGpu);
//            errorCheckCudaThreadSynchronize();

//            K_markSuperVertices_Disjoint(cmd, mdLinks, rootMergingGraphGpu, oaMarkSuperVerticePoint, dev_finishMst);
//            errorCheckCudaThreadSynchronize();

//            K_CompactGraph2_finalDisjoint(mdLinks, rootMergingGraphGpu, oaCompactGraph2);

//            cudaEventRecord(stop, 0);
//            cudaEventSynchronize(stop);
//            cudaEventElapsedTime(&elapsedTimeConnect, start, stop);
//            cudaEventDestroy(start);
//            cudaEventDestroy(stop);
//            cout << "cuda time connect " << elapsedTimeConnect << endl;


//            errorCheckCudaThreadSynchronize();
//            gpuErrchk( cudaPeekAtLastError() );
//            gpuErrchk( cudaDeviceSynchronize() );

//            rootMergingGraphGpu.gpuFreeMem();
//            rootMergingGraphCpu.freeMem();

//#endif


//            cudaEventRecord(stopFindMinConnect, 0);
//            cudaEventSynchronize(stopFindMinConnect);
//            cudaEventElapsedTime(&elapsedTimeFindMinConnect, startFindMinConnect, stopFindMinConnect);
//            cudaEventDestroy(startFindMinConnect);
//            cudaEventDestroy(stopFindMinConnect);


//            gpuTimeTotal += elapsedTimeCompactDisjoint + elapsedTimeFindMin1 + elapsedTimeFindMin2 + elapsedTimeConnect;
//            mstTotalTime += timeCpuCompact + elapsedTimeCompactDisjoint + elapsedTimeFindMin1 +
//                    elapsedTimeFindMin2 + elapsedTimeConnect;

//        }// while mst

//        //free mem
//        cudaFree(dev_finishMst);
//        free(h_finishMst);

//    }// end buildBoruvkaMST parallel all step




//    /*!
//             * \brief wenbao Qiao: template operation on Array
//             *
//             */
//    template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
//              class OperateAdaptor
//              >
//    GLOBAL inline void templateOperationOnArray(NeuralNetLinks<BufferDimension, Point>& nn_source,
//                                                class OperateAdaptor& oa) {

//        KER_CALL_THREAD_BLOCK_1D(b, t,
//                                 BLOCKDIMX, BLOCKDIMX,
//                                 nn_source.grayValueMap.getWidth(),
//                                 nn_source.grayValueMap.getHeight());
//        K_Mst_OperatorOnComponentID _KER_CALL_(b, t) (
//                    nn_source, oa);
//    }


    //! 190916 QWB: add to use one thread one node
    template <class CellularMatrix,
              class CellularMatrix2,
              template<typename, typename> class NeuralNetLinks, class Point,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_projectorNodeSpiralSearch(CellularMatrix& source,
                                                   CellularMatrix2& cible,
                                                   NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                   NeuralNetLinks<BufferDimension, Point>& nn_cible,
                                                   class GetAdaptor& ga,
                                                   class SearchAdaptor& sa,
                                                   class OperateAdaptor& oa,
                                                   int radiusSearchCells) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              nn_source.adaptiveMap.width,
                              nn_source.adaptiveMap.height);
        K_MST_projector_links_nodeSpiralSearch _KER_CALL_(b, t) (
                    source, cible, nn_source, nn_cible, ga, sa, oa, radiusSearchCells);
    }


    //! 190916 QWB: overload cpu version
    template <class CellularMatrix,
              class CellularMatrix2,
              template<typename, typename> class NeuralNetLinks, class Point,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    inline void K_projectorNodeSpiralSearch_cpu(CellularMatrix& source,
                                                CellularMatrix2& cible,
                                                NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                NeuralNetLinks<BufferDimension, Point>& nn_cible,
                                                class GetAdaptor& ga,
                                                class SearchAdaptor& sa,
                                                class OperateAdaptor& oa,
                                                int radiusSearchCells) {

        K_MST_projector_links_nodeSpiralSearch_cpu(source, cible, nn_source, nn_cible, ga, sa, oa, radiusSearchCells);
    }



    /*!
     * \brief 200916 qiao add projector_links_cpu to project nodes into cm on cpu side
     *
     */
    template <class CellularMatrix,
              class CellularMatrix2,
              class NN,
              template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    void mst_refreshCell_cpu(CellularMatrix cm_source,
                             CellularMatrix2 cm_cible,
                             NN nn_source,
                             NeuralNetLinks<BufferDimension, Point> nn_cible,
                             GetAdaptor getAdaptor,
                             SearchAdaptor searchAdaptor,
                             OperateAdaptor operateAdaptor)
    {

        for (int _y = 0; _y < (cm_source.getHeight_cpu()); ++_y) {
            for (int _x = 0; _x < (cm_source.getWidth_cpu()); ++_x) {


                getAdaptor.init(cm_source[_y][_x]);

                do {
                    PointCoord PC(_x, _y);

                    PointCoord ps;
                    bool extracted = false;
                    extracted = getAdaptor.get(cm_source[_y][_x],
                                               nn_source,
                                               ps);

                    if (extracted) {
                        // Spiral search
                        PointCoord minP;
                        bool found = false;

                        found = searchAdaptor.search(cm_cible,
                                                     nn_source,
                                                     nn_cible,
                                                     PC,
                                                     ps,
                                                     minP);


                    }
                } while (getAdaptor.next());
            }
        }

    }//projector_links


    //! 160916 qiao add cpu version to insert values into cm
    template <class CellularMatrix,
              class CellularMatrix2,
              class NN,
              template<typename, typename> class NeuralNetLinks, class Point,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void refreshCell_cpu(CellularMatrix& source,
                                       CellularMatrix2& cible,
                                       NN& nn_source,
                                       NeuralNetLinks<BufferDimension, Point>& nn_cible,
                                       class GetAdaptor& ga,
                                       class SearchAdaptor& sa,
                                       class OperateAdaptor& oa) {
        mst_refreshCell_cpu (
                    source, cible, nn_source, nn_cible, ga, sa, oa);
    }



    //! 141016 QWB: add to find the shortest outgoing edge for each superVertices on CPU side
    template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
    void findMin2_cpu(NeuralNetLinks<BufferDimension, Point>& nn_source) {

        int _w = nn_source.adaptiveMap.width;
        int _h = nn_source.adaptiveMap.height;

        nn_source.fixedLinks.resetValue(0);

        for(int _y = 0; _y< _h; _y++)
            for(int _x = 0; _x < _w; _x++){

                PointCoord ps(_x, _y);

                PointCoord pInitial(-1, -1);
                PointCoord pInitial2D(-1, -1);
                if(nn_source.correspondenceMap[_y][_x] != pInitial &&
                        !nn_source.fixedLinks[_y][_x])
                {

                    superVertexData neiClosCompo;
                    vector<PointCoord> tabO;
                    tabO.clear();
                    vector<PointCoord> tabD;
                    tabD.clear();

                    tabO.push_back(ps);

                    while(tabO.size() > 0){

                        for(int i = 0; i < tabO.size(); i++){

                            PointCoord pCoord = tabO[i];

                            if(!nn_source.fixedLinks[pCoord[1]][pCoord[0]])
                            {
                                nn_source.fixedLinks[pCoord[1]][pCoord[0]] = 1;

                                findShortestLexicoEdgeBetweenComponents(nn_source, pCoord, neiClosCompo);

                                int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                                for(int pLink = 0; pLink < nLinks; pLink++){

                                    PointCoord pLinkOfNode;
                                    nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                                    if(pLinkOfNode != pInitial2D){

                                        PointCoord pCoordLink(-1, -1);
                                        pCoordLink[0] = (int)pLinkOfNode[0];
                                        pCoordLink[1] = (int)pLinkOfNode[1];

                                        tabD.push_back(pCoordLink);
                                    }

                                }// end num links of pCoord

                            }

                        }// end for tabO.size

                        tabO.swap(tabD);
                        tabD.clear();

                    }// end while tabO.size > 0

                    PointCoord couple1 = neiClosCompo.couple1;
                    nn_source.fixedMap[couple1[1]][couple1[0]] = 1;

                }// end if node has not correspondence

            }// end for every node

        nn_source.fixedLinks.resetValue(0);
    }// end find min2 cpu

    //! WB.Q add for EMST
    int minimum(int ps, int fameNode){

        if(ps < fameNode)
            return ps;
        else
            return fameNode;
    }

    //! WB.Q add for EMST
    int maximum(int ps, int fameNode){

        if(ps > fameNode)
            return ps;
        else
            return fameNode;
    }


    //! WB.Q add for EMST
    template <class NetLinkPointCoord>
    void findShortestLexicoEdgeBetweenComponents(NetLinkPointCoord& nn_source,
                                                 PointCoord& ps_,
                                                 superVertexData& neighborClosestComponent)
    {

        PointCoord fammeNode = nn_source.correspondenceMap[ps_[1]][ps_[0]];

        if(fammeNode[0] != -1){

            int fammeNodeID = nn_source.grayValueMap[fammeNode[1]][fammeNode[0]];
            double distance = nn_source.densityMap[ps_[1]][ps_[0]];

            int minimumId = minimum(ps_[0], fammeNode[0]);


            if(fammeNodeID == INFINITY)
                printf("attention ,  do not find neighbor node for ps_, %d, %d ", ps_[0], ps_[1]);

            if(distance < neighborClosestComponent.minCoupleDistance){
                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.couple1 = ps_;
                neighborClosestComponent.couple2 = fammeNode;
                neighborClosestComponent.id2 = fammeNodeID;
                neighborClosestComponent.id1 = minimumId;
            }

            else if(distance == neighborClosestComponent.minCoupleDistance){

                if(fammeNodeID < neighborClosestComponent.id2){

                    neighborClosestComponent.minCoupleDistance = distance;
                    neighborClosestComponent.couple1 = ps_;
                    neighborClosestComponent.couple2 = fammeNode;
                    neighborClosestComponent.id2 = fammeNodeID;
                    neighborClosestComponent.id1 = minimumId;

                }
                else if((fammeNodeID == neighborClosestComponent.id2)
                        && minimumId < neighborClosestComponent.id1 ){

                    neighborClosestComponent.minCoupleDistance = distance;
                    neighborClosestComponent.couple1 = ps_;
                    neighborClosestComponent.couple2 = fammeNode;
                    neighborClosestComponent.id2 = fammeNodeID;
                    neighborClosestComponent.id1 = minimumId;

                }

            }

        }

    }// end find min2 operation in component




    //! 191017 wb.Q add find root
    int findRoot(Grid<GLint> &idGpuMap, int x)
    {
        GLint r = x;
        while(r != idGpuMap[0][r])
        {
            r = idGpuMap[0][r]; // r is the root
        }

        return r;
    }


    //!101017 wb.Q add to connect disjoint sets
    template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
    void connectGraph_cpuDisjoint(NeuralNetLinks<BufferDimension, Point>& nn_source) {

        int _w = nn_source.adaptiveMap.width;
        int _h = nn_source.adaptiveMap.height;

        for(int _y = 0; _y < _h; _y++)
            for (int _x = 0; _x <_w; _x++){

                if(nn_source.fixedMap[_y][_x]){

                    PointCoord ps(_x, _y);

                    PointCoord couple2(-1, -1);
                    couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];
                    PointCoord pTemp2(-1, -1);
                    pTemp2 = couple2;
                    nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);

                    //! 191017 wb.Q joint two disjoint sets
                    int rootPs = findRoot(nn_source.grayValueMap, ps[0]);
                    int rootCouple2 = findRoot(nn_source.grayValueMap, couple2[0]);
                    if(rootPs > rootCouple2)
                    {
                        nn_source.grayValueMap[0][rootPs] = rootCouple2;
                    }
                    else if (rootPs < rootCouple2)
                    {
                        nn_source.grayValueMap[0][rootCouple2] = rootPs;
                    }


                    PointCoord pCouple2(-1, -1);
                    pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];

                    //! wenbao Qiao insert ps into couple2's link list under conditions
                    if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0){
                        PointCoord ps_(-1, -1);
                        ps_ = ps;
                        nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);

                    }
                    if(nn_source.fixedMap[0][couple2[0]] && pCouple2 != ps ){
                        PointCoord ps_(-1, -1);
                        ps_ = ps;
                        nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
                    }


                }// end if winner node

            }// end for every node

    }// end connected graph cpu side





};



}//namespace operators

#endif // SOM_OPERATOR_H

#ifndef EMST_FUNCTORS_H
#define EMST_FUNCTORS_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao, J.C. Creput
 * Creation date : June. 2018
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

#ifdef CUDA_CODE
#include "random_generator.h"
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"

#include "SpiralSearch.h"

#include "Objectives.h"
#include "basic_operations.h"
#include "geometry.h"
#include "CellularMatrix.h"

//! reference EMST components
#include "NeuralNetEMST.h"
#include "SpiralSearchNNL.h"
//#include "adaptator_EMST.h"
//#include "macros_cuda_EMST.h"

#define BLOCKSIZE 256

using namespace std;
using namespace components;

namespace operators
{

/**************** 01June2018,wb.Q transfers original functions from adaptor_EMST to below.************/
//! wb.Q 230916, overload to adatp all nn with type of networklinks and search neighbor cell within radius
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinksWithRadius {

    template <class NN1, class NN2, class IndexCM>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN1& scher,
                            NN2& sched,
                            IndexCM PC,
                            PointCoord ps,
                            PointCoord& minP,
                            int radiusSearchCells) {
        bool ret = false;

        SpiralSearchCMIteratorLinks sps_iter(
                    PC,
                    cm.vgd.getWidth() * cm.vgd.getHeight(),
                    0,
                    MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                    1 //d_step
                    );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP,
                              radiusSearchCells);
        return ret;
    }
};

#ifdef CUDA_CODE
//! QWB 031217 add double links to winner node, [connect graph 2]
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode_disjoint{

    DEVICE_HOST OperateAddDoubleLinkToWinnerNode_disjoint(){}


    DEVICE_HOST int findRoot(Grid<GLint> &idGpuMap, int x)
    {
        GLint r = x;
        while(r != idGpuMap[0][r])
        {
            r = idGpuMap[0][r]; // r is the root
        }

        return r;
    }

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps, Grid<BufferLinkPointCoord>& rootMergingGraphGpu){

        PointCoord couple2(-1);
        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

        PointCoord pCouple2(-1);
        pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];

        PointCoord pTemp2(-1);
        pTemp2 = couple2;
        nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);// wenbaoQiao, necessary to be here

        int rootPs = findRoot(nn_source.grayValueMap, ps[0]);
        int rootCouple2 = findRoot(nn_source.grayValueMap, couple2[0]);
        pTemp2[0] = rootCouple2;
        pTemp2[1] = 0;
        rootMergingGraphGpu[0][rootPs].insert(pTemp2); // wenbaoQiao, rootPs add link of rootCouple2

        //! QWB add double links to winner node, [connect graph 2]
        pTemp2[0] = rootPs;
        pTemp2[1] = 0;
        if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0) {
            PointCoord ps_(-1);
            ps_ = ps;
            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][0]), (float)ps_[0]);
            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][1]), (float)ps_[1]);
            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][0]), pTemp2[0]);
            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][1]), pTemp2[1]);

        }
        else if(nn_source.fixedMap[couple2[1]][couple2[0]]== 1 && pCouple2 != ps)
        {
            PointCoord ps_(-1);
            ps_ = ps;
            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][0]), (float)ps_[0]);
            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][1]), (float)ps_[1]);
            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][0]), pTemp2[0]);
            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][1]), pTemp2[1]);
        }

        if(nn_source.grayValueMap[ps[1]][ps[0]] > nn_source.grayValueMap[couple2[1]][couple2[0]])
        {
            nn_source.fixedMap[ps[1]][ps[0]] = 0;
        }

    } // operate

};
#else
//! wenbao Qiao 191017 add double links to winner node under conditions
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode_disjoint{

    DEVICE_HOST OperateAddDoubleLinkToWinnerNode_disjoint(){}


    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        PointCoord couple2(-1, -1);

        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

        PointCoord pCouple2(-1, -1);
        pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];


        PointCoord pTemp2(-1, -1);
        pTemp2 = couple2;

        nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);


        //! wb.Q if correspondence is not winner
        if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0){
            PointCoord ps_(-1, -1);
            ps_ = ps;
            nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
        }
        //! wb.Q if correspondence is winner but not couple to ps
        if(nn_source.fixedMap[0][couple2[0]] && pCouple2 != ps ){
            PointCoord ps_(-1, -1);
            ps_ = ps;
            nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
        }

    }// operate
};

#endif // gpu cpu version of add mst outgoing edge only once

//! QWB 031216 add operation to mark finnal winner root on merging root graph
template <class Cell>
struct operateMarkMstSuperVertices_finalPointDisjoint{

    DEVICE_HOST operateMarkMstSuperVertices_finalPointDisjoint(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, Grid<BufferLinkPointCoord>& rootMergingGraphGpu, PointCoord ps){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

        int idLeast = idOriginal;
        {
            PointCoord pInitial(-1);

            BufferLinkPointCoordVector nodeAlreadyTraversed;
            nodeAlreadyTraversed.init(MAX_VECTOR_SIZE_MERGRAPH);

            BufferLinkPointCoordVector tabO;
            tabO.init(MAX_VECTOR_SIZE_MERGRAPH);
            BufferLinkPointCoordVector tabD;
            tabD.init(MAX_VECTOR_SIZE_MERGRAPH);

            tabO.insert(ps);

            BufferLinkPointCoordVector* tabO_ = &tabO;
            BufferLinkPointCoordVector* tabD_ = &tabD;
            BufferLinkPointCoordVector* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

            while((*tabO_).numLinks > 0){

                for (int i = 0; i < (*tabO_).numLinks; i++){

                    PointCoord pCoord = (*tabO_).bCell[i];

                    if(pCoord != pInitial)
                    {

                        int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];
                        if(idLeast == infinity ){
                            nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                            idLeast = pCoord[0];
                        }
                        if(idPCoord < idLeast)
                        {
                            idLeast = idPCoord;
                        }

                        PointCoord pInitial2D(-1);

                        int nLinks = rootMergingGraphGpu[pCoord[1]][pCoord[0]].numLinks;
                        for (int pLink = 0; pLink < nLinks; pLink++){

                            PointCoord pLinkOfNode;
                            rootMergingGraphGpu[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                            if(pLinkOfNode != pInitial2D){
                                PointCoord pCoordLink(-1);
                                pCoordLink[0] = (int)pLinkOfNode[0];
                                pCoordLink[1] = (int)pLinkOfNode[1];

                                bool traversed = 0;
                                for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
                                    PointCoord pLinkTemp(-1);
                                    pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
                                    if (pCoordLink == pLinkTemp)
                                        traversed = 1;
                                }
                                if (!traversed){
                                    (*tabD_).insert(pCoordLink);

                                }
                            }
                        }
                    }
                }

                if((*nodeAlreadyTraversed_).numLinks > MAX_VECTOR_SIZE_MERGRAPH)
                    printf("size merging graph %d, %d, %d \n", (*nodeAlreadyTraversed_).numLinks, (*tabO_).numLinks, (*tabD_).numLinks);

                (*nodeAlreadyTraversed_).clearLinks();


                swapClass4(&nodeAlreadyTraversed_, &tabO_);
                swapClass4(&tabO_, &tabD_);
            }
        }

        //! WB.Q mark the representative city
        if(idLeast == idOriginal){
            nn_source.activeMap[0][idLeast] = 1;
        }

    } // operate
};

//! QWB 031216 refresh root id on root merging graph from activeMap
template <class Cell>
struct operateCompactGraph2_final2Disjoint{

    DEVICE_HOST operateCompactGraph2_final2Disjoint(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, Grid<BufferLinkPointCoord>& rootMergingGraph, PointCoord& ps){

        int idLeast = nn_source.grayValueMap[ps[1]][ps[0]];

        PointCoord pInitial(-1);

        BufferLinkPointCoordVector nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE_MERGRAPH);

        BufferLinkPointCoordVector tabO;
        tabO.init(MAX_VECTOR_SIZE_MERGRAPH);
        BufferLinkPointCoordVector tabD;
        tabD.init(MAX_VECTOR_SIZE_MERGRAPH);

        tabO.insert(ps);

        BufferLinkPointCoordVector* tabO_ = &tabO;
        BufferLinkPointCoordVector* tabD_ = &tabD;
        BufferLinkPointCoordVector* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];
                {

                    int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];

                    if(idPCoord > idLeast)
                    {

                        nn_source.grayValueMap[pCoord[1]][pCoord[0]] = idLeast;
                    }

                    int nLinks = rootMergingGraph[pCoord[1]][pCoord[0]].numLinks;
                    for (int pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode;
                        rootMergingGraph[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        if(pLinkOfNode != pInitial)
                        {
                            PointCoord pCoordLink(-1);
                            pCoordLink[0] = pLinkOfNode[0];
                            pCoordLink[1] = pLinkOfNode[1];

                            bool traversed = 0;
                            for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
                                PointCoord pLinkTemp(-1);
                                pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
                                if (pCoordLink == pLinkTemp)
                                    traversed = 1;
                            }
                            if (!traversed){
                                (*tabD_).insert(pCoordLink);
                            }
                        }
                    }
                }
            }
            if((*nodeAlreadyTraversed_).numLinks > MAX_VECTOR_SIZE_MERGRAPH)
                printf("size merging graph %d, %d, %d \n", (*nodeAlreadyTraversed_).numLinks, (*tabO_).numLinks, (*tabD_).numLinks);

            (*nodeAlreadyTraversed_).clearLinks();


            swapClass4(&nodeAlreadyTraversed_, &tabO_);
            swapClass4(&tabO_, &tabD_);

        }// while
    } // operate
};

//! WB.Q add to count time
//! copy source code from https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
void StartCounter(double& PCFreq, __int64& CounterStart)
{

    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart)/1000.0;// millisecond
    //    PCFreq = double(li.QuadPart); // s
    //    PCFreq = double(li.QuadPart)/1000000.0; // microsecond

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter(double PCFreq, __int64 CounterStart)
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-CounterStart)/PCFreq;
}

//!QWB add to test changement in grid
template<class type>
int testGridNum(Grid<type>& testGrid){

    typedef typename Grid<type>::index_type index_type;

    //GLint numTest = testGrid(index_type(0));
    GLint numTest = testGrid.findRoot(testGrid, testGrid.compute_offset(index_type(0)));
    int nComp = 0;
    for (int j = 0; j < testGrid.height; j++ )
        for (int i = 0; i < testGrid.width; i++)
        {
            index_type ps(i, j);
            //GLint r = testGrid(ps);
            GLint r = testGrid.findRoot(testGrid, testGrid.compute_offset(ps));
            if (numTest != r)
                nComp += 1;// testGrid[j][i];
        }
    return nComp;

}

}//namespace operators

#endif // EMST_FUNCTORS_H

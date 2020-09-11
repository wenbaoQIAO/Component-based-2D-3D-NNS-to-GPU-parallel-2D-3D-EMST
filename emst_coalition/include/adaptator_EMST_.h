#ifndef ADAPTATOR_EMST_H
#define ADAPTATOR_EMST_H
/*
 ***************************************************************************
 *
 * Author : Wenbao QIAO
 * Creation date : November. 2016
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

#include "random_generator.h"
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"

#include "SpiralSearchNNL.h"

#include "ViewGrid.h"
#ifdef TOPOLOGIE_HEXA
#include "ViewGridHexa.h"
#endif
#include "NIter.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif

#include "Objectives.h"
#include "basic_operations.h"
#include "geometry.h"
#include "CellularMatrix.h"

// wb.Q add
#define MAX_VECTOR_2OPT_STEP 16384
#define BLOCKSIZE 256
#define GRIDSIZE 256 // 1024 for sw24978
#define MAX_CITIES 1000



using namespace std;
using namespace components;

namespace operators
{

//! qiao 200516 add for searcher nodes contening links
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinks {
    
    template <class NN1, class NN2>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN1& scher,
                            NN2& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;
        
        SpiralSearchCMIteratorLinks sps_iter(
                    PC,
                    cm.vgd.getWidth() * cm.vgd.getHeight(),
                    0,
                    MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                    1
                    );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};


//! WB.Q 201016 add for searcher nodes contening links only cpu version
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinks_cpu {

    template <class NN1, class NN2>
    bool search(CellularMatrix& cm,
                NN1& scher,
                NN2& sched,
                PointCoord PC,
                PointCoord ps,
                PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIteratorLinks sps_iter(
                    PC,
                    cm.vgd.getWidth() * cm.vgd.getHeight(),
                    0,
                    MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                    1
                    );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};



//! qiao 290916 add for enlarge the searching cell radius
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinksGenerateTspFromSom {

    template <class NN1, class NN2>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN1& scher,
                            NN2& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIteratorLinks sps_iter(
                    PC,
                    cm.vgd.getWidth() * cm.vgd.getHeight(),
                    0,
                    cm.vgd.getWidthDual()* cm.vgd.getHeightDual(),
                    1
                    );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};


//! qiao 030916 overload : add for searcher nodes contening links
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinks2 {

    template <class NN1, class NN2>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN1& scher,
                            NN2& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIteratorLinks sps_iter(
                    PC,
                    cm.vgd.getWidth() * cm.vgd.getHeight(),
                    0,
                    MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                    1
                    );
        ret = sps_iter.search(scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};



//! qiao 030916, overload to adatp all nn with type of networklinks
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinks3 {

    template <class NN1, class NN2>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN1& scher,
                            NN2& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIteratorLinks sps_iter(
                    PC,
                    cm.vgd.getWidth() * cm.vgd.getHeight(),
                    0,
                    MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
                    1
                    );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};


////! wb.Q 230916, overload to adatp all nn with type of networklinks and search neighbor cell within radius
//template <class CellularMatrix,
//          class SpiralSearchCMIteratorLinks>
//struct SearchSpiralAdaptorLinksWithRadius {

//    template <class NN1, class NN2>
//    DEVICE_HOST bool search(CellularMatrix& cm,
//                            NN1& scher,
//                            NN2& sched,
//                            PointCoord PC,
//                            PointCoord ps,
//                            PointCoord& minP,
//                            int radiusSearchCells) {
//        bool ret = false;

//        SpiralSearchCMIteratorLinks sps_iter(
//                    PC,
//                    cm.vgd.getWidth() * cm.vgd.getHeight(),
//                    0,
//                    MAX(cm.vgd.getWidthDual(), cm.vgd.getHeightDual()),
//                    1 //d_step
//                    );
//        ret = sps_iter.search(cm,
//                              scher,
//                              sched,
//                              ps,
//                              minP,
//                              radiusSearchCells);
//        return ret;
//    }
//};


//! qiao 230916, overload to work on CPU side
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinksWithRadius_cpu {

    template <class NetLinkPointCoord>
    bool search(CellularMatrix& cm,
                NetLinkPointCoord& scher,
                NetLinkPointCoord& sched,
                PointCoord PC,
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




//! qiao 08081616 add for 2opt
template <class CellularMatrix,
          class SpiralSearchCMIteratorLinks>
struct SearchSpiralAdaptorLinks2opt {
    
    template <class NN, class NetLinkPointCoord>
    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NetLinkPointCoord& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;
        minP = ps;
        return ret = true;
    }
};



//! 190516 wb.Q add for trigger links, correct version, verify (int)(radius - learningRange)070616
template <class Cell>
struct OperateTriggerAdaptorLinksRecursiveVector {
    
    GLfloat alpha;
    GLfloat radius;
    
    DEVICE_HOST OperateTriggerAdaptorLinksRecursiveVector() : alpha(), radius(){}
    
    DEVICE_HOST OperateTriggerAdaptorLinksRecursiveVector(GLfloat a, GLfloat r) : alpha(a), radius(r){}
    
    template <class NN, class NetLinkPointCoord>
    DEVICE_HOST inline void operate(Cell& cell,  NN& nn_source, NetLinkPointCoord& nn_cible, PointCoord p_source, PointCoord p_cible) {
        
        vector<PointCoord> nodeAlreadyTeached;
        int d = 0;
        
        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()
                && p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight()) {
            
            operate(nn_source, nn_cible, p_source, p_cible, nodeAlreadyTeached, d);
        }
    }
    
    template <class NN, class NetLinkPointCoord>
    DEVICE_HOST void operate(NN& nn_source, NetLinkPointCoord& nn_cible, PointCoord p_source, PointCoord p_cible, vector<PointCoord>& nodeAlreadyTeached, int d) {
        
        PointEuclid p = nn_source.adaptiveMap[p_source[1]][p_source[0]];
        
        
        GLfloat alpha_temp = alpha * chap(d, (GLfloat)radius*LARG_CHAP);
        
        // no-matter this node p_cibe has links or not, if it is detected, it can be taught
        if (nn_cible.fixedMap[p_cible[1]][p_cible[0]] == 0) {
            
            PointEuclid n_links = nn_cible.adaptiveMap[p_cible[1]][p_cible[0]];
            
            n_links[0] = n_links[0] + alpha_temp * (p[0] - n_links[0]);
            n_links[1] = n_links[1] + alpha_temp * (p[1] - n_links[1]);
            
            nn_cible.adaptiveMap[p_cible[1]][p_cible[0]] = n_links;
        }
        
        // marck this node as being taught
        nodeAlreadyTeached.push_back(p_cible);
        
        // stop recursive
        if (d > (int)radius) return;

        {
            
            int nLinks = nn_cible.networkLinks[p_cible[1]][p_cible[0]].numLinks;
            
            for (int pLink = 0; pLink < nLinks; pLink++){ // teach current node and its links
                
                PointCoord pLinkOfNode;
                nn_cible.networkLinks[p_cible[1]][p_cible[0]].get(pLink, pLinkOfNode);
                
                PointCoord pCoord(0, 0);
                pCoord[0] = (int)pLinkOfNode[0];
                pCoord[1] = (int)pLinkOfNode[1];
                
                // compare if the current pCoord is already be teached
                bool teached = 0;
                for (int k = 0; k < nodeAlreadyTeached.size(); k ++){
                    PointCoord pLinkTemp(0, 0);
                    pLinkTemp = nodeAlreadyTeached[k];
                    if (pCoord[0] == pLinkTemp[0] && pCoord[1] == pLinkTemp[1])
                        teached = 1;
                }
                
                if (teached)
                    continue;
                
                else {
                    
                    operate(nn_source, nn_cible, p_source, pCoord, nodeAlreadyTeached, d++);
                }
                
            }// for
        }
    }
};//OperateTriggerAdaptorLinks-recursive




//! QWB 070816 add to mark neurons as closest to cities
template <class Cell>
struct OperateProjectCityToNeuron{

    DEVICE_HOST OperateProjectCityToNeuron(){}

    template <class NN, class NetLinkPointCoord>
    DEVICE_HOST void operate(Cell& cell,  NN& nn_source, NetLinkPointCoord& nn_cible, Grid<Point2D>& cityMap, PointCoord p_source, PointCoord p_cible){

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()
                && p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight()
                && nn_cible.activeMap[p_cible[1]][p_cible[0]] == 0){

            //! QWB here I use activeMap as flag
            nn_cible.activeMap[p_cible[1]][p_cible[0]] = 1;
            nn_cible.adaptiveMap[p_cible[1]][p_cible[0]] = nn_source.adaptiveMap[p_source[1]][p_source[0]];
            nn_cible.adaptiveMapOri[p_cible[1]][p_cible[0]] = cityMap[p_source[1]][p_source[0]];
        }
    }
};


//! QWB 070816 add to mark neurons as closest to cities, only on cpu side
template <class Cell>
struct OperateProjectCityToNeuron_cpu{

    OperateProjectCityToNeuron_cpu(){}

    template <class NN, class NetLinkPointCoord>
    void operate(Cell& cell,  NN& nn_source, NetLinkPointCoord& nn_cible, Grid<Point2D>& cityMap, PointCoord p_source, PointCoord p_cible){

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()
                && p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight()
                && nn_cible.activeMap[p_cible[1]][p_cible[0]] == 0){

            //! QWB here I use activeMap as flag
            nn_cible.activeMap[p_cible[1]][p_cible[0]] = 1;
            nn_cible.adaptiveMap[p_cible[1]][p_cible[0]] = nn_source.adaptiveMap[p_source[1]][p_source[0]];
            nn_cible.adaptiveMapOri[p_cible[1]][p_cible[0]] = cityMap[p_source[1]][p_source[0]];
        }
    }
};



#ifdef CUDA_CODE
//! QWB 141016 add operation to reconnect indenpendent tours for mst
template <class Cell, class Distance>
struct OperateConnectMstSuperVertices{

    DEVICE_HOST OperateConnectMstSuperVertices(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                        PointCoord couple1, PointCoord couple2,
                        Grid<BufferLinkPointCoord>& netWorkLinkCopy){

        bool exist = 0;

        int numLinks = nn_source.networkLinks[ps[1]][ps[0]].numLinks;

        for(int k = 0; k < numLinks; k++){

            PointCoord pTLink(-1, -1);
            PointCoord pTLink_(-1, -1);

            pTLink = nn_source.networkLinks[ps[1]][ps[0]].get(k);

            pTLink_[0] = (int)pTLink[0];
            pTLink_[1] = (int)pTLink[1];

            if(couple2 == pTLink_)
                exist = 1;
        }

        if(!exist){

            PointCoord pTemp2(-1, -1);
            pTemp2 = couple2;

            nn_source.networkLinks[ps[1]][ps[0]].insertAtomic(pTemp2);
            netWorkLinkCopy[ps[1]][ps[0]].insertAtomic(pTemp2);
        }

    } // operate


};
#else


// WBQ 11/16 no use
template <class Cell, class Distance>
struct OperateConnectMstSuperVertices{

    DEVICE_HOST OperateConnectMstSuperVertices(){}

    size_t step;

    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                             PointCoord couple1, PointCoord couple2,
                             Grid<BufferLink2D>& netWorkLinkCopy ){

        bool exist = 0;

        int numLinks = nn_source.networkLinks[ps[1]][ps[0]].numLinks;

        for(int k = 0; k < numLinks; k++){

            PointCoord pTLink(-1, -1);
            PointCoord pTLink_(-1, -1);

            pTLink = nn_source.networkLinks[ps[1]][ps[0]].get(k);

            pTLink_[0] = (int)pTLink[0];
            pTLink_[1] = (int)pTLink[1];

            if(couple2 == pTLink_)
                exist = 1;
        }

        if(!exist){

            PointCoord pTemp2(-1, -1);
            pTemp2 = couple2;

            nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);
            netWorkLinkCopy[ps[1]][ps[0]].insert(pTemp2);
        }
    } // operate


};
#endif // gpu cpu version of add mst outgoing edge only once



// WBQ 031216 add single link to v_i
template <class Cell, class Distance>
struct OperateConnectMstSuperVertices_final{

    DEVICE_HOST OperateConnectMstSuperVertices_final(){}

    size_t step;

    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        PointCoord couple2(-1, -1);
        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

        PointCoord pTemp2(-1, -1);
        pTemp2 = couple2;

        nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);

    }
};


// WB.Q add to add single link for pre-winner node
template <class Cell, class Distance>
struct OperateConnectMstSuperVertices_cpu{

    OperateConnectMstSuperVertices_cpu(){}

    size_t step;

    template <class NetLinkPointCoord>
    void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                 Grid<BufferLinkPointCoord>& netWorkLinkCopy ){


        PointCoord couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

        if(nn_source.fixedMap[couple2[1]][couple2[0]]){

            PointCoord couple1 = nn_source.correspondenceMap[couple2[1]][couple2[0]];

            if(ps == couple1){

                bool exist = 0;

                int numLinks = nn_source.networkLinks[ps[1]][ps[0]].numLinks;

                for(int k = 0; k < numLinks; k++){

                    PointCoord pTLink(-1, -1);
                    PointCoord pTLink_(-1, -1);

                    pTLink = nn_source.networkLinks[ps[1]][ps[0]].get(k);

                    pTLink_[0] = (int)pTLink[0];
                    pTLink_[1] = (int)pTLink[1];

                    if(couple2 == pTLink_)
                        exist = 1;
                }

                if(!exist){

                    PointCoord pTemp2(-1, -1);
                    pTemp2 = couple2;

                    nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);
                    netWorkLinkCopy[ps[1]][ps[0]].insert(pTemp2);
                }

            }

        }

    } // operate
};


#ifdef CUDA_CODE
//! QWB 191016 add double links to winner node
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode{

    DEVICE_HOST OperateAddDoubleLinkToWinnerNode(){}


    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                        Grid<BufferLinkPointCoord>& netWorkLinkCopy){

        printf("step4 gpu operation, ");

        PointCoord pLinOfWin(-1, -1);
        pLinOfWin = nn_source.correspondenceMap[ps[1]][ps[0]];


        bool exist = 0;

        int numLinks = nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].numLinks;

        for(int k = 0; k < numLinks; k++){

            PointCoord pTLink(-1, -1);
            PointCoord pTLink_(-1, -1);

            pTLink = nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].get(k);

            pTLink_[0] = (int)pTLink[0];
            pTLink_[1] = (int)pTLink[1];

            if(ps == pTLink_)
                exist = 1;
        }

        // if ps does not exist in its links
        if(!exist){
            PointCoord ps_(-1, -1);
            ps_ = ps;

         if ((atomicAdd(&(nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].numLinks), 1)) < MAX_LINK_SIZE)
            {
                atomicExch(&(nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].bCell[(nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].numLinks) - 1][0]), (float)ps_[0]);
                atomicExch(&(nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].bCell[(nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].numLinks) - 1][1]), (float)ps_[1]);

            }


        }


    } // operate


};
#else
//! QWB 191016 add double links to winner node
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode{

    DEVICE_HOST OperateAddDoubleLinkToWinnerNode(){}


    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                             Grid<BufferLink2D>& netWorkLinkCopy ){

        PointCoord pLinOfWin(-1, -1);

        pLinOfWin = nn_source.correspondenceMap[ps[1]][ps[0]];

        bool exist = 0;

        int numLinks = nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].numLinks;

        for(int k = 0; k < numLinks; k++){

            PointCoord pTLink(-1, -1);
            PointCoord pTLink_(-1, -1);

            pTLink = nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].get(k);

            pTLink_[0] = (int)pTLink[0];
            pTLink_[1] = (int)pTLink[1];

            if(ps == pTLink_)
                exist = 1;
        }
        // if ps does not exist in its links
        if(!exist){
            PointCoord ps_(-1, -1);
            ps_ = ps;
            nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].insert(ps_);
            cout << "step4 insert ps into " << pLinOfWin[0] << ", " << pLinOfWin[1]  << endl;
        }
    }// operate
};

#endif // gpu cpu version of add mst outgoing edge only once


////! QWB 031216 add operation to mark finnal winner root on merging root graph
//template <class Cell>
//struct operateMarkMstSuperVertices_finalPointDisjoint{

//    DEVICE_HOST operateMarkMstSuperVertices_finalPointDisjoint(){}

//    template <class NN, class NetLinkPointCoord>
//    DEVICE void operate(NetLinkPointCoord& nn_source, Grid<BufferLinkPointCoord>& rootMergingGraphGpu, PointCoord ps, float *dev_finishMst){

//        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

//        int idLeast = idOriginal;
//        {
//            PointCoord pInitial(-1, -1);

//            BufferLinkPointCoordVector nodeAlreadyTraversed;
//            nodeAlreadyTraversed.init(MAX_VECTOR_SIZE_MERGRAPH);

//            BufferLinkPointCoordVector tabO;
//            tabO.init(MAX_VECTOR_SIZE_MERGRAPH);
//            BufferLinkPointCoordVector tabD;
//            tabD.init(MAX_VECTOR_SIZE_MERGRAPH);

//            tabO.insert(ps);

//            BufferLinkPointCoordVector* tabO_ = &tabO;
//            BufferLinkPointCoordVector* tabD_ = &tabD;
//            BufferLinkPointCoordVector* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

//            while((*tabO_).numLinks > 0){

//                for (int i = 0; i < (*tabO_).numLinks; i++){

//                    PointCoord pCoord = (*tabO_).bCell[i];

//                    if(pCoord != pInitial)
//                    {

//                        int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];
//                        if(idLeast == infinity ){
//                            nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
//                            idLeast = pCoord[0];
//                        }
//                        if(idPCoord < idLeast)
//                        {
//                            idLeast = idPCoord;
//                        }

//                        PointCoord pInitial2D(-1, -1);

//                        int nLinks = rootMergingGraphGpu[pCoord[1]][pCoord[0]].numLinks;
//                        for (int pLink = 0; pLink < nLinks; pLink++){

//                            PointCoord pLinkOfNode;
//                            rootMergingGraphGpu[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

//                            if(pLinkOfNode != pInitial2D){
//                                PointCoord pCoordLink(-1, -1);
//                                pCoordLink[0] = (int)pLinkOfNode[0];
//                                pCoordLink[1] = (int)pLinkOfNode[1];

//                                bool traversed = 0;
//                                for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
//                                    PointCoord pLinkTemp(-1, -1);
//                                    pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
//                                    if (pCoordLink == pLinkTemp)
//                                        traversed = 1;
//                                }
//                                if (!traversed){
//                                    (*tabD_).insert(pCoordLink);

//                                }
//                            }
//                        }
//                    }
//                }

//                if((*nodeAlreadyTraversed_).numLinks > MAX_VECTOR_SIZE_MERGRAPH)
//                    printf("size merging graph %d, %d, %d \n", (*nodeAlreadyTraversed_).numLinks, (*tabO_).numLinks, (*tabD_).numLinks);

//                (*nodeAlreadyTraversed_).clearLinks();


//                swapClass4(&nodeAlreadyTraversed_, &tabO_);
//                swapClass4(&tabO_, &tabD_);
//            }
//        }

//        //! WB.Q mark the representative city
//        if(idLeast == idOriginal){
//            nn_source.activeMap[0][idLeast] = 1;
//        }

//    } // operate
//};



////! QWB 031216 refresh root id on root merging graph from activeMap
//template <class Cell>
//struct operateCompactGraph2_final2Disjoint{

//    DEVICE_HOST operateCompactGraph2_final2Disjoint(){}

//    template <class NetLinkPointCoord>
//    DEVICE void operate(NetLinkPointCoord& nn_source, Grid<BufferLinkPointCoord>& rootMergingGraph, PointCoord& ps){

//        int idLeast = nn_source.grayValueMap[ps[1]][ps[0]];

//        PointCoord pInitial(-1, -1);

//        BufferLinkPointCoordVector nodeAlreadyTraversed;
//        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE_MERGRAPH);

//        BufferLinkPointCoordVector tabO;
//        tabO.init(MAX_VECTOR_SIZE_MERGRAPH);
//        BufferLinkPointCoordVector tabD;
//        tabD.init(MAX_VECTOR_SIZE_MERGRAPH);

//        tabO.insert(ps);

//        BufferLinkPointCoordVector* tabO_ = &tabO;
//        BufferLinkPointCoordVector* tabD_ = &tabD;
//        BufferLinkPointCoordVector* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

//        while((*tabO_).numLinks > 0){

//            for (int i = 0; i < (*tabO_).numLinks; i++){

//                PointCoord pCoord = (*tabO_).bCell[i];
//                {

//                    int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];

//                    if(idPCoord > idLeast)
//                    {

//                        nn_source.grayValueMap[pCoord[1]][pCoord[0]] = idLeast;
//                    }

//                    int nLinks = rootMergingGraph[pCoord[1]][pCoord[0]].numLinks;
//                    for (int pLink = 0; pLink < nLinks; pLink++){

//                        PointCoord pLinkOfNode;
//                        rootMergingGraph[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

//                        if(pLinkOfNode != pInitial)
//                        {
//                            PointCoord pCoordLink(-1, -1);
//                            pCoordLink[0] = pLinkOfNode[0];
//                            pCoordLink[1] = pLinkOfNode[1];

//                            bool traversed = 0;
//                            for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
//                                PointCoord pLinkTemp(-1, -1);
//                                pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
//                                if (pCoordLink == pLinkTemp)
//                                    traversed = 1;
//                            }
//                            if (!traversed){
//                                (*tabD_).insert(pCoordLink);
//                            }
//                        }
//                    }
//                }
//            }
//            if((*nodeAlreadyTraversed_).numLinks > MAX_VECTOR_SIZE_MERGRAPH)
//                printf("size merging graph %d, %d, %d \n", (*nodeAlreadyTraversed_).numLinks, (*tabO_).numLinks, (*tabD_).numLinks);

//            (*nodeAlreadyTraversed_).clearLinks();


//            swapClass4(&nodeAlreadyTraversed_, &tabO_);
//            swapClass4(&tabO_, &tabD_);

//        }// while
//    } // operate
//};



//#ifdef CUDA_CODE
////! QWB 031217 add double links to winner node, [connect graph 2]
//template <class Cell, class Distance>
//struct OperateAddDoubleLinkToWinnerNode_disjoint{

//    DEVICE_HOST OperateAddDoubleLinkToWinnerNode_disjoint(){}


//    DEVICE_HOST int findRoot(Grid<GLint> &idGpuMap, int x)
//    {
//        GLint r = x;
//        while(r != idGpuMap[0][r])
//        {
//            r = idGpuMap[0][r]; // r is the root
//        }

//        return r;
//    }



//    template <class NetLinkPointCoord>
//    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps, Grid<BufferLinkPointCoord>& rootMergingGraphGpu){

//        PointCoord couple2(-1, -1);
//        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

//        PointCoord pCouple2(-1, -1);
//        pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];

//        PointCoord pTemp2(-1, -1);
//        pTemp2 = couple2;
//        nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);// wenbaoQiao, necessary to be here

//        int rootPs = findRoot(nn_source.grayValueMap, ps[0]);
//        int rootCouple2 = findRoot(nn_source.grayValueMap, couple2[0]);
//        pTemp2[0] = rootCouple2;
//        pTemp2[1] = 0;
//        rootMergingGraphGpu[0][rootPs].insert(pTemp2); // wenbaoQiao, rootPs add link of rootCouple2

//        //! QWB add double links to winner node, [connect graph 2]
//        pTemp2[0] = rootPs;
//        pTemp2[1] = 0;
//        if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0) {
//            PointCoord ps_(-1, -1);
//            ps_ = ps;
//            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][0]), (float)ps_[0]);
//            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][1]), (float)ps_[1]);
//            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][0]), pTemp2[0]);
//            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][1]), pTemp2[1]);

//        }
//        else if(nn_source.fixedMap[couple2[1]][couple2[0]]== 1 && pCouple2 != ps)
//        {
//            PointCoord ps_(-1, -1);
//            ps_ = ps;
//            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][0]), (float)ps_[0]);
//            atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][1]), (float)ps_[1]);
//            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][0]), pTemp2[0]);
//            atomicExch(&(rootMergingGraphGpu[0][rootCouple2].bCell[(rootMergingGraphGpu[0][rootCouple2].numLinks) - 1][1]), pTemp2[1]);
//          }

//        if(nn_source.grayValueMap[ps[1]][ps[0]] > nn_source.grayValueMap[couple2[1]][couple2[0]])
//        {
//            nn_source.fixedMap[ps[1]][ps[0]] = 0;
//        }

//    } // operate

//};
//#else
////! wenbao Qiao 191017 add double links to winner node under conditions
//template <class Cell, class Distance>
//struct OperateAddDoubleLinkToWinnerNode_disjoint{

//    DEVICE_HOST OperateAddDoubleLinkToWinnerNode_disjoint(){}


//    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

//        PointCoord couple2(-1, -1);

//        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

//        PointCoord pCouple2(-1, -1);
//        pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];


//        PointCoord pTemp2(-1, -1);
//        pTemp2 = couple2;

//        nn_source.networkLinks[ps[1]][ps[0]].insert(pTemp2);


//        //! wb.Q if correspondence is not winner
//        if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0){
//            PointCoord ps_(-1, -1);
//            ps_ = ps;
//            nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
//        }
//        //! wb.Q if correspondence is winner but not couple to ps
//        if(nn_source.fixedMap[0][couple2[0]] && pCouple2 != ps ){
//            PointCoord ps_(-1, -1);
//            ps_ = ps;
//            nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
//        }

//    }// operate
//};

//#endif // gpu cpu version of add mst outgoing edge only once


#ifdef CUDA_CODE
//! QWB 031216 add double links to winner node, [connect graph 2]
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode_final{

    DEVICE_HOST OperateAddDoubleLinkToWinnerNode_final(){}


    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        PointCoord couple2(-1, -1);
        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

        PointCoord pCouple2(-1, -1);
        pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];



        // if ps does not exist in its links
        if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0){
            PointCoord ps_(-1, -1);
            ps_ = ps;

            if ((atomicAdd(&(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks), 1)) < MAX_LINK_SIZE)
            {
                atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][0]), (float)ps_[0]);
                atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][1]), (float)ps_[1]);
            }
            printf("add double link ps %d, %d to couple2 %d, %d \n" , ps[0], 0, couple2[0], 0);

        }
        if(nn_source.fixedMap[couple2[1]][couple2[0]] && pCouple2 != ps){
            PointCoord ps_(-1, -1);
            ps_ = ps;

            if ((atomicAdd(&(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks), 1)) < MAX_LINK_SIZE) // copy from Wang's version, this way does not work for multi-thread operation
            {
                atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][0]), (float)ps_[0]);
                atomicExch(&(nn_source.networkLinks[couple2[1]][couple2[0]].bCell[(nn_source.networkLinks[couple2[1]][couple2[0]].numLinks) - 1][1]), (float)ps_[1]);
            }

            printf("add double link ps %d, %d to couple2 %d, %d \n" , ps[0], 0, couple2[0], 0);

        }

    } // operate

};
#else
//! QWB 191016 add double links to winner node
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode_final{

    DEVICE_HOST OperateAddDoubleLinkToWinnerNode_final(){}


    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        PointCoord couple2(-1, -1);

        couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];


        PointCoord pCouple2(-1, -1);
        pCouple2 = nn_source.correspondenceMap[couple2[1]][couple2[0]];


        // if ps does not exist in its links
        if(nn_source.fixedMap[couple2[1]][couple2[0]] == 0){
            PointCoord ps_(-1, -1);
            ps_ = ps;
            nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
            //            printf("add double link ps %d, %d to couple2 %d, %d \n" , ps[0], 0, couple2[0], 0);

        }
        if(nn_source.fixedMap[0][couple2[0]] && pCouple2 != ps ){
            PointCoord ps_(-1, -1);
            ps_ = ps;
            nn_source.networkLinks[couple2[1]][couple2[0]].insert(ps_);
            //            printf("add double link ps %d, %d to couple2 %d, %d \n" , ps[0], 0, couple2[0], 0);

        }

    }// operate
};

#endif // gpu cpu version of add mst outgoing edge only once



//! QWB 191016 add double links to winner node
template <class Cell, class Distance>
struct OperateAddDoubleLinkToWinnerNode_cpu{

    OperateAddDoubleLinkToWinnerNode_cpu(){}


    template <class NetLinkPointCoord>
    void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                 Grid<BufferLinkPointCoord>& netWorkLinkCopy ){

        PointCoord pLinOfWin(-1, -1);

        pLinOfWin = nn_source.correspondenceMap[ps[1]][ps[0]];

        bool exist = 0;

        int numLinks = nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].numLinks;

        for(int k = 0; k < numLinks; k++){

            PointCoord pTLink(-1, -1);
            PointCoord pTLink_(-1, -1);

            pTLink = nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].get(k);

            pTLink_[0] = (int)pTLink[0];
            pTLink_[1] = (int)pTLink[1];

            if(ps == pTLink_)
                exist = 1;

        }
        // if ps does not exist in its links
        if(!exist){
            PointCoord ps_(-1, -1);
            ps_ = ps;
            nn_source.networkLinks[pLinOfWin[1]][pLinOfWin[0]].insert(ps_);

            cout << "step4 insert " << pLinOfWin[0] << ", " << pLinOfWin[1] << " into ps " << endl;
        }

    }// operate
};


//! QWB 260916 add operation to mark independent super vertice
//! only 2-connected nn-links can work in this way
template <class Cell>
struct operateMarkSuperVerticesCpu{

    operateMarkSuperVerticesCpu(){}
    operateMarkSuperVerticesCpu(size_t step_){ step =  step_;}

    size_t step;

    template <class NetLinkPointCoord>
    void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        printf("Operator mark superVertices \n");
        if(nn_source.networkLinks[ps[1]][ps[0]].numLinks == 2){

            PointCoord pLinkOfNode;
            nn_source.networkLinks[ps[1]][ps[0]].get(0, pLinkOfNode);
            PointCoord pco(-1, -1);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];

            nn_source.grayValueMap[ps[1]][ps[0]] = ps[0];
            nn_source.grayValueMap[pco[1]][pco[0]] = ps[0];
            nn_source.fixedMap[pco[1]][pco[0]] = 1;

            PointCoord pco2(-1, -1);
            nn_source.networkLinks[pco[1]][pco[0]].get(0, pLinkOfNode);
            pco2[0] = (int)pLinkOfNode[0];
            pco2[1] = (int)pLinkOfNode[1];

            if(pco2 == ps){
                nn_source.networkLinks[pco[1]][pco[0]].get(1, pLinkOfNode);
                pco2[0] = (int)pLinkOfNode[0];
                pco2[1] = (int)pLinkOfNode[1];
            }

            PointCoord pco2Avant;
            pco2Avant = pco;

            while(pco2 != ps){

                PointCoord pco3(-1, -1);
                nn_source.networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode);
                pco3[0] = (int)pLinkOfNode[0];
                pco3[1] = (int)pLinkOfNode[1];

                if(pco3 == pco2Avant){
                    nn_source.networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode);
                    pco3[0] = (int)pLinkOfNode[0];
                    pco3[1] = (int)pLinkOfNode[1];

                }

                if(pco3 == pco2Avant){
                    //                    cout << "open circle . ps: " << ps[0] << ", " <<  ps[1] << endl;
                    break;
                }

                nn_source.grayValueMap[pco2[1]][pco2[0]] = ps[0];
                nn_source.fixedMap[pco2[1]][pco2[0]] = 1;

                pco2Avant = pco2;
                pco2 = pco3;

            }

        }
        //        else
        //            cout << "Error mark super vertices, no links or bigger than 2. " << endl;


    } // operate
};



//! QWB 131016 add operation to mark independent super vertice for mst from node level
//! any nn-links can work in this way
template <class Cell>
struct operateMarkMstSuperVerticesOneNodeOneThread{

    DEVICE_HOST operateMarkMstSuperVerticesOneNodeOneThread(){}

    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

        //        printf(" refresh id, ps = %d, id_ps = %d  \n", ps[0], idOriginal);
        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_VECTOR_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_VECTOR_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;

        while((*tabO_).numLinks > 0){

            for (size_t i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                // compare if the current pCoord is already be teached
                bool traversed = 0;
                for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = nodeAlreadyTraversed.bCell[k];
                    if (pCoord == pLinkTemp)
                        traversed = 1;
                }
                if (traversed)
                    continue;
                else{

                    nodeAlreadyTraversed.insert(pCoord);

                    //                    printf("link of ps, pCoord = %d \n", pCoord[0]);
                    //                    printf("old id gray = %d \n", nn_source.grayValueMap[ps[1]][ps[0]]);
                    if(pCoord[0] < idOriginal){
                        nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                        idOriginal = pCoord[0];
                        //                        printf("         id < ps, new id = %d \n", nn_source.grayValueMap[ps[1]][ps[0]]);

                    }


                    size_t nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;


                    for (size_t pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode;
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        PointCoord pCoordLink(-1, -1);
                        pCoordLink[0] = (int)pLinkOfNode[0];
                        pCoordLink[1] = (int)pLinkOfNode[1];


                        // compare if the current pCoord is already be teached
                        bool traversed = 0;
                        for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                            PointCoord pLinkTemp(-1, -1);
                            pLinkTemp = nodeAlreadyTraversed.bCell[k];
                            if (pCoordLink == pLinkTemp)
                                traversed = 1;
                        }
                        if (traversed)
                            continue;

                        else
                            (*tabD_).insert(pCoordLink);

                    }

                }

            }

            (*tabO_).clearLinks();//qiao note, here if we just clear the numlinks, the buffer will always has values

#ifdef CUDA_CODE

            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else

            swapClass4(&tabO_, &tabD_);
#endif


        }// while

    } // operate


};


//! QWB 031216 add operation to mark independent super vertice base on paper
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateMarkMstSuperVertices_finalPoint{

    DEVICE_HOST operateMarkMstSuperVertices_finalPoint(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps, bool *dev_finishMst){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];
        int idLeast = idOriginal;
        int size = 0;


        PointCoord pInitial(-1, -1);

        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_LINK_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_LINK_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;
        BufferLinkPointCoord* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){


                PointCoord pCoord = (*tabO_).bCell[i];

                if(pCoord != pInitial){
                    // compare if the current pCoord is already be teached
                    bool traversed = 0;
                    for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
                        PointCoord pLinkTemp(-1, -1);
                        pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
                        if (pCoord == pLinkTemp)
                            traversed = 1;
                    }
                    if (!traversed)
                    {
                        //                        size ++;

                        int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];

                        //                        printf("[1] ps %d, %d ; link pco %d, %d \n", ps[0], ps[1], pCoord[0], pCoord[1]);

                        if(idLeast == 999999 && pCoord[0] < idLeast){
                            nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                            idLeast = pCoord[0];
                        }
                        else if(idPCoord < idLeast) // QWB.031216 should not use if the ID of PS is minimum
                        {
                            idLeast = idPCoord;

                            //                            printf("[1]ps %d, %d, ID %d , pCoord %d, %d, ID %d, idLeast change, new idLeast %d \n" ,
                            //                                   ps[0], ps[1],nn_source.grayValueMap[ps[1]][ps[0]],
                            //                                    pCoord[0], pCoord[1],nn_source.grayValueMap[pCoord[1]][pCoord[0]], idLeast);

                        }

                        PointCoord pInitial2D(-1, -1);

                        int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                        for (int pLink = 0; pLink < nLinks; pLink++){

                            PointCoord pLinkOfNode;
                            nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                            if(pLinkOfNode != pInitial2D){
                                PointCoord pCoordLink(-1, -1);
                                pCoordLink[0] = (int)pLinkOfNode[0];
                                pCoordLink[1] = (int)pLinkOfNode[1];

                                // compare if the current pCoord is already be teached
                                traversed = 0;
                                for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
                                    PointCoord pLinkTemp(-1, -1);
                                    pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
                                    if (pCoordLink == pLinkTemp)
                                        traversed = 1;
                                }
                                if (!traversed){
                                    size ++;
                                    (*tabD_).insert(pCoordLink);
                                    //                                    printf("[1] ps %d, %d,  size %d, %d, \n", ps[0], ps[1], pCoordLink[0], pCoordLink[1]);

                                }
                            }
                        }
                    }

                }
            }

            (*nodeAlreadyTraversed_).clearLinks();
#ifdef CUDA_CODE

            *nodeAlreadyTraversed_ = *tabO_;
            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else


            swapClass4(&nodeAlreadyTraversed_, &tabO_);
            swapClass4(&tabO_, &tabD_);
#endif
        }// while

        //        nn_source.sizeOfComponentMap[ps[1]][ps[0]] = size;

        //! WB.Q mark the finish criterion
        if(size == nn_source.adaptiveMap.width)
            *dev_finishMst = 1;

        //! WB.Q mark the representative city
        if(idLeast == idOriginal){
            nn_source.sizeOfComponentMap[ps[1]][ps[0]] = size;
            nn_source.activeMap[ps[1]][ps[0]] = 1;
        }
    } // operate
};


//! QWB 031216 add operation initialize the initial ID from 999999 to i
template <class Cell>
struct operateInitializeIdOfSuperVertices{

    DEVICE_HOST operateInitializeIdOfSuperVertices(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps, bool *dev_finishMst){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

        if(idOriginal == 999999){
            nn_source.grayValueMap[ps[1]][ps[0]] = ps[0];
        }
    } // operate
};



//! QWB 031216 add operation to mark independent super vertice base on paper
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateMarkMstSuperVertices_final{

    DEVICE_HOST operateMarkMstSuperVertices_final(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps, bool *dev_finishMst){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];
        int idLeast = idOriginal;
        int size = 0;


        PointCoord pInitial(-1, -1);
        PointCoord pInitialPco(-1, -1);

        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_LINK_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_LINK_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                if(pCoord != pInitialPco)
                {

                    //                     compare if the current pCoord is already be teached
                    bool traversed = 0;
                    for (int k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                        PointCoord pLinkTemp(-1, -1);
                        pLinkTemp = nodeAlreadyTraversed.bCell[k];
                        if (pCoord == pLinkTemp)
                            traversed = 1;
                    }
                    if (!traversed)
                    {

                        size ++;

                        nodeAlreadyTraversed.insert(pCoord);

                        int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];

                        //                        printf("[1] ps %d, %d ; link pco %d, %d \n", ps[0], ps[1], pCoord[0], pCoord[1]);

                        if(idLeast == 999999 && pCoord[0] < idLeast){
                            nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                            idLeast = pCoord[0];
                        }
                        else if(idPCoord < idLeast) // QWB.031216 should not use if the ID of PS is minimum
                        {
                            idLeast = idPCoord;

                            //                        printf("[1]ps %d, %d, ID %d , pCoord %d, %d, ID %d, idLeast change, new idLeast %d \n" ,
                            //                               ps[0], ps[1],nn_source.grayValueMap[ps[1]][ps[0]],
                            //                                pCoord[0], pCoord[1],nn_source.grayValueMap[pCoord[1]][pCoord[0]], idLeast);

                        }

                        int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                        for (int pLink = 0; pLink < nLinks; pLink++){

                            PointCoord pLinkOfNode;
                            nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                            if(pLinkOfNode != pInitial)
                            {
                                PointCoord pCoordLink(-1, -1);
                                pCoordLink[0] = (int)pLinkOfNode[0];
                                pCoordLink[1] = (int)pLinkOfNode[1];

                                // compare if the current pCoord is already be teached
                                traversed = 0;
                                for (int k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                                    PointCoord pLinkTemp(-1, -1);
                                    pLinkTemp = nodeAlreadyTraversed.bCell[k];

                                    if (pCoordLink == pLinkTemp)
                                        traversed = 1;
                                }
                                if (!traversed)
                                {
                                    (*tabD_).insert(pCoordLink);
                                    //                                    size ++;
                                    //                                    printf("[1] ps %d, %d,  size %d, %d, \n", ps[0], ps[1], pCoordLink[0], pCoordLink[1]);

                                }
                            }
                        }

                    }
                }
            }

            (*tabO_).clearLinks();
#ifdef CUDA_CODE

            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else

            swapClass4(&tabO_, &tabD_);
#endif
        }// while

        //        nn_source.sizeOfComponentMap[ps[1]][ps[0]] = size;

        if(size == nn_source.adaptiveMap.width)
            *dev_finishMst = 1;

        if(idLeast == idOriginal){
            nn_source.activeMap[ps[1]][ps[0]] = 1;
            nn_source.sizeOfComponentMap[ps[1]][ps[0]] = size;
        }
    } // operate
};



//! QWB 031216 add compact graph kernel 2, 071216 correct for test20
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateCompactGraph2_final{

    DEVICE_HOST operateCompactGraph2_final(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        int idLeast = nn_source.grayValueMap[ps[1]][ps[0]];
        int size = 0;

        PointCoord pInitial(-1, -1);

        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_LINK_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_LINK_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;
        BufferLinkPointCoord* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];


                {

                    size ++;

                    int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];


                    if(idPCoord > idLeast)
                    {
                        nn_source.grayValueMap[pCoord[1]][pCoord[0]] = idLeast;
                    }

                    int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                    for (int pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode;
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        if(pLinkOfNode != pInitial)
                        {
                            PointCoord pCoordLink(-1, -1);
                            pCoordLink[0] = pLinkOfNode[0];
                            pCoordLink[1] = pLinkOfNode[1];

                            //                              printf("[2] ps %d, %d, pLinkOfNode %d, %d , numLink %d, i %d \n", ps[0], ps[1], pCoordLink[0], pCoordLink[1], nLinks, pLink);

                            // compare if the current pCoord is already be teached
                            bool traversed = 0;
                            for (int k = 0; k < (*nodeAlreadyTraversed_).numLinks; k ++){
                                PointCoord pLinkTemp(-1, -1);
                                pLinkTemp = (*nodeAlreadyTraversed_).bCell[k];
                                if (pCoordLink == pLinkTemp)
                                    traversed = 1;
                            }
                            if (!traversed){
                                //                                printf("[2] ps %d, %d, pCo %d, %d , insert %d, %d \n", ps[0], ps[1],pCoord[0], pCoord[1], pCoordLink[0], pCoordLink[1]);

                                (*tabD_).insert(pCoordLink);
                            }
                        }
                    }

                }
            }

            //            printf("num already traversed %d, \n", (*nodeAlreadyTraversed_).numLinks);
            (*nodeAlreadyTraversed_).clearLinks();
#ifdef CUDA_CODE

            *nodeAlreadyTraversed_ = *tabO_;
            *tabO_ = *tabD_;

            //            printf("tabO_ new size %d \n ", (*tabO_).numLinks);

            //            for (int i = 0; i < (*tabO_).numLinks; i++){
            //                PointCoord pCoord = (*tabO_).bCell[i];
            //                printf("tabO_ content %d, %d \n", pCoord[0], pCoord[1]);

            //            }

            (*tabD_).clearLinks();
#else


            swapClass4(&nodeAlreadyTraversed_, &tabO_);
            swapClass4(&tabO_, &tabD_);
#endif

        }// while
    } // operate
};



//! QWB 031216 add compact graph kernel 2, recursive mode
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateCompactGraph2Recursive_final{

    DEVICE_HOST operateCompactGraph2Recursive_final(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        int idLeast = nn_source.grayValueMap[ps[1]][ps[0]];



        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

        if(nn_source.networkLinks[ps[1]][ps[0]].numLinks > 0)
            operateCompactGraph2(nn_source, ps, nodeAlreadyTraversed_, idLeast);

    } // operate


    template <class NetLinkPointCoord>
    DEVICE void operateCompactGraph2(NetLinkPointCoord& nn_source,
                                     PointCoord ps,
                                     BufferLinkPointCoord* nodeAlreadyTraversed_,
                                     int idLeast)
    {

        PointCoord pInitial(-1, -1);

        int nLink = nn_source.networkLinks[ps[1]][ps[0]].numLinks;

        (* nodeAlreadyTraversed_).insert(ps);

        for(int i = 0; i < nLink; i++){

            PointCoord pLinkOfNode(-1, -1);
            nn_source.networkLinks[ps[1]][ps[0]].get(i, pLinkOfNode);

            if(pLinkOfNode != pInitial){

                PointCoord pco(-1, -1);
                pco[0] = (int)pLinkOfNode[0];
                pco[1] = (int)pLinkOfNode[1];

                int idPco = nn_source.grayValueMap[pco[1]][pco[0]];

                if(idPco > idLeast) // QWB.031216 should not use if the ID of PS is minimum
                {
                    nn_source.grayValueMap[pco[1]][pco[0]] = idLeast;
                    //                                         printf("[2 recursive] ps active %d, %d, ID %d , change pCoord %d, %d, new ID %d \n" ,
                    //                                                ps[0], ps[1], idLeast,
                    //                                                 pco[0], pco[1], nn_source.grayValueMap[pco[1]][pco[0]]);
                }



                bool traversed = 0;
                for (int k = 0; k < (* nodeAlreadyTraversed_).numLinks; k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = (* nodeAlreadyTraversed_).bCell[k];

                    if(pco == pLinkTemp)
                        traversed = 1;
                }

                if(!traversed){
                    printf("recursive pco %d, %d \n ", pco[0], pco[1]);
                    operateCompactGraph2(nn_source, pco, nodeAlreadyTraversed_, idLeast);
                }

            }


        }
        return;

    }//operateCompactGraph2
};






//! QWB 241016 add operation to mark independent super vertice for mst from node level
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateMarkMstSuperVertices{

    DEVICE_HOST operateMarkMstSuperVertices(){}

    template <class NetLinkPointCoord>
    DEVICE void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        printf("ps %d, %d \n", ps[0], ps[1]);


        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

        printf("ID ps %d \n", idOriginal);


        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_VECTOR_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_VECTOR_SIZE);

        tabO.insert(ps);
        //        tabO.insertAutomic(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                // compare if the current pCoord is already be teached
                bool traversed = 0;
                for (int k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = nodeAlreadyTraversed.bCell[k];
                    if (pCoord == pLinkTemp)
                        traversed = 1;
                }
                if (traversed)
                    continue;
                else{

                    nodeAlreadyTraversed.insert(pCoord);
                    //                    nodeAlreadyTraversed.insertAutomic(pCoord);

                    int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];

                    printf("ID PCoord %d, pCoord = %d , ps = %d \n", idPCoord, pCoord[0], ps[0]);

                    if(idOriginal == 999999 && pCoord[0] < idOriginal){
                        nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                        idOriginal = pCoord[0];
                    }
                    else if(idPCoord < idOriginal){
                        printf("idOriginal changed %d \n", idOriginal);
                        nn_source.grayValueMap[ps[1]][ps[0]] = idPCoord;
                        idOriginal = idPCoord;
                        printf("ID PCoord %d, pCoord = %d , ps = %d \n", idPCoord, pCoord[0], ps[0]);

                        printf("idOriginal changed %d \n", idOriginal);

                    }
                    else if(idPCoord > idOriginal){
                        nn_source.grayValueMap[pCoord[1]][pCoord[0]] = idOriginal; //QWB other links set the minimum ID

                        printf("ID pCoord changed %d \n", idOriginal);
                    }

                    int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                    for (int pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode;
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        PointCoord pCoordLink(-1, -1);
                        pCoordLink[0] = (int)pLinkOfNode[0];
                        pCoordLink[1] = (int)pLinkOfNode[1];


                        // compare if the current pCoord is already be teached
                        bool traversed = 0;
                        for (int k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                            PointCoord pLinkTemp(-1, -1);
                            pLinkTemp = nodeAlreadyTraversed.bCell[k];
                            if (pCoordLink == pLinkTemp)
                                traversed = 1;
                        }
                        if (traversed)
                            continue;
                        else
                            (*tabD_).insert(pCoordLink);
                        //                            (*tabD_).insertAutomic(pCoordLink);

                    }

                }

            }

            (*tabO_).clearLinks();
#ifdef CUDA_CODE

            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else


            swapClass4(&tabO_, &tabD_);
#endif


        }// while
    } // operate
};




//! QWB 241016 add operation to mark independent super vertice for mst from node level
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateMarkMstSuperVertices_cpu{

    operateMarkMstSuperVertices_cpu(){}

    template <class NetLinkPointCoord>
    void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_VECTOR_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_VECTOR_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                // compare if the current pCoord is already be teached
                bool traversed = 0;
                for (int k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = nodeAlreadyTraversed.bCell[k];
                    if (pCoord == pLinkTemp)
                        traversed = 1;
                }
                if (traversed)
                    continue;
                else{

                    nodeAlreadyTraversed.insert(pCoord);

                    int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];

                    printf("ID PCoord %d, pCoord = %d , ps = %d \n", idPCoord, pCoord[0], ps[0]);


                    if(idOriginal == 999999 && pCoord[0] < idOriginal){
                        nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                        idOriginal = pCoord[0];
                    }
                    else if(idPCoord < idOriginal){
                        printf("idOriginal changed %d \n", idOriginal);
                        nn_source.grayValueMap[ps[1]][ps[0]] = idPCoord;
                        idOriginal = idPCoord;
                        printf("ID PCoord %d, pCoord = %d , ps = %d \n", idPCoord, pCoord[0], ps[0]);

                        printf("idOriginal changed %d \n", idOriginal);

                    }
                    else if(idPCoord > idOriginal){
                        nn_source.grayValueMap[pCoord[1]][pCoord[0]] = idOriginal; //QWB other links set the minimum ID
                    }

                    size_t nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                    for (size_t pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode;
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        PointCoord pCoordLink(-1, -1);
                        pCoordLink[0] = (int)pLinkOfNode[0];
                        pCoordLink[1] = (int)pLinkOfNode[1];


                        // compare if the current pCoord is already be teached
                        bool traversed = 0;
                        for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                            PointCoord pLinkTemp(-1, -1);
                            pLinkTemp = nodeAlreadyTraversed.bCell[k];
                            if (pCoordLink == pLinkTemp)
                                traversed = 1;
                        }
                        if (traversed)
                            continue;
                        else
                            (*tabD_).insert(pCoordLink);

                    }

                }

            }

            (*tabO_).clearLinks();//qiao note, here if we just clear the numlinks, the buffer will always has values

            swapClass4(&tabO_, &tabD_);// ok for cpu, correct

        }// while
    } // operate
};



//! QWB 241016 add operation to mark independent super vertice for mst from node level
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct operateMarkMstSuperVerticesAndMinRadius{

    DEVICE_HOST operateMarkMstSuperVerticesAndMinRadius(){}

    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        int idOriginal = nn_source.grayValueMap[ps[1]][ps[0]];

        float minRadiusOri = nn_source.minRadiusMap[ps[1]][ps[0]];

        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_VECTOR_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_VECTOR_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;

        while((*tabO_).numLinks > 0){

            for (size_t i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                // compare if the current pCoord is already be teached
                bool traversed = 0;
                for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = nodeAlreadyTraversed.bCell[k];
                    if (pCoord == pLinkTemp)
                        traversed = 1;
                }
                if (traversed)
                    continue;
                else{

                    nodeAlreadyTraversed.insert(pCoord);

                    if(nn_source.fixedMap[pCoord[1]][pCoord[0]] && minRadiusOri > nn_source.minRadiusMap[pCoord[1]][pCoord[0]]){ // it is a winner node

                        minRadiusOri = nn_source.minRadiusMap[pCoord[1]][pCoord[0]];
                        nn_source.minRadiusMap[ps[1]][ps[0]] = minRadiusOri;

                    }
                    else{
                        nn_source.minRadiusMap[pCoord[1]][pCoord[0]] = minRadiusOri;
                    }

                    int idPCoord = nn_source.grayValueMap[pCoord[1]][pCoord[0]];
                    //                                        printf("link of ps, pCoord = %d \n", pCoord[0]);
                    //                                        printf("old id ps = %d \n", nn_source.grayValueMap[ps[1]][ps[0]]);
                    //                                        printf("old id pCoor = %d \n", nn_source.grayValueMap[pCoord[1]][pCoord[0]]);
                    if(idOriginal == 999999 && pCoord[0] < idOriginal){
                        nn_source.grayValueMap[ps[1]][ps[0]] = pCoord[0];
                        idOriginal = pCoord[0];
                        // minRadius at the first iteration, should all be 0

                        //                                                printf("         id < ps, new id = %d \n", nn_source.grayValueMap[ps[1]][ps[0]]);
                    }
                    else if(idPCoord < idOriginal){
                        nn_source.grayValueMap[ps[1]][ps[0]] = idPCoord;
                        idOriginal = idPCoord;

                    }
                    else if(idPCoord > idOriginal){
                        nn_source.grayValueMap[pCoord[1]][pCoord[0]] = idOriginal; //QWB other links set the minimum ID
                    }
                    else if(pCoord[0] != idOriginal){
                        nn_source.activeMap[pCoord[1]][pCoord[0]] = 0; // no use 261016 QWB refresh activeMap to improve step3, but it is difficult here
                    }


                    size_t nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                    for (size_t pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode;
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        PointCoord pCoordLink(-1, -1);
                        pCoordLink[0] = (int)pLinkOfNode[0];
                        pCoordLink[1] = (int)pLinkOfNode[1];


                        // compare if the current pCoord is already be teached
                        bool traversed = 0;
                        for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                            PointCoord pLinkTemp(-1, -1);
                            pLinkTemp = nodeAlreadyTraversed.bCell[k];
                            if (pCoordLink == pLinkTemp)
                                traversed = 1;
                        }
                        if (traversed)
                            continue;
                        else
                            (*tabD_).insert(pCoordLink);

                    }

                }

            }

            (*tabO_).clearLinks();//qiao note, here if we just clear the numlinks, the buffer will always has values

#ifdef CUDA_CODE

            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else


            swapClass4(&tabO_, &tabD_);// ok for cpu, correct
#endif


        }// while

    } // operate

};





//! QWB 260916 add operation to mark independent super vertice
//! every node acts as a starting node, and compare all its links one by one to save the minum index
//! only 2-connected nn-links works in this way
template <class Cell>
struct operateMarkSuperVerticesGpu{

    DEVICE_HOST operateMarkSuperVerticesGpu(){}
    DEVICE_HOST operateMarkSuperVerticesGpu(size_t step_){ step =  step_;}

    size_t step;

    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps){

        printf("Operator mark superVertices \n");
        if(nn_source.networkLinks[ps[1]][ps[0]].numLinks == 2){

            PointCoord pLinkOfNode;
            nn_source.networkLinks[ps[1]][ps[0]].get(0, pLinkOfNode);
            PointCoord pco(-1, -1);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];

            nn_source.grayValueMap[ps[1]][ps[0]] = ps[0];
            nn_source.grayValueMap[pco[1]][pco[0]] = ps[0];
            nn_source.fixedMap[pco[1]][pco[0]] = 1;

            PointCoord pco2(-1, -1);
            nn_source.networkLinks[pco[1]][pco[0]].get(0, pLinkOfNode);
            pco2[0] = (int)pLinkOfNode[0];
            pco2[1] = (int)pLinkOfNode[1];

            if(pco2 == ps){
                nn_source.networkLinks[pco[1]][pco[0]].get(1, pLinkOfNode);
                pco2[0] = (int)pLinkOfNode[0];
                pco2[1] = (int)pLinkOfNode[1];
            }

            PointCoord pco2Avant;
            pco2Avant = pco;

            while(pco2 != ps){

                PointCoord pco3(-1, -1);
                nn_source.networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode);
                pco3[0] = (int)pLinkOfNode[0];
                pco3[1] = (int)pLinkOfNode[1];

                if(pco3 == pco2Avant){
                    nn_source.networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode);
                    pco3[0] = (int)pLinkOfNode[0];
                    pco3[1] = (int)pLinkOfNode[1];

                }

                if(pco3 == pco2Avant){
                    //                    cout << "open circle . ps: " << ps[0] << ", " <<  ps[1] << endl;
                    break;
                }

                nn_source.grayValueMap[pco2[1]][pco2[0]] = ps[0];
                nn_source.fixedMap[pco2[1]][pco2[0]] = 1;

                pco2Avant = pco2;
                pco2 = pco3;

            }

        }

    } // operate


};

//! QWB 051016 add operation to refresh independent superVertices to connect them into a graph
//! begin with ps, find the shortest distance, ps_idSource, pco_idNeighbor,
//! superVertice : densityMap stores the shortest distance,
template <class Cell>
struct operateConnectSuperVerticesToGraph{

    operateConnectSuperVerticesToGraph(){}
    operateConnectSuperVerticesToGraph(size_t step_){ step =  step_;}

    size_t step;

    template <class NetLinkPointCoord>
    void operate(NetLinkPointCoord& nn_source, PointCoord ps, NetLinkPointInt& superVertices){

        vector<superVertexCpu> neighborComponentVector; // for each component in superVertices, find all its different neighborId
        neighborComponentVector.clear();

        printf("\n Operator refresh superVertices \n");
        if(nn_source.networkLinks[ps[1]][ps[0]].numLinks == 2){

            PointCoord pLinkOfNode;
            nn_source.networkLinks[ps[1]][ps[0]].get(0, pLinkOfNode);
            PointCoord pco(-1, -1);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];


            //            nn_source.grayValueMap[ps[1]][ps[0]] = ps[0];
            //            nn_source.grayValueMap[pco[1]][pco[0]] = ps[0];
            //            nn_source.fixedMap[pco[1]][pco[0]] = 1;


            // insert the first component into vector
            insertNeighborIntoVector(nn_source, ps, superVertices, neighborComponentVector);
            insertNeighborIntoVector(nn_source, pco, superVertices, neighborComponentVector);

            PointCoord pco2(-1, -1);
            nn_source.networkLinks[pco[1]][pco[0]].get(0, pLinkOfNode);
            pco2[0] = (int)pLinkOfNode[0];
            pco2[1] = (int)pLinkOfNode[1];

            if(pco2 == ps){
                nn_source.networkLinks[pco[1]][pco[0]].get(1, pLinkOfNode);
                pco2[0] = (int)pLinkOfNode[0];
                pco2[1] = (int)pLinkOfNode[1];
            }

            PointCoord pco2Avant;
            pco2Avant = pco;

            while(pco2 != ps){

                PointCoord pco3(-1, -1);
                nn_source.networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode);
                pco3[0] = (int)pLinkOfNode[0];
                pco3[1] = (int)pLinkOfNode[1];

                if(pco3 == pco2Avant){
                    nn_source.networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode);
                    pco3[0] = (int)pLinkOfNode[0];
                    pco3[1] = (int)pLinkOfNode[1];

                }

                if(pco3 == pco2Avant){
                    //                    cout << "open circle . ps: " << ps[0] << ", " <<  ps[1] << endl;
                    break;
                }

                insertNeighborIntoVector(nn_source, pco2, superVertices, neighborComponentVector);

                pco2Avant = pco2;
                pco2 = pco3;

            }

        }
        //        else
        //            cout << "Error mark super vertices, no links or bigger than 2. " << endl;

        // test
        for(int i = 0; i < neighborComponentVector.size(); i++){

            superVertexCpu oneNeighborComponent;
            oneNeighborComponent = neighborComponentVector[i];
        }


    } // operate


    //! wb.Q insert neighbor into vector
    template <class NetLinkPointCoord>
    void insertNeighborIntoVector(NetLinkPointCoord& nn_source, PointCoord ps,
                                  NetLinkPointInt& superVertices,
                                  vector<superVertexCpu>& neighborComponentVector){

        PointCoord closestNode = nn_source.correspondenceMap[ps[1]][ps[0]]; //every node in current component should be registrated its correspondence and compare its distance
        int closestNodeID = nn_source.grayValueMap[closestNode[1]][closestNode[0]];
        cout << "node treated " << ps[0] << ", " << ps[1] << endl;
        cout << "ps id " << nn_source.grayValueMap[ps[1]][ps[0]] << endl;
        cout << "ps corres id : " << closestNodeID << ", ps corres node " << closestNode[0]  << endl;
        cout << "ps distance " << nn_source.densityMap[ps[1]][ps[0]] << endl;

        if(closestNodeID == -1)
            cout << "attention ,  do not find neighbor node for ps " << ps[0] << ", " << ps[1] << endl;
        PointCoord centralNeighborNode;
        centralNeighborNode[0] = closestNodeID;
        centralNeighborNode[1] = 0; // for 1D TSP problem

        superVertexCpu oneNeighborComponent;
        oneNeighborComponent.centralNode = centralNeighborNode;
        oneNeighborComponent.minCoupleDistance = nn_source.densityMap[ps[1]][ps[0]];
        oneNeighborComponent.couple1 = ps;
        oneNeighborComponent.couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];

        // add first, here may exist bug
        if(neighborComponentVector.size() == 0){ // add first node into
            neighborComponentVector.push_back(oneNeighborComponent);
            cout << " " << endl;
        }
        else if(neighborComponentVector.size() > 0){
            bool existComponent = 0;
            for(int i = 0; i < neighborComponentVector.size(); i++){
                // treat neighbor component with same id
                if(centralNeighborNode == neighborComponentVector[i].centralNode){
                    cout << " " << endl;
                    existComponent = 1;
                    if(oneNeighborComponent.minCoupleDistance < neighborComponentVector[i].minCoupleDistance){
                        neighborComponentVector[i].minCoupleDistance = oneNeighborComponent.minCoupleDistance;
                        neighborComponentVector[i].couple1 = oneNeighborComponent.couple1;
                        neighborComponentVector[i].couple2 = oneNeighborComponent.couple2;

                    }
                }
            }
            // add different id component
            if (!existComponent){

                neighborComponentVector.push_back(oneNeighborComponent);
            }

        }

    } // insertNeighborNodeIntoVector

};


//! QWB 051016 add operation to only connect the shortest outgoing distance
//! begin with ps, find the shortest distance, ps_idSource, pco_idNeighbor,
//! superVertice : densityMap stores the shortest distance,
template <class Cell>
struct searchNeighborComponentsWithShortestEdge{

    DEVICE_HOST searchNeighborComponentsWithShortestEdge(){}
    DEVICE_HOST searchNeighborComponentsWithShortestEdge(size_t step_){ step =  step_;}

    size_t step;

    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(NetLinkPointCoord& nn_source, PointCoord ps,
                             NetLinkPointInt& superVertices, PointCoord& couple1, PointCoord& couple2,
                             NetLinkPointCoord& nn_mst){

        superVertexCpu neighborClosestComponent; // for each component in superVertices, find all its different neighborId

        printf("\n Operator refresh superVertices \n");
        if(nn_source.networkLinks[ps[1]][ps[0]].numLinks == 2){

            PointCoord pLinkOfNode;
            nn_source.networkLinks[ps[1]][ps[0]].get(0, pLinkOfNode);
            PointCoord pco(-1, -1);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];

            // insert the first component into neighborClosestComponent
            PointCoord closestNode = nn_source.correspondenceMap[ps[1]][ps[0]];
            if(closestNode[0] != -1){
                PointCoord centralNeighborNode;
                centralNeighborNode[0] = nn_source.grayValueMap[closestNode[1]][closestNode[0]];;
                centralNeighborNode[1] = 0; // for 1D TSP problem
                neighborClosestComponent.minCoupleDistance = nn_source.densityMap[ps[1]][ps[0]];
                neighborClosestComponent.centralNode = centralNeighborNode;
                neighborClosestComponent.couple1 = ps;
                neighborClosestComponent.couple2 = closestNode;
                neighborClosestComponent.id2 = nn_source.grayValueMap[closestNode[1]][closestNode[0]];
            }

            // insert the second component into neighborClosestComponent
            findShortestLexicoEdgeBetweenComponents(nn_source, pco, superVertices, neighborClosestComponent);

            PointCoord pco2(-1, -1);
            nn_source.networkLinks[pco[1]][pco[0]].get(0, pLinkOfNode);
            pco2[0] = (int)pLinkOfNode[0];
            pco2[1] = (int)pLinkOfNode[1];

            if(pco2 == ps){
                nn_source.networkLinks[pco[1]][pco[0]].get(1, pLinkOfNode);
                pco2[0] = (int)pLinkOfNode[0];
                pco2[1] = (int)pLinkOfNode[1];
            }

            PointCoord pco2Avant;
            pco2Avant = pco;

            while(pco2 != ps){

                PointCoord pco3(-1, -1);
                nn_source.networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode);
                pco3[0] = (int)pLinkOfNode[0];
                pco3[1] = (int)pLinkOfNode[1];

                if(pco3 == pco2Avant){
                    nn_source.networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode);
                    pco3[0] = (int)pLinkOfNode[0];
                    pco3[1] = (int)pLinkOfNode[1];

                }

                if(pco3 == pco2Avant){
                    break;
                }

                findShortestLexicoEdgeBetweenComponents(nn_source, pco2, superVertices, neighborClosestComponent);

                pco2Avant = pco2;
                pco2 = pco3;

            }

        }

        // build mst
        if(neighborClosestComponent.minCoupleDistance != INFINITY){
            printf("minDistance %f, couple1 %d, couple2 %d \n", neighborClosestComponent.minCoupleDistance,
                   neighborClosestComponent.couple1[0], neighborClosestComponent.couple2[0]);
            couple1 = neighborClosestComponent.couple1;
            couple2 = neighborClosestComponent.couple2;

            PointCoord pTemp1(-1, -1);
            PointCoord pTemp2(-1, -1);
            pTemp1 = couple1;
            pTemp2 = couple2;
            nn_mst.networkLinks[couple1[1]][couple1[0]].insert(pTemp2);
            nn_mst.networkLinks[couple2[1]][couple2[0]].insert(pTemp1);

        }

        //! transfer les infor of neighbor closest component to current NN-Link-component


    } // operate


    template <class NetLinkPointCoord>
    DEVICE_HOST void findShortestLexicoEdgeBetweenComponents(NetLinkPointCoord& nn_source, PointCoord ps,
                                                             NetLinkPointInt& superVertices,
                                                             superVertexCpu& neighborClosestComponent){

        PointCoord closestNode = nn_source.correspondenceMap[ps[1]][ps[0]]; //every node in current component should be registrated its correspondence and compare its distance

        if(closestNode[0] != -1){ // the node ps find its closest neighbor node with diff id

            int closestNodeID = nn_source.grayValueMap[closestNode[1]][closestNode[0]];
            float distance = nn_source.densityMap[ps[1]][ps[0]];

            if(closestNodeID == INFINITY)
                printf("attention ,  do not find neighbor node for ps, %d, %d ", ps[0], ps[1]);

            PointCoord centralNeighborNode;
            centralNeighborNode[0] = closestNodeID;
            centralNeighborNode[1] = 0; // for 1D TSP problem

            if(distance < neighborClosestComponent.minCoupleDistance){
                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.centralNode = centralNeighborNode;
                neighborClosestComponent.couple1 = ps;
                neighborClosestComponent.couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];
                neighborClosestComponent.id2 = closestNodeID;
            }
            // lexico order
            if(distance == neighborClosestComponent.minCoupleDistance
                    && closestNodeID < neighborClosestComponent.id2){
                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.centralNode = centralNeighborNode;
                neighborClosestComponent.couple1 = ps;
                neighborClosestComponent.couple2 = nn_source.correspondenceMap[ps[1]][ps[0]];
                neighborClosestComponent.id2 = closestNodeID;

            }

        }

    } // insertNeighborNodeIntoVector

};


//! QWB 031216 add compact graph kernel 2
//! any nn-links can work in this way, only the winner node works
template <class Cell>
struct searchMstShortestOutgoingEdge_final{

    DEVICE_HOST searchMstShortestOutgoingEdge_final(){}

    template <class NetLinkPointCoord>
    DEVICE bool search(NetLinkPointCoord& nn_source, PointCoord ps,
                       PointCoord& couple1, PointCoord& couple2){

        bool ret = 0;
        PointCoord pInitial(-1, -1);

        superVertexData neiClosCompo; // for each component in superVertices, find all its different neighborId


        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_LINK_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_LINK_SIZE);

        tabO.insert(ps);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;
        BufferLinkPointCoord* nodeAlreadyTraversed_ = &nodeAlreadyTraversed;

        while((*tabO_).numLinks > 0){

            for (int i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                // begin operations
                findShortestLexicoEdgeBetweenComponents(nn_source, pCoord, neiClosCompo);
                // end operations

                int nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;
                for (int pLink = 0; pLink < nLinks; pLink++){

                    PointCoord pLinkOfNode;
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

            (*nodeAlreadyTraversed_).clearLinks();
#ifdef CUDA_CODE

            *nodeAlreadyTraversed_ = *tabO_;
            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else


            swapClass4(&nodeAlreadyTraversed_, &tabO_);
            swapClass4(&tabO_, &tabD_);
#endif
        }// while

        // build mst, only the ps == winner node, then export the winner couple to do reconnection step
        if(neiClosCompo.minCoupleDistance != INFINITY
                && neiClosCompo.minCoupleDistance != 999999
                && neiClosCompo.couple1 == ps )
        {
            couple1 = neiClosCompo.couple1;
            //            couple2 = neiClosCompo.couple2;
            //            printf("findMin 2, ps %d, %d, couple1 %d, %d \n", ps[0], ps[1], couple1[0], couple1[1]);

            nn_source.fixedMap[couple1[1]][couple1[0]] = 1; //  mark the winner node for current component, used in step4
            ret = 1;
        }

        return ret;
    } // operate



    //! wb.Q find shortest lexico edge between components
    template <class NetLinkPointCoord>
    DEVICE_HOST void findShortestLexicoEdgeBetweenComponents(NetLinkPointCoord& nn_source,
                                                             PointCoord& ps_,
                                                             superVertexData& neighborClosestComponent)
    {

        PointCoord fammeNode = nn_source.correspondenceMap[ps_[1]][ps_[0]];

        if(fammeNode[0] != -1){

            int fammeNodeID = nn_source.grayValueMap[fammeNode[1]][fammeNode[0]];
            double distance = nn_source.densityMap[ps_[1]][ps_[0]];


            if(fammeNodeID == INFINITY)
                printf("attention ,  do not find neighbor node for ps_, %d, %d ", ps_[0], ps_[1]);

            if(distance < neighborClosestComponent.minCoupleDistance){
                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.couple1 = ps_;
                neighborClosestComponent.couple2 = fammeNode;
                neighborClosestComponent.id2 = fammeNodeID;// QWB compare big ID
                neighborClosestComponent.id1 = ps_[0]; // QWB compare small id of curent ps
            }

            else if(distance == neighborClosestComponent.minCoupleDistance){

                if(fammeNodeID < neighborClosestComponent.id2){

                    neighborClosestComponent.minCoupleDistance = distance;
                    neighborClosestComponent.couple1 = ps_;
                    neighborClosestComponent.couple2 = fammeNode;
                    neighborClosestComponent.id2 = fammeNodeID;
                    neighborClosestComponent.id1 = ps_[0];

                }
                else if((fammeNodeID == neighborClosestComponent.id2)
                        && ps_[0] < neighborClosestComponent.id1 ){

                    neighborClosestComponent.minCoupleDistance = distance;
                    neighborClosestComponent.couple1 = ps_;
                    neighborClosestComponent.couple2 = fammeNode;
                    neighborClosestComponent.id2 = fammeNodeID;
                    neighborClosestComponent.id1 = ps_[0];

                }

            }

        }

    }

};


//! QWB 141016 add operator
//! begin with ps, find the shortest distance, ps_idSource, pco_idNeighbor,
//! work for any network
template <class Cell>
struct searchMstShortestOutgoingEdge{

    DEVICE_HOST searchMstShortestOutgoingEdge(){}

    template <class NetLinkPointCoord>
    DEVICE bool search(NetLinkPointCoord& nn_source, PointCoord ps,
                       PointCoord& couple1, PointCoord& couple2){
        bool ret = 0;

        superVertexData neiClosCompo; // for each component in superVertices, find all its different neighborId

        BufferLinkPointCoord nodeAlreadyTraversed;
        nodeAlreadyTraversed.init(MAX_VECTOR_SIZE);

        BufferLinkPointCoord tabO;
        tabO.init(MAX_LINK_SIZE);
        BufferLinkPointCoord tabD;
        tabD.init(MAX_LINK_SIZE);

        tabO.insert(ps);

        PointCoord pLinkInitial(-1, -1);

        BufferLinkPointCoord* tabO_ = &tabO;
        BufferLinkPointCoord* tabD_ = &tabD;

        while((*tabO_).numLinks > 0){

            for (size_t i = 0; i < (*tabO_).numLinks; i++){

                PointCoord pCoord = (*tabO_).bCell[i];

                // compare if the current pCoord is already be teached
                bool traversed = 0;
                for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = nodeAlreadyTraversed.bCell[k];
                    if (pCoord == pLinkTemp)
                        traversed = 1;
                }
                if (traversed)
                    continue;
                else
                {

                    nodeAlreadyTraversed.insert(pCoord);

                    // begin operations
                    findShortestLexicoEdgeBetweenComponents(nn_source, pCoord, neiClosCompo);
                    // end operations

                    size_t nLinks = nn_source.networkLinks[pCoord[1]][pCoord[0]].numLinks;

                    for (size_t pLink = 0; pLink < nLinks; pLink++){

                        PointCoord pLinkOfNode(-1, -1);
                        nn_source.networkLinks[pCoord[1]][pCoord[0]].get(pLink, pLinkOfNode);

                        if(pLinkOfNode == pLinkInitial)
                            continue;
                        else
                        {

                            PointCoord pCoordLink(-1, -1);
                            pCoordLink[0] = (int)pLinkOfNode[0];
                            pCoordLink[1] = (int)pLinkOfNode[1];

                            // compare if the current pCoord is already be teached
                            bool traversed = 0;
                            for (size_t k = 0; k < nodeAlreadyTraversed.numLinks; k ++){
                                PointCoord pLinkTemp(-1, -1);
                                pLinkTemp = nodeAlreadyTraversed.bCell[k];
                                if (pCoordLink == pLinkTemp)
                                    traversed = 1;
                            }
                            if (traversed)
                                continue;

                            else
                            {
                                (*tabD_).insert(pCoordLink);
                                printf("findMin 2, ps %d, %d, pCoordLink %d, %d \n", ps[0], ps[1], pCoordLink[0], pCoordLink[1]);
                            }
                        }

                    }

                }

            }

            (*tabO_).clearLinks();//qiao note, here if we just clear the numlinks, the buffer will always has values

#ifdef CUDA_CODE
            // ok for gpu
            *tabO_ = *tabD_;
            (*tabD_).clearLinks();
#else

            swapClass4(&tabO_, &tabD_);// ok for cpu, correct
#endif

        }// while

        // build mst, only the ps == winner node, then export the winner couple to do reconnection step
        if(neiClosCompo.minCoupleDistance != INFINITY
                && neiClosCompo.minCoupleDistance != 999999
                && neiClosCompo.couple1 == ps )
        {


            couple1 = neiClosCompo.couple1;
            //            couple2 = neiClosCompo.couple2;
            printf("findMin 2, ps %d, %d, couple1 %d, %d \n", ps[0], ps[1], couple1[0], couple1[1]);

            nn_source.fixedMap[ps[1]][ps[0]] = 1; //  mark the winner node for current component, used in step4
            ret = 1;
        }

        return ret;
    } // search shortest outgoing edges


    template <class NetLinkPointCoord>
    DEVICE void findShortestLexicoEdgeBetweenComponents(NetLinkPointCoord& nn_source,
                                                        PointCoord& ps_,
                                                        superVertexData& neighborClosestComponent)
    {

        PointCoord fammeNode = nn_source.correspondenceMap[ps_[1]][ps_[0]];

        if(fammeNode[0] != -1){

            int fammeNodeID = nn_source.grayValueMap[fammeNode[1]][fammeNode[0]];
            double distance = nn_source.densityMap[ps_[1]][ps_[0]];


            if(fammeNodeID == INFINITY)
                printf("attention ,  do not find neighbor node for ps_, %d, %d ", ps_[0], ps_[1]);

            if(distance < neighborClosestComponent.minCoupleDistance){
                neighborClosestComponent.minCoupleDistance = distance;
                neighborClosestComponent.couple1 = ps_;
                neighborClosestComponent.couple2 = fammeNode;
                neighborClosestComponent.id2 = fammeNodeID;// QWB compare big ID
                neighborClosestComponent.id1 = ps_[0]; // QWB compare small id of curent ps
            }

            else if(distance == neighborClosestComponent.minCoupleDistance){

                if(fammeNodeID < neighborClosestComponent.id2){

                    neighborClosestComponent.minCoupleDistance = distance;
                    neighborClosestComponent.couple1 = ps_;
                    neighborClosestComponent.couple2 = fammeNode;
                    neighborClosestComponent.id2 = fammeNodeID;
                    neighborClosestComponent.id1 = ps_[0];

                }
                else if((fammeNodeID == neighborClosestComponent.id2)
                        && ps_[0] < neighborClosestComponent.id1 ){

                    neighborClosestComponent.minCoupleDistance = distance;
                    neighborClosestComponent.couple1 = ps_;
                    neighborClosestComponent.couple2 = fammeNode;
                    neighborClosestComponent.id2 = fammeNodeID;
                    neighborClosestComponent.id1 = ps_[0];

                }

            }

        }

    }

};




// 260516 qiao add to generate neighbor links, repeated links can also be deleted
template <class Cell,
          class NIter>
struct OperateGeneratorNeiborLinksAdaptor {
    
    size_t rayon;
    
    DEVICE_HOST OperateGeneratorNeiborLinksAdaptor(size_t r) : rayon(r){}
    
    DEVICE_HOST OperateGeneratorNeiborLinksAdaptor(){}
    
    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(Cell& cell,  NetLinkPointCoord& nn_source, PointCoord p_source) {
        
        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {
            
            //            nn_source.fixedMap[p_source[1]][p_source[0]] = 1;// qiao for test
            
            NIter ni(p_source, 1, rayon);// neighbor links do not contain the father node
            //                    ni.init(); // father node is included its links
            
            PointCoord pCoord;
            do {
                if (ni.getCurrentDistance() > rayon)
                    break;
                
                pCoord = ni.getNodeIncr();
                
                size_t nLinksTemp = nn_source.networkLinks[p_source[1]][p_source[0]].numLinks;
                
                if (pCoord[0] >= 0 && pCoord[0] < nn_source.adaptiveMap.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < nn_source.adaptiveMap.getHeight()) {
                    
                    
                    PointCoord p2dInt(0, 0);
                    p2dInt[1] = (float)pCoord[1];
                    p2dInt[0] = (float)pCoord[0];
                    
                    if (nLinksTemp == 0){
                        nn_source.networkLinks[p_source[1]][p_source[0]].insert(p2dInt);
                        nn_source.fixedLinks[p_source[1]][p_source[0]] = 2;
                    }
                    
                    else
                    {
                        bool sameLink = 0;
                        for (int i = 0;  i < nLinksTemp; i++){
                            
                            PointCoord pLinkExisting(0, 0);
                            nn_source.networkLinks[p_source[1]][p_source[0]].get(i, pLinkExisting);
                            
                            if (p2dInt == pLinkExisting) {
                                sameLink= 1;
                                break;
                            }
                        }
                        if (!sameLink){
                            nn_source.networkLinks[p_source[1]][p_source[0]].insert(p2dInt);
                        }
                    }
                }
            } while (ni.nextNodeIncr());
        }
        
    }//operate
    
};


// wb.Q compare the difference between node's links, select the most least difference, and compare the seconde most least difference
template <class Cell,
          class Distance>
struct OperateDeleteUnsimilarLinksAdaptor {

    
    DEVICE_HOST OperateDeleteUnsimilarLinksAdaptor(){}
    
    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(Cell& cell,  NetLinkPointCoord& nn_source, PointCoord p_source, float diff) {

        //! getAdaptor each node, traversing its links, compare the ditance between father node and links
        nn_source.fixedLinks[p_source[1]][p_source[0]] = 2;

        Distance dis;

        size_t numLinksFather  =  nn_source.networkLinks[p_source[1]][p_source[0]].numLinks;
        //        cout << "psource " << p_source[0] << ", " << p_source[1] << endl;

        for(int i = 0; i < numLinksFather; i++){

            PointCoord pSon_(0, 0);
            PointCoord pSon(0, 0);
            nn_source.networkLinks[p_source[1]][p_source[0]].get(i, pSon_);
            pSon[0] = (int)pSon_[0];
            pSon[1] = (int)pSon_[1];

            //            cout << "pSon " << pSon[0] << ", " << pSon[1] << endl;

            if(pSon[0] == -1 || pSon[1] == -1)
                continue;
            else
            {

                float difference = dis(pSon, p_source, nn_source, nn_source);

                //                cout << "difference " << difference << endl;

                if(difference > diff){

                    PointCoord pLinkNull(-1, -1);

                    // delete this link of father node
                    nn_source.networkLinks[p_source[1]][p_source[0]].bCell[i] = pLinkNull;
                    nn_source.activeMap[p_source[1]][p_source[0]] = 1;

                    // traversing links in pSon, find p_source, delete
                    size_t numLinksSon = nn_source.networkLinks[pSon[1]][pSon[0]].numLinks;

                    for(int j = 0; j < numLinksSon; j++){

                        PointCoord pSonSon_(0, 0);
                        PointCoord pSonSon(0, 0);
                        nn_source.networkLinks[pSon[1]][pSon[0]].get(j, pSonSon_);
                        pSonSon[0] = pSonSon_[0];
                        pSonSon[1] = pSonSon_[1];

                        if(pSonSon[1] == p_source[1] && pSonSon[0] == p_source[0]){

                            nn_source.networkLinks[pSon[1]][pSon[0]].bCell[j] = pLinkNull;
                            nn_source.activeMap[pSon[1]][pSon[0]] = 1;
                        }
                    }
                }
            }
        }
    }//operate
    
};



//! QWB operate after check searching neighbor nodes with different id_superVertices
template <class Cell,
          class Distance>
struct OperateNeighborNodeWithDiffId {


    DEVICE_HOST OperateNeighborNodeWithDiffId(){}

    template <class NetLinkPointCoord>
    DEVICE_HOST void operate(Cell& cell,  NetLinkPointCoord& nn_source,NetLinkPointCoord& nn_cible,
                             PointCoord p_source, PointCoord p_cible) {

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()
                && p_cible[0] >= 0 && p_cible[0] < nn_cible.adaptiveMap.getWidth()
                && p_cible[1] >= 0 && p_cible[1] < nn_cible.adaptiveMap.getHeight()
                ){
            nn_source.correspondenceMap[p_source[1]][p_source[0]] = p_cible;
        }

    }//operate

};


////! WB.Q add to count time
////! copy source code from https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
//void StartCounter(double& PCFreq, __int64& CounterStart)
//{

//    LARGE_INTEGER li;
//    if(!QueryPerformanceFrequency(&li))
//        cout << "QueryPerformanceFrequency failed!\n";

//    PCFreq = double(li.QuadPart)/1000.0;// millisecond
//    //    PCFreq = double(li.QuadPart); // s
//    //    PCFreq = double(li.QuadPart)/1000000.0; // microsecond

//    QueryPerformanceCounter(&li);
//    CounterStart = li.QuadPart;
//}
//double GetCounter(double PCFreq, __int64 CounterStart)
//{
//    LARGE_INTEGER li;
//    QueryPerformanceCounter(&li);
//    return double(li.QuadPart-CounterStart)/PCFreq;
//}




////!QWB add to test changement in grid
//template<class type>
//int testGridNum(Grid<type> testGrid){
    
//    int numTest = 0;
//    for (int j = 0; j < testGrid.height; j++ )
//        for (int i = 0; i < testGrid.width; i++)
//        {
//            if (testGrid[j][i])
//                numTest += 1;// testGrid[j][i];
//        }
//    return numTest;
    
//}

//!QWB add to test changement in grid
template<class type>
int testGridNumDetail(Grid<type> testGrid){

    int numTest = 0;
    for (int j = 0; j < testGrid.height; j++ )
        for (int i = 0; i < testGrid.width; i++)
        {
            if (testGrid[j][i])
                numTest += testGrid[j][i];
        }
    return numTest;

}

//!QWB add to test changement in grid
template<class type>
int testGridNumDetailBigger1(Grid<type> testGrid){

    int numTest = 0;
    for (int j = 0; j < testGrid.height; j++ )
        for (int i = 0; i < testGrid.width; i++)
        {
            if (testGrid[j][i] > 1)
                numTest += 1;
        }
    return numTest;

}

}//namespace operators

#endif // ADAPTATOR_BASICS_H

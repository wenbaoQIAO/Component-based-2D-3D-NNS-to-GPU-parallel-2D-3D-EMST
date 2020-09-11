#ifndef NEURALNETLINKS_H
#define NEURALNETLINKS_H
/*
 ***************************************************************************
 *
 * Author : Wenbao. Qiao
 * Creation date : Apil. 2016
 * This file contains the standard distributed graph representation with independent adjacency list
 * assigned to each vertex of the graph, namely an adjacency list where each vertex only possesses
 * a collection of its adjacency neighboring vertices.
 * It naturally follows that the graph is doubly linked since each node of a given edge has a link
 * in the node's adjacency list towards to the connected node. We call it as doubly linked vertex list (DLVL)
 * in order to distinguish DLVL from doubly linked list (DLL) or doubly connected edge list (DCEL).
 * For self-organizing irregular network applications, it is also called as "Neural Network Links".
 *
 * If you use this source codes, please reference our publication:
 * Qiao, Wen-bao, and Jean-charles Créput.
 * "Massive Parallel Self-organizing Map and 2-Opt on GPU to Large Scale TSP."
 * International Work-Conference on Artificial Neural Networks. Springer, Cham, 2017.
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <algorithm>
#include <sm_20_atomic_functions.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
//#include <boost/geometry/geometries/point.hpp>


#include "macros_cuda.h"
#include "Node.h"
#include "NeuralNet.h"
#include "Objectives.h"
#include "GridOfNodes.h"


//! WB.Q add
#include "NetLinks.h"
#define useDerive 1 //1 class nnLinks derive from nn
#define infinity 999999
#define decodePiK 1
#define initialPrepareValue -1
#define Viewer 0 //1 do not use viewer, use CellularMatrix,  0 use viewer, need to verify other two places in NeuralNet.h and NetLinks.h


using namespace std;
using namespace components;


// wenbao.Qiao add read/write networkLinks
#define NNG_SFX_CELLULARMATRIX_LINK_IMAGE  ".cmLinks"
#define NN_SFX_ADAPTIVE_ORIGINALMAP_TSP  ".pointsOri"


namespace components
{

//! Wenbao Qiao 090516 add irregular graph representation with doubly linked vertex list (DLVL) data structure
template <class BufferLinkDimension, class Point>
class NeuralNetLinks : public NeuralNet<Point, GLfloat>
{

public:

    Grid<BufferLinkDimension > networkLinks;
    Grid<int> fixedLinks;
    Grid<PointCoord> correspondenceMap;
    Grid<GLfloat> minRadiusMap;
    Grid<Point> adaptiveMapOri;
    Grid<float> sizeOfComponentMap;

public:
    DEVICE_HOST NeuralNetLinks() {}

    DEVICE_HOST NeuralNetLinks(int nnW, int nnH):
        networkLinks(nnW, nnH),
        fixedLinks(nnW, nnH),
        correspondenceMap(nnW, nnH),
        minRadiusMap(nnW, nnH),
        adaptiveMapOri(nnW, nnH),
        sizeOfComponentMap(nnW, nnH),
        NeuralNet<Point, GLfloat>(nnW, nnH)
    { }

    void resize(int w, int h){
        networkLinks.resize(w, h);
        fixedLinks.resize(w, h);
        correspondenceMap.resize(w, h);
        minRadiusMap.resize(w, h);
        adaptiveMapOri.resize(w, h);
        sizeOfComponentMap.resize(w, h);
        NeuralNet<Point, GLfloat>::resize(w, h);
    }

    void freeMem(){
        networkLinks.freeMem();
        fixedLinks.freeMem();
        correspondenceMap.freeMem();
        minRadiusMap.freeMem();
        adaptiveMapOri.freeMem();
        sizeOfComponentMap.freeMem();
        NeuralNet<Point, GLfloat>::freeMem();
    }


    void gpuResize(int w, int h){
        networkLinks.gpuResize(w, h);
        fixedLinks.gpuResize(w, h);
        correspondenceMap.gpuResize(w, h);
        minRadiusMap.gpuResize(w, h);
        adaptiveMapOri.gpuResize(w, h);
        sizeOfComponentMap.gpuResize(w, h);
        NeuralNet<Point, GLfloat>::gpuResize(w, h);
    }

    void setIdentical(NeuralNetLinks& nn) {
        networkLinks.setIdentical(nn.networkLinks);
        fixedLinks.setIdentical(nn.fixedLinks);
        correspondenceMap.setIdentical(nn.correspondenceMap);
        minRadiusMap.setIdentical(nn.minRadiusMap);
        adaptiveMapOri.setIdentical(nn.adaptiveMapOri);
        sizeOfComponentMap.setIdentical(nn.sizeOfComponentMap);
        NeuralNet<Point, GLfloat>::setIdentical(nn);
    }

    void gpuSetIdentical(NeuralNetLinks& nn) {
        networkLinks.gpuSetIdentical(nn.networkLinks);
        fixedLinks.gpuSetIdentical(nn.fixedLinks);
        correspondenceMap.gpuSetIdentical(nn.correspondenceMap);
        minRadiusMap.gpuSetIdentical(nn.minRadiusMap);
        adaptiveMapOri.gpuSetIdentical(nn.adaptiveMapOri);
        sizeOfComponentMap.gpuSetIdentical(nn.sizeOfComponentMap);
        NeuralNet<Point, GLfloat>::gpuSetIdentical(nn);
    }
    void gpuCopyHostToDevice(NeuralNetLinks & gpuNeuralNetLinks){
        this->networkLinks.gpuCopyHostToDevice(gpuNeuralNetLinks.networkLinks);
        this->fixedLinks.gpuCopyHostToDevice(gpuNeuralNetLinks.fixedLinks);
        this->correspondenceMap.gpuCopyHostToDevice(gpuNeuralNetLinks.correspondenceMap);
        this->minRadiusMap.gpuCopyHostToDevice(gpuNeuralNetLinks.minRadiusMap);
        this->adaptiveMapOri.gpuCopyHostToDevice(gpuNeuralNetLinks.adaptiveMapOri);
        this->sizeOfComponentMap.gpuCopyHostToDevice(gpuNeuralNetLinks.sizeOfComponentMap);
        this->objectivesMap.gpuCopyHostToDevice(gpuNeuralNetLinks.objectivesMap);
        this->adaptiveMap.gpuCopyHostToDevice(gpuNeuralNetLinks.adaptiveMap);
        this->activeMap.gpuCopyHostToDevice(gpuNeuralNetLinks.activeMap);
        this->fixedMap.gpuCopyHostToDevice(gpuNeuralNetLinks.fixedMap);
        this->colorMap.gpuCopyHostToDevice(gpuNeuralNetLinks.colorMap);
        this->grayValueMap.gpuCopyHostToDevice(gpuNeuralNetLinks.grayValueMap);
        this->densityMap.gpuCopyHostToDevice(gpuNeuralNetLinks.densityMap);
    }

    void gpuCopyDeviceToHost(NeuralNetLinks & gpuNeuralNetLinks){
        this->networkLinks.gpuCopyDeviceToHost(gpuNeuralNetLinks.networkLinks);
        this->fixedLinks.gpuCopyDeviceToHost(gpuNeuralNetLinks.fixedLinks);
        this->correspondenceMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.correspondenceMap);
        this->minRadiusMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.minRadiusMap);
        this->adaptiveMapOri.gpuCopyDeviceToHost(gpuNeuralNetLinks.adaptiveMapOri);
        this->sizeOfComponentMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.sizeOfComponentMap);
        this->objectivesMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.objectivesMap);
        this->adaptiveMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.adaptiveMap);
        this->activeMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.activeMap);
        this->fixedMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.fixedMap);
        this->colorMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.colorMap);
        this->grayValueMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.grayValueMap);
        this->densityMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.densityMap);
    }
    void gpuCopyDeviceToDevice(NeuralNetLinks & gpuNeuralNetLinks){
        this->networkLinks.gpuCopyDeviceToDevice(gpuNeuralNetLinks.networkLinks);
        this->fixedLinks.gpuCopyDeviceToDevice(gpuNeuralNetLinks.fixedLinks);
        this->correspondenceMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.correspondenceMap);
        this->minRadiusMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.minRadiusMap);
        this->adaptiveMapOri.gpuCopyDeviceToDevice(gpuNeuralNetLinks.adaptiveMapOri);
        this->sizeOfComponentMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.sizeOfComponentMap);
        this->objectivesMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.objectivesMap);
        this->adaptiveMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.adaptiveMap);
        this->activeMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.activeMap);
        this->fixedMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.fixedMap);
        this->colorMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.colorMap);
        this->grayValueMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.grayValueMap);
        this->densityMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.densityMap);
    }
    //!wbQ: read just netLinks
    void readNetLinks(string str){

        int pos = this->getPos(str);
        ifstream fi;
        string str_sub = str.substr(0, pos);
        str_sub.append(NNG_SFX_CELLULARMATRIX_LINK_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        else
        {
            std::cout << "read netLinks from: "<< str_sub << endl;
            int _w = 0;
            int _h = 0;
            fi >> str >> str >> _w;
            fi >> str >> str >> _h;
            networkLinks.resize(_w, _h);
            fixedLinks.resize(_w, _h);
            NeuralNet<Point, GLfloat>::resize(_w, _h);
        }

        char strLink[256];
        while(fi >> strLink){
            int y = 0;
            int x = 0;
            fi  >> strLink >> x >> y;
            if (y > networkLinks.height || x > networkLinks.width){
                cout << "error: fail read network links, links over range." << endl;
                cout << "over range y = " << y << " , over range x = " << x << endl;
                fi.close();
            }
            else
            {
                fi >> networkLinks[y][x];
                this->fixedLinks[y][x] = 1;
            }
        } ;
        fi.close();
    }

    //! 290416 QWB add to just write netLinks, use "write" to write otherMaps
    void writeLinks(string str){

        int pos= this->getPos(str);
        string str_sub;
        ofstream fo;

        if(networkLinks.width != 0 && networkLinks.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NNG_SFX_CELLULARMATRIX_LINK_IMAGE);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write " << str_sub << endl; }
            else
            {
                cout << "write netWorkLinks to: " << str_sub << endl;
                fo << "Width = " << networkLinks.width << " ";
                fo << "Height = " << networkLinks.height << " " << endl;
            }
            for (int y = 0; y < networkLinks.height; y ++)
                for (int x = 0; x < networkLinks.width; x++)
                {
                    if (this->networkLinks[y][x].numLinks > 0){
                        fo << endl << "NodeP = " << x << " " << y;
                        fo << networkLinks[y][x];
                    }
                }
            fo.close();
        }
        else
            cout << "Error writeLinks: this NN does not have netLinks." << endl;
    }

    //! WB.Q 101216 add to read original map for TSP
    void readOriginalMap(string str) {

        int pos = getPos(str);
        ifstream fi;
        //! read adaptiveMapOri
        string str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_ADAPTIVE_ORIGINALMAP_TSP);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read " << str_sub << endl; }
        fi >> adaptiveMapOri;
        fi.close();

    }

    //! WB.Q 101216 add to write original map for TSP
    void writeOriginalMap(string str){

            int pos = getPos(str);
            string str_sub;

            ofstream fo;
            //! write adaptiveMapOri;
            if(adaptiveMapOri.width != 0 && adaptiveMapOri.height != 0){
                str_sub = str.substr(0, pos);
                str_sub.append(NN_SFX_ADAPTIVE_ORIGINALMAP_TSP);
                fo.open(str_sub.c_str());
                if (!fo) {
                    std::cout << "erreur  write " << str_sub << endl; }
                fo << adaptiveMapOri;
                fo.close();
            }
    }



    //!wb.Q: this function computes the total distance of a netLink
    template <class Distance>
    double evaluateWeightOfSingleTree_Recursive()
    {

        this->grayValueMap.resetValue(0); // to count the number of nodes

        double totalWeight = 0;
        cout << ">>>> evaluateWeightOfSingleTree_Recursive: " << endl;

        vector<PointCoord> nodeAlreadyTraversed;
        nodeAlreadyTraversed.clear();

        PointCoord ps(0, 0);

        evaluateWeightOfSingleTreeRecursive<Distance>(ps, nodeAlreadyTraversed, totalWeight);

        return totalWeight;
    }



    //! QWB setting all nodes between node2 and node3 can not execute 2-opt, do not use recursive function
    void setFixedlinksBetweenNode2Node3(PointCoord node1, PointCoord node2, PointCoord node3){

        // get the first node between node2 and node3
        Point2D pLinkOfNode2;
        PointCoord pco2(-1, -1);
        this->networkLinks[node2[1]][node2[0]].get(0, pLinkOfNode2);
        pco2[0] = (int)pLinkOfNode2[0];
        pco2[1] = (int)pLinkOfNode2[1];
        if(pco2 == node1){
            this->networkLinks[node2[1]][node2[0]].get(1, pLinkOfNode2);
            pco2[0] = (int)pLinkOfNode2[0];
            pco2[1] = (int)pLinkOfNode2[1];
        }
        this->fixedMap[node2[1]][node2[0]] = 1;
        this->fixedMap[pco2[1]][pco2[0]] = 1;

        PointCoord nodeAvant;
        nodeAvant = node2;

        while(pco2 != node3){

            PointCoord pcoT(-1, -1);
            this->networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode2);
            pcoT[0] = (int)pLinkOfNode2[0];
            pcoT[1] = (int)pLinkOfNode2[1];
            if(pcoT == nodeAvant){
                this->networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode2);
                pcoT[0] = (int)pLinkOfNode2[0];
                pcoT[1] = (int)pLinkOfNode2[1];
            }

            this->fixedMap[pcoT[1]][pcoT[0]] = 1;
            nodeAvant = pco2;
            pco2 = pcoT;
        }

    }




private:


    //!QWB: this function computes the total distance of a netLink, closed or open 2D ring, spanning tree
    //! every node should be connected at least once
    template <class Distance>
    void evaluateWeightOfTspRecursive(PointCoord psBegin, PointCoord psAvant, PointCoord ps, double& totalWeight)
    {

        this->fixedMap[ps[1]][ps[0]] += 1; // qiao for test how many nodes being evaluated

        Distance dist;
        Point2D pLinkOfNode;
        this->networkLinks[ps[1]][ps[0]].get(0, pLinkOfNode);
        PointCoord pco(-1, -1);
        pco[0] = (int)pLinkOfNode[0];
        pco[1] = (int)pLinkOfNode[1];

        if(pco == psAvant){
            this->networkLinks[ps[1]][ps[0]].get(1, pLinkOfNode);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];
        }

        totalWeight += dist(ps, pco, *this, *this);

        if(pco == psBegin){
            return;
        }
        else
            evaluateWeightOfTspRecursive<Distance>(psBegin, ps, pco, totalWeight);

    }


    //!QWB: this function computes the total distance of a netLink, closed or open 2D ring, spanning tree
    //! every node should be connected at least once
    template <class Distance>
    void evaluateWeightOfSingleTreeRecursive(PointCoord ps, vector<PointCoord>& nodeAlreadyTraversed, double& totalWeight)
    {

        Distance dist;

        Point2D pInitial(-1, -1);

        this->grayValueMap[ps[1]][ps[0]] += 1;// for test if all node are traversed just once
        nodeAlreadyTraversed.push_back(ps);

        for (int i = 0; i < this->networkLinks[ps[1]][ps[0]].numLinks; i ++){

            Point2D pLinkOfNode;
            this->networkLinks[ps[1]][ps[0]].get(i, pLinkOfNode);

            if(pLinkOfNode == pInitial)
                continue;
            else{

                PointCoord pco(0, 0);
                pco[0] = (int)pLinkOfNode[0];
                pco[1] = (int)pLinkOfNode[1];

                // compare if the current pCoord is already be traversed
                bool traversed = 0;
                for (int k = 0; k < nodeAlreadyTraversed.size(); k ++){
                    PointCoord pLinkTemp(-1, -1);
                    pLinkTemp = nodeAlreadyTraversed[k];
                    if (pco[0] == pLinkTemp[0] && pco[1] == pLinkTemp[1])
                        traversed = 1;
                }

                if(traversed)
                    continue;

                else{
                    totalWeight += dist(ps, pco, *this, *this);
                    nodeAlreadyTraversed.push_back(pco);
                    if(networkLinks[pco[1]][pco[0]].numLinks != 0){
                        evaluateWeightOfSingleTreeRecursive<Distance>(pco, nodeAlreadyTraversed, totalWeight);
                    }
                }

            }
        }

    }
};

//! wenbao Qiao 060716 add for using static buffer links, Point2D has operations like ofstream
typedef NeuralNetLinks<BufferLink2D, Point2D> NetLinkPointCoord;
typedef NetLinkPointCoord NNLinkPoint2D;
typedef NeuralNetLinks<BufferLink2D, Point3D> MatNetLinks;
typedef NeuralNetLinks<BufferLink2D, Point2DInt> NetLinkPointInt;


}//namespace componentsEMST
#endif // NEURALNETLINKS_H

#ifndef NETLINKS_H
#define NETLINKS_H
/*
 ***************************************************************************
 *
 * Author : Wenbao. Qiao
 * Creation date : Avril. 2016
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"

//! reference to EMST components
#include "NodeEMST.h"




// qiao add to distinguish using or not using cellularmatrix for netLinks; 0 do not use, 1 use
#define Viewer 0 //1 do not use viewer, use CellularMatrix,  0 use viewer, need to verify other two places in NeuralNet.h and NetLinks.h

#if Viewer
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>

#include "NIterHexa.h"
#include "ViewGridHexa.h"

#endif


//!wb.Q Maximum buffer size indicates the maximum number of links
#define MAX_VECTOR_SIZE 128// 1024 //16384 //1024 //32768 //16384 //1024 //32768 // depend on the size of graph, tsp ,image
#define MAX_LINK_SIZE 20 // 2
#define MAX_VECTOR_SIZE_MERGRAPH 64

#define BUFFER_INCREMENT 256

using namespace std;

namespace components
{

struct DistanceLinks
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(const PointCoord& p1, const PointCoord& p2)
    {
        GLfloat dist = (p1 - p2) * (p1 - p2);
        return sqrt(dist);
    }
};


// QWB 051016 add struct for superVertices on cpu side only, to connect different component into a graph
class superVertexCpu {

public:

    PointCoord centralNode;
    float minCoupleDistance;
    PointCoord couple1;
    PointCoord couple2;
    int id2;

public:
    DEVICE_HOST superVertexCpu() {
        minCoupleDistance = INFINITY;
        PointCoord pInitial(INFINITY, INFINITY);
        centralNode = pInitial;
        id2 = INFINITY;
        couple1 = pInitial;
        couple2 = pInitial;
    }
};

// QWB 141016 add struct for superVertices on cpu side only, to connect different component into a graph
struct superVertexDataMinimum {

public:


    PointCoord couple1;
    float minCoupleDistance;
    int id2;
    int id1;
    int minimumId2;

public:
    DEVICE_HOST superVertexDataMinimum() {
        PointCoord pInitial(-1, -1);
        couple1 = pInitial;
        minCoupleDistance = INFINITY; // WB.Q necessary to initialized with infinitiy
        id2 = INFINITY;
        id1 = INFINITY;
        minimumId2 = INFINITY;
    }
};

// QWB 141016 add struct for superVertices on cpu side only, to connect different component into a graph
struct superVertexData {

public:

    float minCoupleDistance;
    PointCoord couple1;
    PointCoord couple2;
    int id2;
    int id1;

public:
    DEVICE_HOST superVertexData() {
        minCoupleDistance = INFINITY; // WB.Q necessary to initialized with infinitiy
        PointCoord pInitial(-1, -1);
        id2 = INFINITY;
        id1 = INFINITY;
        couple1 = pInitial;
        couple2 = pInitial;
    }
};


// QWB add buffer to contain links
template <class Node>
class BufferNode : public Point<Node, MAX_LINK_SIZE> {

public:

    int length;

public:

    DEVICE_HOST BufferNode() {
        length = MAX_LINK_SIZE;
    }

    DEVICE_HOST bool init() {
        length = MAX_LINK_SIZE;
        return true;
    }
    DEVICE_HOST bool init(int size) {
        length = MAX_LINK_SIZE;
        return true;
    }
    DEVICE_HOST bool incre() {
        return false;
    }
    DEVICE_HOST bool incre(int size) {
        return false;
    }
};


// wb.Q add buffer to contain links
template <class Node>
class BufferLinkVector : public Point<Node, MAX_VECTOR_SIZE> {

public:

    int length;

public:

    DEVICE_HOST BufferLinkVector() {
        length = MAX_VECTOR_SIZE;
    }

    DEVICE_HOST bool init() {
        length = MAX_VECTOR_SIZE;
        return true;
    }
    DEVICE_HOST bool init(int size) {
        length = MAX_VECTOR_SIZE;
        return true;
    }
    DEVICE_HOST bool incre() {
        return false;
    }
    DEVICE_HOST bool incre(int size) {
        return false;
    }
};




//! qiao 060716 add for buffer links static
template <class Node>
struct LinksBufferStaticVector {

    BufferLinkVector<Node> bCell;
    size_t numLinks;

    DEVICE_HOST LinksBufferStaticVector(): numLinks(0) {bCell.init();}


    //! wb.Qiao 070716 add clear function
    DEVICE_HOST inline void clearLinks() {// qiao note, here, if only clear numLinks, the buffer will always has values
        this->numLinks = 0;
        if (!(bCell.init()))
            printf("Cell buffer alloc failed !\n");
        this->bCell = BufferLinkVector<Node>() ;//! qiao 010716 add, sinon, points in bCell._value have the old values
    }

    //! wb.Qiao 070716 add swap two objects
    DEVICE_HOST inline void swapArray(BufferLinkVector<Node> *left, BufferLinkVector<Node> *right){
        if(&left == &right);
        else{
            BufferLinkVector<Node> *temp1Begin = left;
            BufferLinkVector<Node> *temp1end = left + MAX_LINK_SIZE;
            BufferLinkVector<Node> *temp2 = right;
            for (; temp1Begin != temp1end; ++temp1Begin, ++temp2)
                std::iter_swap(temp1Begin, temp2);
        }
    }

    DEVICE_HOST void swapArray2(BufferLinkVector<Node>* left, BufferLinkVector<Node>* right){

        BufferLinkVector<Node> *temp;

        temp = left;
        left = right;
        right = temp;
    }



    DEVICE_HOST LinksBufferStaticVector& operator = (LinksBufferStaticVector tempCp){
        this->numLinks = tempCp.numLinks;
        this->bCell = tempCp.bCell;

        return *this;
    }

    //! To iterate
    DEVICE_HOST inline void init() { numLinks = 0; bCell.init(); }

    DEVICE_HOST inline void init(int size) { numLinks = 0; bCell.init(size); }

    DEVICE_HOST inline bool get(int i, Node& ps) {
        bool ret = i >= 0 && i < this->numLinks;
        if (ret)
            ps = bCell[i];
        return (ret);
    }

    DEVICE_HOST inline Node get(int i) {
        bool ret = i >= 0 && i < this->numLinks;
        if (ret)
            return (bCell[i]);
    }
    DEVICE_HOST inline bool next() {
        return ++numLinks < bCell.length;
    }


    // Insertion in buffer
    DEVICE_HOST bool insert(Node& pc) {
        bool ret = false;
        if (this->numLinks < bCell.length) {
            ret = true;
            (Node&)bCell[this->numLinks] = pc;
            this->numLinks += 1;
        }
        return ret;
    }

    //! wb.Qiao 120516
    friend ifstream& operator>>(ifstream& i, LinksBufferStaticVector & mat){
        if (!i)
            return(i);

        string strLink;
        string strNumLinks = "NumLinks";

        i >>  strLink;
        if( !strcmp(strLink.c_str(),strNumLinks.c_str())){

            i >>  strLink >> mat.numLinks;

            for (int _x = 0; _x < mat.numLinks; _x++){
                i >> mat.bCell[_x];
            }
        }

        return i;
    }

    //! qiao 270416 add operation <<
    friend ofstream& operator<<(ofstream & o, LinksBufferStaticVector const & mat) {
        if (!o)
            return(o);

        o << " NumLinks = " << mat.numLinks << " ";

        if (!o)
            return(o);

        for (int _x = 0; _x < mat.numLinks; _x++)
        {
            o << (Node&)mat.bCell._value[_x];
        }

        o << endl;
        return o;
    }

}; //VectorLinks LinksBufferStaticVector


//! qiao 060716 add for buffer links static
template <class Node>
struct LinksBufferStatic {

    BufferNode<Node> bCell;
    size_t numLinks;

    DEVICE_HOST LinksBufferStatic(): numLinks(0) {bCell.init();}

    //! qiao add copy constructor
    DEVICE_HOST LinksBufferStatic(const LinksBufferStatic & copy){
        numLinks = copy.numLinks;
        bCell = copy.bCell;
    }

    //! qiao 070716 add clear function
    DEVICE_HOST inline void clearLinks() {// qiao note, here, if only clear numLinks, the buffer will always has values
        this->numLinks = 0;
        if (!(bCell.init()))
            printf("Cell buffer alloc failed !\n");
        bCell = BufferNode<Node>() ;//! qiao 010716 add, sinon, points in bCell._value have the old values
    }

    //! qiao 070716 add swap two objects
    DEVICE_HOST inline void swapArray(BufferNode<Node> *left, BufferNode<Node> *right){
        if(&left == &right);
        else{
            BufferNode<Node> *temp1Begin = left;
            BufferNode<Node> *temp1end = left + MAX_LINK_SIZE;
            BufferNode<Node> *temp2 = right;
            for (; temp1Begin != temp1end; ++temp1Begin, ++temp2)
                std::iter_swap(temp1Begin, temp2);
        }
    }

    DEVICE_HOST void swapArray2(BufferNode<Node>* left, BufferNode<Node>* right){

        BufferNode<Node> *temp;

        temp = left;
        left = right;
        right = temp;
    }




    DEVICE_HOST LinksBufferStatic& operator = (LinksBufferStatic tempCp){
        this->numLinks = tempCp.numLinks;
        this->bCell = tempCp.bCell;

        return *this;
    }

    //! To iterate
    DEVICE_HOST inline void init() { numLinks = 0; bCell.init(); }

    DEVICE_HOST inline void init(int size) { numLinks = 0; bCell.init(size); }

    DEVICE_HOST inline bool get(int i, Node& ps) {
        bool ret = i >= 0 && i < this->numLinks;
        if (ret)
            ps = bCell[i];
        return (ret);
    }

    DEVICE_HOST inline Node get(int i) {
        bool ret = i >= 0 && i < this->numLinks;
        if (ret)
            return (bCell[i]);
    }
    DEVICE_HOST inline bool next() {
        return ++numLinks < bCell.length;
    }

    // Insertion in buffer
    DEVICE_HOST bool insert(Node& pc) {
        bool ret = false;
        if (this->numLinks < bCell.length) {
            ret = true;
            (Node&)bCell[this->numLinks] = pc;
            this->numLinks += 1;
        }
        return ret;
    }


#ifdef CUDA_CODE
    // Insertion in buffer
    DEVICE bool insertAtomic(Node& pc) {
        bool ret = false;

        if ((atomicAdd(&(this->numLinks), 1)) < MAX_LINK_SIZE)
        {
            ret = true;
            atomicExch(&(bCell[this->numLinks - 1][0]), pc[0]);
            atomicExch(&(bCell[this->numLinks - 1][1]), pc[1]);
        }

        return ret;
    }
#endif

    //! qiao 120516
    friend ifstream& operator>>(ifstream& i, LinksBufferStatic & mat){
        if (!i)
            return(i);

        string strLink;
        string strNumLinks = "NumLinks";

        i >>  strLink;
        if( !strcmp(strLink.c_str(),strNumLinks.c_str())){

            i >>  strLink >> mat.numLinks;

            for (int _x = 0; _x < mat.numLinks; _x++){
                i >> mat.bCell[_x];
            }
        }

        return i;
    }

    //! qiao 270416 add operation <<
    friend ofstream& operator<<(ofstream & o, LinksBufferStatic const & mat) {
        if (!o)
            return(o);

        o << " NumLinks = " << mat.numLinks << " ";

        if (!o)
            return(o);

        for (int _x = 0; _x < mat.numLinks; _x++)
        {
            o << (Node&)mat.bCell._value[_x];
        }

        o << endl;
        return o;
    }

}; //CellBLinks


//! wenbao Qiao 060716 add for gpu static links
typedef LinksBufferStatic<Point2D> BufferLink2D;
typedef LinksBufferStatic<Point2DInt> BufferIntLink2D;
typedef LinksBufferStatic<Point3D> BufferLink3D;
typedef LinksBufferStatic<PointCoord> BufferLinkPointCoord;
typedef LinksBufferStaticVector<PointCoord> BufferLinkPointCoordVector;


}//namespace components

#endif // CELL_H

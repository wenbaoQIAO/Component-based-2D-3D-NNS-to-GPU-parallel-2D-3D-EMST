#ifndef EMST_OPERATORS_H
#define EMST_OPERATORS_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao, J.C. Créput
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

//#include <cuda_runtime.h>
//#include <cuda.h>
//#include <helper_functions.h>
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <sm_60_atomic_functions.h>
#include <sm_61_intrinsics.h>

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
#include "NIter.h"

//! reference EMST components
#include "NeuralNetEMST.h"
#include "NodeEMST.h"

#define EMST_BLOCK_SIZE 128

using namespace std;
using namespace components;

struct TspResultInfo
{
    GLfloat length = 0;
    GLint size = 0;
    GLfloat timeFlatten = 0;
    GLfloat timeTestTermination = 0;
    GLfloat timeFindNextClosest = 0;
    GLfloat timeFindMinPair = 0;
    GLfloat timeConnectGraphUnion = 0;
    GLfloat timeCumulativeFindNextClosest = 0;
    GLfloat timeCumulativeFindMinPair = 0;
    // wb.Q add
    GLfloat timeCumulativeConnetUnion = 0;
    GLfloat timeCumulativeFlatten = 0;
    GLfloat timeCumulativeTermination = 0;
    string benchMark = "";
};

//#include "MstOperator.h"

namespace operators
{

/*******************************************************
 *  Spiral Search initialisation
 *******************************************************/
template <typename CellularMatrix, typename Grid, typename Grid2 >
KERNEL inline void K_EMST_initializeSpiralSearch(CellularMatrix cm, Grid g_point, Grid2 g_ss) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

    typename Grid::index_type ps(_x, _y, _z);
    if (g_point.valideIndex(ps))
    {
        typedef typename CellularMatrix::index_type index_type_cm;
        typedef typename Grid::index_type index_type;
        typedef typename Grid2::point_type ss_node_type;

        index_type_cm pc = cm.vgd.findCell(g_point(ps));

        if (cm.valideAndPositiveIndex(pc)) {
//            printf("%d , %d , %d \n",ps[0], ps[1], ps[2]);
            cm(pc).insert(ps);
            g_ss(ps).init(pc, ps, 0, MAX(cm.vgd.getWidthDual(),MAX(cm.vgd.getHeightDual(),cm.vgd.getDepthDual())));
//            g_ss(ps).init(pc, ps, 0, cm.vgd.getWidthDual() + cm.vgd.getHeightDual() + cm.vgd.getDepthDual());
        }
        else {
            g_ss(ps) = ss_node_type( index_type_cm(-1),
                                     index_type(-1),
                                     0,
                                     -1);
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/**********************************************************************
 * K_FindNextClosestPoint for each point in Component
 *********************************************************************/
template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_computeOctants(Grid cm,
                                               Grid2 g_dstree,
                                               Grid3 g_point,
                                               Grid4 g_dist,
                                               Grid5 g_corr,
                                               Grid6 g_ss) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).computeOctants(cm, g_dstree, g_point, g_dist, g_corr);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_computeOctant(Grid cm,
                                               Grid2 g_dstree,
                                               Grid3 g_point,
                                               Grid4 g_dist,
                                               Grid5 g_corr,
                                               Grid6 g_ss) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).computeOctant(cm, g_dstree, g_point, g_dist, g_corr);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_findNextClosestPoint(Grid cm,
                                               Grid2 g_dstree,
                                               Grid3 g_point,
                                               Grid4 g_dist,
                                               Grid5 g_corr,
                                               Grid6 g_ss) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).search(cm, g_dstree, g_point, g_dist, g_corr);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/*******************************************************
 *  Cellular Matrix management
 *******************************************************/
template <typename CellularMatrix, typename Grid >
KERNEL inline void K_EMST_refreshCell(CellularMatrix cm, Grid g) {

    KER_SCHED_3D(g.getWidth(), g.getHeight(), g.getDepth())

    typename Grid::index_type ps(_x, _y, _z);
    if (g.valideIndex(ps))
    {
        typename CellularMatrix::index_type pc(-1);

        pc = cm.vgd.findCell(g(ps));

        if (cm.valideAndPositiveIndex(pc))
            cm(pc).insert(ps);
        else
            printf("erreur index %d %d %d \n", pc[0], pc[1], pc[2]);
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL inline void K_EMST_FindNextClosestPoint(Grid cm,
                                               Grid2 g_dstree,
                                               Grid3 g_point,
                                               Grid4 g_dist,
                                               Grid5 g_corr) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        index_type_cm pc;

        // Find cell coordinate of ps
        pc = cm.vgd.findCell(g_point(ps));

        if (cm.valideAndPositiveIndex(pc)) {

            NodeSpiralSearch<index_type_cm, index_type> nss(
                        pc,
                        ps,
                        0,
                        cm.vgd.getWidthDual()+cm.vgd.getHeightDual()
                        );

            nss.search(cm, g_dstree, g_point, g_dist, g_corr);
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/*****************************************************
 * Evaluation
 *****************************************************/
template <typename Grid1, typename Grid2, typename Grid3 >
KERNEL inline void K_EMST_evaluate_ST(Grid1 g_link, Grid2 g_point, Grid3 g_obj) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        g_obj(ps)[obj_distr] = g_link(ps).numLinks;
        g_obj(ps)[obj_length] = 0;

        // Compute sum of outgoing lengths
        for (int i = 0; i < g_link(ps).numLinks; ++i){

            index_type pco(-1);
            g_link(ps).get(i, pco);

            g_obj(ps)[obj_length] +=
            components::DistanceEuclidean<point_type>()(
                        g_point(ps),
                        g_point(pco)
                        );
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <typename Grid1, typename Grid2, typename Grid3, typename Dist >
KERNEL inline void K_EMST_length_ST(Grid1 g_link, Grid2 g_point, Grid3 g_obj, Dist dist) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);

    if (g_link.valideIndex(ps))
    {
        g_obj(ps)[obj_length] = 0;

        // Compute sum of outgoing lengths
        for (int i = 0; i < g_link(ps).numLinks; i ++){

            index_type pco(-1);
            g_link(ps).get(i, pco);

            g_obj(ps)[obj_length] +=
                    dist(
                        g_point(ps),
                        g_point(pco)
                        );
        }

    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/*******************************************************
 * Disjoint Set Tree Initialisation
 ******************************************************/
template <typename Grid >
KERNEL inline void K_EMST_initDisjointSet(Grid g) {

    KER_SCHED_3D(g.getWidth(), g.getHeight(), g.getDepth())

    typename Grid::index_type idx(_x, _y, _z);
    if (g.valideIndex(idx))
    {
        g(idx) = g.compute_offset(idx);// / sizeof(Grid::point_type);
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class CellularMatrix, class Grid >
GLOBAL inline void K_EMST_refreshCell_cpu(CellularMatrix& cm, Grid& g) {

    for (int _z = 0; _z < (g.getDepth()); ++_z) {
    for (int _y = 0; _y < (g.getHeight()); ++_y) {
    for (int _x = 0; _x < (g.getWidth()); ++_x) {

    if (_x < g.getWidth() && _y < g.getHeight() && _z < g.getDepth())
    {
        //Grid::index_type
        PointCoord ps(_x, _y, _z);
        //CellularMatrix::index_type
        PointCoord minP(-1);

        minP = cm.vgd.findCell(g[ps[1]][ps[0]]);
        //printf("index( %d , %d ) : p( %f , %f ) : minP( %d , %d ) \n", ps[0], ps[1], g[ps[1]][ps[0]][0], g[ps[1]][ps[0]][1], minP[0], minP[1]);

        if (minP[0] >= 0 && minP[0] < cm.getWidth() && minP[1] >= 0 && minP[1] < cm.getHeight())
            cm[minP[1]][minP[0]].insert_cpu(ps);
    }

    }}}

}

/*******************************************************
 *  Clear Links
 *******************************************************/

template <typename Grid >
KERNEL inline void K_EMST_clearLinks(Grid g) {

    KER_SCHED_3D(g.getWidth(), g.getHeight(), g.getDepth())

    typename Grid::index_type ps(_x, _y, _z);
    if (g.valideIndex(ps))
    {
        g(ps).clearLinks();
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/**********************************************************************
 * Flatten Disjoint Set Tree
 *********************************************************************/
template <typename Grid >
KERNEL inline void K_EMST_flatten_DST(Grid g_dstree) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

    typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        // find root and set as parent
        //g_dstree(ps) = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
        GLint r = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
        atomicExch(&(g_dstree(ps)), r);
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <typename Grid, typename Grid2 >
KERNEL inline void K_EMST_flatten_DST_1(Grid g_dstree, Grid2 g_parent) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

    typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        GLint r = g_dstree.compute_offset(ps);
        //GLint i = 0;
        while(r != g_dstree(r) /*&& ++i <= 10*/)
        {
            r = g_dstree(r); // r is the root
        }
        g_parent(ps) = r;
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}
template <typename Grid, typename Grid2 >
KERNEL inline void K_EMST_flatten_DST_2(Grid g_dstree, Grid2 g_parent) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

    typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        g_dstree(ps) = g_parent(ps);
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/**********************************************************************
 * Find Min of Component
 *********************************************************************/
template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid9,
          class Grid10,
          class Grid11
          >
KERNEL void K_EMST_findMinInComponentDB(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid3 g_fix,
                                        Grid4 g_corr,
                                        Grid5 g_dist,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid9 g_win,
                                        Grid10 g_dest,
                                        Grid11 g_minD,
                                        int state,
                                        int final_state
                                        ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        index_type win_node(-1);
        index_type dest_node(-1);
        GLfloat minDist = INFINITY;

        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            atomicAdd(&(g_evt(ps)), 1);
        }

        //int state = -2;
        while (state != final_state) {
            if (state == -1) {// visited

                if (atomicAdd(&(g_vnum(ps)), 0) == 0) {

                    state++;

                    index_type p2 = g_corr(ps);
                    if (p2 != index_type(-1)) {
                        win_node = ps;
                        dest_node = p2;
                        minDist = g_dist(ps);
                    }

                    // Compute sum of outgoing lengths
                    for (int i = 0; i < g_link(ps).numLinks; ++i){

                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        if (g_parent.compute_offset(pco) != g_parent(ps)) {

                            index_type w_node = pco.atomicAddition(&(g_win(pco)), 0);
                            index_type d_node = pco.atomicAddition(&(g_dest(pco)), 0);
                            GLfloat minD = atomicAdd(&(g_minD(pco)), 0);

                            if (w_node != index_type(-1) && p2 != index_type(-1)) {
                                if (EMST_isInf(
                                            g_link.compute_offset(w_node),
                                            g_link.compute_offset(d_node),
                                            minD,
                                            g_link.compute_offset(win_node),
                                            g_link.compute_offset(dest_node),
                                            minDist)) {
                                    win_node = w_node;
                                    dest_node = d_node;
                                    minDist = minD;
                                }
                            }
                            else if (p2 == index_type(-1)) {
                                win_node = w_node;
                                dest_node = d_node;
                                minDist = minD;
                            }
                        }//not parent
                    }
                    // Memorize Winner node
                    ps.atomicExchange(&(g_win(ps)), win_node);
                    ps.atomicExchange(&(g_dest(ps)), dest_node);
                    atomicExch(&(g_minD(ps)), minDist);

                    if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                        if (win_node[0] != -1)//index_type(-1))
                            g_fix(win_node) = true;
                    }
                    else
                        atomicAdd(&(g_vnum(g_parent(ps))), -1);
                }
//                else
//                    state++;
            }
            else
            if (state == -2){

                if (atomicAdd(&(g_evt(ps)), 0)) {
                    state++;
                    int numLinks = g_link(ps).numLinks;
                    if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                        numLinks -= 1;
                    }
                    atomicExch(&(g_vnum(ps)), numLinks);
                    for (int i = 0; i < g_link(ps).numLinks; ++i){
                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        // test to avoid parent
                        if (g_parent.compute_offset(pco)
                                != g_parent(ps)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8
          >
KERNEL void K_EMST_findMinDB_1(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent
                                          ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            atomicAdd(&(g_evt(ps)), 1);
        }

        int state = -2;
        while (state != -1) {
            if (state == -2){

                if (atomicAdd(&(g_evt(ps)), 0)) {
                    state++;
                    int numLinks = g_link(ps).numLinks;
                    if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                        numLinks -= 1;
                    }
                    atomicExch(&(g_vnum(ps)), numLinks);
                    for (int i = 0; i < g_link(ps).numLinks; ++i){
                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        // test to avoid parent
                        if (g_parent.compute_offset(pco)
                                != g_parent(ps)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid9,
          class Grid10,
          class Grid11
          >
KERNEL void K_EMST_findMinDB_2(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid3 g_fix,
                                        Grid4 g_corr,
                                        Grid5 g_dist,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid9 g_win,
                                        Grid10 g_dest,
                                        Grid11 g_minD
                                        ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        index_type win_node(-1);
        index_type dest_node(-1);
        GLfloat minDist = INFINITY;

        int state = -1;
        while (state != 0) {
            if (state == -1) {// visited

                if (atomicAdd(&(g_vnum(ps)), 0) == 0) {

                    state++;

                    index_type p2 = g_corr(ps);
                    if (p2 != index_type(-1)) {
                        win_node = ps;
                        dest_node = p2;
                        minDist = g_dist(ps);
                    }

                    // Compute sum of outgoing lengths
                    for (int i = 0; i < g_link(ps).numLinks; ++i){

                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        if (g_parent.compute_offset(pco) != g_parent(ps)) {

                            index_type w_node = pco.atomicAddition(&(g_win(pco)), 0);
                            index_type d_node = pco.atomicAddition(&(g_dest(pco)), 0);
                            GLfloat minD = atomicAdd(&(g_minD(pco)), 0);

                            if (w_node != index_type(-1) && p2 != index_type(-1)) {
                                if (EMST_isInf(
                                            g_link.compute_offset(w_node),
                                            g_link.compute_offset(d_node),
                                            minD,
                                            g_link.compute_offset(win_node),
                                            g_link.compute_offset(dest_node),
                                            minDist)) {
                                    win_node = w_node;
                                    dest_node = d_node;
                                    minDist = minD;
                                }
                            }
                            else if (p2 == index_type(-1)) {
                                win_node = w_node;
                                dest_node = d_node;
                                minDist = minD;
                            }
                        }//not parent
                    }
                    // Memorize Winner node
                    ps.atomicExchange(&(g_win(ps)), win_node);
                    ps.atomicExchange(&(g_dest(ps)), dest_node);
                    atomicExch(&(g_minD(ps)), minDist);

                    if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                        if (win_node[0] != -1)//index_type(-1))
                            g_fix(win_node) = true;
                    }
                    else
                        atomicAdd(&(g_vnum(g_parent(ps))), -1);
                }
//                else
//                    state++;
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid12
          >
KERNEL void K_EMST_findMinDBActivate_1(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid12 g_state
                                          ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

//        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
//            atomicAdd(&(g_evt(ps)), 1);
//        }

        if (g_state(ps) == -2){

            if (atomicAdd(&(g_evt(ps)), 0)||(g_dstree(ps) == g_dstree.compute_offset(ps))) {
                g_state(ps) += 1;
                int numLinks = g_link(ps).numLinks;
                if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                    numLinks -= 1;
                }
                atomicExch(&(g_vnum(ps)), numLinks);
                for (int i = 0; i < g_link(ps).numLinks; ++i){
                    index_type pco(-1);
                    g_link(ps).get(i, pco);

                    // test to avoid parent
                    if (g_parent.compute_offset(pco)
                            != g_parent(ps)) {
                        atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                        atomicAdd(&(g_evt(pco)), 2);
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid9,
          class Grid10,
          class Grid11,
          class Grid12
          >
KERNEL void K_EMST_findMinDBActivate_2(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid3 g_fix,
                                        Grid4 g_corr,
                                        Grid5 g_dist,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid9 g_win,
                                        Grid10 g_dest,
                                        Grid11 g_minD,
                                        Grid12 g_state
                                        ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        index_type win_node(-1);
        index_type dest_node(-1);
        GLdouble minDist = HUGE_VAL;

        if (g_state(ps) == -1) {// visited

            if (g_vnum(ps) == 0) {

                g_state(ps) += 1;

                index_type p2 = g_corr(ps);
                if (p2 != index_type(-1)) {
                    win_node = ps;
                    dest_node = p2;
                    minDist = g_dist(ps);
                }

                // Compute sum of outgoing lengths
                for (int i = 0; i < g_link(ps).numLinks; ++i){

                    index_type pco(-1);
                    g_link(ps).get(i, pco);

                    if (g_parent.compute_offset(pco) != g_parent(ps)) {

                        index_type w_node = g_win(pco);
                        index_type d_node = g_dest(pco);
                        GLdouble minD = g_minD(pco);

                        if (d_node != index_type(-1) && dest_node != index_type(-1)) {

                            GLint w_nodef = g_dstree.compute_offset(w_node);
                            GLint d_nodef = g_dstree.compute_offset(d_node);
                            GLint win_nodef = g_dstree.compute_offset(win_node);
                            GLint dest_nodef = g_dstree.compute_offset(dest_node);

                            GLint id1x = MIN(w_nodef,d_nodef);
                            GLint id2x = MAX(w_nodef,d_nodef);
                            GLint idd1x = MIN(win_nodef,dest_nodef);
                            GLint idd2x = MAX(win_nodef,dest_nodef);

                            if (EMST_isInf(
                                        id1x,
                                        id2x,
                                        minD,
                                        idd1x,
                                        idd2x,
                                        minDist)) {
                                win_node = w_node;
                                dest_node = d_node;
                                minDist = minD;
                            }
                        }
                        else if (dest_node == index_type(-1)) {
                            win_node = w_node;
                            dest_node = d_node;
                            minDist = minD;
                        }
                    }//not parent
                }
                // Memorize Winner node
                ps.atomicExchange(&(g_win(ps)), win_node);
                ps.atomicExchange(&(g_dest(ps)), dest_node);
                atomicExch((GLfloat*)&(g_minD(ps)), minDist);

                if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                    if (win_node[0] != -1)//index_type(-1))
                        g_fix(win_node) = true;
                }
                else
                    atomicAdd(&(g_vnum(g_parent(ps))), -1);
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid12
          >
KERNEL void K_EMST_diffusateDetectCycle(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid12 g_state
                                          ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

//        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
//            atomicAdd(&(g_evt(ps)), 1);
//        }

        if (g_state(ps) == -2){

            bool isRoot = (g_dstree(ps) == g_dstree.compute_offset(ps));

            if (g_evt(ps) || isRoot) {
                g_state(ps) += 1;
                int numLinks = g_link(ps).numLinks;
                if (!isRoot) {
                    numLinks -= 1;
                }
                else
                    atomicAdd(&(g_evt(ps)), 1);
                atomicExch(&(g_vnum(ps)), numLinks);
                for (int i = 0; i < g_link(ps).numLinks; ++i){
                    index_type pco(-1);
                    g_link(ps).get(i, pco);

                    // test to avoid parent
                    if (g_parent.compute_offset(pco)
                            != g_parent(ps)) {
                        if (!g_evt(pco)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                        else
                            atomicAdd(&(g_evt(pco)), 50);
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid12
          >
KERNEL void K_EMST_eliminateCycle(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid12 g_state
                                          ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_state(ps) == -2){

            g_dstree(ps) = g_dstree.compute_offset(ps);
            g_link(ps).clearLinks();

        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8
          >
KERNEL void K_EMST_diffusateDetectCycle_2(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent
                                          ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            atomicAdd(&(g_evt(ps)), 1);
        }

        int state = -2;
        while (state != -1) {
            if (state == -2){

                if (atomicAdd(&(g_evt(ps)), 0)) {
                    state++;
                    int numLinks = g_link(ps).numLinks;
                    if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                        numLinks -= 1;
                    }
                    atomicExch(&(g_vnum(ps)), numLinks);
                    for (int i = 0; i < g_link(ps).numLinks; ++i){
                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        // test to avoid parent
                        if (g_parent.compute_offset(pco)
                                != g_parent(ps)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

template <typename Index, typename Grid1, typename Grid2, typename Grid3 >
DEVICE_HOST void EMST_findMinOutgoingEdge(Index parent, Index p1, Index& win_node, Index& dest_node, GLfloat& minDist, Grid1& g_link, Grid2& g_corr, Grid3& g_dist) {

    typedef typename Grid1::index_type index_type;

    index_type p2 = g_corr(p1);
    if (p2[0] != -1/*index_type(-1)*/) {
        win_node = p1;
        dest_node = p2;
        minDist = g_dist(p1);
    }

    // Compute sum of outgoing lengths
    for (int i = 0; i < g_link(p1).numLinks; ++i){

        index_type pco(-1);
        g_link(p1).get(i, pco);

        if (pco != parent) {

        index_type w_node(-1);
        index_type d_node(-1);
        GLfloat minD = HUGE_VAL;

//        index_type ppco = g_corr(pco);
//        if (ppco[0] != -1/*index_type(-1)*/) {
//            w_node = pco;
//            d_node = ppco;
//            minD = g_dens(pco);
//        }
        if (g_dist(p1) >= 0.0f)
        EMST_findMinOutgoingEdge(p1, pco, w_node, d_node, minD, g_link, g_corr, g_dist);

        if (w_node[0] != -1/*index_type(-1)*/ && p2[0] != -1/*index_type(-1)*/) {
                if (EMST_isInf(
                            g_link.compute_offset(w_node),
                            g_link.compute_offset(d_node),
                            minD,
                            g_link.compute_offset(win_node),
                            g_link.compute_offset(dest_node),
                            minDist)) {
                    win_node = w_node;
                    dest_node = d_node;
                    minDist = minD;
                }
        }
        else if (p2[0] == -1/*index_type(-1)*/) {
            win_node = w_node;
            dest_node = d_node;
            minDist = minD;
        }

        if (g_dist(p1) >= 0.0f)
            g_dist(pco) = -1.0f;
        else if (g_dist(p1) <= -1.0f)
            g_dist(pco) = -2.0f;

        }//not parent
    }
}

template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL void K_EMST_findMinInComponent(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_dist) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps)) == g_dstree.compute_offset(ps)) {
            index_type win_node(-1);
            index_type dest_node(-1);
            GLfloat minDist = HUGE_VAL;

            // Search recursively
            EMST_findMinOutgoingEdge(index_type(-1), ps, win_node, dest_node, minDist, g_link, g_corr, g_dist);

            if (win_node[0] != -1/*index_type(-1)*/)
                g_fix(win_node) = true;
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
void K_EMST_findMinInComponent_cpu(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_dist) {

    typedef typename Grid1::index_type index_type;

    for (int _z = 0; _z < (g_link.getDepth()); ++_z) {
    for (int _y = 0; _y < (g_link.getHeight()); ++_y) {
    for (int _x = 0; _x < (g_link.getWidth()); ++_x) {

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps)) == g_dstree.compute_offset(ps)) {
        //if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            index_type win_node(-1);
            index_type dest_node(-1);
            GLfloat minDist = INFINITY;

            // Search recursively
            EMST_findMinOutgoingEdge(index_type(-1), ps, win_node, dest_node, minDist, g_link, g_corr, g_dist);

            if (win_node != index_type(-1))
                g_fix(win_node) = true;
        }
    }

    }}}
}

/**********************************************************************
 * Connect graph and Union of disjoint set components
 *********************************************************************/
template <typename Grid1, typename Grid2, typename Grid3, typename Grid4 >
KERNEL inline void K_EMST_connectComponentAndUnion(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_fix(ps)){

            index_type pcorr = g_corr(ps);

            g_link(ps).insert(pcorr);

            size_t rootPS = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
            size_t rootPCorr = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(pcorr));

            // test to avoid double insertion
            if (!g_fix(pcorr)
                    || (g_fix(pcorr) && (g_corr(pcorr) != ps))) {
                g_link(pcorr).insert(ps);
                //g_dstree(rootPS) = rootPCorr;
                atomicExch(&(g_dstree(rootPS)), rootPCorr);
            }
            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
                    // Set new disjoint set tree root of new set (component)
                    //g_dstree(rootPS) = rootPCorr;
                    atomicExch(&(g_dstree(rootPCorr)), rootPS);
            }
        }// end if winner node
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

//template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
//KERNEL inline void K_EMST_connectComponentAndUnion_2(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_parent) {

//    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

//    typedef typename Grid1::index_type index_type;

//    index_type ps(_x, _y, _z);
//    if (g_link.valideIndex(ps))
//    {
//        if (g_fix(ps)){

//            index_type pcorr = g_corr(ps);

//            g_link(ps).insert(pcorr);

//            GLint rootPS = g_parent.findRoot(g_parent, g_parent.compute_offset(ps));
//            GLint rootPCorr = g_parent.findRoot(g_parent, g_parent.compute_offset(pcorr));

//            // test to avoid double insertion
//            if (!g_fix(pcorr)
//                    || (g_fix(pcorr) && (g_corr(pcorr) != ps))) {
//                g_link(pcorr).insert(ps);
////                g_dstree(rootPS) = rootPCorr;
//                atomicExch(&(g_dstree(rootPS)), rootPCorr);
//            }
//            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
//                    // Set new disjoint set tree root of new set (component)
////                    g_dstree(rootPCorr) = rootPS;
//                    atomicExch(&(g_dstree(rootPCorr)), rootPS);
//            }
//        }// end if winner node
//    }
//    END_KER_SCHED_3D

//    SYNCTHREADS
//}


template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL inline void K_EMST_connectComponentAndUnion_2(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_parent) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_fix(ps)){

            index_type pcorr = g_corr(ps);

            g_link(ps).insert(pcorr);

            GLint rootPS = g_parent(ps);
            GLint rootPCorr = g_parent(pcorr);

            // reduce global mem access
            bool g_fi = g_fix(pcorr);

            // test to avoid double insertion
            if (!g_fi
                    || (g_fi && (g_corr(pcorr) != ps))) {
                g_link(pcorr).insert(ps);
                g_dstree(rootPS) = rootPCorr;
            }
            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
                    g_dstree(rootPCorr) = rootPS;
            }
        }// end if winner node
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

// wb.q one link network not sure, just for difference in the behave of warp threads
template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL inline void K_EMST_connectComponentAndUnion_2_forFutureOneDirection(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_parent) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_fix(ps)){

            index_type pcorr = g_corr(ps);

            GLint rootPS = g_parent(ps);
            GLint rootPCorr = g_parent(pcorr);

            // reduce global mem access
            bool g_fi = g_fix(pcorr);

            // test to avoid double insertion
            if (!g_fi
                    || (g_fi && (g_corr(pcorr) != ps))) {
                g_link(ps).insert(pcorr);
                g_link(pcorr).insert(ps);
                g_dstree(rootPS) = rootPCorr;
            }
            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
                g_link(ps).insert(pcorr);
                g_dstree(rootPCorr) = rootPS;
            }
        }// end if winner node
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}


//// wb.Q note: this step does not need g_parent
//template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
//KERNEL inline void K_EMST_connectComponentAndUnion_2(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_parent) {

//    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

//    typedef typename Grid1::index_type index_type;

//    index_type ps(_x, _y, _z);
//    if (g_link.valideIndex(ps))
//    {
//        if (g_fix(ps)){

//            index_type pcorr = g_corr(ps);

//            g_link(ps).insert(pcorr);

//            GLint rootPS = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
//            GLint rootPCorr = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(pcorr));

//            // test to avoid double insertion
//            if (!g_fix(pcorr)
//                    || (g_fix(pcorr) && (g_corr(pcorr) != ps))) {
//                g_link(pcorr).insert(ps);
//                g_dstree(rootPS) = rootPCorr;
////                atomicExch(&(g_dstree(rootPS)), rootPCorr);
//            }
//            else if (g_dstree(rootPS) < g_dstree(rootPCorr)) {
//                    // Set new disjoint set tree root of new set (component)
//                    g_dstree(rootPCorr) = rootPS;
////                    atomicExch(&(g_dstree(rootPCorr)), rootPS);
//            }
//        }// end if winner node
//    }
//    END_KER_SCHED_3D

//    SYNCTHREADS
//}

/**
 * \brief Class EMSTOperators
 * Classe that group operators for EMST Building
 **/
template <class CellularMatrixR,
          class CellularMatrixD,
          class NetLink,
          class CellR,
          class CellD,
          class NIter,
          class NIterDual,
          class ViewG,
          class BufferDimension
          >
class EMSTOperators
{

public:
    DEVICE_HOST explicit EMSTOperators(){}

    // Initialize the values in GPU grids EMST
    GLOBAL void gpuResetValue(NetLink& nnLGpu){

//        nnLGpu.distanceMap.gpuResetValue(infinity);// on cpu, infinity turn out to be 1
        nnLGpu.densityMap.gpuResetValue(infinity);// on cpu, infinity turn out to be 1
        nnLGpu.disjointSetMap.gpuResetValue(infinity);
        nnLGpu.activeMap.gpuResetValue(1);// QWB 241016 only for mst, specially initialize activeMap = 1, to mark the thread running shortest outgoing
        nnLGpu.fixedMap.gpuResetValue(1);// QWB 241016, only for mst, specially initialize fixedmap = 1 to mark initial winner nodes
        nnLGpu.minRadiusMap.gpuResetValue(0);
        nnLGpu.sizeOfComponentMap.gpuResetValue(1);
        PointCoord pInitial(-1);
        nnLGpu.correspondenceMap.gpuResetValue(pInitial);
    }

    // wb.Q: initialize values in CPU grids before EMST
    GLOBAL void cpuResetValue(NetLink& nnLCpu){

//        nnLCpu.distanceMap.resetValue(infinity);
        nnLCpu.densityMap.resetValue(infinity);
        nnLCpu.disjointSetMap.resetValue(infinity);
        nnLCpu.activeMap.resetValue(1);// QWB 241016 only for mst, specially initialize activeMap = 1, to mark the thread running shortest outgoing
        nnLCpu.fixedMap.resetValue(1);// QWB 241016, only for mst, specially initialize fixedmap = 1 to mark initial winner nodes
        nnLCpu.minRadiusMap.resetValue(0);
        nnLCpu.sizeOfComponentMap.resetValue(1);
        PointCoord pInitial(-1);
        nnLCpu.correspondenceMap.resetValue(pInitial);
    }

    /**
     * KERNEL CALLS
     */

    template <class CellularMatrix, class Grid, class Grid2 >
    GLOBAL inline void K_initializeSpiralSearch(CellularMatrix& cm, Grid& g_point, Grid2& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_point.getWidth(),
                              g_point.getHeight(),
                              g_point.getDepth());

        K_EMST_initializeSpiralSearch _KER_CALL_(b, t)(cm, g_point, g_ss);
    }

    template <class CellularMatrix, class Grid >
    GLOBAL inline void K_refreshCell(CellularMatrix& cm, Grid& g) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g.getWidth(),
                              g.getHeight(),
                              g.getDepth());

        K_EMST_refreshCell _KER_CALL_(b, t)(cm, g);
    }

    template <class CellularMatrix, class Grid >
    GLOBAL inline void K_refreshCell_cpu(CellularMatrix& cm, Grid& g) {

        K_EMST_refreshCell_cpu (cm, g);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_computeOctants(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr, Grid6& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_EMST_computeOctants _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr, g_ss);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_computeOctant(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr, Grid6& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_EMST_computeOctant _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr, g_ss);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_findNextClosestPoint(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr, Grid6& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_EMST_findNextClosestPoint _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr, g_ss);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_FindNextClosestPoint(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_EMST_FindNextClosestPoint _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr);
    }

    /**
     * Termination test
     */
    template<class Grid>
    bool testTermination(Grid& testGrid){

        typedef typename Grid::index_type index_type;

        //GLint numTest = testGrid(index_type(0));
        GLint numTest = testGrid.findRoot(testGrid, testGrid.compute_offset(index_type(0)));
        int nComp = 0;
        for (int j = 0; j < testGrid.height; j++ )
            for (int i = 0; i < testGrid.width; i++)
            {
                index_type ps(i, j);
                GLint r = testGrid(ps);
                //GLint r = testGrid.findRoot(testGrid, testGrid.compute_offset(ps));
                if (numTest != r)
                    nComp += 1;// testGrid[j][i];
            }
        return nComp;

    }

    template <class Grid1, class Grid2, class Grid3 >
    GLOBAL void K_evaluate_ST(Grid1& g_link, Grid2& g_point, Grid3& g_obj) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        K_EMST_evaluate_ST _KER_CALL_(b, t)(g_link, g_point, g_obj);
    }

    template <class Grid1, class Grid2, class Grid3 >
    GLOBAL void K_length_ST(Grid1& g_link, Grid2& g_point, Grid3& g_obj) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        K_EMST_length_ST _KER_CALL_(b, t)(g_link, g_point, g_obj);
    }

    template <class Grid >
    GLOBAL void K_initDisjointSet(Grid& g) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g.getWidth(),
                              g.getHeight(),
                              g.getDepth());

        K_EMST_initDisjointSet _KER_CALL_(b, t)(g);
    }

    template <class Grid >
    GLOBAL void K_flatten_DST_0(Grid& g_dstree) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_EMST_flatten_DST _KER_CALL_(b, t)(g_dstree);
    }

    template <class Grid, class Grid2 >
    GLOBAL void K_flatten_DST(Grid& g_dstree, Grid2& g_parent) {

        g_parent.gpuResetValue(-1);

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_EMST_flatten_DST_1 _KER_CALL_(b, t)(g_dstree, g_parent);
        g_parent.gpuCopyDeviceToDevice(g_dstree);
    }

    template <class Grid >
    GLOBAL void K_clearLinks(Grid& g) {

         KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g.getWidth(),
                              g.getHeight(),
                              g.getDepth());

        K_EMST_clearLinks _KER_CALL_(b, t) (g);
    }

    template <class Grid1,
              class Grid2,
              class Grid3,
              class Grid4,
              class Grid5,
              class Grid6,
              class Grid7,
              class Grid8,
              class Grid9,
              class Grid10,
              class Grid11,
              class Grid12
              >
    GLOBAL void K_findMinInComponentDB(Grid1& g_link,
                                       Grid2& g_dstree,
                                       Grid3& g_fix,
                                       Grid4& g_corr,
                                       Grid5& g_dist,
                                       Grid6 g_evt,
                                       Grid7 g_vnum,
                                       Grid8 g_parent,
                                       Grid9 g_win,
                                       Grid10 g_dest,
                                       Grid11 g_minD,
                                       Grid12 g_state,
                                       Grid12 g_state_cpu
                                       ) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        g_state.gpuResetValue(-2);
        while (true) {

            K_EMST_findMinDBActivate_1 _KER_CALL_(b, t)(g_link,
                                                         g_dstree,
                                                         g_evt,
                                                         g_vnum,
                                                         g_parent,
                                                         g_state
                                                         );
            g_state_cpu.gpuCopyDeviceToHost(g_state);

            BOp op;
            int result = 0;
            op.K_sumReduction(g_state_cpu, result);
            if (result == -g_link.getWidth())
                break;
        }
        while (true) {

            K_EMST_findMinDBActivate_2 _KER_CALL_(b, t)(g_link,
                                                         g_dstree,
                                                         g_fix,
                                                         g_corr,
                                                         g_dist,
                                                         g_evt,
                                                         g_vnum,
                                                         g_parent,
                                                         g_win,
                                                         g_dest,
                                                         g_minD,
                                                         g_state
                                                         );
            g_state_cpu.gpuCopyDeviceToHost(g_state);

            BOp op;
            int result = 0;
            op.K_sumReduction(g_state_cpu, result);
            if (!result)
                break;
        }
    }

    template <class Grid1,
              class Grid2,
              class Grid6,
              class Grid7,
              class Grid8,
              class Grid12
              >
    GLOBAL void K_diffusateDetectCycleDB(Grid1& g_link,
                                       Grid2& g_dstree,
                                       Grid6 g_evt,
                                       Grid7 g_vnum,
                                       Grid8 g_parent,
                                       Grid12 g_state,
                                       Grid12 g_state_cpu
                                       ) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        int i = 71009;
        int result = 0;
        while (--i) {

            K_EMST_diffusateDetectCycle _KER_CALL_(b, t)(g_link,
                                                         g_dstree,
                                                         g_evt,
                                                         g_vnum,
                                                         g_parent,
                                                         g_state
                                                         );
            g_state_cpu.gpuCopyDeviceToHost(g_state);

            BOp op;
            result = 0;
            op.K_sumReduction(g_state_cpu, result);
            if (result == -g_link.getWidth())
                break;
        }
//        K_EMST_eliminateCycle _KER_CALL_(b, t) (g_link,
//                              g_dstree,
//                              g_evt,
//                              g_vnum,
//                              g_parent,
//                              g_state
//                              );
        cout << "DIFFUSION !!!!!!!!!!!!!!!! " << result << endl;
    }

    template <class Grid1,
              class Grid2,
              class Grid6,
              class Grid7,
              class Grid8,
              class Grid12
              >
    GLOBAL void K_diffusateDetectCycleDB_2(Grid1& g_link,
                                       Grid2& g_dstree,
                                       Grid6 g_evt,
                                       Grid7 g_vnum,
                                       Grid8 g_parent,
                                       Grid12 g_state,
                                       Grid12 g_state_cpu
                                       ) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());


            K_EMST_diffusateDetectCycle_2 _KER_CALL_(b, t)(g_link,
                                                         g_dstree,
                                                         g_evt,
                                                         g_vnum,
                                                         g_parent
                                                         );
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_FindMinInComponent(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr, Grid5& g_dist) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        K_EMST_findMinInComponent _KER_CALL_(b, t)(g_link, g_dstree, g_fix, g_corr, g_dist);
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_FindMinInComponent_cpu(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr, Grid5& g_dist) {

        K_EMST_findMinInComponent_cpu(g_link, g_dstree, g_fix, g_corr, g_dist);
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4 >
    GLOBAL void K_connectComponentAndUnion(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        K_EMST_connectComponentAndUnion _KER_CALL_(b, t)(g_link, g_dstree, g_fix, g_corr);
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_connectComponentAndUnion(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr, Grid5& g_parent) {

        g_dstree.gpuCopyDeviceToDevice(g_parent);

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        K_EMST_connectComponentAndUnion_2 _KER_CALL_(b, t)(g_link, g_dstree, g_fix, g_corr, g_parent);
    }

    template <typename Grid, typename Grid2 >
    GLOBAL void K_createComponentList(Grid& g_link, Grid& g_dstree, Grid2& g_corr) {

        // Init to -1

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_dstree.getWidth(),
                              g_dstree.getHeight(),
                              g_dstree.getDepth());

        K_NEMST_createComponentList _KER_CALL_(b, t)(g_link, g_dstree, g_corr);
    }

    template <class Grid1,
              class Grid2,
              class Grid3,
              class Grid4,
              class Grid5
              >
    GLOBAL void K_findMinPair(Grid1& g_link,
                              Grid2& g_dstree,
                              Grid3& g_fix,
                              Grid4& g_corr,
                              Grid5& g_dist
                              )
    {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                              EMST_BLOCK_SIZE, 1, 1,
                              g_link.getWidth(),
                              g_link.getHeight(),
                              g_link.getDepth());

        K_NEMST_findMinPair _KER_CALL_(b, t)(g_link,
                            g_dstree,
                            g_fix,
                            g_corr,
                            g_dist
                            );
    }
};

}//namespace operators

#endif // EMST_OPERATORS_H

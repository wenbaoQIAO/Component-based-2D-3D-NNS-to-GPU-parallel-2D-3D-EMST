#ifndef NODE_EMST_H
#define NODE_EMST_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao, J.C. Créput
 * Creation date : June. 2016
 * Add a new class Point2DInt
 ***************************************************************************
 */
#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include "macros_cuda.h"

#include <device_atomic_functions.h>
#include <sm_60_atomic_functions.h>
#include <sm_61_intrinsics.h>

//! reference basic componnets
#include "Node.h"
#include "NIter.h"
#include "distances_matching.h"
#include "GridOfNodes.h"
#include "CellularMatrix.h"

#define TEST_CODE 0
#define IS_INF_TEST_AS_PROCEDURE 0
#define SLAB_TECHNIQUE 1
#define NEMST_TRACE 0
#define NEMST_TEST_MY_SLAB 0
#define NEMST_TEST_PREVIOUS 1
#define CREATE_RESTRICTED_COMP_LST 1

using namespace std;

namespace components
{
#if IS_INF_TEST_AS_PROCEDURE
template <typename GLint, typename GLdouble >
DEVICE_HOST inline bool EMST_isInf(GLint id1x, GLint id2x, GLdouble d1, GLint idd1x, GLint idd2x, GLdouble d2) {

    return (d1 == d2) ?
        ((id1x == idd1x) ? (id2x < idd2x) : (id1x < idd1x))
        : d1 < d2;
}
#endif

/***************************************************
 * FIND MIN1 2D CM
 * *************************************************
 */

template<unsigned int N, unsigned int DimCM>
struct Power
{
    enum {Value = N * Power<N, DimCM-1>::Value};
};

template<GLuint N>
struct Power<N, 0>
{
    enum {Value = 1};
};

template <typename IndexCM, typename Index, GLuint DimCM >
struct NodeSpiralSearchMD {

    static const unsigned int TWO_POWER_DIM = Power<2, DimCM>::Value;
    static const unsigned int TWO_POWER_DIM_MOINS_UN = Power<2, DimCM-1>::Value;
    static const unsigned int NFACE = DimCM * 2;
    static const unsigned int NSLAB = TWO_POWER_DIM_MOINS_UN;

    // Slab closed or open
    bool slabC[NFACE][TWO_POWER_DIM_MOINS_UN];
    // Slab point offset with adaptiveMap units
    GLint slabP[NFACE][TWO_POWER_DIM_MOINS_UN];
    GLfloatP slabDP[NFACE][TWO_POWER_DIM_MOINS_UN];

    // Face closed
    bool faceC[NFACE];
    bool searchFinished;

    // Cuurent minimum bound
    //Index minB;
    //GLfloatP minDistB;

    // Cell center in CM
    IndexCM pc;
    // Node that searches
    Index ps;

    GLuint d_min;
    GLuint d_max;

    DEVICE_HOST NodeSpiralSearchMD() {}

    DEVICE_HOST NodeSpiralSearchMD(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            )
        :
          pc(_pc),
          ps(_ps),
          d_min(_d_min),
          d_max(_d_max)
    {}

    DEVICE_HOST void init(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            ) {
          pc = _pc;
          ps = _ps;
          d_min = _d_min;
          d_max = _d_max;
          for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
              for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                  slabC[no_face][no_slab] = false;
                  slabP[no_face][no_slab] = -1;
                  slabDP[no_face][no_slab] = HUGE_VAL;
              }
          }
          // Validation
          for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
              faceC[no_face] = false;
          }
          searchFinished = false;
    }

    template <typename Grid,
              typename Grid2,
              typename Grid3,
              typename Grid4,
              typename Grid5  >
    DEVICE bool search(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr)  {

        typedef typename Grid::index_type index_type_cm;
        typedef IterIndex<DimCM> iterator_type;
        typedef typename Grid::point_type cell_type;
        typedef typename Grid2::index_type index_type;
        typedef typename Grid3::point_type point_type;
        typedef typename point_type::coord_type coord_type;

        bool modified = false;
        bool ret = false;

#if SLAB_TECHNIQUE
        if (this->searchFinished) {
            //printf("all slab-faces closed %d \n", ps[0]);
            return ret;
        }
#endif
        Index ps = this->ps;
#if NEMST_TEST_PREVIOUS
        index_type pcorr = g_corr(ps);
        if (pcorr[0] != -1 && g_dstree(pcorr) != g_dstree(ps)) {
            return true;
        }
        else {
            g_corr(ps) = index_type(-1);
            g_dist(ps) = HUGE_VAL;
        }
#endif
        // Slab validated
        bool faceV[NFACE];
        bool slabC[NFACE][TWO_POWER_DIM_MOINS_UN];
        GLint slabP[NFACE][TWO_POWER_DIM_MOINS_UN];
        GLfloatP slabDP[NFACE][TWO_POWER_DIM_MOINS_UN];

        // Validation
        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            faceV[no_face] = faceC[no_face];
        }

#if SLAB_TECHNIQUE
        bool validation = true;
        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            validation = validation && faceV[no_face];
            if (!validation)
                break;
        }
        if (validation) {
            //printf("searchFinished %d \n", ps[0]);
            this->searchFinished = true;
            return ret;
        }
#endif
        IndexCM pc = this->pc;

        GLuint d_min = this->d_min;
        GLuint d_max = this->d_max;

        point_type pps = g_point(ps);

        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                slabC[no_face][no_slab] = this->slabC[no_face][no_slab];
                slabP[no_face][no_slab] = this->slabP[no_face][no_slab];
                slabDP[no_face][no_slab] = this->slabDP[no_face][no_slab];
            }
        }
        // Radius
        GLuint radius = d_min;
        index_type minPCoord(-1);
        GLdouble minDistance = HUGE_VAL;

        // Current distance to pc
        while (radius <= d_max) {

            bool validation = true;
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
                validation = validation && faceV[no_face];
                if (!validation)
                    break;
            }
            if (validation) {
                //printf("all face validation radius %d \n", radius);
                break;
            }

            // Current face
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {

                if (!faceV[no_face]) {

                    GLuint dir = no_face / 2;
                    GLuint sens = (no_face % 2) ? -1 : 1;

                    index_type_cm radius_base_s(0);
                    radius_base_s[dir] = radius * sens;

                    index_type_cm base_s = pc + radius_base_s;
#if NEMST_TRACE
                    if (ps[0] == 0)
                        printf("base_s %d %d %d \n", base_s[0],
                                base_s[1], base_s[2]);
#endif
                    // Detect face validation
                    if (!cm.valideAndPositiveIndex(base_s)) {
                        faceV[no_face] = true;
                    }

                    // Slab closed validation
                    if (!faceV[no_face]) {
                        bool validation = true;
                        for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                            validation = validation && slabC[no_face][no_slab];
                            if (!validation)
                                break;
                        }
                        if (validation) {
#if SLAB_TECHNIQUE
                            faceC[no_face] = true;
#endif
                            faceV[no_face] = true;
                        }
                    }

                    // Geometric upper bound validation
                    if (!faceV[no_face] && radius > 0 && (minPCoord[0] != -1)) {
                        GLdouble ppsj = pps[dir];

                        // Axis to consider
                        index_type_cm P1(0);
                        index_type_cm P2(0);
                        P2[dir] = 1;

                        point_type p1 = cm.vgd.FEuclid(cm.vgd.FDual(P1));
                        point_type p2 = cm.vgd.FEuclid(cm.vgd.FDual(P2));

                        // Two axis slabs
                        GLdouble size_slab = p2[dir]-p1[dir];

                        point_type pbase_s = cm.vgd.FEuclid(cm.vgd.FDual(base_s));
                        GLdouble dist = abs(pbase_s[dir] - ppsj) - size_slab / 2;
                        dist *= dist;
                        if (dist > minDistance)
                            faceV[no_face] = true;
                    }

                    // If the face has to be searched
                    if (!faceV[no_face]) {
                        // For each slab of the face
                        for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab)
                        {
                            if (!slabC[no_face][no_slab]) {
#if SLAB_TECHNIQUE
                                GLint p = slabP[no_face][no_slab];
                                if (p != -1 && radius > 1) {
                                    // Detect if becoming closed
                                    // Axis to consider
                                    index_type_cm P1(0);
                                    index_type_cm P2(0);
                                    P2[dir] = 1;
                                    point_type p1 = cm.vgd.FEuclid(cm.vgd.FDual(P1));
                                    point_type p2 = cm.vgd.FEuclid(cm.vgd.FDual(P2));
                                    // Two axis slabs
                                    GLdouble size_slab = p2[dir]-p1[dir];

                                    point_type pbase_s = cm.vgd.FEuclid(cm.vgd.FDual(base_s));
                                    pbase_s[dir] = pbase_s[dir] - sens * (size_slab / 2);
                                    point_type pi;
                                    point_type pp = g_point(p);
                                    pp = pp - pps;
                                    pi = pp * 0.5;
                                    pi = pbase_s - pi;
                                    if (pi * pp > 0) {
                                        modified = true;
                                        slabC[no_face][no_slab] = true;
                                        //printf("slab closed %d %d \n", no_face, no_slab);
                                    }
                                }//slabP
#endif
                                if (!slabC[no_face][no_slab]) {
                                    index_type_cm ext_s(radius+1);
                                    ext_s[dir] = 1;

                                    index_type_cm sign_s(-1);
                                    sign_s[dir] = 0;
                                    GLuint mask = 1;
                                    for (int i = 0; i < DimCM; ++i) {
                                        if (i != dir) {
                                            sign_s[i] = (no_slab & mask) ? -1 : 1 ;
                                            mask = mask << 1;
                                        }
                                    }

                                    // Search in current slab
                                    index_type_cm idx(0);
                                    iterator_type iter;
                                    iter.reset(idx);
                                    while (iter.next(idx, ext_s)) {
                                        index_type_cm dept = idx * sign_s;
                                        index_type_cm pcell = base_s + dept;

                                        if (cm.valideAndPositiveIndex(pcell)) {

#if NEMST_TRACE
                                            if (ps[0] == 0)
                                                printf("NSLAB dept %d %d %d %d \n", NSLAB, pcell[0],
                                                        pcell[1], pcell[2]);
#endif
                                            // cell pointer
                                            cell_type* cell = &cm(pcell);

                                            size_t count = 0;
                                            while (count < cell->size)
                                            {
                                                index_type pco = cell->bCell[count];
                                                if (g_corr.valideIndex(pco))
                                                {
                                                    point_type ppco = g_point(pco);

                                                        if (g_dstree(pco) != g_dstree(ps)) {
                                                            GLdouble v =
                                                                    components::DistanceSquaredEuclideanP<point_type>()(
                                                                        pps,
                                                                        ppco
                                                                        );
                                                            if (minPCoord[0] == -1)
                                                            {
                                                                ret = true;
                                                                minPCoord = pco;
                                                                minDistance = v;
                                                                d_min = radius;
                                                            }
                                                            else {
                                                                GLint psf = g_dstree.compute_offset(ps);
                                                                GLint pcof = g_dstree.compute_offset(pco);
                                                                GLint minPCoordf = g_dstree.compute_offset(minPCoord);
                                                                GLint id1x = MIN(psf,pcof);
                                                                GLint id2x = MAX(psf,pcof);
                                                                GLint idd1x = MIN(psf,minPCoordf);
                                                                GLint idd2x = MAX(psf,minPCoordf);
                                                                if (EMST_isInf(
                                                                            id1x,
                                                                            id2x,
                                                                            v,
                                                                            idd1x,
                                                                            idd2x,
                                                                            minDistance))
                                                                {
                                                                    ret = true;
                                                                    minPCoord = pco;
                                                                    minDistance = v;
                                                                }
                                                            }
                                                        }//if ! in component
#if SLAB_TECHNIQUE
                                                        else {
                                                            GLdouble v =
                                                                    components::DistanceSquaredEuclideanP<point_type>()(
                                                                        pps,
                                                                        ppco
                                                                        );
                                                            if (v > 0 && pco != ps) {
                                                                ppco = ppco - pps;
                                                                GLdouble maxd = 0;
                                                                for (GLuint dir = 0; dir < DimCM; ++dir) {
                                                                    if (abs(ppco[dir]) > maxd)
                                                                        maxd = abs(ppco[dir]);
                                                                }

                                                                for (GLuint dir = 0; dir < DimCM; ++dir) {

                                                                    if (abs(ppco[dir]) >= maxd) {
                                                                        GLuint no_f = dir * 2;
                                                                        no_f += (ppco[dir] >= 0) ? 0 : 1;
                                                                        for (GLuint no_s = 0; no_s < NSLAB; ++no_s) {
                                                                            // Check if in slab

                                                                            // Test if reason to test
                                                                            GLint p = slabP[no_f][no_s];
                                                                            GLdouble minD = slabDP[no_f][no_s];

                                                                            if (p == -1 || v < minD)
                                                                            {
                                                                                GLuint mask = 1;
                                                                                bool inSlab = true;
                                                                                for (int i = 0; i < DimCM; ++i) {
                                                                                    if (i != dir) {
                                                                                        GLdouble sign = (no_s & mask) ? -1 : 1 ;
                                                                                        inSlab = inSlab && ((sign*ppco[i]) >= 0);
                                                                                        if (!inSlab)
                                                                                            break;
                                                                                        mask = mask << 1;
                                                                                    }
                                                                                }
                                                                                if (inSlab) {//ppco in slab
                                                                                    GLint pcof = g_point.compute_offset(pco);

                                                                                    if (p == -1)
                                                                                    {
                                                                                        modified = true;
                                                                                        slabP[no_f][no_s] = pcof;
                                                                                        slabDP[no_f][no_s] = v;
                                                                                    }
                                                                                    else {
                                                                                        GLint psf = g_point.compute_offset(ps);
                                                                                        GLint id1x = MIN(psf,pcof);
                                                                                        GLint id2x = MAX(psf,pcof);
                                                                                        GLint idd1x = MIN(psf,p);
                                                                                        GLint idd2x = MAX(psf,p);
                                                                                        if (EMST_isInf(
                                                                                                    id1x,
                                                                                                    id2x,
                                                                                                    v,
                                                                                                    idd1x,
                                                                                                    idd2x,
                                                                                                    minD))
                                                                                        {
                                                                                            modified = true;
                                                                                            slabP[no_f][no_s] = pcof;
                                                                                            slabDP[no_f][no_s] = v;
                                                                                        }
                                                                                    }//else
                                                                                }//if
                                                                            }//(p == -1 || v <= MinD)
                                                                        }//for each slab
                                                                    }//if
                                                                }//for

                                                            }//(v > 0)

                                                        }//in component
#endif
                                                }//if valid
                                                count++;
                                            }//while in cell
                                        }//if valid cell
                                    }//while inside a given slab
                                }//if !slabC
                            }//if !slabC
                        }//for each slab
                    }//if (!faceV[no_face])
                }// face not validated
            }// for face
            radius++;
        }//while r <= d_max

        if (ret)
        {
            g_corr(ps) = minPCoord;
            g_dist(ps) = minDistance;
            this->d_min = d_min;
        }
#if NEMST_TRACE
        else
            if (ps[0] == 0)
                printf("erreur search not found %d %d \n", radius, d_max);
#endif
        if (modified) {
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
                for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                    this->slabC[no_face][no_slab] = slabC[no_face][no_slab];
                    this->slabP[no_face][no_slab] = slabP[no_face][no_slab];
                    this->slabDP[no_face][no_slab] = slabDP[no_face][no_slab];
                }
            }
        }
        return ret;
    }
};

/***************************************************
 * FIND MIN1 2D CM
 * *************************************************
 */

template <typename IndexCM, typename Index >
struct NodeSpiralSearch {

    IndexCM pc;//Coordinates cell center in gdc
    Index ps;//Coordinates cell center in gdc

    size_t d_min;
    size_t d_max;

    DEVICE_HOST NodeSpiralSearch() {}

    DEVICE_HOST NodeSpiralSearch(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            )
        :
          pc(_pc),
          ps(_ps),
          d_min(_d_min),
          d_max(_d_max)
    {}

    DEVICE_HOST void init(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            ) {
          pc = _pc;
          ps = _ps;
          d_min = _d_min;
          d_max = _d_max;
    }

    template <typename Grid,
              typename Grid2,
              typename Grid3,
              typename Grid4,
              typename Grid5  >
    DEVICE bool search(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr)  {

        typedef typename Grid::index_type index_type_cm;
        typedef typename Grid::point_type cell_type;
        typedef typename Grid2::index_type index_type;
        typedef typename Grid3::point_type point_type;
        typedef typename point_type::coord_type coord_type;

        IndexCM pc = this->pc;
        Index ps = this->ps;

        size_t d_min = this->d_min;
        size_t d_max = this->d_max;

        bool ret = false;
        NIterQuad ni(PointCoord(pc[0], pc[1]), d_min, d_max);
        ni.setCurrentDistance(d_min);

        index_type minPCoord(-1);
        GLdouble minDistance = HUGE_VAL;

        do {

            if (ni.getCurrentDistance() > d_max)
                break;

            PointCoord pcoord = ni.getNodeIncr();
            index_type_cm pcell(pcoord[0], pcoord[1]);

            if (cm.valideAndPositiveIndex(pcell)) {

                //! wb.Q change to use pointers
                cell_type* cell = &cm(pcell);

                size_t count = 0;
                while (count < cell->size)
                {
                    index_type pco = cell->bCell[count];
                    if (g_corr.valideIndex(pco))
                    {
                        if(g_dstree(pco) != g_dstree(ps)){

                            GLdouble v =
                            components::DistanceSquaredEuclideanP<point_type>()(
                                        g_point(ps),
                                        g_point(pco)
                                        );

                            if (minPCoord[0] == -1)
                            {
                                ni.setMaxDistance(ni.getCurrentDistance() + (int)ceil((float) ni.getCurrentDistance()*0.5) + 1);
                                ret = true;
                                minPCoord = pco;
                                minDistance = v;
                                //d_min = ni.getCurrentDistance();
                            }
                            else {
                                GLint psf = g_dstree.compute_offset(ps);
                                GLint pcof = g_dstree.compute_offset(pco);
                                GLint minPCoordf = g_dstree.compute_offset(minPCoord);
                                GLint id1x = MIN(psf,pcof);
                                GLint id2x = MAX(psf,pcof);
                                GLint idd1x = MIN(psf,minPCoordf);
                                GLint idd2x = MAX(psf,minPCoordf);
                                if (EMST_isInf(
                                                id1x,
                                                id2x,
                                                v,
                                                idd1x,
                                                idd2x,
                                                minDistance))
                                {
                                    ni.setMaxDistance(ni.getCurrentDistance() + (int)ceil((float) ni.getCurrentDistance()*0.5) + 1);
                                    ret = true;
                                    minPCoord = pco;
                                    minDistance = v;
                                }
                            }
                        }
                    }
                    count++;
                }//while cell
            }
        } while (ni.nextNodeIncr());

        if (ret)
        {
            g_corr(ps) = minPCoord;
            g_dist(ps) = minDistance;
            this->d_min = d_min;
        }

        return ret;
    }

};

/***************************************************
 * FIND MIN1 2D CM sliced spiral search
 * wb.Q transfer old code to this plateform, not totally transformed from old version yet
 * *************************************************
 */

template <typename IndexCM, typename Index >
struct NodeSLicedSpiralSearch {

    IndexCM pc;//Coordinates cell center in gdc
    Index ps;//Coordinates cell center in gdc

    size_t d_min;
    size_t d_max;

    DEVICE_HOST NodeSLicedSpiralSearch() {}

    DEVICE_HOST NodeSLicedSpiralSearch(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            )
        :
          pc(_pc),
          ps(_ps),
          d_min(_d_min),
          d_max(_d_max)
    {}

    DEVICE_HOST void init(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            ) {
          pc = _pc;
          ps = _ps;
          d_min = _d_min;
          d_max = _d_max;
    }

    template <typename Grid,
              typename Grid2,
              typename Grid3,
              typename Grid4,
              typename Grid5  >
    DEVICE bool search(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr)  {

        typedef typename Grid::index_type index_type_cm;
        typedef typename Grid::point_type cell_type;
        typedef typename Grid2::index_type index_type;
        typedef typename Grid3::point_type point_type;
        typedef typename point_type::coord_type coord_type;

        IndexCM pc = this->pc;
        Index ps = this->ps;

        size_t d_min = this->d_min;
        size_t d_max = this->d_max;

        bool ret = false;
        NIterQuad ni(PointCoord(pc[0], pc[1]), d_min, d_max);
        ni.setCurrentDistance(d_min);

        index_type minPCoord(-1);
        GLdouble minDistance = HUGE_VAL;

        do {

            if (ni.getCurrentDistance() > d_max)
                break;

            PointCoord pcoord = ni.getNodeIncr();
            index_type_cm pcell(pcoord[0], pcoord[1]);

            if (cm.valideAndPositiveIndex(pcell)) {

                //! wb.Q change to use pointers
                cell_type* cell = &cm(pcell);

                size_t count = 0;
                while (count < cell->size)
                {
                    index_type pco = cell->bCell[count];
                    if (g_corr.valideIndex(pco))
                    {
                        if(g_dstree(pco) != g_dstree(ps)){

                            GLdouble v =
                            components::DistanceSquaredEuclideanP<point_type>()(
                                        g_point(ps),
                                        g_point(pco)
                                        );

                            if (minPCoord[0] == -1)
                            {
                                ni.setMaxDistance(ni.getCurrentDistance() + (int)ceil((float) ni.getCurrentDistance()*0.5) + 1);
                                ret = true;
                                minPCoord = pco;
                                minDistance = v;
                                //d_min = ni.getCurrentDistance();
                            }
                            else {
                                GLint psf = g_dstree.compute_offset(ps);
                                GLint pcof = g_dstree.compute_offset(pco);
                                GLint minPCoordf = g_dstree.compute_offset(minPCoord);
                                GLint id1x = MIN(psf,pcof);
                                GLint id2x = MAX(psf,pcof);
                                GLint idd1x = MIN(psf,minPCoordf);
                                GLint idd2x = MAX(psf,minPCoordf);
                                if (EMST_isInf(
                                                id1x,
                                                id2x,
                                                v,
                                                idd1x,
                                                idd2x,
                                                minDistance))
                                {
                                    ni.setMaxDistance(ni.getCurrentDistance() + (int)ceil((float) ni.getCurrentDistance()*0.5) + 1);
                                    ret = true;
                                    minPCoord = pco;
                                    minDistance = v;
                                }
                            }
                        }
                    }
                    count++;
                }//while cell
            }
        } while (ni.nextNodeIncr());

        if (ret)
        {
            g_corr(ps) = minPCoord;
            g_dist(ps) = minDistance;
            this->d_min = d_min;
        }

        return ret;
    }

};

/***************************************************
 * FIND MIN2 PROCEDURES
 * *************************************************/
struct Maillon
{
    template <typename Grid >
    DEVICE_HOST void insert(Grid& g_link, GLint mine, GLint prev)
    {
        GLint old, link = g_link(prev);
        do {
            old = link;
            g_link(mine) = old;
            link = atomicCAS(&g_link(prev), link, mine);
        } while (link != old);
    }
};

#if CREATE_RESTRICTED_COMP_LST
// g_list is previously initialized to -1
template <typename Grid, typename Grid2 >
KERNEL void K_NEMST_createComponentList(Grid g_link, Grid g_dstree, Grid2 g_corr) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

    typename Grid::index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_corr(ps)[0] != -1) {
            GLint mine = g_dstree.compute_offset(ps);
            GLint prev = g_dstree(ps);

            if (mine != prev) {
                GLint old;
                GLint link = g_link(prev);
                do {
                    old = link;
                    g_link(mine) = old;
                    link = atomicCAS(&g_link(prev), link, mine);
                } while (link != old);
            }
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}
#else
template <typename Grid >
KERNEL void K_NEMST_createComponentList(Grid g_link, Grid g_dstree) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

    typename Grid::index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        GLint mine = g_dstree.compute_offset(ps);
        GLint prev = g_dstree(ps);

        if (mine != prev) {
            GLint old;
            GLint link = g_link(prev);
            do {
                old = link;
                g_link(mine) = old;
                link = atomicCAS(&g_link(prev), link, mine);
            } while (link != old);
        }
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}
#endif
template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5
          >
KERNEL inline void K_NEMST_findMinPair(Grid1 g_link,
                                       Grid2 g_dstree,
                                       Grid3 g_fix,
                                       Grid4 g_corr,
                                       Grid5 g_dist
                                       )
{
    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

    typename Grid1::index_type ps(_x, _y, _z);

    if (g_link.valideIndex(ps))
    {
        GLint mine = g_dstree.compute_offset(ps);

        if (g_dstree(ps) == mine) { // root
            typedef typename Grid1::index_type index_type;
            typedef typename Grid1::point_type point_glink_type;
            typedef typename Grid4::point_type point_gcorr_type;

            GLint win_node = -1;
            GLint dest_node = -1;
            GLdouble minDist = HUGE_VAL;
            GLint w_node = -1;
            GLint d_node = -1;
            GLdouble minD = HUGE_VAL;

            GLint pco = mine;
            while (pco != -1) {

                w_node = -1;
                d_node = -1;
                minD = HUGE_VAL;
                index_type pco2 = g_link.back_offset(pco);
                index_type ppco = g_corr(pco2);
                if (ppco[0] != -1) {
                    w_node = pco;
                    d_node = g_dstree.compute_offset(ppco);
                    minD = g_dist(pco);
                }

                if ((d_node != -1) && (dest_node != -1)) {

                    GLint id1x = MIN(w_node,d_node);
                    GLint id2x = MAX(w_node,d_node);
                    GLint idd1x = MIN(win_node,dest_node);
                    GLint idd2x = MAX(win_node,dest_node);

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
                else if ((dest_node == -1) && (d_node != -1)) {
                    win_node = w_node;
                    dest_node = d_node;
                    minDist = minD;
                }

                // Next element of the list
                pco = g_link(pco);
            }//while

            // Write result
            if (win_node != -1)//index_type(-1))
                g_fix(win_node) = 1;
        }//root
    }
    END_KER_SCHED_3D

    SYNCTHREADS
}

/*************************************************************
 * Compute closest slab points
 * ***********************************************************/
template <typename IndexCM, typename Index, GLuint DimCM >
struct NodeComputeOctantMD {

    static const unsigned int TWO_POWER_DIM = Power<2, DimCM>::Value;
    static const unsigned int TWO_POWER_DIM_MOINS_UN = Power<2, DimCM-1>::Value;
    static const unsigned int NFACE = DimCM * 2;
    static const unsigned int NSLAB = TWO_POWER_DIM_MOINS_UN;

    // Slab point offset with adaptiveMap units
    GLint slabP[NFACE][TWO_POWER_DIM_MOINS_UN];
    GLfloatP slabDP[NFACE][TWO_POWER_DIM_MOINS_UN];

    // Cell center in CM
    IndexCM pc;
    // Node that searches
    Index ps;

    GLuint d_min;
    GLuint d_max;

    DEVICE_HOST NodeComputeOctantMD() {}

    DEVICE_HOST NodeComputeOctantMD(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            )
        :
          pc(_pc),
          ps(_ps),
          d_min(_d_min),
          d_max(_d_max)
    {}

    DEVICE_HOST void init(
            IndexCM _pc,
            Index _ps,
            int _d_min,
            int _d_max
            ) {
          pc = _pc;
          ps = _ps;
          d_min = _d_min;
          d_max = _d_max;
          for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
              for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                  slabP[no_face][no_slab] = -1;
                  slabDP[no_face][no_slab] = HUGE_VAL;
              }
          }
    }

    template <typename Grid,
              typename Grid2,
              typename Grid3,
              typename Grid4,
              typename Grid5  >
    DEVICE bool computeOctants(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr)  {

        typedef typename Grid::index_type index_type_cm;
        typedef IterIndex<DimCM> iterator_type;
        typedef typename Grid::point_type cell_type;
        typedef typename Grid2::index_type index_type;
        typedef typename Grid3::point_type point_type;
        typedef typename point_type::coord_type coord_type;

        bool modified = false;
        bool ret = false;

        Index ps = this->ps;

        // Slab validated
        bool faceV[NFACE];
        bool slabC[NFACE][TWO_POWER_DIM_MOINS_UN];

        GLint slabP[NFACE][TWO_POWER_DIM_MOINS_UN];
        GLfloatP slabDP[NFACE][TWO_POWER_DIM_MOINS_UN];

        // Validation
        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            faceV[no_face] = false;
        }

        IndexCM pc = this->pc;

        GLuint d_min = this->d_min;
        GLuint d_max = this->d_max;

        point_type pps = g_point(ps);

        bool allinit = true;
        bool atleastonezero = false;
        bool nozero = true;
        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                slabC[no_face][no_slab] = false;
                slabP[no_face][no_slab] = this->slabP[no_face][no_slab];
                slabDP[no_face][no_slab] = this->slabDP[no_face][no_slab];
                if (slabDP[no_face][no_slab] == 0) {
                    nozero = false;
                    atleastonezero = true;
                }
                else if (slabP[no_face][no_slab] != -1)
                    allinit = false;
            }
        }
        if (atleastonezero) {
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
                for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                    slabP[no_face][no_slab] = -1;
                    slabDP[no_face][no_slab] = HUGE_VAL;
                }
            }

        }
        if (nozero && !allinit)
            return true;
        // Radius
        GLuint radius = d_min;

        // Current distance to pc
        while (radius <= d_max) {

            bool validation = true;
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
                validation = validation && faceV[no_face];
                if (!validation)
                    break;
            }
            if (validation) {
                //printf("all face validation radius %d \n", radius);
                break;
            }

            // Current face
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {

                if (!faceV[no_face]) {

                    GLuint dir = no_face / 2;
                    GLuint sens = (no_face % 2) ? -1 : 1;

                    index_type_cm radius_base_s(0);
                    radius_base_s[dir] = radius * sens;

                    index_type_cm base_s = pc + radius_base_s;
                    // Detect face validation
                    if (!cm.valideAndPositiveIndex(base_s)) {
                        faceV[no_face] = true;
                        for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                            //if (slabP[no_face][no_slab] != -1) {
                                slabC[no_face][no_slab] = true;
                            //}
                        }
                    }

                    // Geometric upper bound validation
                    if (!faceV[no_face] && radius > 0) {
                        GLdouble ppsj = pps[dir];

                        // Axis to consider
                        index_type_cm P1(0);
                        index_type_cm P2(0);
                        P2[dir] = 1;

                        point_type p1 = cm.vgd.FEuclid(cm.vgd.FDual(P1));
                        point_type p2 = cm.vgd.FEuclid(cm.vgd.FDual(P2));

                        // Two axis slabs
                        GLdouble size_slab = p2[dir]-p1[dir];

                        point_type pbase_s = cm.vgd.FEuclid(cm.vgd.FDual(base_s));
                        GLdouble dist = abs(pbase_s[dir] - ppsj) - size_slab / 2;
                        dist *= dist;
                        for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                            if (slabP[no_face][no_slab] != -1/* && slabP[no_face][no_slab] != slabMinB[no_face][no_slab]*/) {

                                if (dist > slabDP[no_face][no_slab]) {
                                    slabC[no_face][no_slab] = true;
                                }

                            }
                        }
                    }// Geometric upper bound validation

                    // Slab closed validation
                    if (!faceV[no_face]) {
                        bool validation = true;
                        for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                            validation = validation && slabC[no_face][no_slab];
                            if (!validation)
                                break;
                        }
                        if (validation) {
                            faceV[no_face] = true;
                        }
                    }

                    // If the face has to be searched
                    if (!faceV[no_face]) {
                        // For each slab of the face
                        for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab)
                        {
                            if (!slabC[no_face][no_slab]) {
                            index_type_cm ext_s(radius+1);
                            ext_s[dir] = 1;

                            index_type_cm sign_s(-1);
                            sign_s[dir] = 0;
                            GLuint mask = 1;
                            for (int i = 0; i < DimCM; ++i) {
                                if (i != dir) {
                                    sign_s[i] = (no_slab & mask) ? -1 : 1 ;
                                    mask = mask << 1;
                                }
                            }

                            // Search in current slab
                            index_type_cm idx(0);
                            iterator_type iter;
                            iter.reset(idx);
                            while (iter.next(idx, ext_s)) {
                                index_type_cm dept = idx * sign_s;
                                index_type_cm pcell = base_s + dept;

                                if (cm.valideAndPositiveIndex(pcell)) {
                                    // cell pointer
                                    cell_type* cell = &cm(pcell); // wb.Q iterate in different manner for 3D space

                                    size_t count = 0;
                                    while (count < cell->size)
                                    {
                                        index_type pco = cell->bCell[count];
                                        if (g_corr.valideIndex(pco) && pco != ps)
                                        {
                                            point_type ppco = g_point(pco);

                                            GLdouble v =
                                                    components::DistanceSquaredEuclideanP<point_type>()(
                                                        pps,
                                                        ppco
                                                        );

                                            if (v != 0 || (v == 0 & g_dstree(pco) != g_dstree(ps))) {
                                                ppco = ppco - pps;
                                                GLdouble maxd = 0;
                                                for (GLuint dir = 0; dir < DimCM; ++dir) {
                                                    if (abs(ppco[dir]) > maxd)
                                                        maxd = abs(ppco[dir]);
                                                }

                                                for (GLuint dir = 0; dir < DimCM; ++dir) {

                                                    if (abs(ppco[dir]) >= maxd) {
                                                        GLuint no_f = dir * 2;
                                                        no_f += (ppco[dir] >= 0) ? 0 : 1;
                                                        for (GLuint no_s = 0; no_s < NSLAB; ++no_s) {
                                                            // Check if in slab
                                                            GLint p = slabP[no_f][no_s];
                                                            GLdouble minD = slabDP[no_f][no_s];

                                                            GLuint mask = 1;
                                                            bool inSlab = true;
                                                            for (int i = 0; i < DimCM; ++i) {
                                                                if (i != dir) {
                                                                    GLdouble sign = (no_s & mask) ? -1 : 1 ;
                                                                    inSlab = inSlab && ((sign*ppco[i]) >= 0);
                                                                    if (!inSlab)
                                                                        break;
                                                                    mask = mask << 1;
                                                                }
                                                            }
                                                            if (inSlab) {//ppco in slab
                                                                GLint pcof = g_point.compute_offset(pco);

                                                                if (p == -1/* && slabDP[no_f][no_s] != 0*/)
                                                                {
                                                                    modified = true;
                                                                    ret = true;
                                                                    slabP[no_f][no_s] = pcof;
                                                                    slabDP[no_f][no_s] = v;
                                                                    if (p == -1)
                                                                        d_min = MIN(d_min, radius);
                                                                }
                                                                else {
                                                                    GLint psf = g_point.compute_offset(ps);
                                                                    GLint id1x = MIN(psf,pcof);
                                                                    GLint id2x = MAX(psf,pcof);
                                                                    GLint idd1x = MIN(psf,p);
                                                                    GLint idd2x = MAX(psf,p);
                                                                    if (EMST_isInf(
                                                                                id1x,
                                                                                id2x,
                                                                                v,
                                                                                idd1x,
                                                                                idd2x,
                                                                                minD))
                                                                    {
                                                                        modified = true;
                                                                        ret = true;
                                                                        slabP[no_f][no_s] = pcof;
                                                                        slabDP[no_f][no_s] = v;
                                                                    }
                                                                }
                                                            }//if
                                                        }//for each slab
                                                    }//if
                                                }//for
                                            }
                                        }//if valid
                                        count++;
                                    }//while in cell
                                }//if valid cell
                            }//while inside a given slab
                            }//if
                        }//for each slab
                    }//if (!faceV[no_face])
                }// face not validated
            }// for face
            radius++;
        }//while r <= d_max

        if (ret)
        {
            this->d_min = d_min;
        }
        if (modified) {
            for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
                for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                    this->slabP[no_face][no_slab] = slabP[no_face][no_slab];
                    this->slabDP[no_face][no_slab] = slabDP[no_face][no_slab];
                }
            }
        }
        return ret;
    }

    template <typename Grid,
              typename Grid2,
              typename Grid3,
              typename Grid4,
              typename Grid5  >
    DEVICE bool search(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr)  {

        typedef typename Grid::index_type index_type_cm;
        typedef IterIndex<DimCM> iterator_type;
        typedef typename Grid::point_type cell_type;
        typedef typename Grid2::index_type index_type;
        typedef typename Grid3::point_type point_type;
        typedef typename point_type::coord_type coord_type;

        Index ps = this->ps;

        // Test previous
        index_type pcorr = g_corr(ps);
        if (pcorr[0] != -1 && g_dstree(pcorr) != g_dstree(ps)) {
            return true;
        }
        else {
            g_corr(ps) = index_type(-1);
            g_dist(ps) = HUGE_VAL;
        }

        // Slab validated
        GLint slabP[NFACE][TWO_POWER_DIM_MOINS_UN];
        GLfloatP slabDP[NFACE][TWO_POWER_DIM_MOINS_UN];

        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {
                slabP[no_face][no_slab] = this->slabP[no_face][no_slab];
                slabDP[no_face][no_slab] = this->slabDP[no_face][no_slab];
            }
        }
        index_type minPCoord(-1);
        GLint minP = -1;
        GLdouble minDistance = HUGE_VAL;

        for (GLuint no_face = 0; no_face < NFACE; ++no_face) {
            for (GLuint no_slab = 0; no_slab < NSLAB; ++no_slab) {

                GLint p = slabP[no_face][no_slab];

                if (p != -1)
                {
                    index_type pco = g_point.back_offset(p);

                    if (g_dstree(pco) != g_dstree(ps)) {
                        GLdouble v = slabDP[no_face][no_slab];
                        if (minPCoord[0] == -1)
                        {
                            minPCoord = pco;
                            minP = p;
                            minDistance = v;
                        }
                        else {
                            GLint psf = g_dstree.compute_offset(ps);
                            GLint pcof = g_dstree.compute_offset(pco);
                            GLint minPCoordf = g_dstree.compute_offset(minPCoord);
                            GLint id1x = MIN(psf,pcof);
                            GLint id2x = MAX(psf,pcof);
                            GLint idd1x = MIN(psf,minPCoordf);
                            GLint idd2x = MAX(psf,minPCoordf);
                            if (EMST_isInf(
                                        id1x,
                                        id2x,
                                        v,
                                        idd1x,
                                        idd2x,
                                        minDistance))
                            {
                                minPCoord = pco;
                                minP = p;
                                minDistance = v;
                            }
                        }
                    }//if ! in component
                }
            }
        }
        g_corr(ps) = minPCoord;
        g_dist(ps) = minDistance;

        return true;
    }//search

};

}//namespace components


#endif // NODE_EMST_H

#ifndef VIEWGRID_H
#define VIEWGRID_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
//! JCC 290315 : suppressed
//#include <boost/geometry/geometries/point.hpp>

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "geometry.h"

#include "NIter.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif

//! HW 25/03/15 : conditional compilation switch
#define HW_TEST 0
//#define HW_FINDCELL_GEOMETRY
#define JCC_FINDCELL_GEOMETRY
//#define HW_FINDCELL_GEOMETRY_WITH_ANGLE_MORT
//#define HW_FINDCELL_SPIRAL_SEARCH

using namespace std;
using namespace components;
using namespace geometry_base;

namespace components
{

/*!
 * \defgroup Grid
 * \brief Espace de nommage components
 * Il comporte les views
 */
/*! @{*/
template <size_t DimP, size_t DimG>
class ViewGrid
{
public:
    typedef PointE<DimP> point_type;
    typedef Index<DimG> index_type;
    typedef index_type extents_type; // extends are dimensions lengths

protected:
    // this replaces w, h
    extents_type extents;
    extents_type extents_base;
    extents_type extents_dual;

    //! \brief _R
    //! level _R = radius _R
    size_t _R;
    //! \brief Level 1 map.
    //! Low level map on which to make a zoom.
    //! Center point of map level 0
    index_type pc;
    size_t w;
    size_t h;

    //! \brief Level R (zoom map) called "Base Map".
    //! The first map of Base/Dual hierarchical recursivity.
    //! Center point of Base Map at level R
    index_type PC;
    size_t W;
    size_t H;

    //! \brief Dual of Base Map called "Dual Map".
    //! Center point of Dual Map
    index_type PCD;
    size_t WD;
    size_t HD;

    //! \brief Dual of Dual Map called "Dual of Dual Map".
    //! Center point of Dual of Dual Map
    //! JCC : to come
    index_type PCDD;
    size_t WDD;
    size_t HDD;

public:
    DEVICE_HOST ViewGrid() {}

    DEVICE_HOST ViewGrid(index_type& pc, extents_type& exts, int level=25) :
        pc(pc),
        extents(exts),
        _R(level)
    {
        w = extents[0];
      if (DimG >= 2)
          h = extents[1];
    }

    DEVICE_HOST ViewGrid(index_type pc, size_t width, size_t height, int level=25) :
        pc(pc),
        w(width),
        h(height), _R(level)
    { }
    /*! @name Get the sizes of the maps.
     * \brief Return width and height of the maps.
     * @{
     */
    DEVICE_HOST GLint getLevel() {
        return _R;
    }
    DEVICE_HOST index_type getCenter() {
        return pc;
    }
    DEVICE_HOST extents_type getExtents() {
        return extents;
    }
    DEVICE_HOST extents_type getExtentsBase() {
        return extents_base;
    }

    DEVICE_HOST extents_type getExtentsDual() {
        return extents_dual;
    }

    DEVICE_HOST size_t getWidth() {
        return w;
    }

    DEVICE_HOST size_t getHeight() {
        return h;
    }

    DEVICE_HOST size_t getWidthLowLevel() {
        return w;
    }
    DEVICE_HOST size_t getHeightLowLevel() {
        return h;
    }

    DEVICE_HOST index_type getCenterBase() {
        return PC;
    }
    DEVICE_HOST size_t getWidthBase() {
        return W;
    }
    DEVICE_HOST size_t getHeightBase() {
        return H;
    }

    DEVICE_HOST index_type getCenterDual() {
        return PCD;
    }
    DEVICE_HOST size_t getWidthDual() {
        return WD;
    }
    DEVICE_HOST size_t getHeightDual() {
        return HD;
    }

    DEVICE_HOST size_t getDepthDual() {
        if (DimG >= 3)
            return extents_dual[2];
        else
            return 1;
    }

    // QWB add cpu version for refreshCell
    DEVICE_HOST size_t getWidthDual_cpu() {
        return WD;
    }
    // QWB add cpu version for refreshCell
    DEVICE_HOST size_t getHeightDual_cpu() {
        return HD;
    }

    DEVICE_HOST PointCoord getCenterDualDual() {
        return PCDD;
    }
    DEVICE_HOST size_t getWidthDualDual() {
        return WDD;
    }
    DEVICE_HOST size_t getHeightDualDual() {
        return HDD;
    }
    //! @}

    //! HW 10/03/15: Maybe the virtual functions should be better removed.
    //! Otherwise, the command
    //! g1[_y][_x] = vg.FEuclid(vg.FDual(PointCoord(_x,_y))); (at line 582)
    //! always does not work. I guess the reasons may have somthing
    //! to do with the following descriptions in CUDA C Programming Guide:
    //! "
    //! E.2.6.3. Virtual Functions
    //! When a function in a derived class overrides a virtual function in a base class,
    //! the execution space qualifiers (i.e., __host__, __device__) on the overridden and
    //! overriding functions must match.
    //! It is not allowed to pass as an argument to a __global__ function an object of
    //! a class with virtual functions.
    //! "
    //! However, I am confused why the commands
    //! g1[_y][_x] = vg.FEuclid(PointCoord(_x,_y)); (at line 578)
    //! g1[_y][_x] = vg.FEuclid(vg.F(PointCoord(_x,_y))); (at line 580)
    //! could work, since they are all in the same case trying to
    //! override, on the device side, the virtual functions which were created on host side.
    //! So, for now, I just comment the virtual functions.

    //! HW 09/04/15: Do not use virtual methods, otherwise GPU version will go wrong.
    // NO virtual for CUDA code
#ifndef CUDA_CODE
    //! \brief Base to low level coordinate conversion
    DEVICE_HOST virtual PointCoord F(PointCoord P) = 0;

    //! \brief Dual map to level one coordinate map
    DEVICE_HOST virtual PointCoord FDual(PointCoord PD) = 0;

    //! \brief Dual map to base map (at level R)
    DEVICE_HOST virtual PointCoord FDualToBase(PointCoord PD) = 0;

    //! \brief Dual to level one coordinate conversion
    DEVICE_HOST virtual PointCoord FDualToDual(PointCoord PD) = 0;

    //! \brief Transform low level grid point
    //! to its Euclidean coordinates if the grid is regular
    DEVICE_HOST virtual Point2D FEuclid(PointCoord P) = 0;

    //! \brief Return low level coordinates
    DEVICE_HOST virtual PointCoord FRound(Point2D p) = 0;

    //! \brief Return containing dual cell coord.
    DEVICE_HOST virtual PointCoord findCell(Point2D p) = 0;
#endif
};

template <size_t DimP, size_t DimG>
class ViewGridQuadMD : public ViewGrid<DimP, DimG>
{
    typedef ViewGrid<DimP, DimG> super_type;
public:
    typedef typename super_type::point_type point_type;
    typedef typename super_type::index_type index_type;
    typedef typename super_type::extents_type extents_type;

    DEVICE_HOST ViewGridQuadMD() : ViewGrid<DimP,DimG>() {}

    DEVICE_HOST ViewGridQuadMD(index_type& pc, extents_type& exts, int level=25) :
        ViewGrid<DimP,DimG>(pc, exts, level)
    {
        this->PC = ((this->pc + (this->_R - 1)) / this->_R);

        this->extents_base = (this->PC + 1) + ((this->extents - this->pc) + (this->_R - 1)) / this->_R;
        this->W = this->extents_base[0];
        this->H = this->extents_base[1];

        this->PCD = this->PC / 2;

        this->extents_dual = (this->PCD + 1) + ((this->extents_base - this->PC) + 1) / 2;
        this->WD = this->extents_dual[0];
        this->HD = this->extents_dual[1];
    }

    DEVICE_HOST ViewGridQuadMD(index_type pc, size_t width, size_t height, int level=25) :
        ViewGrid<DimP,DimG>(pc, width, height, level)
    {
        //! JCC : correction
        this->PC[0] = (this->pc[0] + this->_R - 1) / this->_R;
        this->PC[1] = (this->pc[1] + this->_R - 1) / this->_R;

        //this->PC[0] = (this->pc[0] + (R - this->pc[0]) % R) / R;
        //this->PC[1] = (this->pc[1] + (R - this->pc[1]) % R) / R;

        this->W = this->PC[0] + 1 + (this->w - this->pc[0] + this->_R - 1) / this->_R;
        this->H = this->PC[1] + 1 + (this->h - this->pc[1] + this->_R - 1) / this->_R;

        this->PCD[0] = this->PC[0] / 2;
        this->PCD[1] = this->PC[1] / 2;

        this->WD = this->PCD[0] + 1 + (this->W - this->PC[0] + 1) / 2;
        this->HD = this->PCD[1] + 1 + (this->H - this->PC[1] + 1) / 2;
    }

    //! \brief Base to low level coordinate conversion
    DEVICE_HOST index_type F(index_type p) {

        p *= this->_R;
        p -= this->PC * this->_R;
        p += this->pc;

        return p;
    }

    //! \brief Dual map to level one coordinate map
    DEVICE_HOST index_type FDual(index_type PD) {
        return this->F(this->FDualToBase(PD));
    }

    //! \brief Dual map to base map (at level R)
    DEVICE_HOST index_type FDualToBase(index_type p) {

        p *= 2;
        p += this->PC % 2;

        return p;
    }

    //! \brief Dual to level one coordinate conversion
    DEVICE_HOST index_type FDualToDual(index_type PD) {

        return PD;
    }

    //! \brief Transform low level grid point
    //! to its Euclidean coordinates if the grid is regular
    DEVICE_HOST point_type FEuclid(index_type P) {
        point_type p(0);

        for (size_t i = 0; i < MIN(DimP,DimG); ++i)
            p[i] = P[i];

        return p;
    }

    //! \brief Return low level coordinates
    DEVICE_HOST index_type FRound(point_type p) {
        index_type P(0);

        for (GLuint i = 0; i < MIN(DimP,DimG); ++i)
            P[i] = floor(p[i] + 0.5);

        return P;
    }

    //! \brief Return containing dual cell coord.
    DEVICE_HOST index_type findCell(point_type ps) {
        index_type P(0);

        // Two axis with slabs to consider
        index_type P1(0);
        point_type p1 = this->FEuclid(this->FDual(P1));

        for (GLuint dir = 0; dir < MIN(DimP,DimG); ++dir) {
            index_type P2(0);
            P2[dir] = 1;

            point_type p2 = this->FEuclid(this->FDual(P2));

            GLdouble size_slab = p2[dir]-p1[dir];

            // Compute cell rank along u
            GLdouble li = ps[dir] - p1[dir] + size_slab/2;
            P[dir] = (GLint) floor(li / size_slab);
        }
        return P;
    }

    //! \brief Return containing dual cell coord.
    DEVICE_HOST index_type findCell3D(point_type ps) {
        index_type P;

        // Two axis with slabs to consider
        index_type P1(0,0,0);
        index_type P2(1,0,0);
        index_type P3(0,1,0);
        index_type P4(0,0,1);

        point_type p1 = this->FEuclid(this->FDual(P1));
        point_type p2 = this->FEuclid(this->FDual(P2));
        point_type p3 = this->FEuclid(this->FDual(P3));
        point_type p4 = this->FEuclid(this->FDual(P4));

        // Two axis slabs
        double size_slab_u = p2[0]-p1[0];
        double size_slab_v = p3[1]-p1[1];

        // Compute cell rank along u axis
        double li = ps[0] - p1[0] + size_slab_u/2;
        P[0] = (int) floor(li / size_slab_u);

        // Compute cell rank along v axis
        double lii = ps[1] - p1[1] + size_slab_v/2;
        P[1] = (int) floor(lii / size_slab_v);

        if (DimP >= 3 && DimG >= 3) {
            double size_slab_w = p4[2]-p1[2];
            // Compute cell rank along w axis
            double liii = ps[2] - p1[2] + size_slab_w/2;
            P[2] = (int) floor(liii / size_slab_w);
        }
        return P;
    }

    //! \brief Return containing dual cell coord.
    DEVICE_HOST index_type findCell2D(point_type ps) {
        index_type P;

        // Two axis with slabs to consider
        index_type P1(0,0);
        index_type P2(1,0);
        index_type P3(0,1);

        // Passing in euclidean level using geometry
        Point_2 p(ps[0],ps[1]);

        // Two axis in euclidean plane
        Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
        Point_2 p2((this->FEuclid(this->FDual(P2)))[0], (this->FEuclid(this->FDual(P2)))[1]);
        Point_2 p3((this->FEuclid(this->FDual(P3)))[0], (this->FEuclid(this->FDual(P3)))[1]);

        // Two axis slabs
        double size_slab_u = p2.x()-p1.x();
        double size_slab_v = p3.y()-p1.y();

        // Compute cell rank along u axis
        double li = p.x() - p1.x() + size_slab_u/2;
        P[0] = (int) floor(li / size_slab_u);

        //! HW 16/03/15 : correction
        // Compute cell rank along v axis
        double lii = p.y() - p1.y() + size_slab_v/2;
        P[1] = (int) floor(lii / size_slab_v);

        return P;
    }

//    DEVICE_HOST PointCoord findCell(Point3D ps) {
//        PointCoord P;

//        // Two axis with slabs to consider
//        PointCoord P1(0,0);
//        PointCoord P2(1,0);
//        PointCoord P3(0,1);

//        // Passing in euclidean level using geometry
//        Point_2 p(ps[0],ps[1]);

//        // Two axis in euclidean plane
//        Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
//        Point_2 p2((this->FEuclid(this->FDual(P2)))[0], (this->FEuclid(this->FDual(P2)))[1]);
//        Point_2 p3((this->FEuclid(this->FDual(P3)))[0], (this->FEuclid(this->FDual(P3)))[1]);

//        // Two axis slabs
//        double size_slab_u = p2.x()-p1.x();
//        double size_slab_v = p3.y()-p1.y();

//        // Compute cell rank along u axis
//        double li = p.x() - p1.x() + size_slab_u/2;
//        P[0] = (int) floor(li / size_slab_u);

//        //! HW 16/03/15 : correction
//        // Compute cell rank along v axis
//        double lii = p.y() - p1.y() + size_slab_v/2;
//        P[1] = (int) floor(lii / size_slab_v);

//        return P;
//    }

    //! QWB add for refreshCell on CPU side\brief Return containing dual cell coord.
    PointCoord findCell_cpu(Point2D ps) {
        PointCoord P;

        // Two axis with slabs to consider
        PointCoord P1(0,0);
        PointCoord P2(1,0);
        PointCoord P3(0,1);

        // Passing in euclidean level using geometry
        geometry::Point_2 p(ps[0],ps[1]);

        // Two axis in euclidean plane
        geometry::Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
        geometry::Point_2 p2((this->FEuclid(this->FDual(P2)))[0], (this->FEuclid(this->FDual(P2)))[1]);
        geometry::Point_2 p3((this->FEuclid(this->FDual(P3)))[0], (this->FEuclid(this->FDual(P3)))[1]);

        // Two axis slabs
        double size_slab_u = p2.x()-p1.x();
        double size_slab_v = p3.y()-p1.y();

        // Compute cell rank along u axis
        double li = p.x() - p1.x() + size_slab_u/2;
        P[0] = (int) floor(li / size_slab_u);

        //! HW 16/03/15 : correction
        // Compute cell rank along v axis
        double lii = p.y() - p1.y() + size_slab_v/2;
        P[1] = (int) floor(lii / size_slab_v);

        return P;
    }

    PointCoord findCell_cpu(Point3D ps) {
        PointCoord P;

        // Two axis with slabs to consider
        PointCoord P1(0,0);
        PointCoord P2(1,0);
        PointCoord P3(0,1);

        // Passing in euclidean level using geometry
        geometry::Point_2 p(ps[0],ps[1]);

        // Two axis in euclidean plane
        geometry::Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
        geometry::Point_2 p2((this->FEuclid(this->FDual(P2)))[0], (this->FEuclid(this->FDual(P2)))[1]);
        geometry::Point_2 p3((this->FEuclid(this->FDual(P3)))[0], (this->FEuclid(this->FDual(P3)))[1]);

        // Two axis slabs
        double size_slab_u = p2.x()-p1.x();
        double size_slab_v = p3.y()-p1.y();

        // Compute cell rank along u axis
        double li = p.x() - p1.x() + size_slab_u/2;
        P[0] = (int) floor(li / size_slab_u);

        //! HW 16/03/15 : correction
        // Compute cell rank along v axis
        double lii = p.y() - p1.y() + size_slab_v/2;
        P[1] = (int) floor(lii / size_slab_v);

        return P;
    }

};//ViewGridQuadMD

typedef ViewGridQuadMD<2,2> ViewGridQuad;

class ViewGridTetra : public ViewGrid<2,2>
{
    typedef ViewGrid<2, 2> super_type;
public:
    typedef typename super_type::point_type point_type;
    typedef typename super_type::index_type index_type;
    typedef typename super_type::extents_type extents_type;

    DEVICE_HOST ViewGridTetra() : ViewGrid<2,2>() {}

    DEVICE_HOST ViewGridTetra(index_type pc, size_t width, size_t height, int level=25) :
        ViewGrid<2,2>(pc, width, height, level)
    {
        //! JCC : correction
        this->PC[0] = (this->pc[0] + _R - 1) / _R;
        this->PC[1] = 2 * (this->pc[1] + 2 * _R - 1) / (2 * _R);

        //! JCC 13/03/15 : modif
        if (pc[0] % 2 == 1) {
            this->PC[1] += 1;
        }

        this->W = this->PC[0] + 1 + (this->w - this->pc[0] + _R - 1) / _R;
        this->H = this->PC[1] + 1 + (this->h - this->pc[1] + _R - 1) / _R;

        this->PCD[0] = this->PC[0] / 2;
        this->PCD[1] = this->PC[1];

        this->WD = this->PCD[0] + 1 + (this->W - this->PC[0] + 1) / 2;
        //! HW 17/03/15 : correction
        this->HD = this->H;
    }

    //! \brief Base to low level coordinate conversion
    DEVICE_HOST index_type F(index_type P) {
        int X, Y, x, y;

        X = P[0];
        Y = P[1];

        x = X * _R - (this->PC[0] * _R - this->pc[0]);
        y = Y * _R - (this->PC[1] * _R - this->pc[1]);

        return index_type(x, y);
    }

    //! \brief Dual map to level one coordinate map
    DEVICE_HOST index_type FDual(index_type PD) {
        return this->F(this->FDualToBase(PD));
    }

    //! \brief Dual map to base map (at level R)
    DEVICE_HOST index_type FDualToBase(index_type PD) {
        int XD, YD, X, Y;

        XD = PD[0];
        YD = PD[1];

        //! HW 16/3/15 modif
        X = 2 * XD + this->PC[0] % 2 - (YD % 2 ? 1 : 0);
        Y = YD;

        return index_type(X, Y);
    }

    //! \brief Dual to level one coordinate conversion
    DEVICE_HOST index_type FDualToDual(index_type PD) {
        index_type P;

        return P;
    }

    //! \brief Transform low level grid point
    //! to its Euclidean coordinates if the grid is regular
    DEVICE_HOST point_type FEuclid(index_type P) {
        point_type p;

        p[0] = P[0];
        p[1] = P[1];
        return p;
    }

    //! \brief Return low level coordinates
    DEVICE_HOST index_type FRound(point_type p) {
        index_type P;

        P[1] = (int) floor(p[1] + 0.5);
        P[0] = (int) floor(p[0] + 0.5);

        return P;
    }

    //! JCC : Proposition of Tetra findCell
    //! based on goTo functions with 8 directions
    //! not 4 (right, upright, up, upleft
    //! \brief Return containing dual cell coord.
    DEVICE_HOST index_type findCell(point_type ps) {
        index_type P;

        // Two axis with slabs to consider
        index_type P1(0,0);
        index_type P2(1,1);
        index_type P3(0,1);

        // Pass in euclidean level using geometry
        Point_2 p(ps[0],ps[1]);

        Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
        Point_2 p2((this->FEuclid(this->FDual(P2)))[0], (this->FEuclid(this->FDual(P2)))[1]);
        Point_2 p3((this->FEuclid(this->FDual(P3)))[0], (this->FEuclid(this->FDual(P3)))[1]);

        // Two axis
        Vector_2 u(p1,p2);//down right
        //! HW 17/03/15 : correction
        Vector_2 v(p1,p3);//down left

        double size_slab_u = sqrt(u*u);
        double size_slab_v = size_slab_u;

        // Intersection (p,u) axis with (p1,v) axis
        Point_2 pi = geometry::intersect(p,u,p1,v);

        // Compute length (pi,p)+size_slab_u/2
        Vector_2 vi(pi,p);
        double li = sqrt(vi.squared_length()) + size_slab_u/2; // double li = sqrt(vi) + size_slab_u/2;
        // Compute cell rank along u axis
        int c_rank_u = (int) floor(li / size_slab_u);

        // Compute lengh (pi,p1)+size_slab_v/2
        Vector_2 vii(pi,p1);
        double lii = sqrt(vii.squared_length()) + size_slab_v/2; // double lii = sqrt(vii) + size_slab_v/2;
        // Compute cell rank along v axis
        int c_rank_v = (int) floor(lii / size_slab_v);

        // Use NIterNIterHexaDual to find the coords in Dual Grid
        NIterTetraDual niter(P1, 0, 0);

        //! HW 19/03/15 : modif
        if (v * Vector_2(p1,pi) >= 0)
            // Move down right, then down left
            P = niter.goTo<1>(niter.goTo<7>(P1, c_rank_u),c_rank_v);
        else
            // Move down right, then up right
            P = niter.goTo<5>(niter.goTo<7>(P1, c_rank_u),c_rank_v);

        return P;
    }
};//ViewGridTetra

//! KERNEL FUNCTION
//! Kernel call by object
//! Standard Kernel call with objects
enum Level { LOW_LEVEL, BASE, DUAL };

template<class ViewGrid, class Grid, Level LEVEL>
KERNEL void K_VG_initializeIntoPlane(ViewGrid vg, Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight())
    {
        if (LEVEL == LOW_LEVEL)
            g1[_y][_x] = vg.FEuclid(PointCoord(_x,_y));
        if (LEVEL == BASE)
            g1[_y][_x] = vg.FEuclid(vg.F(PointCoord(_x,_y)));
        if (LEVEL == DUAL)
            g1[_y][_x] = vg.FEuclid(vg.FDual(PointCoord(_x,_y)));
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 15.06.15 : Add K_VG_initializeIntoPlaneWithFlow
template<class ViewGrid, class Grid, Level LEVEL>
KERNEL void K_VG_initializeIntoPlaneWithFlow(ViewGrid vg, Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight())
    {
        if (LEVEL == LOW_LEVEL) {
            g1[_y][_x][0] += (vg.FEuclid(PointCoord(_x,_y))).get(0);
            g1[_y][_x][1] += (vg.FEuclid(PointCoord(_x,_y))).get(1);
        }
        if (LEVEL == BASE) {
            g1[_y][_x][0] += (vg.FEuclid(vg.F(PointCoord(_x,_y)))).get(0);
            g1[_y][_x][1] += (vg.FEuclid(vg.F(PointCoord(_x,_y)))).get(1);
        }
        if (LEVEL == DUAL) {
            g1[_y][_x][0] += (vg.FEuclid(vg.FDual(PointCoord(_x,_y)))).get(0);
            g1[_y][_x][1] += (vg.FEuclid(vg.FDual(PointCoord(_x,_y)))).get(1);
        }
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! KERNEL FUNCTION
template<class ViewGrid,
         class Grid, class Node,
         class NIter>
KERNEL void K_NIterViewGrid(ViewGrid vg, Grid g1)
{
    size_t width = vg.getWidthDual();
    size_t height = vg.getHeightDual();

    KER_SCHED(width, height)

    if (_x < width && _y < height) {

        PointCoord pc = vg.FDual(PointCoord(_x, _y));
        PointEuclid pc0 = vg.FEuclid(pc);
        NIter ni(pc, 0, 18);
        do {
            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            PointCoord pCoord;
            pCoord = ni.getNodeIncr();
            if (pCoord[0] >= 0 && pCoord[0] < g1.getWidth()
                    && pCoord[1] >= 0 && pCoord[1] < g1.getHeight()) {
                PointCoord p1 = pCoord;//vg.F(pCoord);
                PointEuclid p0 = vg.FEuclid(p1);
                if (p1[0]>=0&&p1[0]<g1.getWidth()
                        &&p1[1]>=0&&p1[1]<g1.getHeight()) {
                    g1[p1[1]][p1[0]] = pc0;//Node(pc0[0],pc0[1]);
                }
            }
        } while (ni.nextNodeIncr());
    }

    END_KER_SCHED

    SYNCTHREADS;
}

#if HW_TEST

template<class ViewGrid, class Grid>
KERNEL void K_VG_findCell0(ViewGrid vg, Grid g1)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

//    if (_x < g1.getWidth() && _y < g1.getHeight())
//    if ((_x < 40 && _y < 140 && _x > 30 && _y > 130) ||
//        (_x < 140 && _y < 40 && _x > 130 && _y > 30) ||
//        (_x < 40 && _y < 40 && _x > 30 && _y > 30))
    if (_x == 130 && _y == 130)
    {
        PointCoord pc = vg.FDual(vg.findCell(vg.FEuclid(PointCoord(_x, _y))));
        PointEuclid pc0 = vg.FEuclid(pc);
//        printf("_x = %d, _y = %d, PCD = ", _x, _y);
//        (vg.findCell(vg.FEuclid(PointCoord(_x, _y)))).printInt();
//        printf(" pc = ", _x, _y);
//        pc.printInt();
//        printf(" pc0 = ", _x, _y);
//        pc0.printFloat();
//        printf("\n");
        g1[_y][_x] = pc0;

    }

    END_KER_SCHED

    SYNCTHREADS;
}

template<class ViewGrid,
         class Grid, class Node,
         class NIter>
KERNEL void K_VG_findCell(ViewGrid vg, Grid g1)
{
    size_t width = vg.getWidthDual();
    size_t height = vg.getHeightDual();

    KER_SCHED(width, height)

    if (_x < width && _y < height)
    {
        PointCoord pc = vg.FDual(PointCoord(_x, _y));
        PointEuclid pc0 = vg.FEuclid(pc);
        //! JCC 200315 : no absolute value
        //! created access method //NIter ni(pc, 0, 23);
        NIter ni(pc, 0, vg.getLevel()-1);
        do {
            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            PointCoord pCoord;
            pCoord = ni.getNodeIncr();
            if (pCoord[0] >= 0 && pCoord[0] < g1.getWidth()
                    && pCoord[1] >= 0 && pCoord[1] < g1.getHeight()) {
                PointCoord p1 = pCoord;//vg.F(pCoord);
                PointEuclid p0 = vg.FEuclid(p1);
                PointEuclid pc2 = vg.FEuclid(vg.FDual(vg.findCell(p0)));
                if (p1[0]>=0&&p1[0]<g1.getWidth()
                        &&p1[1]>=0&&p1[1]<g1.getHeight()) {
                    g1[p1[1]][p1[0]] = pc2;//Node(pc0[0],pc0[1]);
                }
            }
        } while (ni.nextNodeIncr());
    }

    END_KER_SCHED

    SYNCTHREADS;
}

#endif

// Hongjian's Debug
////! Test program
template <class ViewGrid,
          class Grid,
          size_t SXX,
          size_t SYY>
class TestViewGrid {

    Grid initGrid;
    PointCoord initPoint;
    char* fileSolution;
public:

    TestViewGrid(Grid g1, PointCoord n1, char* fileSolution) : initGrid(g1), initPoint(n1), fileSolution(fileSolution) {}

    void initialize(Grid g1, PointCoord n1, char* fileSolution) {
        initGrid = g1;
        initPoint = n1;
        fileSolution = fileSolution;
    }

    void run() {
        int devID = 0;
#ifdef CUDA_CODE
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&devID);

        if (error != cudaSuccess)
        {
            printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        }

        error = cudaGetDeviceProperties(&deviceProp, devID);

        if (deviceProp.computeMode == cudaComputeModeProhibited)
        {
            fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
            exit(EXIT_SUCCESS);
        }

        if (error != cudaSuccess)
        {
            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        }
        else
        {
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        }
#endif
        cout << "debut test ViewGrid ..." << endl;
        const size_t SX = SXX, SY = SYY;

        // ViewGrid creation
        ViewGrid vg(initPoint, SX, SY);

        // Creation de view grid grille en local
        Grid gd(vg.getWidth(), vg.getHeight());
        Grid gd_base(vg.getWidthBase(), vg.getHeightBase());
        Grid gd_dual(vg.getWidthDual(), vg.getHeightDual());

        cout << "Creation de grilles sur device GPU ..." << endl;

        // Creation de grilles sur device GPU
        // Create grids
        Grid gpu_gd;
        gpu_gd.gpuResize(vg.getWidth(), vg.getHeight());
        Grid gpu_gd_base;
        gpu_gd_base.gpuResize(vg.getWidthBase(), vg.getHeightBase());
        Grid gpu_gd_dual;
        gpu_gd_dual.gpuResize(vg.getWidthDual(), vg.getHeightDual());

#ifdef CUDA_CODE
        // cuda timer
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
#endif
        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
        double x0;
        x0 = clock();

        cout << "Appel du Kernel ..." << endl;

        // Copie des grilles CPU -> GPU
        gd.gpuCopyHostToDevice(gpu_gd);
        gd_base.gpuCopyHostToDevice(gpu_gd_base);
        gd_dual.gpuCopyHostToDevice(gpu_gd_dual);

        // Low level initialization
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              gpu_gd.getWidth(),
                              gpu_gd.getHeight());
        K_VG_initializeIntoPlane<ViewGrid, Grid, LOW_LEVEL > _KER_CALL_(b, t) (
                    vg,
                    gpu_gd);

        // Base level initialization
        KER_CALL_THREAD_BLOCK(b2, t2,
                              4, 4,
                              gpu_gd_base.getWidth(),
                              gpu_gd_base.getHeight());
        K_VG_initializeIntoPlane<ViewGrid, Grid, BASE> _KER_CALL_(b2, t2) (
                    vg,
                    gpu_gd_base);

        // Dual level initialization
        KER_CALL_THREAD_BLOCK(b3, t3,
                              4, 4,
                              gpu_gd_dual.getWidth(),
                              gpu_gd_dual.getHeight());
        K_VG_initializeIntoPlane<ViewGrid, Grid, DUAL> _KER_CALL_(b3, t3) (
                    vg,
                    gpu_gd_dual);

        // Kernel call with class parameters
//        KER_CALL_THREAD_BLOCK(b4, t4,
//                              4, 4,
//                              vg.getWidthBase(),
//                              vg.getHeightBase());
//        K_NIterViewGrid<ViewGrid, Grid, Point2D, NIterHexa> _KER_CALL_(b4, t4) (vg, gpu_gd);
//        K_NIterViewGrid<ViewGrid, Grid, Point2D, NIterTetra> _KER_CALL_(b4, t4) (vg, gpu_gd);
//        K_NIterViewGrid<ViewGrid, Grid, Point2D, NIterQuad> _KER_CALL_(b4, t4) (vg, gpu_gd);

#if HW_TEST
        // findCell
        KER_CALL_THREAD_BLOCK(b5, t5,
                              4, 4,
                              vg.getWidth(),
                              vg.getHeight());
//        K_VG_findCell0<ViewGrid, Grid> _KER_CALL_(b5, t5) (vg, gpu_gd);
//          K_VG_findCell<ViewGrid, Grid, Point2D, NIterHexa> _KER_CALL_(b5, t5) (vg, gpu_gd);
//        K_VG_findCell<ViewGrid, Grid, Point2D, NIterTetra> _KER_CALL_(b5, t5) (vg, gpu_gd);
//        K_VG_findCell<ViewGrid, Grid, Point2D, NIterQuad> _KER_CALL_(b5, t5) (vg, gpu_gd);
#endif
        // Copie du resultat GPU -> CPU
        gd.gpuCopyDeviceToHost(gpu_gd);
        gd_base.gpuCopyDeviceToHost(gpu_gd_base);
        gd_dual.gpuCopyDeviceToHost(gpu_gd_dual);

        cout << "Affichage du resultat a la console ..." << endl;

        // Writting files
        ofstream fo;
        std::string str = fileSolution;
        std::size_t pos = str.find(".");
        std::size_t pos1 = str.find("_");
        if (pos == std::string::npos)
          pos = str.length();
        cout << "file " << fileSolution << endl;
        cout << "pos " << pos << endl;
        string str_sub;

        str_sub = str.substr(0, pos);
        ostringstream os; os<<"_";
        str_sub.append(os.str());
        str_sub.append("ll.grid2dpts");

        fo.open(str_sub.c_str());
        if (fo) {
            fo << gd;
            fo.close();
        }
        else
            cout << "pb file" << endl;

        str_sub = str.substr(0, pos);
        os.str(""); os<<"_";
        str_sub.append(os.str());
        str_sub.append("b.grid2dpts");

        fo.open(str_sub.c_str());
        if (fo) {
            fo << gd_base;
            fo.close();
        }
        else
            cout << "pb file" << endl;

        str_sub = str.substr(0, pos);
        os.str(""); os<<"_";
        str_sub.append(os.str());
        str_sub.append("d.grid2dpts");

        fo.open(str_sub.c_str());
        if (fo) {
            fo << gd_dual;
            fo.close();
        }
        else
            cout << "pb file" << endl;

        // Console
        PointCoord pc = vg.getCenter();
        cout << pc[0] << " " << pc[1] << endl;
        PointCoord PC = vg.F(vg.getCenterBase());
        cout << PC[0] << " " << PC[1] << endl;
        PointCoord PCD = vg.FDual(vg.getCenterDual());
        cout << PCD[0] << " " << PCD[1] << endl;

        cout << vg.FEuclid(pc)[0] << " " << vg.FEuclid(pc)[1] << endl;
        cout << vg.FEuclid(PC)[0] << " " << vg.FEuclid(PC)[1] << endl;
        cout << vg.FEuclid(PCD)[0] << " " << vg.FEuclid(PCD)[1] << endl;

        // cpu timer
        cout << "CPU Time : " << (clock() - x0)/CLOCKS_PER_SEC << endl;

        // cuda timer
#ifdef CUDA_CODE
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
        cout << "GPU Execution Time: " <<  elapsedTime << " ms" << endl;
        cout << endl;
#endif
        // Explicit
        gpu_gd.gpuFreeMem();
        gd.freeMem();
    }
};

}//namespace components
//! @}
#endif // VIEWGRID_H

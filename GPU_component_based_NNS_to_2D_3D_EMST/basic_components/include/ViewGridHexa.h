#ifndef VIEWGRIDHEXA_H
#define VIEWGRIDHEXA_H
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

#include "ViewGrid.h"

#include "NIter.h"
//#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
//#endif

using namespace std;
using namespace geometry_base;

namespace components
{

//template <int R>
class ViewGridHexa : public ViewGrid<2,2>
{
public:
    DEVICE_HOST ViewGridHexa() : ViewGrid<2,2>() {}

    DEVICE_HOST ViewGridHexa(PointCoord pc, size_t width, size_t height, int level=25) :
        ViewGrid<2,2>(pc, width, height, level)
    {
        //! HW 04/03/15: Add "this->" in front of each member variables
        //! JCC : correction
        this->PC[0] = (this->pc[0] + _R - 1) / _R;
        this->PC[1] = (this->pc[1] + 2 * _R - 1) / _R;

        //! JCC 17/04/15 : modif
        if (this->PC[1] % 2 == 0) {
            this->PC[1] += 1;
        }

        this->W = this->PC[0] + 1 + (this->w - this->pc[0] + _R - 1) / _R;
        this->H = this->PC[1] + 1 + (this->h - this->pc[1] + _R - 1) / _R;

        //! JCC : correction
        this->PCD[0] = this->PC[0] / 3 + (this->PC[0] % 3 == 2 ? 1 : 0);
        this->PCD[1] = this->PC[1];

        this->WD = this->PCD[0] + 1 + (this->W - this->PC[0] + 1) / 3;
        this->HD = this->H;
    }

    //! \brief Base to low level coordinate conversion
    DEVICE_HOST PointCoord F(PointCoord P) {
        int X, Y, x, y;

        X = P[0];
        Y = P[1];

        y = Y * _R - (this->PC[1] * _R - this->pc[1]);
        x = X * _R - (this->PC[0] * _R - this->pc[0]);

        //! JCC 17/04/15 : modif
        if (Y % 2 == 0) {
            x += _R / 2 + (_R % 2)*((y+_R+1) % 2)*((y+_R+1) % 2);
        }
        //! JCC 17/04/15 : suppressed
        //if (Y % 2 == 1) {
        //    x -= _R / 2 - (_R % 2)*((y+_R) % 2)*((y+_R) % 2);
        //}

        return PointCoord(x, y);
    }

    //! JCC : changing FDual
    //! \brief Dual map to level one coordinate map
    DEVICE_HOST PointCoord FDual(PointCoord PD) {
        return this->F(this->FDualToBase(PD));
    }

    //! \brief Dual map to base map (at level R)
    DEVICE_HOST PointCoord FDualToBase(PointCoord PD) {
        int XD, YD, X, Y;

        XD = PD[0];
        YD = PD[1];

        Y = YD;

        //! JCC 13/3/15 modif
        X = 3 * (XD - (this->PC[0] % 3 == 2 ? 1 : 0))
                + this->PC[0] % 3;

        //! HW 03/08/15 modif
        //X -= 1;// JCC 19/09/17 suppression

        //! JCC 13/03/15 : modif
        if (Y % 2 == 0) {
            X += 1;
        }

        return PointCoord(X, Y);
    }

    //! \brief Dual to level one coordinate conversion
    DEVICE_HOST PointCoord FDualToDual(PointCoord PD) {
        PointCoord P;

        return P;
    }

    //! \brief Transform low level grid point
    //! to its Euclidean coordinates if the grid is regular
    DEVICE_HOST Point2D FEuclid(PointCoord P) {
        Point2D p;

        p[0] = P[0] + 0.5 * (P[1] % 2 == 0 ? 1 : 0);
        p[1] = P[1];

        return p;
    }

    //! \brief Return low level coordinates
    DEVICE_HOST PointCoord FRound(Point2D p) {
        PointCoord P;

        P[1] = (int) floor(p[1] + 0.5);
        P[0] = (int) floor(p[0] + 0.5 - 0.5 * (P[1] % 2 == 0 ? 1 : 0));

        return P;
    }

#ifdef HW_FINDCELL_SPIRAL_SEARCH

    //! \brief Return containing dual cell coord.
    DEVICE_HOST PointCoord findCell(Point2D ps) {
        PointCoord P;

        //        int x = (int)((ps[0]/R)/3+0.5);
        //        int y = (int)((ps[1]/R)+0.5);

        int y = (int)(ps[1] + 0.5);
        int x = (int)(ps[0] - 0.5 * (y % 2 == 0 ? 1 : 0) + 0.5);
        x /= R;
        y /= R;
        x /= 3;

        PointCoord P0(x,y);
        Point_2 p(ps[0],ps[1]);
        Point_2 p0((this->FEuclid(this->FDual(P0)))[0], (this->FEuclid(this->FDual(P0)))[1]);
        Vector_2 v0(p0,p);
        double lmin = sqrt(v0.squared_length());
        P = P0;

        NIterHexaDual ni(P0, 0, 3);
        do {
            // Ready to next distance
            PointCoord pCoord;
            pCoord = ni.getNodeIncr();
            if (pCoord[0] >= 0 && pCoord[0] < this->WD
                    && pCoord[1] >= 0 && pCoord[1] < this->HD) {
                PointCoord P1 = pCoord;
                Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
                Vector_2 v1(p1,p);
                double l1 = sqrt(v1.squared_length());
                if (l1 < lmin)
                {
                    P = P1;
                    lmin = l1;
                }
            }
        } while (ni.nextNodeIncr());

        return P;
    }
#endif

#ifdef HW_FINDCELL_GEOMETRY

    //! \brief Return containing dual cell coord.
    DEVICE_HOST PointCoord findCell(Point2D ps) {
        PointCoord P;

        PointCoord P00(0,0);
        Point_2 p00((this->FEuclid(this->FDual(P00)))[0], (this->FEuclid(this->FDual(P00)))[1]);
        float offsetX = R - p00.x();
        float offsetY = R - p00.y();
        ps[0] = ps[0] + offsetX;
        ps[1] = ps[1] + offsetY;

        // position of the big rectangle
        int x = (int)floor(ps[0] / (3 * R));
        int y = (int)floor(ps[1] / (2 * R));
        PointCoord P1(x, 2 * y);
        NIterHexaDual niter(P1, 0, 0);

        // first column, upper part -> No.1 slab
        if (ps[0] >= (x * 3 * R) && ps[0] < ((float)(x * 3 * R) + 0.5f * (float)R)
                && ps[1] >= (y * 2 * R) && ps[1] < (y * 2 * R + R))
        {
            Point_2 p(ps[0], ps[1]);
            Point_2 p0((x * 3 * R), (y * 2 * R));
            Point_2 p1((float)(x * 3 * R) + 0.5f * (float)R, (y * 2 * R));
            Point_2 p2((x * 3 * R), (y * 2 * R + R));
            if (geometry::ifAtSameSideBis(p, p0, p1, p2))
                // move up left
                P = niter.goTo<2>(P1, 1);
            else
                P = P1;
            return P;
        }
        // first column, bottom part -> No.2 slab
        if (ps[0] >= (x * 3 * R) && ps[0] < ((float)(x * 3 * R) + 0.5f * (float)R)
                && ps[1] >= (y * 2 * R + R) && ps[1] < ((y + 1) * 2 * R))
        {
            Point_2 p(ps[0], ps[1]);
            Point_2 p0((x * 3 * R), ((y + 1) * 2 * R));
            Point_2 p1((float)(x * 3 * R) + 0.5f * (float)R, ((y + 1) * 2 * R));
            Point_2 p2((x * 3 * R), (y * 2 * R + R));
            if (geometry::ifAtSameSideBis(p, p0, p1, p2))
                // move down left
                P = niter.goTo<1>(P1, 1);
            else
                P = P1;
            return P;
        }
        // second column -> No.3 slab
        if (ps[0] >= ((float)(x * 3 * R) + 0.5f * (float)R) && ps[0] < ((float)(x * 3 * R) + 1.5f * (float)R)
                && ps[1] >= (y * 2 * R) && ps[1] < ((y + 1) * 2 * R))
        {
            return P1;
        }
        // third column, upper part -> No.4 slab
        if (ps[0] >= ((float)(x * 3 * R) + 1.5f * (float)R) && ps[0] < (x * 3 * R + 2 * R)
                && ps[1] >= (y * 2 * R) && ps[1] < (y * 2 * R + R))
        {
            Point_2 p(ps[0], ps[1]);
            Point_2 p0((x * 3 * R + 2 * R), (y * 2 * R));
            Point_2 p1((float)(x * 3 * R) + 1.5f * (float)R, (y * 2 * R));
            Point_2 p2((x * 3 * R + 2 * R), (y * 2 * R + R));
            if (geometry::ifAtSameSideBis(p, p0, p1, p2))
                // move up right
                P = niter.goTo<4>(P1, 1);
            else
                P = P1;
            return P;
        }
        // third column, bottom part -> No.5 slab
        if (ps[0] >= ((float)(x * 3 * R) + 1.5f * (float)R) && ps[0] < (x * 3 * R + 2 * R)
                && ps[1] >= (y * 2 * R + R) && ps[1] < ((y + 1) * 2 * R))
        {
            Point_2 p(ps[0], ps[1]);
            Point_2 p0((x * 3 * R + 2 * R), ((y + 1) * 2 * R));
            Point_2 p1((float)(x * 3 * R) + 1.5f * (float)R, ((y + 1) * 2 * R));
            Point_2 p2((x * 3 * R + 2 * R), (y * 2 * R + R));
            if (geometry::ifAtSameSideBis(p, p0, p1, p2))
                // move down right
                P = niter.goTo<5>(P1, 1);
            else
                P = P1;
            return P;
        }
        // fourth column, upper part -> No.6 slab
        if (ps[0] >= (x * 3 * R + 2 * R) && ps[0] < ((x + 1) * 3 * R)
                && ps[1] >= (y * 2 * R) && ps[1] < (y * 2 * R + R))
        {
            // move up right
            P = niter.goTo<4>(P1, 1);
            return P;
        }
        // fourth column, bottom part -> No.7 slab
        if (ps[0] >= (x * 3 * R + 2 * R) && ps[0] < ((x + 1) * 3 * R)
                && ps[1] >= (y * 2 * R + R) && ps[1] < ((y + 1) * 2 * R))
        {
            // move down right
            P = niter.goTo<5>(P1, 1);
            return P;
        }

        return P1;
    }
#endif

#ifdef JCC_FINDCELL_GEOMETRY

    //! \brief Return containing dual cell coord.
    DEVICE_HOST PointCoord findCell(Point2D ps) {
        PointCoord P;

        P[0] = (GLint) floor((ps[0]-FEuclid(FDual(PointCoord(0,0)))[0]) / (3*_R)) + 1;
        P[1] = (GLint) floor((ps[1]-FEuclid(FDual(PointCoord(0,0)))[1]) / (2*_R)) * 2 + 1;

        NIterHexa niter(FDualToBase(P), 0, 0);
        Point_2 pp0(FEuclid(F(niter.goTo<0>(FDualToBase(P), 1))));
        Point_2 pp1(FEuclid(F(niter.goTo<1>(FDualToBase(P), 1))));
        Point_2 pp2(FEuclid(F(niter.goTo<2>(FDualToBase(P), 1))));
        Point_2 pp3(FEuclid(F(niter.goTo<3>(FDualToBase(P), 1))));
        Point_2 pp4(FEuclid(F(niter.goTo<4>(FDualToBase(P), 1))));
        Point_2 pp5(FEuclid(F(niter.goTo<5>(FDualToBase(P), 1))));

        Vector_2 vp0(pp0, pp1);
        Vector_2 vp2(pp2, pp3);
        Vector_2 vp3(pp3, pp4);
        Vector_2 vp5(pp5, pp0);

        vp0 = vp0.perpendicular();
        vp2 = vp2.perpendicular();
        vp3 = vp3.perpendicular();
        vp5 = vp5.perpendicular();

        Point_2 pps(ps);

        NIterHexaDual niterd(P, 0, 0);

        if (ps[1] <= FEuclid(FDual(P))[1]) {
            Vector_2 vps(pp0,pps);
            Vector_2 vps2(pp2,pps);
            if (vp0 * vps > 0)
                P = niterd.goTo<4>(P, 1);
            else if (vp2 * vps2 > 0)
                P = niterd.goTo<2>(P, 1);
        }
        else {
            Vector_2 vps(pp3,pps);
            Vector_2 vps2(pp5,pps);
            if (vp3 * vps >= 0)
                P = niterd.goTo<1>(P, 1);
            else if (vp5 * vps2 >= 0)
                P = niterd.goTo<5>(P, 1);
        }
        return P;
    }

#endif

#ifdef HW_FINDCELL_GEOMETRY_WITH_ANGLE_MORT

    //! \brief Return containing dual cell coord.
    DEVICE_HOST PointCoord findCell(Point2D ps) {
        PointCoord P;

        // Two axis with slabs to consider
        PointCoord P1(0,0);
        PointCoord P2(1,1);
        PointCoord P3(0,1);
        PointCoord P4(0,2);

        // Pass in euclidean level using geometry
        Point_2 p(ps[0],ps[1]);

        Point_2 p1((this->FEuclid(this->FDual(P1)))[0], (this->FEuclid(this->FDual(P1)))[1]);
        Point_2 p2((this->FEuclid(this->FDual(P2)))[0], (this->FEuclid(this->FDual(P2)))[1]);
        Point_2 p3((this->FEuclid(this->FDual(P3)))[0], (this->FEuclid(this->FDual(P3)))[1]);
        Point_2 p4((this->FEuclid(this->FDual(P4)))[0], (this->FEuclid(this->FDual(P4)))[1]);

        // Two axis
        Vector_2 u(p1,p2);//down right
        Vector_2 v(p1,p3);//down left
        Vector_2 w(p1,p4);//down

        double size_slab_u = sqrt(u*u);
        double size_slab_v = size_slab_u;
        double size_slab_w = sqrt(w*w);

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
        NIterHexaDual niter(P1, 0, 0);

        if (v * Vector_2(p1, pi) >= 0)
            // Move down right, then down left
            P = niter.goTo<1>(niter.goTo<5>(P1, c_rank_u),c_rank_v);
        else
            // Move down right, then up right
            P = niter.goTo<4>(niter.goTo<5>(P1, c_rank_u),c_rank_v);

        // Verification with last vertical axis slab
        // Compute cell rank along w axis
        Point2D Peuclid = FEuclid(FDual(P));
        if (ps[1] < Peuclid[1] - size_slab_w/2)
            P = niter.goTo<3>(P, 1);
        else
            if (ps[1] >= Peuclid[1] + size_slab_w/2)
            P = niter.goTo<0>(P, 1);

        return P;
    }
#endif

};//ViewGridHexa


}//namespace components
//! @}
#endif // VIEWGRIDHEXA_H

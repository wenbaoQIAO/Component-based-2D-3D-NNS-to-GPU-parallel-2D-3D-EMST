#ifndef FILTERS_H
#define FILTERS_H
/*
 ***************************************************************************
 *
 * Author : H. Wang, J.C. Creput
 * Creation date : Apr. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <vector>
#include <iterator>

#include "macros_cuda.h"
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "ImageRW.h"

#define POSTPROCESS_OPTICAL_FLOW  1
#define EDGE_THRESHOLD 0.5
#define OCCLUSION_THRESHOLD 10.0f
#define FIX_OCCLU_MAX_RADIUS 40
#define MF_WINDOW_RADIUS 5
#define WMF_COE_SPA 18.0f
#define WMF_COE_COL 0.1f

using namespace std;
using namespace components;

namespace components
{

template <class T>
DEVICE_HOST void mirror(T w, T h, int& x, int& y)
{
    if (x < 0) x = ABS(x + 1);
    if (y < 0) y = ABS(y + 1);
    if (x >= w) x = w * 2 - x - 1;
    if (y >= h) y = h * 2 - y - 1;
}

//==============================================================================
// RGB2XYZ
//
// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
template <class T>
DEVICE_HOST void RGB2XYZ(
        T	sR,
        T	sG,
        T	sB,
        double&	X,
        double&	Y,
        double&	Z)
{
    double R = sR/255.0;
    double G = sG/255.0;
    double B = sB/255.0;

    double r, g, b;

    if(R <= 0.04045)	r = R/12.92;
    else				r = pow((R+0.055)/1.055,2.4);
    if(G <= 0.04045)	g = G/12.92;
    else				g = pow((G+0.055)/1.055,2.4);
    if(B <= 0.04045)	b = B/12.92;
    else				b = pow((B+0.055)/1.055,2.4);

    X = r*0.4124564 + g*0.3575761 + b*0.1804375;
    Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
    Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

template <class Image>
KERNEL void K_F_RGB2LAB(Image cmap)
{
    KER_SCHED(cmap.getWidth(), cmap.getHeight())

    if (_x < cmap.getWidth() && _y < cmap.getHeight())
    {
        //------------------------
        // sRGB to XYZ conversion
        //------------------------
        double X, Y, Z;
        RGB2XYZ(cmap[_y][_x][0], cmap[_y][_x][1], cmap[_y][_x][2], X, Y, Z);

        //------------------------
        // XYZ to LAB conversion
        //------------------------
        double epsilon = 0.008856;	//actual CIE standard
        double kappa   = 903.3;		//actual CIE standard

        double xr = X/0.950456;	//reference white
        double yr = Y/1.0;		//reference white
        double zr = Z/1.088754;	//reference white

        double fx, fy, fz;
        if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        else				fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        else				fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        else				fz = (kappa*zr + 16.0)/116.0;

        cmap[_y][_x][0] = 116.0*fy-16.0; //l
        cmap[_y][_x][1] = 500.0*(fx-fy); //a
        cmap[_y][_x][2] = 200.0*(fy-fz); //b
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 080815 : filter for left-right cross checking
// Grid<Point2D> flowMap is used to save the flow value for further use convenience
template <class NN>
KERNEL void K_F_LR_cross_checking(NN nn,  Grid<Point2D> rAdaptiveMap, Grid<Point2D> oriMap, Grid<Point2D> flowMap)
{
    KER_SCHED(nn.adaptiveMap.getWidth(), nn.adaptiveMap.getHeight())

    if (_x < nn.adaptiveMap.getWidth() && _y < nn.adaptiveMap.getHeight())
    {
        // Here the activeMap of the NN is employed to mark if the pixel is occluded or not
        nn.activeMap[_y][_x] = false;

        Motion left = nn.adaptiveMap[_y][_x] - oriMap[_y][_x];
        Motion right = rAdaptiveMap[_y][_x] - oriMap[_y][_x];
        flowMap[_y][_x] = left;

        float error = fabsf(left[0] + right[0]) + fabsf(left[1] + right[1]);
        if (error > OCCLUSION_THRESHOLD)
            nn.activeMap[_y][_x] = true;
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 080815 : filter for fixing occluded pixel
//! Assign an invalid pixel to the lowest disparity value of the spatially closest nonoccluded pixel
template <class NIter>
KERNEL void K_F_fix_occluded_pixels(Grid<Point2D> flowMap, Grid<bool> ifOccluded)
{
    KER_SCHED(flowMap.getWidth(), flowMap.getHeight())

    if (_x < flowMap.getWidth() && _y < flowMap.getHeight())
    {
        if (ifOccluded[_y][_x])
        {
            NIter niter;
            PointCoord pc(_x, _y);
            bool fixed = false;
            float lowest = 999999.999;
            PointCoord lowest_pos;
            int r = 0;
            while (r < FIX_OCCLU_MAX_RADIUS) // maximum searching radius
            {
                niter.initialize(pc, r, r+1);

                do {
                    PointCoord pco = niter.get();
                    if (pco[0] >= 0 && pco[0] < flowMap.getWidth() && pco[1] >= 0 && pco[1] < flowMap.getHeight())
                    {
                        if (!ifOccluded[pco[1]][pco[0]])
                        {
                            float val = fabsf(flowMap[pco[1]][pco[0]][0]) + fabsf(flowMap[pco[1]][pco[0]][1]);
                            if (val < lowest)
                            {
                                fixed = true;
                                lowest = val;
                                lowest_pos = pco;
                            }
                        }
                    }

                } while (niter.next());

                if(fixed)
                {
                    flowMap[_y][_x] = flowMap[lowest_pos[1]][lowest_pos[0]];
                    break;
                }
                else
                    r++;
            }

            if (!fixed)
                printf("Error when fixing occluded pixels : radius is too small !\n");
        }
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 080815 : 2D median filter for fixing occluded pixel
template <class G2D>
KERNEL void K_F_median_filter(G2D flowMap, Grid<bool> ifOccluded)
{
    KER_SCHED(flowMap.getWidth(), flowMap.getHeight())

    if (_x < flowMap.getWidth() && _y < flowMap.getHeight())
    {
        if (ifOccluded[_y][_x])
        {
            int k = 0;
            const int R = MF_WINDOW_RADIUS;
            float window[(2 * R + 1) * (2 * R + 1)];
            int half = ((2 * R + 1) * (2 * R + 1) / 2) + 1;

            // ----------------- flow value in X direction -----------------
            // Pick up window elements
            for (int j = _y - R; j <= _y + R; ++j)
            {
                for (int i = _x - R; i <= _x + R; ++i)
                {
                    int __i = i;
                    int __j = j;
                    mirror(flowMap.getWidth(), flowMap.getHeight(), __i, __j);
                    window[k++] = flowMap[__j][__i][0];
                }
            }
            // Order elements (only half of them)
            // Selection Sort
            for (int j = 0; j < half; ++j)
            {
                // Find position of minimum element
                int min = j;
                for (int l = j + 1; l < (2 * R + 1) * (2 * R + 1); ++l)
                {
                    if (window[l] < window[min])
                    {
                        min = l;
                    }
                }
                // Put found minimum element in its place
                const float temp = window[j];
                window[j] = window[min];
                window[min] = temp;
            }
            // Get result - the middle element
            flowMap[_y][_x][0] = window[half - 1];

#if POSTPROCESS_OPTICAL_FLOW
            // ----------------- flow value in Y direction -----------------
            // Pick up window elements
            k = 0;
            for (int j = _y - R; j <= _y + R; ++j)
            {
                for (int i = _x - R; i <= _x + R; ++i)
                {
                    int __i = i;
                    int __j = j;
                    mirror(flowMap.getWidth(), flowMap.getHeight(), __i, __j);
                    window[k++] = flowMap[__j][__i][1];
                }
            }
            // Order elements (only half of them)
            // Selection Sort
            for (int j = 0; j < half; ++j)
            {
                // Find position of minimum element
                int min = j;
                for (int l = j + 1; l < (2 * R + 1) * (2 * R + 1); ++l)
                {
                    if (window[l] < window[min])
                    {
                        min = l;
                    }
                }
                // Put found minimum element in its place
                const float temp = window[j];
                window[j] = window[min];
                window[min] = temp;
            }
            // Get result - the middle element
            flowMap[_y][_x][1] = window[half - 1];
#endif
        }
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 080815 : 2D weighted median filter for fixing occluded pixel
//! The weighted median filter is applied to the occulded pixels using the bilateral filter weights.
template <class NN>
KERNEL void K_F_weighted_median_filter(NN nnr, Grid<Point2D> flowMap)
{
    KER_SCHED(flowMap.getWidth(), flowMap.getHeight())

    if (_x < flowMap.getWidth() && _y < flowMap.getHeight())
    {
        if (nnr.activeMap[_y][_x])
        {
            int k, index;
            const int R = MF_WINDOW_RADIUS;
            float window[(2 * R + 1) * (2 * R + 1)];
            float window_weight[(2 * R + 1) * (2 * R + 1)];
            float sum_weight, accum_weight;

            // ----------------- flow value in X direction -----------------
            // Pick up window elements
            k = 0;
            sum_weight = 0.0f;
            for (int j = _y - R; j <= _y + R; ++j)
            {
                for (int i = _x - R; i <= _x + R; ++i)
                {
                    int __i = i;
                    int __j = j;
                    mirror(flowMap.getWidth(), flowMap.getHeight(), __i, __j);
                    window[k] = flowMap[__j][__i][0];

                    float dis_spatial = (_x - i) * (_x - i) + (_y - j) * (_y - j);
                    dis_spatial /= (float)(WMF_COE_SPA * WMF_COE_SPA);
                    float dis_weight =
                            (nnr.colorMap[_y][_x][0] - nnr.colorMap[__j][__i][0]) * (nnr.colorMap[_y][_x][0] - nnr.colorMap[__j][__i][0]) +
                            (nnr.colorMap[_y][_x][1] - nnr.colorMap[__j][__i][1]) * (nnr.colorMap[_y][_x][1] - nnr.colorMap[__j][__i][1]) +
                            (nnr.colorMap[_y][_x][2] - nnr.colorMap[__j][__i][2]) * (nnr.colorMap[_y][_x][2] - nnr.colorMap[__j][__i][2]);
                    dis_weight /= (float)(WMF_COE_COL * WMF_COE_COL);
                    window_weight[k] = exp(-(dis_spatial + dis_weight));
                    sum_weight += window_weight[k];
                    k++;
                }
            }
            // Order elements (only half of them)
            // Selection Sort
            for (int j = 0; j < (2 * R + 1) * (2 * R + 1); ++j)
            {
                // Find position of minimum element
                int min = j;
                for (int l = j + 1; l < (2 * R + 1) * (2 * R + 1); ++l)
                {
                    if (window[l] < window[min])
                    {
                        min = l;
                    }
                }
                // Put found minimum element in its place
                const float temp = window[j];
                const float temp_weight = window_weight[j];
                window[j] = window[min];
                window_weight[j] = window_weight[min];
                window[min] = temp;
                window_weight[min] = temp_weight;
            }
            // Get result - the middle element according to the weights
            accum_weight = 0.0f;
            index = 0;
            while (accum_weight < (sum_weight / 2.0f))
            {
                accum_weight += window_weight[index++];
            }
            flowMap[_y][_x][0] = window[index - 1];

#if POSTPROCESS_OPTICAL_FLOW
            // ----------------- flow value in Y direction -----------------
            // Pick up window elements
            k = 0;
            sum_weight = 0.0f;
            for (int j = _y - R; j <= _y + R; ++j)
            {
                for (int i = _x - R; i <= _x + R; ++i)
                {
                    int __i = i;
                    int __j = j;
                    mirror(flowMap.getWidth(), flowMap.getHeight(), __i, __j);
                    window[k] = flowMap[__j][__i][1];

                    float dis_spatial = (_x - i) * (_x - i) + (_y - j) * (_y - j);
                    dis_spatial /= (float)(WMF_COE_SPA * WMF_COE_SPA);
                    float dis_weight =
                            (nnr.colorMap[_y][_x][0] - nnr.colorMap[__j][__i][0]) * (nnr.colorMap[_y][_x][0] - nnr.colorMap[__j][__i][0]) +
                            (nnr.colorMap[_y][_x][1] - nnr.colorMap[__j][__i][1]) * (nnr.colorMap[_y][_x][1] - nnr.colorMap[__j][__i][1]) +
                            (nnr.colorMap[_y][_x][2] - nnr.colorMap[__j][__i][2]) * (nnr.colorMap[_y][_x][2] - nnr.colorMap[__j][__i][2]);
                    dis_weight /= (float)(WMF_COE_COL * WMF_COE_COL);
                    window_weight[k] = exp(-(dis_spatial + dis_weight));
                    sum_weight += window_weight[k];
                    k++;
                }
            }
            // Order elements (only half of them)
            // Selection Sort
            for (int j = 0; j < (2 * R + 1) * (2 * R + 1); ++j)
            {
                // Find position of minimum element
                int min = j;
                for (int l = j + 1; l < (2 * R + 1) * (2 * R + 1); ++l)
                {
                    if (window[l] < window[min])
                    {
                        min = l;
                    }
                }
                // Put found minimum element in its place
                const float temp = window[j];
                const float temp_weight = window_weight[j];
                window[j] = window[min];
                window_weight[j] = window_weight[min];
                window[min] = temp;
                window_weight[min] = temp_weight;
            }
            // Get result - the middle element according to the weights
            accum_weight = 0.0f;
            index = 0;
            while (accum_weight < (sum_weight / 2.0f))
            {
                accum_weight += window_weight[index++];
            }
            flowMap[_y][_x][1] = window[index - 1];
#endif
        }
    }

    END_KER_SCHED

    SYNCTHREADS;
}

// edge detector
template <class NN>
KERNEL void K_F_detectColorEdges(NN nn)
{
    KER_SCHED(nn.colorMap.getWidth(), nn.colorMap.getHeight())

    if (_x < nn.colorMap.getWidth() && _y < nn.colorMap.getHeight())
    {
        int ix0, iy0;

        // upper middle
        ix0 = _x;
        iy0 = MAX(0, (_y - 1));
        Point3D um = nn.colorMap[iy0][ix0];

        // middle left
        ix0 = MAX(0, (_x - 1));
        iy0 = _y;
        Point3D ml = nn.colorMap[iy0][ix0];

        // middle right
        ix0 = MIN((nn.colorMap.getWidth() - 1), (_x + 1));
        iy0 = _y;
        Point3D mr = nn.colorMap[iy0][ix0];

        // lower middle
        ix0 = _x;
        iy0 = MIN((nn.colorMap.getHeight() - 1), (_y + 1));
        Point3D lm = nn.colorMap[iy0][ix0];

        DistanceSquaredEuclidean<Point3D> dis;
        GLfloat dx = dis(ml, mr);
        GLfloat dy = dis(um, lm);

        nn.densityMap[_y][_x] = fabsf(dx) + fabsf(dy);
//        nn.densityMap[_y][_x] = dx*dx + dy*dy;
    }

    END_KER_SCHED

    SYNCTHREADS;
}

template <class NN>
KERNEL void K_F_detectColorEdgesSobel(NN nn)
{
    KER_SCHED(nn.colorMap.getWidth(), nn.colorMap.getHeight())

    if (_x < nn.colorMap.getWidth() && _y < nn.colorMap.getHeight())
    {
        int ix0, iy0;

        // upper left
        ix0 = MAX(0, (_x - 1));
        iy0 = MAX(0, (_y - 1));
        Point3D ul = nn.colorMap[iy0][ix0];

        // upper middle
        ix0 = _x;
        iy0 = MAX(0, (_y - 1));
        Point3D um = nn.colorMap[iy0][ix0];

        // upper right
        ix0 = MIN((nn.colorMap.getWidth() - 1), (_x + 1));
        iy0 = MAX(0, (_y - 1));
        Point3D ur = nn.colorMap[iy0][ix0];

        // middle left
        ix0 = MAX(0, (_x - 1));
        iy0 = _y;
        Point3D ml = nn.colorMap[iy0][ix0];

        // middle right
        ix0 = MIN((nn.colorMap.getWidth() - 1), (_x + 1));
        iy0 = _y;
        Point3D mr = nn.colorMap[iy0][ix0];

        // lower left
        ix0 = MAX(0, (_x - 1));
        iy0 = MIN((nn.colorMap.getHeight() - 1), (_y + 1));
        Point3D ll = nn.colorMap[iy0][ix0];

        // lower middle
        ix0 = _x;
        iy0 = MIN((nn.colorMap.getHeight() - 1), (_y + 1));
        Point3D lm = nn.colorMap[iy0][ix0];

        // lower right
        ix0 = MIN((nn.colorMap.getWidth() - 1), (_x + 1));
        iy0 = MIN((nn.colorMap.getHeight() - 1), (_y + 1));
        Point3D lr = nn.colorMap[iy0][ix0];

        Point3D dxx = (ur + (mr * 2) + lr - ul - (ml * 2) - ll);
        Point3D dyy = (ul + (um * 2) + ur - ll - (lm * 2) - lr);

        GLfloat dx = sqrt(dxx[0]*dxx[0] + dxx[1]*dxx[1] + dxx[2]*dxx[2]);
        GLfloat dy = sqrt(dyy[0]*dyy[0] + dyy[1]*dyy[1] + dyy[2]*dyy[2]);

        nn.densityMap[_y][_x] = dx + dy;
//        nn.densityMap[_y][_x] = dx*dx + dy*dy;
//        nn.densityMap[_y][_x] *= nn.densityMap[_y][_x];
//        nn.densityMap[_y][_x] *= nn.densityMap[_y][_x];
    }

    END_KER_SCHED

    SYNCTHREADS;
}

template <class NN>
KERNEL void K_F_setEdgeAndDensity(NN md,
                                  Image emap,
                                  Image dmap,
                                  GLfloat max,
                                  GLfloat min)
{
    KER_SCHED(md.densityMap.getWidth(), md.densityMap.getHeight())

    if (_x < md.densityMap.getWidth() && _y < md.densityMap.getHeight())
    {
        // init
        md.activeMap[_y][_x] = false;
        GLfloat scale = (max - min);
        GLfloat unit = (max - min) / 256.0f;

        // set activeMap
        if (((md.densityMap[_y][_x] - min) / scale) > EDGE_THRESHOLD)
        {
            md.activeMap[_y][_x] = true;
            emap[_y][_x][0] = 0.0f;
            emap[_y][_x][1] = 1.0f; // 255.0f;
            emap[_y][_x][2] = 0.0f;
        }
//        dmap[_y][_x][0] = 0.0f;
        dmap[_y][_x][0] = MIN(((md.densityMap[_y][_x] - min) / unit), 255.0f);
        dmap[_y][_x][1] = MIN(((md.densityMap[_y][_x] - min) / unit), 255.0f);
        dmap[_y][_x][2] = MIN(((md.densityMap[_y][_x] - min) / unit), 255.0f);
#if NORMALIZATION
        dmap[_y][_x][0] /= 255.0f;
        dmap[_y][_x][1] /= 255.0f;
        dmap[_y][_x][2] /= 255.0f;
#endif
//        dmap[_y][_x][2] = 0.0f;
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//! HW 080815 : filter for left-right cross checking
GLOBAL inline void K_LR_cross_checking(NN& nn, Grid<Point2D>& rAdaptiveMap, Grid<Point2D>& oriMap, Grid<Point2D>& flowMap) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn.adaptiveMap.getWidth(),
                          nn.adaptiveMap.getHeight());

    K_F_LR_cross_checking _KER_CALL_(b, t) (nn, rAdaptiveMap, oriMap, flowMap);

}

//! HW 080815 : filter for fixing occluded pixel
//! Assign an invalid pixel to the lowest disparity value of the spatially closest nonoccluded pixel
template <class NIter>
GLOBAL inline void K_fix_occluded_pixels(Grid<Point2D>& flowMap, Grid<bool>& ifOccluded) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          flowMap.getWidth(),
                          flowMap.getHeight());

    K_F_fix_occluded_pixels<NIter> _KER_CALL_(b, t) (flowMap, ifOccluded);

}

//! HW 080815 : 2D median filter for fixing occluded pixel
GLOBAL inline void K_median_filter(Grid<Point2D>& flowMap, Grid<bool>& ifOccluded) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          flowMap.getWidth(),
                          flowMap.getHeight());

    K_F_median_filter _KER_CALL_(b, t) (flowMap, ifOccluded);

}

//! HW 080815 : 2D weighted median filter for fixing occluded pixel
//! The weighted median filter is applied to the occulded pixels using the bilateral filter weights.
GLOBAL inline void K_weighted_median_filter(NN& nnr, Grid<Point2D>& flowMap) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          flowMap.getWidth(),
                          flowMap.getHeight());

    K_F_weighted_median_filter _KER_CALL_(b, t) (nnr, flowMap);

}

enum gradientType { average, sobelEdge };

template<gradientType gType>
GLOBAL inline void K_detectColorEdges(NN& nn) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          nn.colorMap.getWidth(),
                          nn.colorMap.getHeight());

    if (gType == average) {
        K_F_detectColorEdges _KER_CALL_(b, t) (nn);
    } else if (gType == sobelEdge) {
        K_F_detectColorEdgesSobel _KER_CALL_(b, t) (nn);
    }
}

GLOBAL inline void K_RGB2LAB(Image& colorMap) {

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          colorMap.getWidth(),
                          colorMap.getHeight());

    K_F_RGB2LAB _KER_CALL_(b, t) (colorMap);
}


template<class NN>
void setEdgeAndWriteDensityImage(NN& md, string fname)
{
    int width = md.densityMap.getWidth();
    int height = md.densityMap.getHeight();
    Grid<GLfloat> tmp_dmap;
    tmp_dmap.resize(width, height);
    tmp_dmap.gpuCopyDeviceToHost(md.densityMap);
    GLfloat max = 0.0f;
    GLfloat min = 999999999.9f;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            if (tmp_dmap[y][x] > max)
                max = tmp_dmap[y][x];
            if (tmp_dmap[y][x] < min)
                min = tmp_dmap[y][x];
        }
    tmp_dmap.freeMem();
    Image tmp_dimage;
    Image tmp_eimage;
    tmp_dimage.gpuResize(width, height);
    tmp_eimage.gpuResize(width, height);
#ifdef CUDA_CODE
    tmp_eimage.gpuMemSetValue(0);
#else
    tmp_eimage.gpuMemSet(Point3D(0.0f,0.0f,0.0f));
#endif

    KER_CALL_THREAD_BLOCK(b, t,
                          4, 4,
                          width,
                          height);

    K_F_setEdgeAndDensity _KER_CALL_(b, t) (md, tmp_eimage, tmp_dimage, max, min);

    Image tmp_dimage2;
    Image tmp_eimage2;
    tmp_dimage2.resize(width, height);
    tmp_eimage2.resize(width, height);
    tmp_dimage2.gpuCopyDeviceToHost(tmp_dimage);
    tmp_eimage2.gpuCopyDeviceToHost(tmp_eimage);

    // Save image
    IRW irw;
    string str = fname;
    int pos = irw.getPos(str);
    string s = str.substr(0, pos);
    s.insert(0, "Gradient");
    s.append(".colors");
    std::ofstream fo;
    fo.open(s.c_str(), ofstream::out);
    if (fo)
        fo << tmp_dimage2;
    irw.write(s, tmp_dimage2);
    fo.close();

    string s2 = str.substr(0, pos);
    s2.insert(0, "Edge");
    s2.append(".colors");
    fo.open(s2.c_str(), ofstream::out);
    if (fo)
        fo << tmp_eimage2;
    irw.write(s2, tmp_eimage2);
    fo.close();

    tmp_dimage.gpuFreeMem();
    tmp_eimage.gpuFreeMem();
    tmp_dimage2.freeMem();
    tmp_eimage2.freeMem();
}


// The following function only works on CPU version.
template<class NN>
void writeDensityImage(NN& md, string fname)
{
    GLfloat max = 0.0f;
    GLfloat min = 999999999.9f;
    for (int y = 0; y < md.densityMap.getHeight(); y++)
        for (int x = 0; x < md.densityMap.getWidth(); x++)
        {
            if (md.densityMap[y][x] > max)
                max = md.densityMap[y][x];
            if (md.densityMap[y][x] < min)
                min = md.densityMap[y][x];

            // init activeMap. Note that here the activeMap is used to decide if the pixel is edge or not.
            md.activeMap[y][x] = false;
        }
    GLfloat scale = (max - min);
    GLfloat unit = (max - min) / 255.0f;
    NN tempDensity, temEdge;
    tempDensity.resize(md.densityMap.getWidth(), md.densityMap.getHeight());
    tempDensity.colorMap.resetValue(Point3D(0.0,0.0,0.0));
    temEdge.resize(md.densityMap.getWidth(), md.densityMap.getHeight());
    temEdge.colorMap.resetValue(Point3D(0.0,0.0,0.0));
    for (int y = 0; y < md.densityMap.getHeight(); y++)
        for (int x = 0; x < md.densityMap.getWidth(); x++)
        {
            // set activeMap
            if (((md.densityMap[y][x] - min) / scale) > EDGE_THRESHOLD)
            {
                md.activeMap[y][x] = true;
                temEdge.colorMap[y][x][1] = 1.0f; //255.0f;
            }
            tempDensity.colorMap[y][x][1] = MIN(((md.densityMap[y][x] - min) / unit), 255.0f);
#if NORMALIZATION
            tempDensity.colorMap[y][x][1] /= 255.0f;
#endif
        }

    // Save image
    IRW irw;
    string str = fname;
    int pos = irw.getPos(str);
    string s = str.substr(0, pos);
    s.insert(0, "G");
    s.append(".colors");
    std::ofstream fo;
    fo.open(s.c_str(), ofstream::out);
    if (fo)
        fo << tempDensity.colorMap;
    irw.write(s, tempDensity);
    fo.close();

    string s2 = str.substr(0, pos);
    s2.insert(0, "E");
    s2.append(".colors");
    fo.open(s2.c_str(), ofstream::out);
    if (fo)
        fo << temEdge.colorMap;
    irw.write(s2, temEdge);
    fo.close();

    tempDensity.freeMem();
    temEdge.freeMem();
}

}//namespace components

#endif // FILTERS_H

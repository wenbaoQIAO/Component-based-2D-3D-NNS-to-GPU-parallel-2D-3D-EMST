#ifndef SPIRAL_SEARCH_NNL_H
#define SPIRAL_SEARCH_NNL_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, Wenbao Qiao
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include "NeuralNetEMST.h"
#include "SpiralSearch.h"

#define TEST_CODE 0

using namespace std;

namespace components
{

template <
        class Cell,
        class NIter,
        class IndexCM
        >
struct SpiraSearchEMST {

    size_t start;
    size_t d_min;
    size_t d_max;
    size_t d_step;

    IndexCM pc;//Coordinates cell center in gdc

    GLfloat minDistance;

    DEVICE_HOST SpiraSearchEMST(
            IndexCM pc,
            GLfloat md,
            size_t d_min,
            size_t d_max,
            size_t d_step
            )
        :
          pc(pc),
          minDistance(md),
          start(0),
          d_min(d_min),
          d_max(d_max),
          d_step(d_step)
    {}

    template <class CM, class NN >
    DEVICE_HOST bool search(CM& gdc, NN& scher,
                            NN& sched, PointCoord ps,
                            PointCoord& minPCoord, int radiusSearchCells)  {

        bool ret = false;
        NIter ni(pc, 0, d_max);

        minPCoord[0] = -1;
        minPCoord[1] = -1;
        minDistance = INFINITY;

        do {

            if (ni.getCurrentDistance() > d_max)
                break;

            PointCoord pcell = ni.getNodeIncr();

            if (gdc.valideAndPositiveIndex(pcell)) {

                //! wb.Q change to use pointers
                Cell* cell = &gdc(pcell);

                bool ret = false;
                size_t count = 0;
                while (count < cell->size)
                {
                    PointCoord pco = cell->bCell[count];
                    if (scher.correspondenceMap.valideIndex(pco))
                    {
                        if(sched.disjointSetMap.findRoot(scher.disjointSetMap, scher.disjointSetMap.compute_offset(pco))
                                != scher.disjointSetMap.findRoot(scher.disjointSetMap, scher.disjointSetMap.compute_offset(ps))){

                            typedef typename NN::point_type point_type;
                            GLfloat v =
                            components::DistanceEuclidean<point_type>()(
                                        scher.adaptiveMap(ps),
                                        sched.adaptiveMap(pco)
                                        );

                            if (minPCoord[0] == -1)
                            {
                                ni.setMaxDistance(MAX(d_min, ni.getCurrentDistance() + (int)ceil((float) ni.getCurrentDistance()*0.1) + 1));
                                ret = true;
                                minPCoord = pco;
                                minDistance = v;
                                scher.correspondenceMap(ps) = pco;
                                scher.densityMap(ps) = v;
                            }
                            else
                                if (v < minDistance )
                                {
                                    ni.setMaxDistance(MAX(d_min, ni.getCurrentDistance() + (int)ceil((float) ni.getCurrentDistance()*0.1) + 1));
                                    ret = true;
                                    minPCoord = pco;
                                    minDistance = v;
                                    scher.correspondenceMap(ps) = pco;
                                    scher.densityMap(ps) = v;
                                }
                                else
                                    if (v == minDistance && pco[0] < minPCoord[0])
                                    {
                                        ni.setMaxDistance(MAX(d_min,ni.getCurrentDistance()  + (int)ceil((float) ni.getCurrentDistance()*0.1) + 1));
                                        ret = true;
                                        minPCoord = pco;
                                        minDistance = v;
                                        scher.correspondenceMap(ps) = pco;
                                        scher.densityMap(ps) = v;
                                    }
                        }
                    }
                    count++;
                }//while cell
            }
        } while (ni.nextNodeIncr());

        return ret;
    }

};

//! 261016 wenbao Qiao add searching neighbor nodes with different id_superVertices considering last radius lexico
//! after this step, evey node find its closest non-same-component node (minPCoord) and has its euclidean distance in distanceMap
//! the radius of last iteration is registrated and can be used in next iteration
template <
        class Cell,
        class NIter,
        class IndexCM
        >
struct SpiraSearchEMST_WQ {

    size_t start;
    size_t d_min;
    size_t d_max;
    size_t d_step;

    IndexCM pc;//Coordinates cell center in gdc

    GLfloat minDistance;

    DEVICE_HOST SpiraSearchEMST_WQ(
            IndexCM pc,
            GLfloat md,
            size_t d_min,
            size_t d_max,
            size_t d_step
            )
        :
          pc(pc),
          minDistance(md),
          start(0),
          d_min(d_min),
          d_max(d_max),
          d_step(d_step)
    {}

    template <class CM, class NN1, class NN2>
    DEVICE_HOST bool search(CM& gdc, NN1& scher,
                            NN2& sched, PointCoord ps,
                            PointCoord& minPCoord, int radiusSearchCells)  {


        // wb.Q 131016 add to initialize minDistance
        minDistance = INFINITY;

        // wb.Q 261016 refresh minRadius
        //        d_min = scher.minRadiusMap[ps[1]][ps[0]];

        //                if(d_min - 1 > 0)
        //                    d_min = d_min - 1;
        //                else
        //                    d_min = 0;
        //                d_min = (d_min - 1 > 0)? d_min -1 : 0;

        bool ret = false;
        NIter ni(pc, 0, d_max);
        //        NIter ni(pc, d_min, d_max);

        minPCoord[0] = -1;
        minPCoord[1] = -1;

        do {

            if (ni.getCurrentDistance() > d_max)
                break;

            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < gdc.getWidth()
                    && pco[1] >= 0 && pco[1] < gdc.getHeight()) {

                //! wb.Q change to use pointers
                Cell* cell = &gdc[pco[1]][pco[0]];

                (*cell).minDistance = minDistance;

                PointCoord minCellP(-1);

                //! WB.Q change this searching with different ID of superVertices and compare equale weight
                if ((*cell).searchClosestDiffIdEqualWeightAuMinimum(scher, sched, ps, minCellP)) {
                    if (minPCoord[0] == -1)
                    {
                        ni.setMaxDistance(MAX(d_min, ni.getCurrentDistance() + d_step+1));
                        ret = true;

                        minDistance = (*cell).getMinDistance();
                        scher.distanceMap[ps[1]][ps[0]] = minDistance;
                        minPCoord = (*cell).getMinPCoord();
                        scher.correspondenceMap[ps[1]][ps[0]] = minCellP;
                        scher.minRadiusMap[ps[1]][ps[0]] = ni.getCurrentDistance();
                    }
                    else
                    if ((*cell).getMinDistance() < minDistance)
                    {
                        ni.setMaxDistance(MAX(d_min, ni.getCurrentDistance() + d_step+1));
                        ret = true;

                        minDistance = (*cell).getMinDistance();
                        scher.distanceMap[ps[1]][ps[0]] = minDistance;
                        minPCoord = (*cell).getMinPCoord();
                        scher.correspondenceMap[ps[1]][ps[0]] = minCellP;
                        scher.minRadiusMap[ps[1]][ps[0]] = ni.getCurrentDistance();
                    }
                    else
                    if((*cell).getMinDistance() == minDistance)
                    {
                        if(minCellP[0] < minPCoord[0]) {

                            ni.setMaxDistance(MAX(d_min, ni.getCurrentDistance()+d_step+1));

                            ret = true;

                            minDistance = (*cell).getMinDistance();
                            scher.distanceMap[ps[1]][ps[0]] = minDistance;
                            minPCoord = (*cell).getMinPCoord();
                            scher.correspondenceMap[ps[1]][ps[0]] = minCellP;
                            scher.minRadiusMap[ps[1]][ps[0]] = ni.getCurrentDistance();
                        }
                    }
                }//if cell
            }
        } while (ni.nextNodeIncr());

        return ret;
    }

};//SpiralSearchWithDiffMinRadiusToFindNodesWithDiffIdLexico



}//namespace components

#endif // SPIRAL_SEARCH_NNL_H

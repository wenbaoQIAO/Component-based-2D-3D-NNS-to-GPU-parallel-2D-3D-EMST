#ifndef CGAL_BOOST_ADAPTATORS_H
#define CGAL_BOOST_ADAPTATORS_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2014
 *
 ***************************************************************************
 */
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/register/point.hpp>

#include <boost/assign.hpp>

//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Cartesian.h>

#include <CGAL/Vector_2.h>
#include <CGAL/Segment_2.h>
#include <CGAL/Line_2.h>
#include <CGAL/Circle_2.h>
#include <CGAL/Bbox_2.h>
#include <CGAL/Iso_rectangle_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Aff_transformation_2.h>

//using namespace CGAL;
//typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Cartesian<double> K;

typedef K::Point_2 Point_2;

/*!
 * \defgroup CGALBOOST Structures pour compatibilite BOOST
 * \brief Ce module correspond a l'espace de nommage boost.
 * Il permet de rendre compatible les structures de classe point
 * avec la librairie Boost. Schema valable aussi bien pour les classes
 * proprietaires que CGAL.
 */
/*! @{*/
namespace boost
{
    namespace geometry
    {
        namespace traits
        {
            // Adapt Point_2 to Boost.Geometry

            template<> struct tag<Point_2>
            { typedef point_tag type; };

            template<> struct coordinate_type<Point_2>
            { typedef K::FT type; };

            template<> struct coordinate_system<Point_2>
            { typedef cs::cartesian type; };

            template<> struct dimension<Point_2> : boost::mpl::int_<2> {};

            template<>
            struct access<Point_2, 0>
            {
                static K::FT get(Point_2 const& p)
                {
                    return p.x();
                }

                static void set(Point_2& p, K::FT const& value)
                {
                    p = Point_2(value, p.y());
                }
            };

            template<>
            struct access<Point_2, 1>
            {
                static K::FT get(Point_2 const& p)
                {
                    return p.y();
                }

                static void set(Point_2& p, K::FT const& value)
                {
                    p = Point_2(p.x(), value);
                }
            };
        }
    }
} // namespace boost::geometry::traits
//! @}

#endif // CGAL_BOOST_ADAPTATORS_H

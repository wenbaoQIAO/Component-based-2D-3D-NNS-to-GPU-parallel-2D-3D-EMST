#ifndef GEOMETRY_CGAL_BOOST_H
#define GEOMETRY_CGAL_BOOST_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2014
 *
 ***************************************************************************
 */

#include "cgal_boost_adaptators.h"

#include <boost/geometry.hpp>

/*!
 * \defgroup FoncCGAL Fonctions geometriques basees sur la librairie CGAL
 * \brief Ce module correspond a l'espace de nommage geometry_cgal_boost.
 * Il comporte l'ensemble des fonctions geometriques et classes d'objets
 * geometriques de base telles que fournies par CGAL ou implementees avec CGAL et Boost.
 */
/*! @{*/
namespace geometry_cgal_boost
{
class Rect;

/*! @name Classes fournies par CGAL et Boost
 * \brief Renommage des types d'objets geometriques de base.
 */
//! @{
typedef K::Point_2 Point_2;
typedef K::Segment_2 Segment_2;
typedef K::Vector_2 Vector_2;
typedef CGAL::Bbox_2 Bbox_2;
typedef K::Circle_2 Circle_2;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Aff_transformation_2<K> Transformation_2;

// Declare un polyline boost
typedef boost::geometry::model::linestring<Point_2> Polyline_B;
// Declare un polygone boost
typedef boost::geometry::model::ring<Point_2> Polygon_B;
// Classe specifique because CGAL Rect non pertinent
typedef Rect Rect_2;
//! @}
}

namespace geometry_cgal_boost
{
//! Classe rectangle de base
class Rect {
    Point_2 _min;

    double _width;
    double _height;
public:
    Rect() : _min(Point_2(0,0)), _width(0), _height(0) {}
    Rect(Point_2 min, double width, double height) : _min(min), _width(width), _height(height) {}

    inline double xmin() { return _min.x(); }
    inline double ymin() { return _min.y(); }
    inline double width() const { return _width; }
    inline double height() const { return _height; }
    friend inline const Point_2& min BOOST_PREVENT_MACRO_SUBSTITUTION (const Rect& r);
};

inline const Point_2& min BOOST_PREVENT_MACRO_SUBSTITUTION (const Rect& r) {
    return r._min;
}

/*!
 * \brief Renvoie le barycentre d'un polygone
 * \param p polygone d'entree
 * \return Point barycentre
 */
template <class Polygon_2>
inline Point_2 barycentre(Polygon_2& p) {
    Point_2 c;

    Polygon_B p_b(p.container().begin(), p.container().end());
    p_b.push_back(p.container().front());

    boost::geometry::centroid(p_b, c);

    return c;
}

//! \brief contains
//! \param p polygone conteneur
//! \param r rectangle inclu
//! \return vrai si p contient r
template <class Polygon_2, class Rect_2>
inline bool contains(Polygon_2& p, const Rect_2& r) {
    bool ret = false;
    Point_2 p_min = min(r);
    if (p.has_on_negative_side(p_min)
            && p.has_on_negative_side(Point_2(p_min.x()+r.width(), p_min.y()))
            && p.has_on_negative_side(Point_2(p_min.x()+r.width(), p_min.y()+r.height()))
            && p.has_on_negative_side(Point_2(p_min.x(), p_min.y()+r.height()))
            )
        ret = true;

    return ret;
}

/*! @name Transformations
 * \brief Translation et rotation.
 */
//! @{
enum Trans {
    TRANSLATION,
    ROTATION,
    ROTATION_TRANSLATION,
    TRANSLATION_ROTATION
};

inline Transformation_2 getTransformation(Trans type, Vector_2 v, double sin, double cos) {

    Transformation_2 translate(CGAL::TRANSLATION, v);
    Transformation_2 rotate(CGAL::ROTATION, sin, cos);
    Transformation_2 transf;
    if (type == ROTATION_TRANSLATION)
        transf = translate * rotate;
    else if (type == TRANSLATION_ROTATION)
        transf = rotate * translate;

    return transf;
}

//! Transforme polyline boost
inline Polyline_B transform(Transformation_2& transf, const Polyline_B& p)
{
    Polyline_B r;

    Polyline_B::const_iterator first1 = p.begin();
    Polyline_B::const_iterator last1 = p.end();
    while (first1 != last1) {
        r.push_back(transf(*first1));
        ++first1;
    }
    return r;
}

inline Polygon_2 transform(const Transformation_2& transf, const Polygon_2& p)
{
  return CGAL::transform(transf, p);
}

inline Segment_2 transform(const Transformation_2& transf, const Segment_2& s)
{
  return s.transform(transf);
}

inline Point_2 transform(const Transformation_2& transf, const Point_2& p)
{
  return transf.transform(p);
}

//! @}

//! Indique si le polygone est simple
inline bool is_simple(const Polygon_2& p)
{
    return p.is_simple();
}

//! Indique si le polygone est oriente "clockwise"
inline bool is_clockwise_oriented(const Polygon_2& p)
{
    return p.is_clockwise_oriented();
}

//! Renverse l'orientation du polygone
inline void reverse_orientation(Polygon_2& p)
{
    p.reverse_orientation();
}


inline bool less_x(const Point_2 &p, const Point_2 &q)
{
    return CGAL::less_x(p, q);
}

inline bool less_y(const Point_2 &p, const Point_2 &q)
{
    return CGAL::less_y(p, q);
}

}//namespace geometry_cgal_boost
//! @}

#endif // GEOMETRY_CGAL_BOOST_H

#ifndef GEOMETRY_COMMON_H
#define GEOMETRY_COMMON_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2014
 *
 ***************************************************************************
 */

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#include "geometry_prop.h"

#ifdef UTILISE_GEOMETRIE_CGAL
#include "geometry_cgal_boost.h"
#endif

#ifdef UTILISE_GEOMETRIE_CGAL
namespace geometry = geometry_cgal_boost;
#else
namespace geometry = geometry_prop;
#endif

typedef geometry_p::Point_2 Point_2;
typedef geometry_p::Segment_2 Segment_2;
typedef geometry_p::Vector_2 Vector_2;
typedef geometry_p::Polyline_2 Polyline_2;
typedef geometry_p::Polygon_2 Polygon_2;

// Declare transformation
typedef geometry_p::Bbox_2 Bbox_2;
// Declare transformation
typedef geometry_p::Circle_2 Circle_2;
// Declare transformation
typedef geometry_p::Transformation_2 Transformation_2;

// Declare un rectangle proprietaire
typedef geometry_p::Rect_2 Rect_2;



/*! \brief Structure definissant le positionnement courant
 * d'un element.
 */
struct PositElement
{
	//! \brief position en x,y
	Point_2 position;  //position courante du point de reference element
	//! \brief angle avec l'horisontale
	double angle;  //angle de rotation en degres

	PositElement() : position(Point_2(0, 0)), angle(0) {}
	PositElement(Point_2 p, double a) : position(p), angle(a) {}
};

/*! Foncteur transformant un boost polyline en
 * vecteur de segments
 */
template <typename Segment>
struct gather_segment_to_vector
{
	// Remember that if coordinates are integer, the length might be floating point
	// So use "double" for integers. In other cases, use coordinate type
	typedef typename boost::geometry::select_most_precise
		<
		typename boost::geometry::coordinate_type<Segment>::type,
		double
		>::type type;

	vector<Segment_2> v;

	// Initialize min and max
	gather_segment_to_vector() {}

	// This operator is called for each segment
	inline void operator()(Segment const& s)
	{
		Segment_2 s2(s.first, s.second);
		v.push_back(s2);
	}
};

#endif // GEOMETRY_COMMON_H

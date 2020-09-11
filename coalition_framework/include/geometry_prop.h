#ifndef GEOMETRY_PROP_H
#define GEOMETRY_PROP_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2014
 *
 ***************************************************************************
 */
#include <iostream>
#include <vector>


#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry.hpp>
#include <qglobal.h>

//#include "Multiout.h"

using namespace std;

#pragma region Extensions ToDouble(), ToInt()
struct ToDouble
{
    double operator()(std::string const & str)
    {
        return boost::lexical_cast<double>(str.c_str());
    }
};

struct ToInt
{
    int operator()(std::string const & str)
    {
        return boost::lexical_cast<int>(str.c_str());
    }
};

struct ToString
{
    std::string operator()(double const & dbl)
    {
        return boost::lexical_cast<std::string>(dbl);
    }
};

#pragma endregion

/*!
 * \defgroup FoncProp Fonctions geometriques proprietaires
 * \brief Ce module correspond a l'espace de nommage geometry_prop.
 * Il comporte l'ensemble des fonctions geometriques et classes d'objets
 * geometriques de base. Seules les fonctions d'intersection, union, et area
 * sont implementees via Boost.
 */
 /*! @{*/
namespace geometry_prop
{
#pragma region Forward declaration
    class Vector;
#pragma endregion

    //! \brief Classe representant un point 2D
    class Point
    {
        double _x;
        double _y;
    public:
        Point()
        {}
        Point(const double x, const double y) : _x(x), _y(y)
        {}
        //! Acces x
        double x() const
        {
            return _x;
        }
        //! Acces y
        double y() const
        {
            return _y;
        }

        //! Ajout d'un vecteur ou translation de vecteur v
        inline Point operator+(const Vector& v) const;//corps place après class Vector
        //! Soustraction vecteur ou translation de vecteur -v
        inline Point operator-(const Vector& v) const;//corps place après class Vector
        //! Affichage
        inline friend ostream& operator<<(ostream& o, Point& p)
        {
            o << p.x() << " " << p.y() << endl;
            return o;
        }
    };

    //! \brief Classe representant un vecteur 2D
    class Vector
    {
        double _x;
        double _y;
    public:
        Vector()
        {}
        Vector(const double x, const double y) : _x(x), _y(y)
        {}
        Vector(const Point& source, const Point& target) : _x(target.x() - source.x()), _y(target.y() - source.y())
        {}
        //! Acces x
        double x() const
        {
            return _x;
        }
        //! Acces y
        double y() const
        {
            return _y;
        }
        //! Produit scalaire
        double operator*(const Vector& v) const
        {
            return _x*v.x() + _y*v.y();
        }
        //! Produit par un scalaire
        Vector operator*(double d) const
        {
            return Vector(_x*d, _y*d);
        }
        //! Adition
        Vector operator+(const Vector& v) const
        {
            return Vector(_x + v.x(), _y + v.y());
        }
        //! Soustraction
        Vector operator-(const Vector& v) const
        {
            return Vector(_x - v.x(), _y - v.y());
        }
        //! Perpendiculaire sens trigo
        Vector perpendicular() const
        {
            return Vector(-_y, _x);
        }
        //! Norme au carre
        double squared_length() const
        {
            return _x *_x + _y *_y;
        }
        //! Norme
        double length() const
        {
            return sqrt(squared_length());
        }
    };

    /*!
     * \param v vecteur translation
     * \return point translate
     */
    inline Point Point::operator+(const Vector& v) const
    {
        return Point(_x + v.x(), _y + v.y());
    }

    /*!
     * \param v vecteur translation opposee
     * \return point translate
     */
    inline Point Point::operator-(const Vector& v) const
    {
        return Point(_x - v.x(), _y - v.y());
    }

    inline bool intersect(const Point& p0,
                          const Vector& u,
                          const Point& p1,
                          const Vector& v, Point& pi)
    {

        // Using Determinant Kramer technic
        double det = u.y()*v.x() - v.y()*u.x();

        if (det == 0)
            return false;

        //! HW 17/03/15 : modif
        double det_x = (u.y()*p0.x() - u.x()*p0.y())*v.x()
            - u.x()*(v.y()*p1.x() - v.x()*p1.y());

        double det_y = (u.y()*p0.x() - u.x()*p0.y())*v.y()
            - u.y()*(v.y()*p1.x() - v.x()*p1.y());

        // det is != 0
        pi = Point(det_x / det, det_y / det);

        return true;
    }

}//geometry_prop

namespace boost
{
    namespace geometry
    {
        namespace traits
        {
            // Adapt Point to Boost.Geometry

            template<> struct tag<geometry_prop::Point>
            {
                typedef point_tag type;
            };

            template<> struct coordinate_type<geometry_prop::Point>
            {
                typedef double type;
            };

            template<> struct coordinate_system<geometry_prop::Point>
            {
                typedef cs::cartesian type;
            };

            template<> struct dimension<geometry_prop::Point> : boost::mpl::int_<2>
            {};

            template<>
            struct access<geometry_prop::Point, 0>
            {
                static double get(geometry_prop::Point const& p)
                {
                    return p.x();
                }

                static void set(geometry_prop::Point& p, double const& value)
                {
                    p = geometry_prop::Point(value, p.y());
                }
            };

            template<>
            struct access<geometry_prop::Point, 1>
            {
                static double get(geometry_prop::Point const& p)
                {
                    return p.y();
                }

                static void set(geometry_prop::Point& p, double const& value)
                {
                    p = geometry_prop::Point(p.x(), value);
                }
            };
        }
    }
} // namespace boost::geometry::traits

namespace geometry_prop
{
#pragma region Forward declaration
    class Point;
    class Vector;
    class Segment;
    class Bbox;
    class Polyline;
    class Polygon;
    class Circle;
    class Rect;

    //! Foncteur des transformations (rotation, translation)
    struct Transformation;

    typedef Point Point_2;
    typedef Vector Vector_2;

    //typedef K::Segment_2 Segment_2;
    typedef Segment Segment_2;

    typedef Bbox Bbox_2;

    typedef Polyline Polyline_2;
    typedef Polygon Polygon_2;

    typedef Circle Circle_2;
    typedef Rect Rect_2;
    typedef Transformation Transformation_2;

    // Declare un polyline boost
    typedef boost::geometry::model::linestring<Point> Polyline_B;
    // Declare un polygone boost
    typedef boost::geometry::model::ring<Point> Polygon_B;
#pragma endregion

#pragma region Segment, BBox, Polygon
    /** Segment défini entre deux points : une source et une target.
     * Le segment est orienté de la source vers la target.
     */
    class Segment
    {
        Point _source;
        Point _target;
    public:
        Segment()
        {}
        Segment(const Point& source, const Point& target) : _source(source), _target(target)
        {}
        Point source() const
        {
            return _source;
        }
        Point target() const
        {
            return _target;
        }
        Vector to_vector() const
        {
            return Vector(_target.x() - _source.x(), _target.y() - _source.y());
        }
        //! Longueur (target - source)
        double length() const
        {
            return sqrt(squared_length());
        }
        //! Longueur au carré (target - source)
        double squared_length() const
        {
            return (_target.x() - _source.x())*(_target.x() - _source.x())
                + (_target.y() - _source.y())*(_target.y() - _source.y());
        }
        //! Inverse point target/source
        Segment opposite()
        {
            return Segment(_target, _source);
        }
    };
    /** Bounding-box (boite englobante)
     */
    class Bbox
    {
        Point_2 _min;
        Point_2 _max;
        double _width;
        double _height;
    public:
        Bbox() : _min(Point_2(0, 0)), _max(Point_2(0, 0)), _width(0), _height(0)
        {}
        Bbox(Point_2 min, double width, double height)
            : _min(min),
            _max(Point_2(min.x() + width, min.y() + height)),
            _width(width),
            _height(height)
        {}

        Bbox(Point_2 min, Point_2 max)
            : _min(min),
            _max(max),
            _width(max.x() - min.x()),
            _height(max.y() - min.y())
        {}

        Bbox(double xmin,
             double ymin,
             double xmax,
             double ymax)
            : _min(Point_2(xmin, ymin)),
            _max(Point_2(xmax, ymax)),
            _width(xmax - xmin),
            _height(ymax - ymin)
        {}

        double xmin()
        {
            return _min.x();
        }
        double ymin()
        {
            return _min.y();
        }
        double xmax()
        {
            return _max.x();
        }
        double ymax()
        {
            return _max.y();
        }
        double width()
        {
            return _width;
        }
        double height()
        {
            return _height;
        }
    };

    class Polyline : public Polyline_B
    {
    public:
        Polyline() : Polyline_B()
        {}
        Polyline(const_iterator begin, const_iterator end) : Polyline_B(begin, end)
        {}
        typedef vector<Point_2>::iterator Vertex_iterator;
        typedef vector<Point_2>::const_iterator Vertex_const_iterator;


        //! Calcule les intersections entre deux polylines
        void intersection(const Polyline &poly2, vector<Point> &output) const
        {
            boost::geometry::intersection((const Polyline_B&)*this, (const Polyline_B&)poly2, (vector<Point>&)output);
        }
        //! Calcule les intersections entre deux polylines
        static void intersection(const Polyline &poly1, const Polyline &poly2, vector<Point> &output)
        {
            //output.clear();
            boost::geometry::intersection((const Polyline_B&)poly1, (const Polyline_B&)poly2, (vector<Point>&)output);
        }

        //! Calcule les intersections internes à un polyline
        bool selfintersects() const
        {
            return boost::geometry::intersects<Polyline_B>((const Polyline_B&)*this);
        }
        //! Calcule les intersections internes à un polyline
        static bool selfintersects(const Polyline &poly)
        {
            return poly.selfintersects();
        }

    };

    class Polygon : public Polygon_B
    {
    public:
        Polygon() : Polygon_B()
        {}
        Polygon(const_iterator begin, const_iterator end) : Polygon_B(begin, end)
        {}
        typedef vector<Point_2>::iterator Vertex_iterator;
        typedef vector<Point_2>::const_iterator Vertex_const_iterator;
        Vertex_iterator vertices_begin()
        {
            return begin();
        }
        Vertex_iterator vertices_end()
        {
            return end();
        }

        

        //! Get bounding box
        Bbox_2 bbox() const
        {
            boost::geometry::model::box<Point_2> bb;
            // Ferme le polygone par ajout du premier point
            ((vector<Point_2>&) *this).push_back(front());
            // Calcule de l'enveloppe
            boost::geometry::envelope((Polygon_B&)*this, bb);
            // Annule la fermeture du polygone
            ((vector<Point_2>&) *this).pop_back();
            // Definie la bbox
            return Bbox_2(bb.min_corner().x(), bb.min_corner().y(), bb.max_corner().x(), bb.max_corner().y());
        }
        //! Point en dehors du polygone ?
        bool has_on_positive_side(const Point_2& p) const
        {
            // Contraire à être à l'intérieur
            return !has_on_negative_side(p);
        }
        //! Point dans le polygone ?
        bool has_on_negative_side(const Point_2& p) const
        {
            // Ferme le polygone par ajout du premier point
            ((vector<Point_2>&) *this).push_back(front());
            // Verifie que le point est couvert par le polygone
            bool r = boost::geometry::covered_by(p, (Polygon_B&)*this);
            // Annule la fermeture du polygone
            ((vector<Point_2>&) *this).pop_back();
            return r;
        }
        //! Polygone fermé ? (dernier point == premier point)
        /**
         * \return false si dernier==premier (fermé), sinon true
         */
        bool is_simple() const
        {
            vector<Point_2>& p = (vector<Point_2>&) *this;
            bool is_simple = true;
            if (p.size() > 1)
            {
                Point_2 p1 = p[0];
                Point_2 p2 = p[p.size() - 1];
                if (p1.x() == p2.x() && p1.y() == p2.y())
                {
                    is_simple = false;
                }
            }
            return is_simple;
        }
        //! Calcule l'orientation du polygone
        /**
         * \return true si orienté clockwise
         */
        bool is_clockwise_oriented() const
        {
            // http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
            // Sum over the edges, (x2 − x1)(y2 + y1). If the result is positive
            // the curve is clockwise, if it's negative the curve is counter-clockwise.
            // (The result is twice the enclosed area, with a +/- convention.)
            // The area under a line segment equals its average height(y2+y1)/2
            // times its horizontal length(x2-x1). Notice the sign convention in x
            bool is_clockwise = true;
            double total = 0;
            vector<Point_2>& p = (vector<Point_2>&) *this;
            for (uint i = 0; i < p.size(); i++)
            {
                Point_2 p1 = p[i];
                uint j = 0;
                if (i + 1 < p.size())
                    j = i + 1;
                Point_2 p2 = p[j];
                total += (p2.x() - p1.x())*(p2.y() + p1.y());
            }
            if (total < 0)
                is_clockwise = false;
            return is_clockwise;
        }
        vector<Point_2>& container() const
        {
            return (vector<Point_2>&) *this;
        }
        //! Cherche le vertex le plus haut ou le plus bas (comparaison suivant y)
        Vertex_const_iterator topbottom_vertex(bool isTop) const
        {
            typedef Vertex_iterator Vi;
            // Prends le premier point comme référence
            Vi result = container().begin();
            double m = ((Point_2&)*(result)).y();

            for (Vi i = container().begin(); i != container().end(); ++i)
            {
                if ((isTop && ((*i).y() >= m)) ||	// Cherche le plus haut?
                    (!isTop && ((*i).y() <= m)))	// Cherche le plus bas?
                {
                    result = i;
                    m = (*i).y();
                }
            }
            return result;
        }
        //! Cherche le vertex le plus haut
        Vertex_const_iterator top_vertex() const
        {
            return topbottom_vertex(true);
        }
        //! Cherche le vertex le plus bas
        Vertex_const_iterator bottom_vertex() const
        {
            topbottom_vertex(false);
        }
        //! Cherche le vertex le plus à gauche ou à droite (comparaison suivant x)
        Vertex_const_iterator leftright_vertex(bool isLeft) const
        {
            typedef Vertex_iterator Vi;
            // Prends le premier point comme référence
            Vi result = container().begin();
            double m = ((Point_2&)*(result)).x();

            for (Vi i = container().begin(); i != container().end(); ++i)
            {
                if ((isLeft && ((*i).x() <= m)) ||
                    (!isLeft && ((*i).x() >= m)))
                {
                    result = i;
                    m = (*i).x();
                }
            }
            return result;
        }
        Vertex_const_iterator left_vertex() const
        {
            return leftright_vertex(true);
        }

        Vertex_const_iterator right_vertex() const
        {
            return leftright_vertex(false);
        }

        void reverse_orientation()
        {
            if (this->size() > 1)
            {
                Polygon_2::Vertex_iterator it = this->vertices_begin();
                std::reverse(++it, this->vertices_end());
            }
        }
        

        string toStdString() const
        {
            string txt = "";
            for (uint i=0; i<this->size(); i++)
            {
                txt.append(boost::lexical_cast<std::string>((*this)[i].x()));
                txt.append(", ");
                txt.append(boost::lexical_cast<std::string>((*this)[i].y()));
                txt.append("; ");
            }
            return txt;
        }



#pragma region Adapteurs Boost
        // Barycentre polygone
        Point_2 barycentre() const
        {
            Point_2 c;
            Polygon_B p_b(this->container().begin(), this->container().end());
            p_b.push_back(this->container().front());
            boost::geometry::centroid(p_b, c);
            return c;
        }

        //! Calcule l'aire du polygone
        double area() const
        {
            // Calcule de l'aire
            return boost::geometry::area((const Polygon_B&)*this);
        }
        //! Calcule l'aire du polygone
        static double area(const Polygon &poly1)
        {
            // Calcule de l'aire
            return boost::geometry::area((const Polygon_B&)poly1);
        }
        //! Calcule les intersections entre deux polygones
        void intersection(const Polygon &poly2, vector<Polygon> &output) const
        {
            boost::geometry::intersection((const Polygon_B&)*this, (const Polygon_B&)poly2, (vector<Polygon_B>&)output);
        }
        //! Calcule les intersections entre deux polygones
        static void intersection(const Polygon &poly1, const Polygon &poly2, vector<Polygon> &output)
        {
            //output.clear();
            boost::geometry::intersection((const Polygon_B&)poly1, (const Polygon_B&)poly2, (vector<Polygon_B>&)output);
        }

        //! Calcule les intersections internes à un polygone
        bool selfintersects() const
        {
            return boost::geometry::intersects<Polygon_B>((const Polygon_B&)*this);
        }
        //! Calcule les intersections internes à un polygone
        static bool selfintersects(const Polygon &poly)
        {
            return poly.selfintersects();
        }
        // Difference de polygones
        void differences(const Polygon &poly2, vector<Polygon> &output) const
        {
            boost::geometry::difference((const Polygon_B&)*this, (const Polygon_B&)poly2, (vector<Polygon_B>&)output);
        }
        // Difference de polygones
        static void differences(const Polygon &poly1, const Polygon &poly2, vector<Polygon> &output)
        {
            boost::geometry::difference((const Polygon_B&)poly1, (const Polygon_B&)poly2, (vector<Polygon_B>&)output);
        }
        // Union de polygones
        void unions(const Polygon &poly2, vector<Polygon> &output) const
        {
            boost::geometry::union_((const Polygon_B&)*this, (const Polygon_B&)poly2, (vector<Polygon_B>&)output);
        }
        // Union de polygones
        static void unions(const Polygon &poly1, const Polygon &poly2, vector<Polygon> &output)
        {
            boost::geometry::union_((const Polygon_B&)poly1, (const Polygon_B&)poly2, (vector<Polygon_B>&)output);
        }

#pragma endregion

    };
#pragma endregion

#pragma region Fonctions friend polygones
    /*! @name Fonctions de base
     * \brief Fonctions de geometrie 2D de base.
     */
     //! @{

     //! Indique si le polygone est simple = non fermé (premier et dernier points sont différents)
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
    //! Construction d'une chaine contenant tous les points du polygone séparés par des ";"
    inline string toStdString(const Polygon_2& p)
    {
        return p.toStdString();
    }

    /*!
     * \brief Renvoie le barycentre d'un polygone
     * \param p polygone d'entree
     * \return Point barycentre
     */
    inline Point_2 barycentre(const Polygon_2& p)
    {
        return p.barycentre();
    }

    class Circle
    {
        Point_2 _center;
        double _squared_radius;
        double _radius;
    public:
        Circle()
        {}
        Circle(Point_2 center, double squared_radius) : _center(center), _squared_radius(squared_radius)
        {
            _radius = sqrt(_squared_radius);
        }
        Point_2 center() const
        {
            return _center;
        }
        double squared_radius() const
        {
            return _squared_radius;
        }
        double radius() const
        {
            return _radius;
        }
        Bbox_2 bbox() const
        {
            return Bbox_2(
                _center.x() - _radius, _center.y() - _radius,
                _center.x() + _radius, _center.y() + _radius
                );
        }
    };

    class Rect
    {
        Point_2 _min;

        double _width;
        double _height;
    public:
        Rect() : _min(Point_2(0, 0)), _width(0), _height(0)
        {}
        Rect(Point_2 min, double width, double height) : _min(min), _width(width), _height(height)
        {}

        inline double xmin()
        {
            return _min.x();
        }
        inline double ymin()
        {
            return _min.y();
        }
        inline double width() const
        {
            return _width;
        }
        inline double height() const
        {
            return _height;
        }
        friend inline const Point_2& min BOOST_PREVENT_MACRO_SUBSTITUTION(const Rect& r);
    };

    inline const Point_2& min BOOST_PREVENT_MACRO_SUBSTITUTION(const Rect& r)
    {
        return r._min;
    }

    template <class Polygon_2, class Rect_2>
    inline bool contains(Polygon_2& p, const Rect_2& r)
    {
        bool ret = false;
        Point_2 p_min = min(r);
        if (p.has_on_negative_side(p_min)
            && p.has_on_negative_side(Point_2(p_min.x() + r.width(), p_min.y()))
            && p.has_on_negative_side(Point_2(p_min.x() + r.width(), p_min.y() + r.height()))
            && p.has_on_negative_side(Point_2(p_min.x(), p_min.y() + r.height()))
            )
            ret = true;

        return ret;
    }

    /*!
     * \brief Calcul l'angle oriente (sens trigo) par rapport au vecteur direction
     * \ingroup geometrie proprietaire
     * \param s segment
     * \param dir direction pour calcul d'angle
     * \return l'angle orienté
     */
    template <class Segment_2, class Vector_2>
    inline double compute_angle(const Segment_2& s, const Vector_2& dir)
    {
        double angle;
        // Longeur segment
        double l1 = s.squared_length();
        // Cas dégénéré, longueur nulle => angle nul
        if (l1<FLT_EPSILON)
        {
            return 0.0;
        }
        Vector_2 v = s.to_vector();
        l1 = sqrt(l1);
        // Produit scalaire avec vecteur unitaire horizontal
        double sp = v * dir;
        double l2 = dir.squared_length();
        l2 = sqrt(l2);
        double cosalpha = sp / l1 / l2;
        // Angle
        angle = acos(cosalpha) * 180.0 / M_PI;//en degres
        // Determiner l'orientation de l'angle positif|negatif
        const Vector_2 v_dir(-dir.y(), dir.x());
        double sp_2 = v * v_dir;
        if (sp_2 < 0)
            angle = -angle;

        return angle;
    }

    inline bool less_x(const Point_2 &p, const Point_2 &q)
    {
        return p.x() < q.x();
    }

    inline bool less_y(const Point_2 &p, const Point_2 &q)
    {
        return p.y() < q.y();
    }

    //! Construit un vecteur de segments
    void polylineToVecteurSegment2(const Polyline_2 poly, vector<Segment_2>& vec);

    //! @}
#pragma endregion

#pragma region Transformations du plan
/*! @name Transformations
 * \brief Rotations et translations.
 */
 //! @{
    enum Trans
    {
        //! Translation seule
        TRF_TRANSLATION,
        //! Rotation d'un angle autour d'un centre de rotation quelconque
        TRF_ROTATION,
        //! Rotation autour de (0,0) puis translation => translate( rotate( obj ) )
        TRF_ROTATION_TRANSLATION,
        //! Translation puis rotation autour de (0,0) => rotate( translate( obj ) )
        TRF_TRANSLATION_ROTATION
    };

    struct Transformation
    {
        Trans _tag;
        Vector_2 _v;
        double _sin;
        double _cos;

        Transformation()
        {}

        Transformation(const Trans tag,
                       const Vector_2 v,
                       const double sin,
                       const double cos
                       ) : _tag(tag), _v(v), _sin(sin), _cos(cos)
        {}

        // This operator is called for each segment
        Point_2 operator()(const Point_2& p) const
        {
            Point_2 r;
            switch (_tag)
            {
                case TRF_ROTATION_TRANSLATION:
                    r = translate(rotate(p));
                    break;
                case TRF_TRANSLATION_ROTATION:
                    r = rotate(translate(p));
                    break;
                case TRF_ROTATION:
                    r = rotate_2(p);
                    break;
                case TRF_TRANSLATION:
                    r = translate(p);
                    break;
                default:
                    r = p;
                    break;
            }
            return r;
        }
        //! Translation suivant p
        inline Point_2 translate(const Point_2& p) const
        {
            return Point_2(
                p.x() + _v.x(),
                p.y() + _v.y()
                );
        }
        //! Rotation du point p autour de (0,0), selon l'angle interne sin/cos
        inline Point_2 rotate(const Point_2& p) const
        {
            return Point_2(
                p.x() * _cos - p.y() * _sin,
                p.x() * _sin + p.y() * _cos
                );
        }
        //! Rotation du point p autour du point v, selon l'angle interne sin/cos
        inline Point_2 rotate_2(const Point_2& p) const
        {
            return Point_2(
                (p.x() - _v.x()) * _cos - (p.y() - _v.y()) * _sin + _v.x(),
                (p.x() - _v.x()) * _sin + (p.y() - _v.y()) * _cos + _v.y()
                );
        }
        inline bool isIdentity()
        {
            switch (_tag)
            {
                case TRF_ROTATION_TRANSLATION:
                case TRF_TRANSLATION_ROTATION:
                    if (_v.squared_length() == 0.0)
                        return true;
                    if (_cos == 1.0 && _sin == 0.0)
                        return true;
                    break;
                case TRF_ROTATION:
                    if (_cos == 1.0 && _sin == 0.0)
                        return true;
                    break;
                case TRF_TRANSLATION:
                    if (_v.squared_length() == 0.0)
                        return true;
                    break;
                default:
                    break;
            }
            return false;
        }
    };

    //! Retourne la transformation équivalente
    inline Transformation getTransformation(Trans tag, Vector_2 v, double sin, double cos)
    {
        return Transformation(tag, v, sin, cos);
    }

    /*!
     * \brief Transformation en rotation et translation
     * \ingroup transformations
     * \param transf foncteur transformation
     * \param geometry objet a transforme
     * \return objet transforme
     */
    inline Polygon_2 transform(const Transformation& transf, const Polygon_2& p)
    {
        typedef vector<Point_2>::const_iterator Pi;
        Polygon_2 result;
        for (Pi i = p.container().begin(); i != p.container().end(); ++i)
        {
            Point_2 p = transf(*i);
            result.push_back(p);
        }
        return result;
    }

    /*!
     * \brief Transformation en rotation et translation
     * \ingroup transformations
     */
    inline Segment_2 transform(const Transformation& transf, const Segment_2& s)
    {
        return Segment_2(transf(s.source()), transf(s.target()));
    }

    /*!
     * \brief Transformation en rotation et translation
     * \ingroup transformations
     */
    inline Point_2 transform(const Transformation& transf, const Point_2& p)
    {
        return transf(p);
    }

    //! Transforme polyline boost
    inline Polyline_2 transform(const Transformation& transf, const Polyline_2& p)
    {
        Polyline_2 r;

        Polyline_2::const_iterator first1 = p.begin();
        Polyline_2::const_iterator last1 = p.end();
        while (first1 != last1)
        {
            r.push_back(transf(*first1));
            ++first1;
        }
        return r;
    }

    /*!
     * \brief Retournement vertical / position en x ("flip" autour de la coordonnée X)
     * \ingroup transformations
     * \param centre abcisse verticale de symetrie
     * \param geometry objet a transforme
     * \return objet transforme
     */
     //! Flip le point par rapport à l'axe X donné
    inline Point_2 retourner(double centre, const Point_2& p)
    {
        return Point_2(2 * centre - p.x(), p.y());
    }
    //! Flip le polygone par rapport à l'axe X donné
    inline Polygon_2 retourner(double centre, const Polygon_2& p)
    {
        typedef vector<Point_2>::const_iterator Pi;
        Polygon_2 result;
        for (Pi i = p.container().begin(); i != p.container().end(); ++i)
        {
            result.push_back(retourner(centre, *i));
        }
        bool IsClockwise = is_clockwise_oriented(result);
        if (!IsClockwise)
        {
            reverse_orientation(result);
        }
        return result;
    }
    //! Flip le segment par rapport à l'axe X donné
    inline Segment_2 retourner(double centre, const Segment_2& s)
    {
        return Segment_2(retourner(centre, s.source()), retourner(centre, s.target()));
    }

    //! Flip le polyline Boost par rapport à l'axe X donné
    inline Polyline_2 retourner(double centre, const Polyline_2& p)
    {
        Polyline_2 r;

        Polyline_2::const_iterator first1 = p.begin();
        Polyline_2::const_iterator last1 = p.end();
        while (first1 != last1)
        {
            r.push_back(retourner(centre, *first1));
            ++first1;
        }
        return r;
    }

    //! Inverse la liste de point du polyline
    //inline Polyline_B reverse(const Polyline_B& p) {
    //{
    //    Polyline_B rr;
    //    if (p.size() > 1) {
    //        Polyline_B::const_iterator first1 = p.begin();
    //        Polyline_B::const_iterator last1 = p.end();
    //        while (last1 != first1) {
    //            rr.push_back(*last1);
    //            --last1;
    //        }
    //    }
    //    return rr;
    //}

    //! @}
#pragma endregion

#pragma region Intersection de cercles, droites
    /*! @name Intersection de cercles
     * \brief Fonctions d'intersection 2D de base.
     */
     //! @{

     /** <summary>Calcule l'intersection entre un grand cercle autour duquel tourne un petit
      * cercle, et un segment (cas de recherche de contact entre roue et chemin
      * quand on pivote autour de l'axe de l'autre roue).
      * S'il y a deux solutions de contact, le 'sens' permet de choisir laquelle
      * retenir en fonction du côté de la roue (gauche ou droit). Si sens==0, on
      * retient la solution ayant la plus grande coordonnée X (à droite).
      *</summary>
      * <param name="sens">=1 solution la plus à gauche, = 0 solution la plus à droite</param>
      * <param name='X0'>[in] position X du centre roue</param>
      * <param name='Y0'>[in] position Y du centre roue</param>
      * <param name='R0_carre'>[in] rayon^2 grand cercle entraxe</param>
      * <param name='X1'>[in] position X autre roue</param>
      * <param name='Y1'>[in] position Y autre roue</param>
      * <param name='R1_carre'>[in] rayon^2 petite roue</param>
      * <param name='X'>[out] position X du contact</param>
      * <param name='Y'>[out] position Y du contact</param>
      * <return>true si contact trouvé, false sinon</return>
      */
    bool intersectCercleCercle(const int sens,
                               const double X0,
                               const double Y0,
                               const double R0_carre,
                               const double X1,
                               const double Y1,
                               const double R1_carre,
                               double& X,
                               double& Y
                               );

    bool intersectCercleCercle(const Vector_2 vSensDirection,
                               const double X0,
                               const double Y0,
                               const double R0_carre,
                               const double X1,
                               const double Y1,
                               const double R1_carre,
                               double& X,
                               double& Y
                               );

     /** <summary>Coupe un cercle par une droite supportée par un segment et
     * renvoie le segment représenté par le premier point d'intersection et le
     * dernier point.
     * Si le cercle est tangent, renvoie un segment constitué d'un point.</summary>
     * <param name="cRot">[in] centre du cercle</param>
     * <param name="d_carre">[in] rayon^2 du cercle</param>
     * <param name='seg'>[in] segment support de la droite</param>
     * <param name='seg_cut'>[out] segment composé des 2 points d'intersection</param>
     * <return>true si intersection, false si aucune</return>
     */
    bool intersectCercleDroite(const Point_2 cRot,
                               const double d_carre,
                               const Segment_2 seg,
                               Segment_2& seg_cut);

    /** <summary>Calcule l'intersection entre un cercle et une droite représentée par
     * y=a*x+b et renvoie le point d'intersection.
     * S'il y a deux solutions de contact, le 'sens' permet de choisir laquelle
     * retenir en fonction du côté de la roue (gauche ou droit). Si sens==0, on
     * retient la solution ayant la plus grande coordonnée X.</summary>
     * Si le cercle est tangent, renvoie un segment constitué d'un point.</summary>
     *
     * <param name="sens">[in] choix du point =0 pour Xmax, =1 pour Xmin</param>
     * <param name='X0'>[in] centre du cercle</param>
     * <param name='Y0'>[in] centre du cercle</param>
     * <param name='d_carre'>[in] rayon^2 cercle</param>
     * <param name='a'>[in] pente de la droite</param>
     * <param name='b'>[in] ordonnées à l'origine de la droite</param>
     * <param name='X'>[out] position X du contact</param>
     * <param name='Y'>[out] position Y du contact</param>
     * <return>true si intersection, false si aucune</return>
     */
    bool intersectCercleDroite(const int sens,
                               const double X0,
                               const double Y0,
                               const double d_carre,
                               const double a,
                               const double b,
                               double& XC,
                               double& YC
                               );

    /** <summary>Calcule l'intersection entre une verticale X=cste et un cercle
     * C(X1,Y1,R1) et renvoie le point d'intersection le plus haut suivant l'axe
     * Y.</summary>
     * <param name="X0">[in] parametre de la droite verticale</param>
     * <param name='X1'>[in] centre du cercle</param>
     * <param name='Y1'>[in] centre du cercle</param>
     * <param name='R1_carre'>[in] rayon^2 cercle</param>
     * <param name='X'>[out] position X=X0 du contact haut</param>
     * <param name='Y'>[out] position Ymax du contact haut</param>
     * <return>true si intersection, false si aucune</return>
     */
    bool intersectVerticaleCercle(const double X0,
                                  const double X1,
                                  const double Y1,
                                  const double R1_carre,
                                  double& X,
                                  double& Y
                                  );

    /** Calcule l'intersection entre une verticale et une droite représentée par
     * y=a*x+b et renvoie le point d'intersection.
     * <param name='X0'>[in] point de la verticale </param>
     * <param name='Y0'>[in] non utilisé</param>
     * <param name='a'>[in] pente de la droite</param>
     * <param name='b'>[in] ordonnées à l'origine de la droite</param>
     * <param name='XC'>[out] position X du contact</param>
     * <param name='YC'>[out] position Y du contact</param>
     * <return>true si intersection, false si aucune</return>
     */
    bool intersectVerticaleDroite(const double X0,
                                  const double Y0,
                                  const double a,
                                  const double b,
                                  double& XC,
                                  double& YC
                                  );
    //! @}
#pragma endregion
#if 0
    /*!
     * Foncteur : transforme un polylyne en vecteur de segments
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

        gather_segment_to_vector()
        {}

        // This operator is called for each segment
        inline void operator()(Segment const& s)
        {
            Segment_2 s2(s.first, s.second);
            v.push_back(s2);
        }
    };

#pragma region Intersections, union, diff de polygones

    /*! @name Intersections de polygones
    * \brief Fonctions d'intersection de polygones 2D de base.
    */
    //! @{
    /*!
    \brief \brief_calc{area} \brief_strategy
    \ingroup area
    \details \details_calc{area} \brief_strategy. \details_strategy_reasons
    \tparam Geometry \tparam_geometry
    \param geometry \param_geometry
    \return \return_calc{area}
    */
    template <typename Geometry>
    inline double area(Geometry const& geometry)
    {
        return boost::geometry::area(geometry);
    }

    /*!
    \brief \brief_calc2{intersection}
    \ingroup intersection
    \details \details_calc2{intersection, spatial set theoretic intersection}.
    \tparam Geometry1 \tparam_geometry
    \tparam Geometry2 \tparam_geometry
    \tparam GeometryOut Collection of geometries (e.g. std::vector, std::deque, boost::geometry::multi*) of which
    the value_type fulfills a \p_l_or_c concept, or it is the output geometry (e.g. for a box)
    \param geometry1 \param_geometry
    \param geometry2 \param_geometry
    \param geometry_out The output geometry, either a multi_point, multi_polygon,
    multi_linestring, or a box (for intersection of two boxes)

    */
    template
        <
        typename Geometry1,
        typename Geometry2,
        typename GeometryOut
        >
        inline bool intersection(Geometry1 const& geometry1,
                                 Geometry2 const& geometry2,
                                 GeometryOut& geometry_out)
    {
        return boost::geometry::intersection(geometry1, geometry2, geometry_out);
    }

    /*!
    \brief_calc2{difference}
    \ingroup difference
    \details \details_calc2{difference, spatial set theoretic difference}.
    \tparam Geometry1 \tparam_geometry
    \tparam Geometry2 \tparam_geometry
    \tparam Collection \tparam_output_collection
    \param geometry1 \param_geometry
    \param geometry2 \param_geometry
    \param output_collection the output collection

    */
    template
        <
        typename Geometry1,
        typename Geometry2,
        typename Collection
        >
        inline void difference(Geometry1 const& geometry1,
                               Geometry2 const& geometry2, Collection& output_collection)
    {
        return boost::geometry::difference(geometry1, geometry2, output_collection);
    }

    /*!
    \brief Combines two geometries which each other
    \ingroup union
    \details \details_calc2{union, spatial set theoretic union}.
    \tparam Geometry1 \tparam_geometry
    \tparam Geometry2 \tparam_geometry
    \tparam Collection output collection, either a multi-geometry,
    or a std::vector<Geometry> / std::deque<Geometry> etc
    \param geometry1 \param_geometry
    \param geometry2 \param_geometry
    \param output_collection the output collection
    \note Called union_ because union is a reserved word.
    */
    template
        <
        typename Geometry1,
        typename Geometry2,
        typename Collection
        >
        inline void union_(Geometry1 const& geometry1,
                           Geometry2 const& geometry2,
                           Collection& output_collection)
    {
        boost::geometry::union_(geometry1, geometry2, output_collection);
    }
    //! @}
#pragma endregion
#endif

#pragma region Conversion, verification de polygone & polyline
    
    /**
     * Fermer un polygon en ajoutant le premier point à la fin de la liste
     */
    Polygon_2 getClosedPolygon(const Polygon_2 & polygon);

    /**
     * Parser une liste de valeur dans une liste de points
     */
    void extractVector(std::string& std_str, std::vector<Point_2> &pts);
    /**
     * Parser une liste de valeur dans un polygone
     */
    void extractContour(std::string& std_str, Polygon_2& p);
    /**
     * Parser une liste de valeur dans un polyline
     */
    void extractChemin(std::string& std_str, Polyline_2& p);

    /**
     * Normaliser un polygon : orientation avant-arriere
     * \param p polygon
     */
    bool verifyAndUpdatePolygon(Polygon_2& pol);

    /**
     * Normaliser un polyline : orientation avant-arriere
     * \param p polyline
     */
    bool verifyAndUpdatePolyline(Polyline_2& pol);

    /**
     * Normaliser un polyline : orientation avant-arriere
     * \param p polyline
     */
    void normaliseCroissantPolyline(Polyline_2& pol);
#pragma endregion


}//namespace geometry_prop
//! @}

#endif // GEOMETRY_PROP_H

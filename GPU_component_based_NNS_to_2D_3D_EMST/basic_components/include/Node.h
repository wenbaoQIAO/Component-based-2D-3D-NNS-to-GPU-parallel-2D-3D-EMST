#ifndef NODE_H
#define NODE_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang, A. Mansouri
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "lib_global.h"
#include "macros_cuda.h"

#define TEST_CODE 0

using namespace std;

typedef short GLshort;
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;

typedef GLfloat GLfloatP;

typedef GLint GrayValue;
typedef GLdouble IntensityValue;

#include <cstddef>

namespace components
{

/*!
 * \defgroup Node
 * \brief Espace de nommage components
 * Il comporte les nodes (point, neurone)
 */
/*! @{*/

/*!
 * \brief Basic point class, having coordinates defined in a neutral way (idem Boost)
 */
template<typename TypeCoordinate, std::size_t Dimension>
class Point
{

protected:
public:// wb.Q for easy output network link results
    TypeCoordinate _value[Dimension];
public:
    //! @brief Default constructor
    DEVICE_HOST inline Point() {}

    DEVICE_HOST inline Point(Point const& p2) {
        if (Dimension >= 1)
            _value[0] = p2._value[0];
        if (Dimension >= 2)
            _value[1] = p2._value[1];
        if (Dimension >= 3)
            _value[2] = p2._value[2];
    }

    //! @brief Affectation
    DEVICE_HOST Point& operator=(Point const& p2) {
        if (Dimension >= 1)
            _value[0] = p2._value[0];
        if (Dimension >= 2)
            _value[1] = p2._value[1];
        if (Dimension >= 3)
            _value[2] = p2._value[2];
        return *this;
    }

    //! @brief Constructor
    DEVICE_HOST explicit inline Point(TypeCoordinate const& v0, TypeCoordinate const& v1, TypeCoordinate const& v2 = 0) {
        if (Dimension >= 1)
            _value[0] = v0;
        if (Dimension >= 2)
            _value[1] = v1;
        if (Dimension >= 3)
            _value[2] = v2;
    }

    //! @brief Constructor
    DEVICE_HOST explicit inline Point(TypeCoordinate const& v0) {
        if (Dimension >= 1)
            _value[0] = v0;
        if (Dimension >= 2)
            _value[1] = v0;
        if (Dimension >= 3)
            _value[2] = v0;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate& operator[](std::size_t i) {
        return _value[i];
    }

    //! @brief Get data adress
    template <std::size_t K>
    DEVICE_HOST inline TypeCoordinate* getData() const {
        return _value;
    }

    //! @brief Get coordinate
    template <std::size_t K>
    DEVICE_HOST inline TypeCoordinate const& get() const {
        return _value[K];
    }

    //! @brief Set coordinate
    template <std::size_t K>
    DEVICE_HOST inline void set(TypeCoordinate const& value) {
        _value[K] = value;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate const& get(std::size_t const i) const {
        return _value[i];
    }

    //! @brief Set coordinatev for loop only
    DEVICE_HOST inline void set(std::size_t const i, TypeCoordinate const& value) {
        _value[i] = value;
    }

    DEVICE_HOST inline Point& operator+=(Point& p2) {
        if (Dimension >= 1)
            _value[0] += p2._value[0];
        if (Dimension >= 2)
            _value[1] += p2._value[1];
        if (Dimension >= 3)
            _value[2] += p2._value[2];
        return *this;
    }
    DEVICE_HOST inline Point& operator-=(Point& p2) {
        if (Dimension >= 1)
            _value[0] -= p2._value[0];
        if (Dimension >= 2)
            _value[1] -= p2._value[1];
        if (Dimension >= 3)
            _value[2] -= p2._value[2];
        return *this;
    }

    DEVICE_HOST inline Point& operator*=(Point& p2) {
        if (Dimension >= 1)
            _value[0] *= p2._value[0];
        if (Dimension >= 2)
            _value[1] *= p2._value[1];
        if (Dimension >= 3)
            _value[2] *= p2._value[2];
        return *this;
    }

    DEVICE_HOST inline Point& operator*=(TypeCoordinate p2) {
        if (Dimension >= 1)
            _value[0] *= p2;
        if (Dimension >= 2)
            _value[1] *= p2;
        if (Dimension >= 3)
            _value[2] *= p2;
        return *this;
    }

    DEVICE_HOST inline Point& operator/=(TypeCoordinate p2) {
        if (Dimension >= 1)
            _value[0] /= p2;
        if (Dimension >= 2)
            _value[1] /= p2;
        if (Dimension >= 3)
            _value[2] /= p2;
        return *this;
    }

    DEVICE_HOST inline Point& operator/=(size_t p2) {
        if (Dimension >= 1)
            _value[0] /= p2;
        if (Dimension >= 2)
            _value[1] /= p2;
        if (Dimension >= 3)
            _value[2] /= p2;
        return *this;
    }

    //! @brief Affectation, wb.Q add ==
    DEVICE_HOST inline bool operator ==(Point const& p2) {
        bool ret = true;
        if (Dimension >= 1 && _value[0] != p2._value[0])
            ret = false;
        if (Dimension >= 2 && _value[1] != p2._value[1])
            ret = false;
        if (Dimension >= 3 && _value[2] != p2._value[2])
            ret = false;
        return ret;
    }

    //! @brief Affectation, wb.Q add !=
    DEVICE_HOST inline bool operator!=(Point const& p2) {
        return !(*this == p2);
    }

    DEVICE_HOST inline friend Point operator+(const Point& p1, const Point& p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] + p2._value[0];
        if (Dimension >= 2)
            p._value[1] = p1._value[1] + p2._value[1];
        if (Dimension >= 3)
            p._value[2] = p1._value[2] + p2._value[2];
        return p;
    }

    DEVICE_HOST inline friend Point operator+(const Point& p1, const TypeCoordinate p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] + p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] + p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] + p2;
        return p;
    }

    DEVICE_HOST inline friend Point operator-(const Point& p1, const Point& p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] - p2._value[0];
        if (Dimension >= 2)
            p._value[1] = p1._value[1] - p2._value[1];
        if (Dimension >= 3)
            p._value[2] = p1._value[2] - p2._value[2];
        return p;
    }

    //! Scalar product
    DEVICE_HOST inline friend TypeCoordinate operator*(const Point& p1, const Point& p2) {
        TypeCoordinate f = 0.0;
        if (Dimension >= 1)
            f += p1._value[0] * p2._value[0];
        if (Dimension >= 2)
            f += p1._value[1] * p2._value[1];
        if (Dimension >= 3)
            f += p1._value[2] * p2._value[2];
        return f;
    }

    DEVICE_HOST inline friend Point operator*(const Point& p1, const TypeCoordinate p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] * p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] * p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] * p2;
        return p;
    }

    DEVICE_HOST inline friend Point operator*(const Point& p1, const size_t p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] * p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] * p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] * p2;
        return p;
    }

    DEVICE_HOST inline friend Point operator%(const Point& p1, const size_t p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] % p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] % p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] % p2;
        return p;
    }

    DEVICE_HOST inline friend Point operator/(const Point& p1, const TypeCoordinate p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] / p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] / p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] / p2;
        return p;
    }

    DEVICE_HOST inline friend Point operator/(const Point& p1, const size_t p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] / p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] / p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] / p2;
        return p;
    }

    DEVICE_HOST inline void printInt() {
        if (Dimension >= 1)
            printf("%d ", (int)_value[0]);
        if (Dimension >= 2)
            printf("%d ", (int)_value[1]);
        if (Dimension >= 3)
            printf("%d ", (int)_value[2]);
    }

    DEVICE_HOST inline void printFloat() {
        if (Dimension >= 1)
            printf("%f ", (float)_value[0]);
        if (Dimension >= 2)
            printf("%f ", (float)_value[1]);
        if (Dimension >= 3)
            printf("%f ", (float)_value[2]);
    }

    //! Scalar product
    DEVICE_HOST inline friend TypeCoordinate fabs(const Point& p) {
        TypeCoordinate f = 0.0f;
        if (Dimension >= 1)
            f += std::fabs(p._value[0]);
        if (Dimension >= 2)
            f += std::fabs(p._value[1]);
        if (Dimension >= 3)
            f += std::fabs(p._value[2]);
        return f;
    }

};//Point

typedef Point<GLint, 1> Point1DInt;
typedef Point<GLfloat, 1> Point1D;

template<std::size_t Dimension>
class Index : public Point<GLint, Dimension> {
public:
    typedef GLint coord_type;
    typedef Point<GLint, Dimension> super_type;

    //! Constructeurs
    DEVICE_HOST explicit inline Index(GLint const& v0,
                                           GLint const& v1,
                                           GLint const& v2 = 0) : super_type(v0, v1, v2) {}

    DEVICE_HOST explicit inline Index(GLint const& v0) : super_type(v0) {}

    //! Default Cons
    DEVICE_HOST inline Index() : super_type() {}

    //! Cons copy
    DEVICE_HOST inline Index(Index const& p) : super_type(p){}

    //! @brief Affectation
    DEVICE_HOST Index& operator=(super_type const& p2) {
        Point::operator=(p2);//((Point&)*this) = p2;//
        return *this;
    }

//    //! @brief Get coordinate for loop only
//    DEVICE_HOST inline GLint& operator[](std::size_t const i) {
//        return _value[i];
//    }

    //! @brief Affectation
    DEVICE_HOST Index atomicExchange(Index* p1, Index p2) {
        Index pRet(0);
        if (Dimension >= 1)
            pRet[0] = atomicExch(&(p1->_value[0]), p2._value[0]);
        if (Dimension >= 2)
            pRet[1] = atomicExch(&(p1->_value[1]), p2._value[1]);
        if (Dimension >= 3)
            pRet[2] = atomicExch(&(p1->_value[2]), p2._value[2]);
        return pRet;
    }

    DEVICE_HOST Index atomicAddition(Index* p1, int p2) {
        Index pRet(0);
        if (Dimension >= 1)
            pRet[0] = atomicAdd(&(p1->_value[0]), p2);
        if (Dimension >= 2)
            pRet[1] = atomicAdd(&(p1->_value[1]), p2);
        if (Dimension >= 3)
            pRet[2] = atomicAdd(&(p1->_value[2]), p2);
        return pRet;
    }

    friend ostream& operator<<(ostream& o, Index& p) {
        if (Dimension >= 1)
            o << p._value[0] << " ";
        if (Dimension >= 2)
            o << p._value[1] << " ";
        if (Dimension >= 3)
            o << p._value[2] << " ";
        return o;
    }

    friend ofstream& operator<<(ofstream& o, Index& p) {
        if (Dimension >= 1)
            o << p._value[0] << " ";
        if (Dimension >= 2)
            o << p._value[1] << " ";
        if (Dimension >= 3)
            o << p._value[2] << " ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Index& p) {
        if (Dimension >= 1)
            i >> p._value[0];
        if (Dimension >= 2)
            i >> p._value[1];
        if (Dimension >= 3)
            i >> p._value[2];
        return i;
    }

    DEVICE_HOST inline friend Index operator%(const Index& p1, const size_t p2) {
        Index p;
        (super_type&) p = (super_type&) p1 % p2;
        return p;
    }

    DEVICE_HOST inline Index operator-=(Index p2) {
        (super_type&) *this -= p2;
        return *this;
    }

    DEVICE_HOST inline Index operator+=(Index p2) {
        (super_type&) *this += p2;
        return *this;
    }

    DEVICE_HOST inline friend Index operator+(Index const& p1, Index const& p2) {
        Index p;
        (super_type&) p = (super_type&) p1 + (super_type&) p2;
        return p;
    }

    DEVICE_HOST inline friend Index operator-(Index const& p1, Index const& p2) {
        Index p;
        (super_type&) p = (super_type&) p1 - (super_type&) p2;
        return p;
    }

//    //! @brief Affectation, wb.Q add ==
//    DEVICE_HOST inline bool operator ==(Point const& p2) {
//        return (Point&) *this == (Point&) p2;
//     }

//    //! @brief Affectation, wb.Q add !=
//    DEVICE_HOST inline bool operator !=(Point const& p2) {
//        return (Point&) *this != p2;
//    }

//    //! Scalar product
//    DEVICE_HOST inline friend GLfloat operator*(PointCoord const& p1, PointCoord const& p2) {
//        GLint p;
//        p = (Point&) p1 * (Point&) p2;
//        return p;
//    }

    //! Scalar mult
    DEVICE_HOST inline friend Index operator*(Index const& p1, size_t const& p2) {
        Index p;
        (super_type&) p = (super_type&) p1 * p2;
        return p;
    }

    DEVICE_HOST inline friend Index operator*(const Index& p1, const Index& p2) {
        Index p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] * p2._value[0];
        if (Dimension >= 2)
            p._value[1] = p1._value[1] * p2._value[1];
        if (Dimension >= 3)
            p._value[2] = p1._value[2] * p2._value[2];
        return p;
    }

//    //! mult
//    DEVICE_HOST inline friend Index operator*(Index const& p1, Index const& p2) {
//        Index p;
//        (super_type&) p = (super_type&) p1 * (super_type&) p2;
//        return p;
//    }

//    //! Scalar div
//    DEVICE_HOST inline friend PointCoord operator/(PointCoord const& p1, GLint const p2) {
//        PointCoord p;
//        (Point&) p = (Point&) p1 / p2;
//        return p;
//    }
};

template<std::size_t Dimension>
class PointE : public Point<GLfloatP, Dimension> {
public:
    typedef GLfloatP coord_type;
    typedef Point<GLfloatP, Dimension> super_type;

    //! Constructeurs
    DEVICE_HOST explicit inline PointE(GLfloatP const& v0,
                                           GLfloatP const& v1,
                                           GLfloatP const& v2 = 0) : super_type(v0, v1, v2) {}
    DEVICE_HOST explicit inline PointE(GLfloatP const& v0) : super_type(v0) {}

    //! Default Cons
    DEVICE_HOST inline PointE() : super_type() {}

    //! Cons Copy
    DEVICE_HOST inline PointE(PointE const& p) : super_type(p){}

//    //! @brief Affectation
//    DEVICE_HOST PointE& operator=(PointE const& p2) {
//        Point::operator=(p2);//((Point&)*this) = p2;//
//        return *this;
//    }

//    //! @brief Affectation wb.Q copy from PointCoord to Point2D
//    DEVICE_HOST PointE& operator=(Index<Dimension> const& p2) {
//        if (Dimension >= 1)
//            this->_value[0] = (float)p2._value[0];
//        if (Dimension >= 2)
//            this->_value[1] = (float)p2._value[1];
//        if (Dimension >= 3)
//            this->_value[2] = (float)p2._value[2];
//        return *this;
//    }

    friend ostream& operator<<(ostream& o, PointE& p) {
        if (Dimension >= 1)
            o << p._value[0] << " ";
        if (Dimension >= 2)
            o << p._value[1] << " ";
        if (Dimension >= 3)
            o << p._value[2] << " ";
        return o;
    }

    friend ofstream& operator<<(ofstream& o, PointE& p) {
        if (Dimension >= 1)
            o << p._value[0] << " ";
        if (Dimension >= 2)
            o << p._value[1] << " ";
        if (Dimension >= 3)
            o << p._value[2] << " ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, PointE& p) {
        if (Dimension >= 1)
            i >> p._value[0];
        if (Dimension >= 2)
            i >> p._value[1];
        if (Dimension >= 3)
            i >> p._value[2];
        return i;
    }

    DEVICE_HOST inline friend PointE operator+(PointE const& p1, PointE const& p2) {
        PointE p;
        (super_type&) p = (super_type&) p1 + (super_type&) p2;
        return p;
    }

    DEVICE_HOST inline friend PointE operator-(PointE const& p1, PointE const& p2) {
        PointE p;
        (super_type&) p = (super_type&) p1 - (super_type&) p2;
        return p;
    }

//    //! Scalar product
//    DEVICE_HOST inline friend GLfloatP operator*(Point2D const& p1, Point2D const& p2) {
//        GLfloatP p;
//        p = (Point&) p1 * (Point&) p2;
//        return p;
//    }

    //! Scalar mult
    DEVICE_HOST inline friend PointE operator*(PointE const& p1, GLfloatP const p2) {
        PointE p;
        (super_type&) p = (super_type&) p1 * p2;
        return p;
    }

//    //! Scalar div
//    DEVICE_HOST inline friend Point2D operator/(Point2D const& p1, GLfloatP const p2) {
//        Point2D p;
//        (Point&) p = (Point&) p1 / p2;
//        return p;
//    }
};
#if TEST_CODE
class PointCoord : public Index<2> {
public:
    //! Constructeurs
    DEVICE_HOST inline PointCoord() : Index() {}
    DEVICE_HOST inline PointCoord(PointCoord const& p) : Index(p){}
    DEVICE_HOST explicit inline PointCoord(GLint const& v0,
                                        GLint const& v1 = 0) : Index(v0, v1) {}

//    //! @brief Affectation
//    DEVICE_HOST PointCoord& operator=(PointCoord const& p2) {
//        Point::operator=(p2);//((Point&)*this) = p2;//
//        return *this;
//    }

    friend ofstream& operator<<(ofstream& o, PointCoord& p) {
        o << p._value[0] << " " << p._value[1] << " ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, PointCoord& p) {
        i >> p._value[0] >> p._value[1];
        return i;
    }

    DEVICE_HOST inline friend PointCoord operator+(PointCoord const& p1, PointCoord const& p2) {
        PointCoord p;
        (Index&) p = (Index&) p1 + (Index&) p2;
        return p;
    }

    DEVICE_HOST inline friend PointCoord operator-(PointCoord const& p1, PointCoord const& p2) {
        PointCoord p;
        (Index&) p = (Index&) p1 - (Index&) p2;
        return p;
    }

//    //! Scalar product
//    DEVICE_HOST inline friend GLfloatP operator*(PointCoord const& p1, PointCoord const& p2) {
//        GLint p;
//        p = (Point&) p1 * (Point&) p2;
//        return p;
//    }

//    //! Scalar mult
//    DEVICE_HOST inline friend PointCoord operator*(PointCoord const& p1, GLint const p2) {
//        PointCoord p;
//        (Point&) p = (Point&) p1 * p2;
//        return p;
//    }

//    //! Scalar div
//    DEVICE_HOST inline friend PointCoord operator/(PointCoord const& p1, GLint const p2) {
//        PointCoord p;
//        (Point&) p = (Point&) p1 / p2;
//        return p;
//    }
};

class Point2D : public PointE<2> {
public:
    //! Constructeurs
    DEVICE_HOST inline Point2D() : PointE() {}
    DEVICE_HOST inline Point2D(Point2D const& p) : PointE(p){}
    DEVICE_HOST explicit inline Point2D(GLfloatP const& v0,
                                        GLfloatP const& v1 = 0) : PointE(v0, v1) {}

    //! @brief Affectation
//    DEVICE_HOST Point2D& operator=(Point2D const& p2) {
//        Point::operator=(p2);//((Point&)*this) = p2;//
//        return *this;
//    }

    //! @brief Affectation wb.Q copy from PointCoord to Point2D
    DEVICE_HOST Point2D& operator=(PointCoord const& p2) {
        this->_value[0] = (float)p2._value[0];
        this->_value[1] = (float)p2._value[1];
        return *this;
    }

    friend ofstream& operator<<(ofstream& o, Point2D& p) {
        o << p._value[0] << " " << p._value[1] << " ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Point2D& p) {
        i >> p._value[0] >> p._value[1];
        return i;
    }

    DEVICE_HOST inline friend Point2D operator+(Point2D const& p1, Point2D const& p2) {
        Point2D p;
        (PointE&) p = (PointE&) p1 + (PointE&) p2;
        return p;
    }

    DEVICE_HOST inline friend Point2D operator-(Point2D const& p1, Point2D const& p2) {
        Point2D p;
        (PointE&) p = (PointE&) p1 - (PointE&) p2;
        return p;
    }

//    //! Scalar product
//    DEVICE_HOST inline friend GLfloatP operator*(Point2D const& p1, Point2D const& p2) {
//        GLfloatP p;
//        p = (Point&) p1 * (Point&) p2;
//        return p;
//    }

//    //! Scalar mult
//    DEVICE_HOST inline friend Point2D operator*(Point2D const& p1, GLfloatP const p2) {
//        Point2D p;
//        (Point&) p = (Point&) p1 * p2;
//        return p;
//    }

//    //! Scalar div
//    DEVICE_HOST inline friend Point2D operator/(Point2D const& p1, GLfloatP const p2) {
//        Point2D p;
//        (Point&) p = (Point&) p1 / p2;
//        return p;
//    }
};

class Point3D : public PointE<3> {
public:
    //! Constructeurs
    DEVICE_HOST    inline Point3D() : PointE() {}
    DEVICE_HOST    inline Point3D(Point3D const& p) : PointE(p){}
    DEVICE_HOST    explicit inline Point3D(GLfloatP const& v0,
                                           GLfloatP const& v1 = 0,
                                           GLfloatP const& v2 = 0) : PointE(v0, v1, v2) {}

    //! @brief Affectation
//    DEVICE_HOST Point3D& operator=(Point3D const& p2) {
//        (PointE&) *this = p2;//Point::operator=(p2);
//        return *this;
//    }

    DEVICE_HOST inline friend Point3D operator+(Point3D const& p1, Point3D const& p2) {
        Point3D p;
        (PointE&) p = (PointE&) p1 + (PointE&) p2;
        return p;
    }

    DEVICE_HOST inline friend Point3D operator-(Point3D const& p1, Point3D const& p2) {
        Point3D p;
        (PointE&) p = (PointE&) p1 - (PointE&) p2;
        return p;
    }

//    //! Scalar product
//    DEVICE_HOST inline friend GLfloatP operator*(Point3D const& p1, Point3D const& p2) {
//        GLfloatP p;
//        p = (PointE&) p1 * (PointE&) p2;
//        return p;
//    }

    //! Scalar mult
    DEVICE_HOST inline friend Point3D operator*(Point3D const& p1, GLfloatP const p2) {
        Point3D p;
        (Point&) p = (Point&) p1 * p2;
        return p;
    }

//    //! Scalar div
//    DEVICE_HOST inline friend Point3D operator/(Point3D const& p1, GLfloatP const p2) {
//        Point3D p;
//        (Point&) p = (Point&) p1 / p2;
//        return p;
//    }

    friend ostream& operator<<(ostream& o, Point3D& p) {
        o << p._value[0] << " " << p._value[1] << " " << p._value[2] << " ";
        return o;
    }

    friend ofstream& operator<<(ofstream& o, Point3D& p) {
        o << p._value[0] << " " << p._value[1] << " " << p._value[2] << " ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Point3D& p) {
        i >> p._value[0] >> p._value[1] >> p._value[2];
        return i;
    }
};
#endif

typedef Index<2> PointCoord;
typedef PointE<2> Point2D;
typedef PointE<3> Point3D;
typedef PointE<2> PointEuclid;
typedef PointE<2> Motion;

//! @}

#if TEST_CODE
//! Test program
class Test {
public:
    void run() {
        cout << "... begin test ..." << endl;

        Point1D p1(10), p2(15), p3;
        p3 = p1 + p2;
        p3 = p1 - p2;
        Point1D p4 = p3;//cons copie
        Point1D p5 = p4;
        cout << "point p = " << p5.get<0>() << endl;

        Point2D pp1(10, 15), pp2(15, 20), pp3;
        pp3 = pp1;
        Point2D pp;
        pp = pp1 + pp2;
        pp2 = pp1 - pp2;
        pp3 = pp1 * pp3;
        Point2D pp4 = pp3;//cons copie
        Point2D pp5 = pp2;
        cout << "point p = (" << pp5.get<0>() << ", " <<  pp5.get<1>() << ")" << endl;
        pp5 = pp4;
        cout << "point p = (" << pp5.get<0>() << ", " <<  pp5.get<1>() << ")" << endl;

        Point3D ppp1(10, 15, 20), ppp2(15, 20, 25), ppp3;
        ppp3 = ppp1;
        Point3D ppp;
        ppp = ppp1 + ppp2;
        ppp2 = ppp1 - ppp2;
        ppp3 = ppp1 * ppp3;
        Point3D ppp4 = ppp3;//cons copie
        Point3D ppp5 = ppp2;
        cout << "point p = (" << ppp5.get<0>() << ", " <<  ppp5.get<1>()  << ", " <<  ppp5.get<2>() << ")" << endl;
        ppp5 = ppp4;
        cout << "point p = (" << ppp5.get<0>() << ", " <<  ppp5.get<1>()  << ", " <<  ppp5.get<2>() << ")" << endl;
        cout << "... end test ..." << endl << endl;
    }
};
#endif

}//namespace components


#endif // NODE_H

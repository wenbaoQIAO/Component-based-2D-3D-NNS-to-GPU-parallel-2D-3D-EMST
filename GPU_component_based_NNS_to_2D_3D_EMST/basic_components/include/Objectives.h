#ifndef OBJECTIVES_H
#define OBJECTIVES_H
/*
 ***************************************************************************
 *
 * Author : H. Wang, J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include "ConfigParams.h"
#include "Node.h"
#include "macros_cuda.h"

using namespace std;

namespace components
{

template<typename TypeCoordinate, std::size_t Dimension>
class Objectives;

typedef Objectives<GLdouble, 9> AMObjectives;
//typedef Objectives<GLfloat, 9> AMObjectives;
enum AMObjNames { obj_distr, obj_length, obj_sqr_length, obj_cost, obj_sqr_cost, obj_cost_window, obj_sqr_cost_window, obj_smoothing, obj_gd_error };

//! HW 01.06.15 : add
//! Types of objectives that will be evaluated
struct ActiveObj {
    float distr;
    float length;
    float sqr_length;
    float cost;
    float sqr_cost;
    float cost_window;
    float sqr_cost_window;
    float smoothing;
    float gd_error;

    DEVICE_HOST ActiveObj() {
        distr = 0.0f;
        length = 0.0f;
        sqr_length = 0.0f;
        cost = 0.0f;
        sqr_cost = 0.0f;
        cost_window = 0.0f;
        sqr_cost_window = 0.0f;
        smoothing = 0.0f;
        gd_error = 0.0f;
    }

    DEVICE_HOST ActiveObj(float _distr,
                          float _length,
                          float _sqr_length,
                          float _cost,
                          float _sqr_cost,
                          float _cost_window,
                          float _sqr_cost_window,
                          float _smoothing,
                          float _gd_error) :
        distr(_distr),
        length(_length),
        sqr_length(_sqr_length),
        cost(_cost),
        sqr_cost(_sqr_cost),
        cost_window(_cost_window),
        sqr_cost_window(_sqr_cost_window),
        smoothing(_smoothing),
        gd_error(_gd_error) {}


    DEVICE_HOST void readParameters(std::string const& name, ConfigParams& params) {
        params.readConfigParameter(name,"obj_distr", distr);
        params.readConfigParameter(name,"obj_length", length);
        params.readConfigParameter(name,"obj_sqr_length", sqr_length);
        params.readConfigParameter(name,"obj_cost", cost);
        params.readConfigParameter(name,"obj_sqr_cost", sqr_cost);
        params.readConfigParameter(name,"obj_cost_window", cost_window);
        params.readConfigParameter(name,"obj_sqr_cost_window", sqr_cost_window);
        params.readConfigParameter(name,"obj_smoothing", smoothing);
        params.readConfigParameter(name,"obj_gd_error", gd_error);
    }//readParameters

};

template<typename TypeCoordinate, std::size_t Dimension>
class Objectives
{
protected:
    TypeCoordinate _objectives[Dimension];
    TypeCoordinate _weights[Dimension];

public:
    /*! @name Objectifs et criteres du probleme
     * @{
     */

    //! @brief Default constructor
    DEVICE_HOST inline Objectives()  {
        if (Dimension >= 1) {
            _objectives[0] = 0;
            _weights[0] = 0;
        }
        if (Dimension >= 2) {
            _objectives[1] = 0;
            _weights[1] = 0;
        }
        if (Dimension >= 3) {
            _objectives[2] = 0;
            _weights[2] = 0;
        }
        if (Dimension >= 4) {
            for (size_t i = 3; i < Dimension; ++i) {
                _objectives[i] = 0;
                _weights[i] = 0;
            }
        }
    }

    //! @brief Constructor
    DEVICE_HOST explicit inline Objectives(TypeCoordinate const& v0,
                               TypeCoordinate const& v1 = 0,
                               TypeCoordinate const& v2 = 0) {

        if (Dimension >= 1)
            _objectives[0] = v0;
        if (Dimension >= 2)
            _objectives[1] = v1;
        if (Dimension >= 3)
            _objectives[2] = v2;
        if (Dimension >= 4) {
            for (size_t i = 3; i < Dimension; ++i)
                _objectives[i] = v0;
        }
    }

//    //! @brief Affectation
//    DEVICE_HOST explicit Objectives(Objectives const& p2) {
//        for (size_t i = 0; i < Dimension; ++i)
//            _objectives[i] = p2._objectives[i];
//    }

    DEVICE_HOST inline void init() {

        if (Dimension >= 1) {
            _objectives[0] = 0;
            _weights[0] = 0;
        }
        if (Dimension >= 2) {
            _objectives[1] = 0;
            _weights[1] = 0;
        }
        if (Dimension >= 3) {
            _objectives[2] = 0;
            _weights[2] = 0;
        }
        if (Dimension >= 4) {
            for (size_t i = 3; i < Dimension; ++i) {
                _objectives[i] = 0;
                _weights[i] = 0;
            }
        }
    }

//    //! @brief Affectation
//    DEVICE_HOST Objectives& operator=(Objectives const& p2) {
//        for (size_t i = 0; i < Dimension; ++i)
//            _objectives[i] = p2._objectives[i];
//        return *this;
//    }

    //! @brief Auto add
    DEVICE_HOST Objectives& operator=(TypeCoordinate const& v0) {
        for (size_t i = 0; i < Dimension; ++i)
            _objectives[i] = v0;
        return *this;
    }

    //! @brief Auto add
    DEVICE_HOST Objectives& operator+=(Objectives const& p2) {
        for (size_t i = 0; i < Dimension; ++i)
            _objectives[i] += p2._objectives[i];
        return *this;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate& operator[](std::size_t const i) {
        return _objectives[i];
    }

    //! @brief Get coordinate
    template <std::size_t K>
    DEVICE_HOST inline TypeCoordinate const& get() const {
        return _objectives[K];
    }

    //! @brief Set coordinate
    template <std::size_t K>
    DEVICE_HOST inline void set(TypeCoordinate const& value) {
        _objectives[K] = value;
    }
    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate const& get(std::size_t const i) const {
        return _objectives[i];
    }

    //! @brief Set coordinatev for loop only
    DEVICE_HOST inline void set(std::size_t const i, TypeCoordinate const& value) {
        _objectives[i] = value;
    }
    template <std::size_t K>
    DEVICE_HOST inline TypeCoordinate const& get_weights() const {
        return _weights[K];
    }

    //! @brief Set coordinate
    template <std::size_t K>
    DEVICE_HOST inline void set_weights(TypeCoordinate const& value) {
        _weights[K] = value;
    }
    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate const& get_weights(std::size_t const i) const {
        return _weights[i];
    }

    //! @brief Set coordinatev for loop only
    DEVICE_HOST inline void set_weights(std::size_t const i, TypeCoordinate const& value) {
        _weights[i] = value;
    }

    //! HW 20/05/15 : add copy weights
    DEVICE_HOST inline void copy_weights(Objectives const& p2) {
        for (size_t i = 0; i < Dimension; ++i)
            _weights[i] += p2._weights[i];
    }

    //! @}

    /*!
     * \return valeur de la fonction objectif agregative
     */
    DEVICE_HOST TypeCoordinate computeObjectif() {

        TypeCoordinate objectif;

        objectif = 0;
        for (int i = 0; i < Dimension; ++i) {
            objectif += get(i) * get_weights(i);
        }

        return objectif;
    }

    /*!
     * \param best solution comparee
     * \return vrai si objectif de l'appelant (ie la solution courante) est inferieur ou egal a celui de la solution comparee
     */
    DEVICE_HOST bool isBest(Objectives* best) {
        bool res = false;

        if (computeObjectif() <= best->computeObjectif())
            res = true;

        return res;
    }

    /*!
     * \return vrai si solution admissible
     */
    DEVICE_HOST bool isSolution() {
        bool ret = true;
        for (int i = 0; i < Dimension; ++i) {
            if (get(i) > 0) {
                ret = false;
                break;
            }
        }
        return ret;
    }//isSolution
};

}//namespace components

#endif // OBJECTIVES_H

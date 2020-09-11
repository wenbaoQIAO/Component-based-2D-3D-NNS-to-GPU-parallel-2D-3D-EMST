#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
/*!
 * \defgroup RandomGen Generateur de nombre aleatoire
 * \brief initialisation, generation d'entier (int),
 * generation de flottant (double).
 * @{
 */
//!
inline void aleat_initialize(unsigned int seed)
{
    srand (seed);
}

inline void aleat_initialize(void)
{
    srand (time (NULL));
}

inline int aleat_int(double a, double b)
{
    double ret = ( (double)rand()/(double)RAND_MAX * (b-a) + a );
    int ret_2 = (int) ret;
    return ret_2;

}

inline double aleat_double(double a, double b) {
    double ret = ( (double)rand()/(double)RAND_MAX * (b-a) + a );
    return ret;
}

inline float aleat_float(float a, float b) {
    double ret = ( (float)rand()/(float)RAND_MAX * (b-a) + a );
    return ret;
}

inline float randomNum(float min, float max)
{
    return ((float)rand() / (float)RAND_MAX * (max - min) + min);
}

inline int myRound(float a)
{
    return (a >= 0) ? (int)(a+0.5) : (int)(a-0.5);
}

//! @}

#endif // RANDOM_GENERATOR_H

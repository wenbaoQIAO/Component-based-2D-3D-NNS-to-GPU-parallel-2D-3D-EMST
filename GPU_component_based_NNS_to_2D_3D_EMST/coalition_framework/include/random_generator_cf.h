#ifndef RANDOM_GENERATOR_CF_H
#define RANDOM_GENERATOR_CF_H

#include <stdlib.h>
#include <time.h>

#include <QtGlobal>
#include <QTime>

namespace random_cf
{

/*!
* \defgroup RandomGen Generateur de nombre aleatoire
* \brief initialisation, generation d'entier (int),
* generation de flottant (double).
* @{
*/

//! Initialise le gnrateur alatoire pour les distributions gaussiennes et exponentielles
void zigset(unsigned long jsrseed);
//! Nombre alatoire selon distribution exponential de density exp(-x), x>0
float rexp(void);
//! Nombre alatoire selon distribution gaussienne centre rduite (0; 1)
float rnor(void);
//! Nombre alatoire selon distribution gaussienne centre en mu, cart-type sigma
float rnor2(float mu, float sigma);

//! Initialise le gnrateur alatoire selon la graine donne en argument
inline void aleat_initialize(int seed)
{
    // Version QT - reentrant
    qsrand(seed);
    //zigset(seed);
}

//! Recupre l'heure courante pour la gnration alatoire d'une graine
inline int aleat_get_time(void)
{
    // Version QT - reentrant
    QTime time = QTime::currentTime();
    return (time.msec() + time.second()*1000);
}

//! Genere un nombre alatoire flotant entre [a et b]
inline double aleat_double(double a, double b)
{
    double ret = ((double)qrand() / (double)RAND_MAX * (b - a) + a);
    return ret;
}

//! Genere un nombre alatoire entier entre [a et b]
inline int aleat_int(int a, int b)
{
    if (a>b)
        return a;
    if ((b-a)==0)
        return a;
    return (qrand()%(b-a+1)) + a;
}

/**<summary>Teste l'ocurrence d'une probabilit entre 0 et 1.0 (test < ou <=)
 * </summary>
 */
inline bool occurs(double proba, bool less_equal = false)
{
    if (less_equal) {
        return (aleat_double(0.0, 1.0) <= proba);
    } else {
        return (aleat_double(0.0, 1.0) < proba);
    }
}

//! @}

}//namespace random

#endif // RANDOM_GENERATOR_H

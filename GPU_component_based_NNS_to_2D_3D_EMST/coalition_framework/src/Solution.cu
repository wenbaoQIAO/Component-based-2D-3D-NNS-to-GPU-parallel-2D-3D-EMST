#include "Solution.h"

#include "random_generator_cf.h"
#include "Calculateur.h"
#include "Multiout.h"

#define round(x) ((fabs(ceil(x) - (x)) < fabs(floor(x) - (x))) ? ceil(x) : floor(x))

int Solution::cptInstance = 0;

void Solution::initialize(char* data, char* sol, char* stats)
{
    fileData = data;
    fileSolution = sol;
    fileStats = stats;

    initialize();
}

void Solution::initialize()
{
        this->global_objectif = g_ConfigParameters->generationNumber;
}

void Solution::initialize(NN& md, NN& mr)
{
    lout << "INITIALISATION" << std::endl;


    lout << "INITIALISATION DONE" << std::endl;
}

void Solution::clone(Solution* sol)
{
    (*sol).fileData = (*this).fileData;
    (*sol).fileSolution = (*this).fileSolution;
    (*sol).fileStats = (*this).fileStats;

    (*sol).t0 = (*this).t0;
    (*sol).tf = (*this).tf;
    (*sol).x0 = (*this).x0;
    (*sol).xf = (*this).xf;

#ifdef CUDA_CODE
    (*sol).start = (*this).start;
    (*sol).stop = (*this).stop;
#endif
    (*sol).global_objectif = (*this).global_objectif;

    (*this).md.clone((*sol).md);
    (*this).mr.clone((*sol).mr);
    (*sol).initialize((*sol).md, (*sol).mr);
    (*sol).evaluate();

}

void Solution::setIdentical(Solution* sol)
{
    (*sol).fileData = (*this).fileData;
    (*sol).fileSolution = (*this).fileSolution;
    (*sol).fileStats = (*this).fileStats;

    (*sol).t0 = (*this).t0;
    (*sol).tf = (*this).tf;
    (*sol).x0 = (*this).x0;
    (*sol).xf = (*this).xf;

#ifdef CUDA_CODE
    (*sol).start = (*this).start;
    (*sol).stop = (*this).stop;
#endif
    (*sol).global_objectif = (*this).global_objectif;

    (*this).md.setIdentical((*sol).md);
    (*this).mr.setIdentical((*sol).mr);
}

void Solution::initEvaluate()
{
//    this->global_objectif = numeric_limits<double>::max();
}

double Solution::evaluate()
{
    //-------------------------------------------------------------------------
    // Mise à jour positions vehicules, chemins, volumes
    //-------------------------------------------------------------------------
    initEvaluate();

    // Calcul objectif global
    computeObjectif();
    return global_objectif;
}//evaluate

/*!
 * \return valeur de la fonction objectif agregative
 */
double Solution::computeObjectif(void)
{
    global_objectif -= 1;

    return global_objectif;
}

/*!
 * \param best solution comparee
 * \return vrai si objectif de l'appelant (ie la solution courante) est inferieur ou egal a celui de la solution comparee
 */
bool Solution::isBest(Solution* best)
{
    bool res = false;

    if (computeObjectif() <= best->computeObjectif())
        res = true;

    return res;
}

/*!
 * \return vrai si solution admissible
 */
bool Solution::isSolution()
{
    bool res = false;
    if (this->global_objectif <= 0)    {
        res = true;
    }
    return res;
}//isSolution


#include "Solution.h"
#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"

/** Operateurs de changement de solution courante.
 *
 */

void Solution::initConstruct()
{
}//initConstruct

/** Construction Sequentielle
 */
void Solution::constructSolutionSeq()
{
    lout << "CONSTRUCTION ..." << endl;
}

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
bool Solution::operator_1() {
    bool ret = true;

    global_objectif = computeObjectif();

    return ret;
}//operator_1

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
bool Solution::operator_2() {
    bool noUsed = true;

    return noUsed;
}//operator_1

bool Solution::generateNeighbor()
{
    int no_op = 0;
    double totalCapacity = 0;
    while (no_op < g_ConfigParameters->probaOperators.size()) {
        totalCapacity += g_ConfigParameters->probaOperators[no_op];
        no_op += 1;
    }

    // Tirage aleatoire par "roulette"
    double d = random::aleat_double(0, totalCapacity);

     // Determiner no d'operateur
    no_op = -1;
    double t_sise = 0;
    int size = g_ConfigParameters->probaOperators.size();
    for (int k = 0; k < size; k++) {
        t_sise += g_ConfigParameters->probaOperators[k];
        //cout << "probaOperators " << g_ConfigParameters->probaOperators[k] << endl;
        if (d < t_sise) {
            no_op = k;
            break;
        }
    }
    if (no_op == -1)
        cout << "PB TIRAGE OPERATEUR !!! " << g_ConfigParameters->probaOperators.size() << endl;
    else
        cout << "Choix operator : " << no_op << endl;

    // Appliquer l'operateur ...
    if (applyOperator(no_op))
    {
        this->computeObjectif();
        if (this->global_objectif < 0)
        {
            lout << "ERROR!!! OPERATEUR num." << no_op << " A DONNE OBJECTIF NEGATIF : " << this->global_objectif << endl;
        }
    }
    return true;
}//generateNeighbor

bool Solution::applyOperator(int i)
{
    bool ret = false;
    switch (i)
    {
        case 0:
            break ;
        case 1:
            break ;
        case 2:
            break ;
        case 3:
            break ;
        case 4:
            break ;
        case 5:
            break ;
        case 6:
            break ;
        case 7:
            break ;
    }
    ret = operator_1();
    return ret;
}

int Solution::nbrOperators() const
{
    return g_ConfigParameters->probaOperators.size();
}


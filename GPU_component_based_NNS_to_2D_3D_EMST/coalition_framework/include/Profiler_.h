#ifndef PROFILER_H
#define PROFILER_H

#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <string>
#include <math.h>
using namespace std;



/* Statistiques lors de la recherche de solution.
 * Utilisé pour instrumenter le code et les heuristiques (opérateurs/proba).
 */
class Profiler
{
protected:
#pragma region Membres prives/protected
    //! Flux de sortie
    ofstream *FileStream;
    //! Ouverture du fichier de sortie (texte) pour statistiques
    void OpenFile();
    //! Fermeture du fichier de sortie (texte) pour statistiques
    void CloseFile();
#pragma endregion


public:
#pragma region Sortie sur fichier
    //! Nom du fichier de sortie
    string FileName;
#pragma endregion

#pragma region Nombre d appels aux operateurs
    //! 
    long CntOrdoSwapInterLot;
    //!
    long CntOrdoSwapIntraLot;
    //!
    long CntSwapIntraLot;
    //!
    long CntActivateRetourne;
    //!
    long CntActivateRemove;
    //!
    long CntActivateStructures;
    //!
    long CntInvertRandomVehicleDirection;
    //!
    long CntMapVehicle;
    //!
    long CntApplyRandomVehicleMovement;
    //!
    long CntApplyContact;
#pragma endregion

#pragma region Nombre d evaluations et comparaison
    //! Evaluations complètes
    long CntEvalComplete;
    //! Comparaison véchicule à véhicule selon ordre de chargement robot
    long CntComparaison;
#pragma endregion


#pragma region Operations de geometrie
    //! Plaquage roue par translation random
    long CntPlaquageRoueByTranslationRandom;
    //! Plaquage roue par translation sur chemins
    long CntPlaquageRoueByTranslationChemins;
    //! Plaquage roue par translation
    long CntPlaquageRoueByTranslation;
    //! Plaquage roue par rotation sur chemins
    long CntPlaquageRoueByRotationChemins;
    //! Plaquage roue par rotation
    long CntPlaquageRoueByRotation;
    //! Evaluations complètes
    long CntEvalOverlapsPolygonStructures;
#pragma endregion

#pragma region proba de mouvement
    long CntNbTranslation;
    long CntNbRotation;
    double SumTranslation;
    double SumRotation;
    double SumTranslation2;
    double SumRotation2;

    long CntNbRelatif;

    long CntNbSuperPas;
    long CntNbPas;
    double SumPas;
    double SumSuperPas;
    double SumPas2;
    double SumSuperPas2;

    // http://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation

    double MeanTranslation()
    {
        return (this->SumTranslation) / this->CntNbTranslation;
    }
    double MeanRotation()
    {
        return (this->SumRotation) / this->CntNbRotation;
    }

    double StdDevTranslation()
    {
        return sqrt(this->CntNbTranslation * this->SumTranslation2 - this->SumTranslation*this->SumTranslation)/this->CntNbTranslation;
    }
    double StdDevRotation()
    {
        return sqrt(this->CntNbRotation * this->SumRotation2 - this->SumRotation*this->SumRotation)/this->CntNbRotation;
    }
    double MeanPas()
    {
        return (this->SumPas) / this->CntNbPas;
    }
    double MeanSuperPas()
    {
        return (this->SumSuperPas) / this->CntNbSuperPas;
    }

    double StdDevPas()
    {
        return sqrt(this->CntNbPas * this->SumPas2 - this->SumPas*this->SumPas)/this->CntNbPas;
    }
    double StdDevSuperPas()
    {
        return sqrt(this->CntNbSuperPas * this->SumSuperPas2 - this->SumSuperPas*this->SumSuperPas)/this->CntNbSuperPas;
    }

#pragma endregion

#pragma region Constructeurs/init
    Profiler()
    {
        FileStream = new ofstream();
        FileName = "profile.stats";
        Init();
    }
    ~Profiler()
    {
        delete FileStream;
    }

    void Init()
    {
        CntOrdoSwapInterLot = 0;
        CntOrdoSwapIntraLot = 0;
        CntSwapIntraLot = 0;
        CntActivateRetourne = 0;
        CntActivateRemove = 0;
        CntActivateStructures = 0;
        CntInvertRandomVehicleDirection = 0;
        CntMapVehicle = 0;
        CntApplyRandomVehicleMovement = 0;
        CntApplyContact = 0;
        CntEvalComplete = 0;
        CntPlaquageRoueByTranslationRandom = 0;
        CntPlaquageRoueByTranslationChemins = 0;
        CntPlaquageRoueByTranslation = 0;
        CntPlaquageRoueByRotationChemins = 0;
        CntPlaquageRoueByRotation = 0;
        CntEvalOverlapsPolygonStructures = 0;


        CntNbTranslation = 0;
        CntNbRotation = 0;
        CntNbSuperPas = 0;
        CntNbPas = 0;
        CntNbRelatif = 0;
        SumTranslation = 0;
        SumRotation = 0;
        SumTranslation2 = 0;
        SumRotation2 = 0;
    }
#pragma endregion

#pragma region Sortie texte sur disque ou stream
    //! Ecriture de l'entete pour statistiques dans le fichier
    void WriteHeaderStatisticsToFile();
    //! Ecriture de l'entete pour statistiques dans la sortie
    void WriteHeaderStatistics(std::ostream& output);
    //! Ecriture des valeurs de criteres/objectifs dans le fichier
    void WriteStatisticsToFile(int iteration);
    //! Ecriture des valeurs de criteres/objectifs dans la sortie
    void WriteStatistics(int iteration, std::ostream& o);
#pragma endregion
};

//! Compteurs sur les opérateurs et les évaluations, utilisé pour instrumenter le code.
extern Profiler ProfilerOp;

#endif // PROFILER_H

#include "Profiler.h"
#include "time.h"
#include "config/ConfigParamsCF.h"

Profiler ProfilerOp;

void Profiler::OpenFile()
{
    FileStream->open(FileName.c_str(), ios::app);
    if (!FileStream->rdbuf()->is_open())
    {
        cerr << "Unable to open file " << FileName << "CRITICAL ERROR" << endl;
        exit(-1);
    }
}

void Profiler::CloseFile()
{
    FileStream->close();
}

void Profiler::WriteHeaderStatisticsToFile()
{
    OpenFile();
    WriteHeaderStatistics(*FileStream);
    CloseFile();
}

void Profiler::WriteHeaderStatistics(std::ostream& output)
{
    time_t today = time(0);
    struct tm today_localtime = *localtime(&today);

    // Banner line
    output << "#"
        // Version
        << "Imbricator"
        << " version " << VERSION_CALCULATEUR
        // Compilation
        << " built on " << __DATE__
        << " " << __TIME__
        // Time when executed
        << ", current time " << today_localtime.tm_year+1900
        << "-" << today_localtime.tm_mon+1
        << "-" << today_localtime.tm_mday
        << " " << today_localtime.tm_hour
        << ":" << today_localtime.tm_min
        << ":" << today_localtime.tm_sec
        << endl;

    // Columns line
    output << "iteration" << "\t"
        << "CntOrdoSwapInterLot" << "\t"
        << "CntOrdoSwapIntraLot" << "\t"
        << "CntSwapIntraLot" << "\t"
        << "CntActivRet" << "\t"
        << "CntActivStr" << "\t"
        << "CntInvRandVehDir" << "\t"
        << "CntMapVehicle" << "\t"
        << "CntApplyRandVehMov" << "\t"
        << "CntApplyContact" << "\t"
        << "CntEvalComplete" << "\t"
        << "CntPlaqRoueByTranslRand" << "\t"
        << "CntPlaqRoueByTranslChemins" << "\t"
        << "CntPlaqRoueByTransl" << "\t"
        << "CntPlaqRoueByRotChemins" << "\t"
        << "CntPlaqRoueByRot" << "\t"
        << "CntEvalOverlPolygonStr" << "\t"

        << "%CntOrdoSwapInterLot" << "\t"
        << "%CntOrdoSwapIntraLot" << "\t"
        << "%CntSwapIntraLot" << "\t"
        << "%CntActivateRetourne" << "\t"
        << "%CntActivateRemove" << "\t"
        << "%CntActivateStructures" << "\t"
        << "%CntInvRandVehDir" << "\t"
        << "%CntMapVehicle" << "\t"
        << "%CntApplyRandVehMov" << "\t"
        << "%CntApplyContact" << "\t"

        << "%CntNbTranslation" << "\t"
        << "%CntNbRotation" << "\t"
        << "%CntNbSuperPas" << "\t"
        << "%CntNbPas" << "\t"
        << "%CntNbRelatif" << "\t"
        << "%MeanTranlation" << "\t"
        << "%MeanRotation" << "\t"
        << "%StdDevTranslation" << "\t"
        << "%StdDevRotation" << "\t"
        << "%MeanPas" << "\t"
        << "%MeanSuperPas" << "\t"
        << "%StdDevPas" << "\t"
        << "%StdDevSuperPas" << "\t"
        << endl;
}

void Profiler::WriteStatisticsToFile(int iteration)
{
    OpenFile();
    WriteStatistics(iteration, *FileStream);
    CloseFile();
}

void Profiler::WriteStatistics(int iteration, std::ostream& output)
{
    output << iteration << "\t"
        << this->CntOrdoSwapInterLot << "\t"
        << this->CntOrdoSwapIntraLot << "\t"
        << this->CntSwapIntraLot << "\t"
        << this->CntActivateRetourne << "\t"
        << this->CntActivateRemove << "\t"
        << this->CntActivateStructures << "\t"
        << this->CntInvertRandomVehicleDirection << "\t"
        << this->CntMapVehicle << "\t"
        << this->CntApplyRandomVehicleMovement << "\t"
        << this->CntApplyContact << "\t"
        << this->CntEvalComplete << "\t"
        << this->CntPlaquageRoueByTranslationRandom << "\t"
        << this->CntPlaquageRoueByTranslationChemins << "\t"
        << this->CntPlaquageRoueByTranslation << "\t"
        << this->CntPlaquageRoueByRotationChemins << "\t"
        << this->CntPlaquageRoueByRotation << "\t"
        << this->CntEvalOverlapsPolygonStructures << "\t";

    // Pourcent oprateurs
    double scale_op = (this->CntOrdoSwapInterLot
                       + this->CntOrdoSwapIntraLot
                       +this->CntSwapIntraLot
                       +this->CntActivateRetourne
                       +this->CntActivateRemove
                       +this->CntActivateStructures
                       +this->CntInvertRandomVehicleDirection
                       +this->CntMapVehicle
                       +this->CntApplyRandomVehicleMovement
                       //+this->CntApplyContact
                       )*0.01;
    output <<  this->CntOrdoSwapInterLot/scale_op << "\t"
        << this->CntOrdoSwapIntraLot/scale_op << "\t"
        << this->CntSwapIntraLot/scale_op << "\t"
        << this->CntActivateRetourne/scale_op << "\t"
        << this->CntActivateRemove/scale_op << "\t"
        << this->CntActivateStructures/scale_op << "\t"
        << this->CntInvertRandomVehicleDirection/scale_op << "\t"
        << this->CntMapVehicle/scale_op << "\t"
        << this->CntApplyRandomVehicleMovement/scale_op << "\t"
        << this->CntApplyContact/scale_op << "\t";

    output <<  this->CntNbTranslation << "\t"
        << this->CntNbRotation << "\t"
        << this->CntNbSuperPas << "\t"
        << this->CntNbPas << "\t"
        << this->CntNbRelatif << "\t"
        << this->MeanTranslation() << "\t"
        << this->MeanRotation() << "\t"
        << this->StdDevTranslation() << "\t"
        << this->StdDevRotation() << "\t"
        << this->MeanPas() << "\t"
        << this->MeanSuperPas() << "\t"
        << this->StdDevPas() << "\t"
        << this->StdDevSuperPas() << "\t";

    output << endl;
}

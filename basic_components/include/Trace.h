#ifndef TRACE_H
#define TRACE_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H.Wanf, A.Mansouri
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include "ConfigParams.h"
#include "Objectives.h"
#include <iomanip>

#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64

#ifdef CUDA_CODE
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

using namespace std;

//! WH 040815 : for optical flow result statistics
#ifdef POPIP_COALTION
struct flowResultInfo
{
    int leftRight; // 0-left; 1-right; 2-post-processing
    int initMode;
    int iteNum;
    int cellRadius;
    int lsMode;
    float costWei;
    float costWindowWei;
    float smoothWei;
    float objValBefore;
    float objValAfter;
    float objValDataBefore;
    float objValDataAfter;
    float objValSmoothBefore;
    float objValSmoothAfter;
    float GTErrBefore;
    float GTErrAfter;
    float runtime;
    float CUDA_runtime;
};

#else
struct flowResultInfo
{
    int leftRight; // 0-left; 1-right; 2-post-processing
    int initMode;
    int iteNum;
    int cellRadius;
    int lsMode;
    float costWei;
    float costWindowWei;
    float smoothWei;
    float objValBefore;
    float objValAfter;
    float objValDataBefore;
    float objValDataAfter;
    float objValSmoothBefore;
    float objValSmoothAfter;
    float GTErrBefore;
    float GTErrAfter;
    float runtime;
    float CUDA_runtime;
} _flowResultInfo;

#endif

namespace components
{

class Trace
{
    //! Objectives to trace
    AMObjectives objs;

    //! Fichier de sortie avec valeurs de criteres et objectifs de la solution
    char* fileStats;
    //! Flux de sortie ouvert pour statistiques
    std::ofstream* OutputStream;

    //! Calcul duree d'execution
    time_t t0;
    //! Calcul duree d'execution
    time_t tf;
    //! Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
    double x0;
    //! Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
    double xf;

    // cuda timer
#ifdef CUDA_CODE
    cudaEvent_t start, stop;
#endif
public:
    explicit Trace() : OutputStream(new ofstream) {}
    explicit Trace(AMObjectives objs) : objs(objs), OutputStream(new ofstream) {}

    //! \brief Destructeur.
    ~Trace(){
        delete OutputStream;
    }

    //! Initialisations
    void initialize(char* stats) {
        fileStats = stats;
        OutputStream->open(fileStats, ios::app);
        initialize();
    }

    void setObjs(AMObjectives objs) {
        this->objs = objs;
    }

    void initialize() {
        if (!OutputStream->rdbuf()->is_open())
        {
            cerr << "Unable to open file " << fileStats << "CRITICAL ERROR" << endl;
            exit(-1);
        }

        //! WH 040815 : for optical flow result statistics
        initHeaderStatisticsFlow(*OutputStream);

//        initHeaderStatistics(*OutputStream);

        time(&t0);
        x0 = clock();

#ifdef CUDA_CODE
        int devID = 0;
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&devID);
        if (error != cudaSuccess)
        {
            printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        }
        error = cudaGetDeviceProperties(&deviceProp, devID);
        if (deviceProp.computeMode == cudaComputeModeProhibited)
        {
            fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
            exit(EXIT_SUCCESS);
        }
        if (error != cudaSuccess)
        {
            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        }
        else
        {
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
                   devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        }
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
#endif
    }

    void initHeaderStatistics(std::ostream& o) {
        o  	<< "iteration" << "\t"
            << "objective_1" << "\t"
            << "objective_2" << "\t"
            << "objective_3" << "\t"
            << "duree(s)" << "\t"
            << "duree(s.xx)" << "\t"
            << "cuda_duree(ms)" << endl;

    }//initHeaderStatistics

    //! WH 040815 : for optical flow result statistics
    void initHeaderStatisticsFlow(std::ostream& o) {
        ifstream fin(fileStats);
        string s;
        fin >> s;
        if(s.length() == 0) {
        o  	<< "L?R?P?" << setw(15)
            << "initMode" << setw(15)
            << "iteration" << setw(15)
            << "cellRadius" << setw(15)
            << "lsMode" << setw(15)
            << "costWei" << setw(15)
            << "costWindowWei" << setw(15)
            << "smoothWei" << setw(15)
            << "objValBefore" << setw(15)
            << "objValAfter" << setw(15)
            << "DataBefore" << setw(15)
            << "DataAfter" << setw(15)
            << "SmoothBefore" << setw(15)
            << "SmoothAfter" << setw(15)
            << "GTErrBefore" << setw(15)
            << "GTErrAfter" << setw(15)
            << "runtime" << setw(15)
            << "CUDA_runtime" << endl;
        }

    }//initHeaderStatistics

    //! WH 040815 : for optical flow result statistics
    void writeStatisticsFlow(flowResultInfo &_flowInfo, std::ostream& o) {
        o  	<< _flowInfo.leftRight << setw(15)
            << _flowInfo.initMode << setw(15)
            << _flowInfo.iteNum << setw(15)
            << _flowInfo.cellRadius << setw(15)
            << _flowInfo.lsMode << setw(15)
            << _flowInfo.costWei << setw(15)
            << _flowInfo.costWindowWei << setw(15)
            << _flowInfo.smoothWei << setw(15)
            << _flowInfo.objValBefore << setw(15)
            << _flowInfo.objValAfter << setw(15)
            << _flowInfo.objValDataBefore << setw(15)
            << _flowInfo.objValDataAfter << setw(15)
            << _flowInfo.objValSmoothBefore << setw(15)
            << _flowInfo.objValSmoothAfter << setw(15)
            << _flowInfo.GTErrBefore << setw(15)
            << _flowInfo.GTErrAfter << setw(15)
            << _flowInfo.runtime << setw(15)
            << _flowInfo.CUDA_runtime // << setw(15)
            << endl;
    }//writeStatistics

    void writeStatisticsFlow(flowResultInfo &_flowInfo) {

        writeStatisticsFlow(_flowInfo, *OutputStream);
    }


    void writeStatistics(int iteration, std::ostream& o) {

#ifdef CUDA_CODE
        // cuda timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
#endif
        o  	<< iteration << "\t"
            << objs[obj_distr] << "\t"
            << objs[obj_length] << "\t"
            << objs[obj_sqr_length] << "\t"
            << time(&tf) - t0 << "\t"
            << (clock() - x0)/CLOCKS_PER_SEC << "\t"
#ifdef CUDA_CODE
            << elapsedTime << "\t"
#endif
            << endl;
    }//writeStatistics


    //! HW 08/05/15 : for superpixel paper experiment
    void writeStatistics(int w, int h, double tCons, double tImpr, float cost, std::ostream& o) {
        o  	<< "matcherWid" << "\t"
            << "matcherHei" << "\t"
            << "timeCons" << "\t"
            << "timeImpr" << "\t"
            << "colorCost" << "\t"
            << endl;
        o  	<< w << "\t" << "\t"
            << h << "\t" << "\t"
            << tCons << "\t" << "\t"
            << tImpr << "\t" << "\t"
            << cost << "\t" << "\t"
            << endl;
    }//writeStatistics
    void writeStatistics(int w, int h, double tCons, double tImpr, float cost) {

        writeStatistics(w, h, tCons, tImpr, cost, *OutputStream);
    }

    void writeStatistics(int iteration) {

        writeStatistics(iteration, *OutputStream);
    }

    void closeStatistics() {
        OutputStream->close();
    }

};

#ifdef POPIP_COALTION
int gettimeofday(struct timeval * tp, struct timezone * tzp);
#else
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#endif
}//namespace components

#endif // TRACE_H

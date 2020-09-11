#include <cmath>
#include "Solution.h"
#include "Multiout.h"

#define TEST_CODE   0

#define round(x) ((fabs(ceil(x) - (x)) < fabs(floor(x) - (x))) ? ceil(x) : floor(x))

//template <typename T> std::string tostr(const T& t)
//{
//    std::ostringstream os; os << t; return os.str();
//}

std::ofstream* Solution::OutputStream = NULL;

void Solution::readSolution()
{
    readSolution(fileData);
}

void Solution::readSolution(const char* file)
{
    lout << "BEGIN READ " << file << std::endl;

    // Initialize NN netwoks CPU/GPU
    // Image read/write
    IRW irw;
    irw.read(fileData, mr, md, false);
    initialize(md, mr);
}

void Solution::writeSolution()
{
    writeSolution(fileSolution);
}

void Solution::writeSolution(const char* file)
{
    lout << "BEGIN WRITE " << file << std::endl;

    // Save solution
    //! Here since md and mr are with the same size and topology,
    //! the md.adaptiveMap is identical to the original mr.adaptiveMap

    mr.write(file);

    lout << "WRITTEN" << std::endl;
}

void Solution::openStatisticsFile()
{
    // Ouverture du fichier traitement en mode append
    OutputStream->open(fileStats, ios::app);
    if (!OutputStream->rdbuf()->is_open())
    {
        cerr << "Unable to open file " << fileStats << "CRITICAL ERROR" << endl;
        exit(-1);
    }
}

void Solution::closeStatisticsFile()
{
    OutputStream->close();
}

void Solution::initStatisticsFile()
{
    time(&t0);
    x0 = clock();
#ifdef CUDA_CODE
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    openStatisticsFile();
    writeHeaderStatistics(*OutputStream);
    closeStatisticsFile();
}

void Solution::writeHeaderStatistics(std::ostream& o)
{
    o << "iteration" << "\t"
      << "global_objectif" << "\t"
      << "duree(s)" << "\t"
      << "duree(s.xx)" << "\t"
#ifdef CUDA_CODE
      << "cuda_duree(ms)" << "\t"
#endif
      << "is_solution" << "\t";

    o << endl;

}

void Solution::writeStatistics(int iteration, std::ostream& o)
{
#ifdef CUDA_CODE
        // cuda timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
#endif

    o << iteration << "\t"
      << this->global_objectif << "\t"
      << time(&tf) - t0 << "\t"
      << (clock() - x0) / CLOCKS_PER_SEC << "\t"
#ifdef CUDA_CODE
      << elapsedTime << "\t"
#endif
      << this->isSolution() << "\t";

    o << endl << endl;
}

void Solution::writeStatisticsToFile(int iteration)
{
    openStatisticsFile();
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}




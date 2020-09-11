#ifndef INSTANCEGENERATOR_H
#define INSTANCEGENERATOR_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : janvier 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <QtCore>
#include <QDomDocument>

#include "config\ConfigParamsCF.h"
#include "geometry_prop.h"

#ifdef UTILISE_GEOMETRIE_CGAL
#include "geometry_cgal_boost.h"
#endif

#ifdef UTILISE_GEOMETRIE_CGAL
namespace geometry = geometry_cgal_boost;
#else
namespace geometry_p = geometry_prop;
#endif

// Declare point application
typedef geometry_p::Point_2 Point_2;
// Declare polygone application
typedef geometry_p::Polygon_2 Polygon_2;
// Declare un polygone boost
typedef geometry_p::Polygon_B Polygon_B;

/*! \class InstanceGenerator
 * \brief Generateur automatique d'instances.
 * Genere un ensemble d'instances a partir d'une description source.
*/
class InstanceGenerator {
    /*!
     * @name Fichiers
     * @{
     */
    char* fileData;
    char* fileSolution;
    char* fileStats;
    std::ofstream* OutputStream;
    //! @}

    /*!
     * @name Statistiques
     * @{
     */
    time_t t0, tf; // calcul de la duree
    double x0, xf; // time in ms via clock() / CLOCKS_PER_SEC
    //! @}

    /*!
     * @name Configuration
     * @{
     */
    config::ConfigParamsCF* params;
    //! @}

    /*!
     * @name Configuration
     * \brief Elements specifiques au generateur
     * @{
     */
    //! \brief Taux de saturation.
    double saturation;
    //! @}

    /*!
     * @name Documents XML
     * @{
     */
    //! docXml document XML source
    QDomDocument* docXml;
    //! docXml document XLM sortie
    QDomDocument* docXmlOutput;
    //! @}

    /*!
     * @name Tableaux XML
     * \brief Modeles, tableau de sortie.
     * @{
     */
    //!
    vector<int> inputModeles;
    vector<int> outputVector;
    //! @}

public:
    InstanceGenerator() : docXml(new QDomDocument()),
                          docXmlOutput(new QDomDocument()){}
    ~InstanceGenerator() { delete docXml;
                           delete docXmlOutput; }

    /*!
     * @name Controles
     * @{
     */
    void initialize(char* data, char* sol, char* stats, config::ConfigParamsCF* params);
    void initialize(char* data, char* sol, char* stats);
    void initialize();

    void init();
    void run(void);
    void generate(int no_test);
    //! @}

    /*!
     * @name Gestion XML
     * @{
     */
    void readDocXmlFromFile(char* fileName);
    void writeDocXmlToFile(char* file, QDomDocument* docXml);

    void parseDomXml(QDomDocument* docXml, vector<int>& inputModeles);
    void updateDomXml(QDomDocument* docXml, vector<int>& outputVector);
    void extractContour(std::string& std_str, Polygon_2& p);
    //! @}

    /*!
     * @name Statistiques de sortie
     * @{
     */
    void initStatistics();
    void initHeaderStatistics(std::ostream& o);
    void writeStatistics(int iteration);
    void writeStatistics(int iteration, std::ostream& o);
    void closeStatistics();
    //! @}

};


#endif // INSTANCEGENERATOR_H

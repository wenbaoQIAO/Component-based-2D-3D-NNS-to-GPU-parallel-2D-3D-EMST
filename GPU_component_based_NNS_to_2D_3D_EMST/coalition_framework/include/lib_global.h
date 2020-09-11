#ifndef LIB_GLOBAL_H
#define LIB_GLOBAL_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : mars 2014
 *
 ***************************************************************************
 */

#include <QtCore/qglobal.h>

//JCC 8/3/2018 modif pour compilation statique

#if defined(LIB_LIBRARY_DLL)
#if defined(LIB_LIBRARY)
#  define LIBSHARED_EXPORT Q_DECL_EXPORT
#else
#  define LIBSHARED_EXPORT Q_DECL_IMPORT
#endif
#else
#  define LIBSHARED_EXPORT
#endif
#endif // LIB_GLOBAL_H

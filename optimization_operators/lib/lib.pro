#-------------------------------------------------
#
# Project created by QtCreator 2014-05-17T18:44:07
#
#-------------------------------------------------

QT       += xml
QT       -= gui

CONFIG += console
CONFIG += exceptions rtti
CONFIG +=
#cuda

cuda:DEFINES += CUDA_CODE

INCLUDEPATH  += ../../../basic_components/include

#Si win32-msvc2010 ou win32-g++ (minGW)
win32 {
        win32-g++:QMAKE_CXXFLAGS += -msse2 -mfpmath=sse
        TARGET = ../../application/bin/libApplication
}

#Si linux-arm-gnueabi-g++ pour cross-compile vers linux et/ou raspberry pi
unix {
        CONFIG += shared
#static
        QMAKE_CXXFLAGS +=
        DEFINES += QT_ARCH_ARMV6
        TARGET = ../application/bin/libApplication
}

TEMPLATE = lib

arm {
    INCLUDEPATH  += ../include /home/pi/boost_1_55_0
}
else {
    INCLUDEPATH  += ../include C:/boost_1_55_0
}

#utile pour QT librairie export
DEFINES += LIB_LIBRARY

cgal:LIBS += -LD:\CGAL-4.3_NMAKE_RELEASE\lib -LC:\boost_1_55_0\lib32-msvc-10.0

SOURCES +=

HEADERS +=\
    ../include/random_generator.h \
    ../include/SomOperator.h \
    ../include/CellularMatrix.h \
    ../include/distance_functors.h \
    ../include/adaptator_basics.h \
    ../include/Evaluation.h \
    ../include/LocalSearch.h

cgal:HEADERS +=

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

#CUDA_SOURCES += ../src/main.cpp
#cuda_code.cu
######################################################
#
# For ubuntu, add environment variable into the project.
# Projects->Build Environment
# LD_LIBRARY_PATH = /usr/local/cuda/lib
#
######################################################

CUDA_FLOAT    = float
CUDA_ARCH     = -gencode arch=compute_20,code=sm_20

win32:{
  #Do'nt use the full path.
  #Because it is include the space character,
  #use the short name of path, it may be NVIDIA~1 or NVIDIA~2 (C:/Progra~1/NVIDIA~1/CUDA/v5.0),
  #or use the mklink to create link in Windows 7 (mklink /d c:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0").
#  CUDA_DIR      = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.0"
  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v5.0"
#  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  QMAKE_LIBDIR += $$CUDA_DIR/lib/Win32
  INCLUDEPATH  += $$CUDA_DIR/include D:\creput\AUT14\gpu_work\popip\basic_components\components\include $$CUDA_DIR/include C:/Qt/qt-everywhere-opensource-src-4.8.4/include C:/Qt/qt-everywhere-opensource-src-4.8.4/include/QtOpenGL ../include C:/boost_1_55_0

#$$QTDIR/include/QtOpenGL
#  LIBS         += -L$$CUDA_DIR/lib/x64 -lcuda -lcudart
  LIBS         += -lcuda -lcudart
# -L$$CUDA_DIR/lib/Win32

# Add the necessary libraries
#  CUDA_LIBS = cuda cudart
#  NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
  QMAKE_LFLAGS_DEBUG    = /DEBUG /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
#  QMAKE_LFLAGS_RELEASE  =         /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
}
unix:{

  INCLUDEPATH  += ../include /usr/local/Trolltech/Qt-4.8.6/include
  INCLUDEPATH  += /home/abdo/boost_1_57_0

  CUDA_DIR      = /usr/local/cuda-7.0
  QMAKE_LIBDIR += $$CUDA_DIR/lib64
  INCLUDEPATH  += $$CUDA_DIR/include
  LIBS += -lcudart -lcuda
  QMAKE_CXXFLAGS += -std=c++0x
  INCLUDEPATH  += $$CUDA_DIR/samples/common/inc
}

DEFINES += "CUDA_FLOAT=$${CUDA_FLOAT}"

NVCC_OPTIONS = --use_fast_math -DCUDA_FLOAT=$${CUDA_FLOAT}
cuda:NVCC_OPTIONS += -DCUDA_CODE
#NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

QMAKE_EXTRA_COMPILERS += cuda

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CONFIG(release, debug|release) {
  OBJECTS_DIR = ./release
  cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}
CONFIG(debug, debug|release) {
  OBJECTS_DIR = ./debug
  cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -D_DEBUG -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}

#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o

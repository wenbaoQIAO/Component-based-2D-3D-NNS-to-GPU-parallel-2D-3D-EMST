#-------------------------------------------------
#
# Project created by QtCreator 2014-05-17T18:44:07
#
#-------------------------------------------------

QT       += xml
QT       -= gui

CONFIG += console
CONFIG += exceptions rtti
CONFIG += bit64
CONFIG += c++11
#CONFIG += static
CONFIG += staticlib

CONFIG += topo_hexa
CONFIG += cuda
#CONFIG += separate_compilation

topo_hexa:DEFINES += TOPOLOGIE_HEXA
separate_compilation:DEFINES += SEPARATE_COMPILATION
cuda:DEFINES += CUDA_CODE
#DEFINES += POPIP_COALITION
#DEFINES += LIB_LIBRARY_DLL

#Sans Shadow Build
#OUT_PATH=../../bin/libCalculateur
#Shadow Build au niveau qui précède
OUT_PATH=../../bin/libCalculateur

c++11:DEFINES += _USE_MATH_DEFINES

#Si win32-msvc2010 ou win32-g++ (minGW)
win32 {
        win32-g++:QMAKE_CXXFLAGS += -msse2 -mfpmath=sse
        TARGET = $$OUT_PATH
}

#Si linux-arm-gnueabi-g++ pour cross-compile vers linux et/ou raspberry pi
unix {
        CONFIG += shared
#static
        QMAKE_CXXFLAGS +=
#-mfpu=vfp
#-mfloat-abi=hard
        DEFINES += QT_ARCH_ARMV6
        TARGET = $$OUT_PATH
}

TEMPLATE = lib

#utile pour QT librairie export
#DEFINES += LIB_LIBRARY

SOURCES += \
    ../src/AgentMetaSolver.cpp \
    ../src/geometry_prop.cpp \
    ../src/Profiler.cpp \
    ../src/random_generator.cpp \
    ../src/config/ConfigParamsCF.cpp \
    ../../basic_components/src/ConfigParams.cpp \
    ../src/Multiout.cpp

HEADERS +=\
    ../include/lib_global.h \
    ../include/Threads.h \
    ../include/LocalSearch.h \
    ../include/LocalSearch.inl \
    ../include/AgentMetaSolver.h \
    ../include/AgentMetaSolver.inl \
    ../include/ConfigParams.h \
    ../include/config/chameleon.h \
    ../include/Profiler.h \
    ../include/Solution.h \
    ../include/Calculateur.h \
    ../include/geometry_prop.h \
    ../include/config/ConfigParamsCF.h \
    ../include/random_generator_cf.h \
    ../../basic_components/include/ConfigParams.h \
    ../include/Multiout.h

cgal:HEADERS +=

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

OTHER_FILES += \
    ../src/Calculateur.cu

CUDA_SOURCES += \
    ../src/Calculateur.cu

separate_compilation {
OTHER_FILES +=\
    ../src/Solution.cu \
    ../src/SolutionRW.cu \
    ../src/SolutionOperators.cu
CUDA_SOURCES +=\
    ../src/Solution.cu \
    ../src/SolutionRW.cu \
    ../src/SolutionOperators.cu
}
######################################################
#
# For ubuntu, add environment variable into the project.
# Projects->Build Environment
# LD_LIBRARY_PATH = /usr/local/cuda/lib
#
######################################################

CUDA_FLOAT    = float
CUDA_ARCH     = -gencode arch=compute_61,code=sm_61
#CUDA_ARCH     = -gencode arch=compute_20,code=sm_20
#CUDA_ARCH     = -gencode arch=compute_12,code=sm_12

win32:{

  CUDA_DIR = "C:/Progra~1/NVIDIA~2/CUDA/v9.1"
  QTDIR = C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
  BOOST_PATH=C:/boost_1_66_0/

  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  INCLUDEPATH  += ../include
  INCLUDEPATH  += $$BOOST_PATH
  INCLUDEPATH  += $$CUDA_DIR/include
  INCLUDEPATH  += $$QTDIR/include $$QTDIR/include/QtCore  $$QTDIR/include/QtXml
  INCLUDEPATH  += ../../optimization_operators/include
  INCLUDEPATH  += ../../basic_components/include
  INCLUDEPATH  += "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.1\common\inc"
  LIBS         +=  -lcuda  -lcudart

# Add the necessary libraries
#  CUDA_LIBS = cuda cudart
#  NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
#  QMAKE_LFLAGS_DEBUG    = /DEBUG /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
#QMAKE_LFLAGS_RELEASE  =
#        /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
}
DEFINES += "CUDA_FLOAT=$${CUDA_FLOAT}"

cuda:NVCC_OPTIONS += -DCUDA_CODE
#-DPOPIP_COALITION
separate_compilation {
cuda:NVCC_OPTIONS += -DSEPARATE_COMPILATION
}
NVCC_OPTIONS += --use_fast_math -DCUDA_FLOAT=$${CUDA_FLOAT}
#cuda:NVCC_OPTIONS += --x cu
#cuda:NVCC_OPTIONS += --dc --x cu
#NVCCFLAGS     = -static
#--compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

NVCCFLAG_COMMON = $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH

QMAKE_EXTRA_COMPILERS += cudaIntr

CONFIG(release, debug|release) {
  OBJECTS_DIR = ./release
bit64:cudaIntr.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
else:cudaIntr.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}
CONFIG(debug, debug|release) {
  OBJECTS_DIR = ./debug
bit64:cudaIntr.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
else:cudaIntr.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}

#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o

## Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cudaLnk
# Cuda linker (required for dynamic parallelism)
# Prepare the linking compiler step
separate_compilation {
CUDA_OBJ = $${OBJECTS_DIR}/Calculateur_cuda.o \
$${OBJECTS_DIR}/Solution_cuda.o \
$${OBJECTS_DIR}/SolutionRW_cuda.o
#$${OBJECTS_DIR}/Trace_cuda.o
}
else {
CUDA_OBJ = $${OBJECTS_DIR}/Calculateur_cuda.o
#$${OBJECTS_DIR}/Trace_cuda.o
}

cudaLnk.CONFIG += combine	# Generate 1 output file

OBJECTS_DIR = ./release

CONFIG(release, debug|release) {

separate_compilation {
bit64:cudaLnk.commands = $$CUDA_DIR/bin/nvcc $$NVCCFLAG_COMMON \
                -dlink $${OBJECTS_DIR}/Calculateur_cuda.o \
    $${OBJECTS_DIR}/Solution_cuda.o \
    $${OBJECTS_DIR}/SolutionRW_cuda.o -o $${OBJECTS_DIR}/link.o
}
else {
bit64:cudaLnk.commands = $$CUDA_DIR/bin/nvcc $$NVCCFLAG_COMMON \
                -dlink $${OBJECTS_DIR}/Calculateur_cuda.o \
                -o $${OBJECTS_DIR}/link.o
}
}
#cudaLnk.dependency_type = TYPE_C
cudaLnk.depend_command = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cudaLnk.depends = $$CUDA_OBJ
cudaLnk.input = CUDA_SOURCES $${OBJECTS_DIR}/Calculateur_cuda.o
cudaLnk.output = $${OBJECTS_DIR}/link.o


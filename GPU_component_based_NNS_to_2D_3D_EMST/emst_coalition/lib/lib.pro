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

#CONFIG += topo_hexa
CONFIG += cuda
#CONFIG += separate_compilation

topo_hexa:DEFINES += TOPOLOGIE_HEXA
separate_compilation:DEFINES += SEPARATE_COMPILATION
cuda:DEFINES += CUDA_CODE
cuda:DEFINES += CUDA_ATOMIC
#DEFINES += POPIP_COALITION

#Shadow Build au niveau qui précède
OUT_PATH=../../bin/libCalculateurEMST

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
        TARGET = ../../bin/libCalculateurEMST
}

TEMPLATE = lib

#utile pour QT librairie export
#DEFINES += LIB_LIBRARY

SOURCES +=

HEADERS +=\
    ../include/SolutionEMST.h \
    ../include/CalculateurEMST.h \
    ../include/NeuralNetEMST.h \
    ../include/EMSTOperators.h \
    ../include/BufferLink.h \
    ../../basic_components/include/GridOfNodes.h \
    ../include/ConverterNetLink.h \
    ../include/EMSTFunctors.h \
    ../include/NodeEMST.h \
    ../../basic_components/include/Node.h \
    ../include/InputRW.h \
    ../../basic_components/include/Cell.h \
    ../../optimization_operators/include/CellularMatrix.h \
    ../../basic_components/include/ViewGrid.h \
    ../../basic_components/include/Objectives.h \
    ../../basic_components/include/macros_cuda.h

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

OTHER_FILES +=

CUDA_SOURCES += \
    ../src/CalculateurEMST.cu

separate_compilation {
OTHER_FILES +=\
    ../src/SolutionEMST.cu \
    ../src/SolutionEMSTRW.cu \
    ../src/SolutionEMSTOperators.cu
CUDA_SOURCES +=\
    ../src/SolutionEMST.cu \
    ../src/SolutionEMSTRW.cu \
    ../src/SolutionEMSTOperators.cu
}

CUDA_FLOAT    = float
#CUDA_ARCH     = -gencode arch=compute_50,code=sm_50
CUDA_ARCH     = -gencode arch=compute_61,code=sm_61
#CUDA_ARCH     = -gencode arch=compute_12,code=sm_12

win32:{

  #Do'nt use the full path.
  #Because it is include the space character,
  #use the short name of path, it may be NVIDIA~1 or NVIDIA~2 (C:/Progra~1/NVIDIA~1/CUDA/v5.0),
  #or use the mklink to create link in Windows 7 (mklink /d c:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0").
#  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v9.1"
#  QTDIR = C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
#  BOOST_PATH=C:/boost_1_66_0/

#For wenbaoQiao computer configuration
  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v10.1"#"C:/Progra~1/NVIDIA~2/CUDA/v9.1"
  QTDIR = C:\QT5.12_static #C:\QT5.10_static #C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
  BOOST_PATH= C:/boost_1_66_0/ #C:/boost_1_59_0

  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  INCLUDEPATH  += ../include
  INCLUDEPATH  += $$BOOST_PATH
  INCLUDEPATH  += $$CUDA_DIR/include
  INCLUDEPATH  += $$QTDIR/include $$QTDIR/include/QtCore  $$QTDIR/include/QtXml
  INCLUDEPATH  += ../../basic_components/include
  INCLUDEPATH  += ../../optimization_operators/include
  INCLUDEPATH  += ../../coalition_framework/include
  INCLUDEPATH  += "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc"
  LIBS         +=  -lcuda  -lcudart
}

#DEFINES += "CUDA_FLOAT=$${CUDA_FLOAT}"

cuda:NVCC_OPTIONS += -DCUDA_CODE -DCUDA_ATOMIC
separate_compilation {
cuda:NVCC_OPTIONS += -DSEPARATE_COMPILATION
}

NVCC_OPTIONS += --use_fast_math
#-DCUDA_FLOAT=$${CUDA_FLOAT}

#--fmad false
#-DCUDA_FLOAT=$${CUDA_FLOAT}
#cuda:NVCC_OPTIONS += --x cu
#cuda:NVCC_OPTIONS += --dc --x cu
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

# Tell Qt that we want add more stuff to the Makefile
#QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr




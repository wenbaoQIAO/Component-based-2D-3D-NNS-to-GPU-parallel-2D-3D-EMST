QT       += xml
QT       += gui
TEMPLATE = app
CONFIG += console
CONFIG += bit64
CONFIG += c++11
CONFIG += topo_hexa
CONFIG += cuda
#CONFIG += static

topo_hexa:DEFINES += TOPOLOGIE_HEXA
cuda:DEFINES += CUDA_CODE
cuda:DEFINES += CUDA_ATOMIC

unix {
        CONFIG +=
#static
        DEFINES += QT_ARCH_ARMV6
        TARGET = bin/application
}
win32 {
        TARGET = ../../bin/application
}

SOURCES +=
#../src/main.cpp

OTHER_FILES += src/main.cu

#LIBS += -L$$PWD/../bin/
#-llibOperators

CUDA_SOURCES += src/main.cu
#cuda_code.cu

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

#  LIBS_COMPONENTS_DIR = "D:/creput/AUT14/gpu_work/popip/basic_components/components/bin"
#  LIBS_OPERATORS_DIR = "D:/creput/AUT14/gpu_work/popip/optimization_operators/operators/bin"

  #Do'nt use the full path.
  #Because it is include the space character,
  #use the short name of path, it may be NVIDIA~1 or NVIDIA~2 (C:/Progra~1/NVIDIA~1/CUDA/v5.0),
  #or use the mklink to create link in Windows 7 (mklink /d c:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0").
#  CUDA_DIR      = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.0"
#  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v5.0"
#  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v8.0"
  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v9.1"
#  QTDIR = C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
  QTDIR = C:\Qt\Qt5.9.1\5.9.1\msvc2015_64
  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  INCLUDEPATH  += include ../include
  INCLUDEPATH  += $$CUDA_DIR/include
  INCLUDEPATH  += $$QTDIR/include
  INCLUDEPATH  += C:/boost_1_66_0
  INCLUDEPATH  += ../../optimization_operators/include
  INCLUDEPATH  += ../../basic_components/include
  INCLUDEPATH  += "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.1\common\inc"
#  QMAKE_LIBDIR += $$CUDA_DIR/lib/Win32 $$LIBS_OPERATORS_DIR $$LIBS_COMPONENTS_DIR
#  INCLUDEPATH  += $$CUDA_DIR/include C:\QT\qt-everywhere-opensource-src-4.8.4\include C:\QT\qt-everywhere-opensource-src-4.8.4\include\QtCore C:\QT\qt-everywhere-opensource-src-4.8.4\include\QtXml C:\QT\qt-everywhere-opensource-src-4.8.4\include\QtOpenGL  C:/boost_1_55_0
#  INCLUDEPATH  += include
#  INCLUDEPATH  += D:\creput\AUT14\gpu_work\popip\optimization_operators\include
#  INCLUDEPATH  += D:\creput\AUT14\gpu_work\popip\basic_components\include
#  INCLUDEPATH  += D:\creput\AUT14\gpu_work\popip\adaptive_meshing\include
#$$QTDIR/include/QtOpenGL
#  LIBS         += -L$$CUDA_DIR/lib/x64 -lcuda -lcudart
  LIBS         +=  -lcuda  -lcudart
#-LC:\Qt\qt-everywhere-opensource-src-4.8.4\lib -lQtGui -LC:\Qt\Qt5.9.1\5.9.1\Qt5.9_static\lib
#-llibOperators
# -L$$CUDA_DIR/lib/Win32

# Add the necessary libraries
#  CUDA_LIBS = cuda cudart
#  NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
#  QMAKE_LFLAGS_DEBUG    = /DEBUG /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
#QMAKE_LFLAGS_RELEASE  =
#        /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
}
unix:{

  ##############################################################################
  # Here to add the specific QT and BOOST paths according to your Linux system.
  # For H.W's system
  INCLUDEPATH  += ../include
  INCLUDEPATH  += /usr/local/Trolltech/Qt-4.8.6/include
  INCLUDEPATH  += /home/abdo/boost_1_57_0
  CUDA_DIR      = /usr/local/cuda-7.0
  QMAKE_LIBDIR += $$CUDA_DIR/lib64
  INCLUDEPATH  += $$CUDA_DIR/include
  LIBS += -lcudart -lcuda
  QMAKE_CXXFLAGS += -std=c++0x
  INCLUDEPATH  += ../optimization_operators/include
  INCLUDEPATH  += ../basic_components/include
  INCLUDEPATH  += ../../adaptive_meshing/include
  INCLUDEPATH  += $$CUDA_DIR/samples/common/inc
}

DEFINES += "CUDA_FLOAT=$${CUDA_FLOAT}"

NVCC_OPTIONS = --use_fast_math -DCUDA_FLOAT=$${CUDA_FLOAT}
cuda:NVCC_OPTIONS += -DCUDA_CODE
NVCCFLAGS     = -static
#--compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

QMAKE_EXTRA_COMPILERS += cuda

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CONFIG(release, debug|release) {
  OBJECTS_DIR = ./release
bit64:cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
else:cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}
CONFIG(debug, debug|release) {
  OBJECTS_DIR = ./debug
bit64:cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
else:cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}

#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o

HEADERS += \
    include/TestCellular.h \
    include/TestSom.h \
    include/TestSomTSP.h

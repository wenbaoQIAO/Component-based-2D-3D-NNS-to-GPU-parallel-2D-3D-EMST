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
DEFINES += POPIP_COALITION

#Sans Shadow Build
#OUT_PATH=../../bin/libCalculateur
#Shadow Build au niveau qui précède
OUT_PATH=../../bin/libCoalition

#Choix de configuration
# Pour utiliser bibliothèque CGAL
# (variable d'environneement QMAKESPEC=win32-msvc2010 uniquement)
# faire CONFIG += cgal
# Pour la cross-compilation de windows vers raspberry pi
# mettre QMAKESPEC=linux-arm-gnueabi-g++ ou bien mettre
# chemin vers mkspec dans le qmake ou le Kit (si QTCreator), exemple :
# qmake -spec C:\Qt..\mkspecs\linux-arm-gnueabi-g++
# Pour compilation directe sur support raspberry pi
# faire CONFIG += arm

#cgal:DEFINES += UTILISE_GEOMETRIE_CGAL
c++11:DEFINES += _USE_MATH_DEFINES

#Si win32-msvc2010 ou win32-g++ (minGW)
win32 {
        win32-g++:QMAKE_CXXFLAGS += -msse2 -mfpmath=sse
#        win32-g++:DEFINES += DOUBLE_PRECISION_ROUNDING
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

#arm {
#    INCLUDEPATH  += ../include /home/pi/boost_1_55_0
#}
#else {
#    INCLUDEPATH  += ../include $$BOOST_PATH
#cgal:INCLUDEPATH  += D:/CGAL-4.3_NMAKE_RELEASE/include D:/CGAL-4.3/include
#}

#utile pour QT librairie export
DEFINES += LIB_LIBRARY

#win32:LIBS += -LC:\Tools\boost_1_55_0MSVC\lib32-msvc-10.0 -llibboost_thread-vc100-mt-1_55

cgal:LIBS += -LD:\CGAL-4.3_NMAKE_RELEASE\lib -LC:\boost_1_55_0\lib32-msvc-10.0

SOURCES += \
    ../src/AgentMetaSolver.cpp \
    ../src/geometry_prop.cpp \
    ../src/config/ConfigParamsCF.cpp \
    ../../basic_components/src/ConfigParams.cpp

HEADERS +=\
    ../include/Threads.h \
    ../include/LocalSearch.h \
    ../include/LocalSearch.inl \
    ../include/AgentMetaSolver.h \
    ../include/AgentMetaSolver.inl \
    ../include/ConfigParams.h \
    ../include/config/chameleon.h \
    ../include/Solution.h \
    ../include/Calculateur.h \
    ../include/geometry_prop.h \
    ../include/config/ConfigParamsCF.h \
    ../include/random_generator_cf.h \
    ../../basic_components/include/ConfigParams.h

cgal:HEADERS +=

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

OTHER_FILES +=

CUDA_SOURCES +=

separate_compilation {
OTHER_FILES +=
CUDA_SOURCES +=
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
  QTDIR = C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
  BOOST_PATH=C:/boost_1_66_0/

#  QTDIR = C:\Qt\Qt5.9.1\5.9.1\msvc2015_64
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

cuda:NVCC_OPTIONS += -DCUDA_CODE -DPOPIP_COALITION
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

#DISTFILES += \
#    ../src/Solution.cu

#DISTFILES += \
#    ../src/Calculateur.cu \
#    ../src/SolutionOperateurs.cu \
#    ../src/SolutionRW.cu \
#    ../../basic_components/src/Trace.cu

#    cuda.input    = CUDA_SOURCES
#     cuda.output   = ${QMAKE_FILE_BASE}_cuda.o
#     cuda.commands = $$CUDA_DIR/bin/nvcc.exe -dlink $$NVCC_OPTIONS $$CUDA_INC $$LIBSCUDA \
#                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
#                    --compile -cudart static -DWIN32 -D_MBCS \
#                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
#                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
#                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#     cuda.dependency_type = TYPE_C
#     QMAKE_EXTRA_COMPILERS += cuda


##------------------------- Cuda intermediat compiler
## Prepare intermediat cuda compiler
#cudaIntr.input = CUDA_SOURCES
##cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
#cudaIntr.output = ${QMAKE_FILE_BASE}_cuda.o

#cudaIntr.commands = 	$CUDA_DIR/bin/nvcc $NVCCFLAG_COMMON \
#			-dc $NVCCFLAGS $CUDA_INC $LIBS \
#			${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

##Set our variable out. These obj files need to be used to create the link obj file and used in our final gcc compilation
#cudaIntr.variable_out = CUDA_OBJ
#cudaIntr.variable_out += OBJECTS

# Tell Qt that we want add more stuff to the Makefile
#QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr

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
# Tweak arch according to your hw's compute capability
#bit64:cudaLnk.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 64 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

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
#cudaLnk.input = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
cudaLnk.depends = $$CUDA_OBJ
cudaLnk.input = CUDA_SOURCES $${OBJECTS_DIR}/Calculateur_cuda.o
# \
#$${OBJECTS_DIR}/Solution_cuda.o \
#$${OBJECTS_DIR}/SolutionRW_cuda.o \
#$${OBJECTS_DIR}/Trace_cuda.o
cudaLnk.output = $${OBJECTS_DIR}/link.o
#cudaLnk.input = CUDA_SOURCES
#cudaLnk.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o


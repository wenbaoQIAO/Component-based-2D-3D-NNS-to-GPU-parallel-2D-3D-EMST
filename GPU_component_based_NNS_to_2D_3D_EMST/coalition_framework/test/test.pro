QT       += xml
QT       += gui
TEMPLATE = app
CONFIG += console
CONFIG += exceptions
#CONFIG += static

CONFIG(debug, debug|release) {
    OUT_PATH=../../debug86/calculateur
} else {
#Shadow Build dans le rep module
    OUT_PATH=../../bin/calculateur
}

unix {
        CONFIG +=
#static
        DEFINES += QT_ARCH_ARMV6
        TARGET = $$OUT_PATH
}
win32 {
        TARGET = $$OUT_PATH
}

#DEFINES += POPIP_COALITION
#DEFINES += LIB_LIBRARY_DLL

SOURCES += ../src/main.cpp

BOOST_PATH=C:/boost_1_66_0/

INCLUDEPATH  += ../include
INCLUDEPATH  += ../../optimization_operators/include
INCLUDEPATH  += ../../basic_components/include
INCLUDEPATH  += $$BOOST_PATH

CONFIG(debug, debug|release) {
    LIBS += -L$$PWD/../bin/ -llibCalculateur
} else {
    LIBS += -L$$PWD/../bin/ -llibCalculateur
}

  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v9.1"
  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  LIBS         +=  -lcuda  -lcudart

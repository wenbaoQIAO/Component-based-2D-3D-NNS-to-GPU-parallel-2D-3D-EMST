QT       += xml
QT       += gui
TEMPLATE = app
CONFIG += console
CONFIG += exceptions
#CONFIG += static

CONFIG(debug, debug|release) {
#    OUT_PATH=bin/imbricateur
    OUT_PATH=../../debug86/calculateur
} else {
#    OUT_PATH=../../release86/imbricateur
#Shadow Build au niveau qui precede "imbricateur"
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

SOURCES += ../src/main.cpp

BOOST_PATH= C:/boost_1_66_0/

#BOOST_PATH= C:/boost_1_59_0

INCLUDEPATH  += ../include
INCLUDEPATH  += ../../basic_components/include
INCLUDEPATH  += ../../optimization_operators/include
INCLUDEPATH  += ../../coalition_framework/include
INCLUDEPATH  += $$BOOST_PATH

CONFIG(debug, debug|release) {
    LIBS += -L$$PWD/../bin/ -llibCalculateurEMST
} else {
    LIBS += -L$$PWD/../../coalition_framework/bin/ -llibCoalition
    LIBS += -L$$PWD/../bin/ -llibCalculateurEMST
}

CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v10.1"
QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
LIBS         +=  -lcuda  -lcudart

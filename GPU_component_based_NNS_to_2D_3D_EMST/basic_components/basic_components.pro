QT       += xml
QT       += gui
TEMPLATE = app
CONFIG += console
CONFIG += bit64
CONFIG += c++11
CONFIG += topo_hexa
CONFIG += cuda
CONFIG += static

topo_hexa:DEFINES += TOPOLOGIE_HEXA
cuda:DEFINES += CUDA_CODE

HEADERS += \
    ../basic_components/include/basic_operations.h \
    ../basic_components/include/Cell.h \
    ../basic_components/include/CellularMatrix.h \
    ../basic_components/include/ConfigParams.h \
    ../basic_components/include/Converter.h \
    ../basic_components/include/Converter_new.h \
    ../basic_components/include/distances_matching.h \
    ../basic_components/include/filters.h \
    ../basic_components/include/geometry.h \
    ../basic_components/include/GridOfNodes.h \
    ../basic_components/include/GridPatch.h \
    ../basic_components/include/ImageRW.h \
    ../basic_components/include/lib_global.h \
    ../basic_components/include/macros_cuda.h \
    ../basic_components/include/NeuralNet.h \
    ../basic_components/include/NIter.h \
    ../basic_components/include/NIter1D.h \
    ../basic_components/include/NIterHexa.h \
    ../basic_components/include/Node.h \
    ../basic_components/include/Node_new.h \
    ../basic_components/include/Objectives.h \
    ../basic_components/include/random_generator.h \
    ../basic_components/include/SpiralSearch.h \
    ../basic_components/include/Trace.h \
    ../basic_components/include/ViewGrid.h \
    ../basic_components/include/ViewGridHexa.h

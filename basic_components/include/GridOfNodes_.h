#ifndef MULTI_GRID_H
#define MULTI_GRID_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, A. Mansouri, W. Qiao
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
#include "macros_cuda.h"
#include "Node.h"
#include "Objectives.h"

#include <boost/multi_array.hpp>
typedef   boost::multi_array<double, 2> array;

#define TEST_CODE 0
#define TEST_PITCH 1 // in that case STRIDE_ALIGNMENT=64 (GPU side)
#define INTERPOLATION 1

#define MIXTE_CPU_GPU_OBJECT 1

// Choix fonction acces
#define MG_ACCESS_ITERATIF  1
#define MG_ACCESS_SCALAR_PRODUCT  0

using namespace std;

namespace components
{

/*! @name Allocateurs CPU/GPU.
 * \brief Le choix est effectue par template.
 */
//! @{
//! Allocateur local CPU/DEVICE
template <class Node>
struct Allocator2DLocal {

    DEVICE_HOST Node* allocMem(size_t width, size_t height, size_t depth, size_t& pitch) {

        Node* _data;

        pitch = width * sizeof(Node);

        _data = new Node[width * height * depth];

        return _data;
    }

    DEVICE_HOST Node* allocMem(size_t width, size_t height, size_t& pitch) {

        Node* _data;

        pitch = width * sizeof(Node);

        _data = new Node[width * height];

        return _data;
    }

    DEVICE_HOST void freeMem(Node* _data) {

        if (_data != NULL) {
            delete [] _data;
            _data = NULL;
        }
    }
};

//! Allocateur GPU

template <class Node>
struct Allocator2DGPU {

    Node* allocMem(size_t width, size_t height, size_t depth, size_t& pitch) {

            Node* _data;
    #ifdef CUDA_CODE
            cudaMallocPitch(
                        (void**)&_data,
                        &pitch,
                        sizeof(Node) * width,
                        height*depth);
    #else
            _data = Allocator2DLocal<Node>().allocMem(width, height, pitch);
    #endif
            return _data;
        }

Node* allocMem(size_t width, size_t height, size_t& pitch) {

        Node* _data;
#ifdef CUDA_CODE
        cudaMallocPitch(
                    (void**)&_data,
                    &pitch,
                    sizeof(Node) * width,
                    height);
#else
        _data = Allocator2DLocal<Node>().allocMem(width, height, pitch);
#endif
        return _data;
    }

    void freeMem(Node* _data) {
#ifdef CUDA_CODE
        if (_data != NULL) {
            cudaFree(_data);
            _data = NULL;
        }
#else
        if (_data)
            delete [] _data;
#endif
    }

};

//template <class Grid, class Node>
template<
        template<typename> class Grid,
        class Node
        >
KERNEL void K_resetValue(Grid<Node> g, Node node)
{
    KER_SCHED(g.getWidth(), g.getHeight())

    if (_x < g.getWidth() && _y < g.getHeight())
    {
        g[_y][_x] = node;
    }

    END_KER_SCHED

    SYNCTHREADS
}//K_resetValue

//! @}

template <std::size_t Dim>
struct IterIndex {
    template <typename index_type2>
    DEVICE_HOST void reset(index_type2&  idx) {
        for (int i = 1; i < Dim; ++i)
            idx[i] = 0;
        idx[0] = -1;
    }

    template <typename index_type2>
    DEVICE_HOST bool next(index_type2& idx, index_type2& extents) {
        IterIndex<Dim-1> idx1;
        if (idx1.next(idx, extents))
            return true;
        else {
            if (idx[Dim-1] < extents[Dim-1] - 1) {
                idx[Dim-1]++;
                idx1.reset(idx);
                idx1.next(idx, extents);
                return true;
            }
            else return false;
        }
    }
};

template <>
struct IterIndex<1> {
    template <typename index_type2>
    DEVICE_HOST void reset(index_type2&  idx) {
        idx[0] = -1;
    }

    template <typename index_type2>
    DEVICE_HOST bool next(index_type2& idx, index_type2& extents) {
        if (idx[0] < extents[0] - 1) {
            idx[0]++;
            return true;
        }
        else
            return false;
    }
};

/*!
 * \defgroup Grilles de nodes
 * \brief Espace de nommage components
 * Il comporte les multi-grilles (grids)
 */
/*! @{*/

/*! \brief Classe definissant la structure d'une grille de "nodes")
 *
 */
template <class Node, std::size_t Dim>
struct MultiGrid
{
protected:

    Node* _data;

    typedef Allocator2DLocal<Node> Alloc;
    typedef Allocator2DGPU<Node> GPUAlloc;

    Alloc alloc;
    GPUAlloc gpu_alloc;

public:

    typedef Index<Dim> index_type;
    typedef index_type extents_type;
    typedef typename index_type::coord_type coord_type;
    typedef IterIndex<Dim> iterator_type;

    // Dimensions de la grille
    extents_type extents;   // extends are dimensions lengths
    extents_type extents_pitch;   // extends are dimensions lengths with extents_pitch[0] = pitch
    extents_type extents_strides;   // strides are multiplicative factors with strides[0]=1,
                            // stride{1]=pitch, stride[2]=extents[1]*pitch,
                            // stride[3]=extents[2]*extents[1]*pitch, etc..

    // Dimensions pour Dim <= 3
    size_t length_in_bytes; // length in bytes
    size_t length; // length in nodes units
    size_t width;
    size_t height;
    size_t depth;
    size_t others; // product of other dimensions >= 4
    size_t pitch; // counted in bytes

    //! Default constructor is private (do not access)
    DEVICE_HOST explicit MultiGrid() :
        width(0),
        height(0),
        depth(0),
        pitch(0),
        _data(NULL) {}

    DEVICE_HOST explicit MultiGrid(int w, int h) :
        width(w),
        height(h),
        depth(1),
        others(1)
    {
        _data = alloc.allocMem(width, height, pitch);
    }

    void clone(MultiGrid& grid) {
        grid.resize(width, height);
        grid.assign(*this);
    }

    DEVICE_HOST inline size_t getDimension() { return Dim; }

    DEVICE_HOST void gpuClone(MultiGrid& grid) {
        grid.gpuResize(width, height);
        gpuCopyDeviceToDevice(grid);
    }

    inline void setIdentical(MultiGrid& grid) {
        grid.assign(*this);
    }

    DEVICE_HOST inline void gpuSetIdentical(MultiGrid& grid) {
        gpuCopyDeviceToDevice(grid);
    }

    DEVICE_HOST void iterInit() {
    }

    DEVICE_HOST void iterInit(index_type& idx) {
        iterator_type().reset(idx);
    }

    DEVICE_HOST bool iterNext(index_type& idx) {

        return iterator_type().next(idx, extents);
    }

    DEVICE_HOST Node* getData() { return _data; }

    DEVICE_HOST size_t getWidth() const {
        return width;
    }

    DEVICE_HOST size_t getHeight() const {
        return height;
    }

    DEVICE_HOST size_t getDepth() const {
        return depth;
    }

    // QWB add cpu version only
    size_t getWidth_cpu() const {
        return width;
    }

    // QWB add cpu version only
    size_t getHeight_cpu() const {
        return height;
    }

    DEVICE_HOST size_t getPitch() {
        return pitch;
    }

    DEVICE_HOST Node& operator()(index_type& idx, extents_type& str) const {
        size_t offset = 0;
#if MG_ACCESS_ITERATIF
        {
          size_t n = 1;
          while (n != Dim) {
            offset += idx[n] * str[n];
            ++n;
          }
        }
#endif
#if MG_ACCESS_SCALAR_PRODUCT
        if (Dim >= 2)
            offset += (size_t)idx[1] * (size_t)str[1];
        if (Dim >= 3)
            offset += (size_t)idx[2] * (size_t)str[2];
        if (Dim >= 4) {
            size_t n = 3;
            while (n != Dim) {
              offset += (size_t)idx[n] * (size_t)str[n];
              ++n;
            }
          }
#endif
        return *((Node*)((char*)_data + offset) + (size_t)idx[0]);
    }

    template <typename index_type, typename extents_type>
    DEVICE_HOST Node* compute_address(index_type& idx, extents_type& str) const {
        size_t offset = 0;
#if MG_ACCESS_ITERATIF
        {
          size_t n = 1;
          while (n != Dim) {
            offset += idx[n] * str[n];
            ++n;
          }
        }
#endif
#if MG_ACCESS_SCALAR_PRODUCT
        if (Dim >= 2)
            offset += (size_t)idx[1] * (size_t)str[1];
        if (Dim >= 3)
            offset += (size_t)idx[2] * (size_t)str[2];
        if (Dim >= 4) {
            size_t n = 3;
            while (n != Dim) {
              offset += (size_t)idx[n] * (size_t)str[n];
              ++n;
            }
          }
#endif
        return (Node*)((char*)_data + offset) + idx[0];
    }

    template <typename StrideList, typename ExtentList>
    DEVICE_HOST void compute_strides(StrideList& stride_list, ExtentList& extent_list)
    {
      // invariant: stride = the stride for dimension n
      size_t stride = 1;
      for (size_t n = 0; n != Dim; ++n) {
        // The stride for this dimension is the product of the
        // lengths of the ranks minor to it.
        stride_list[n] = stride;

        stride *= extent_list[n];
      }
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline Node* operator[](std::size_t y) {
        return ((Node*)((char*)_data + y * pitch));
    }

    /*! @name Globales functions specific for controling the GPU.
     * \brief Memory allocation and communication. Useful
     * for mixte utilisation.
     * @{
     */

    DEVICE_HOST void allocMem() {
        _data = alloc.allocMem(width, height, pitch);
    }

    DEVICE_HOST void freeMem() {
        alloc.freeMem(_data);
    }

    DEVICE_HOST void resize(int w, int h) {
        extents_type exts;
        exts[0] = w;
        exts[1] = h;
        resize(exts);
    }

    DEVICE_HOST void resize(extents_type& exts) {
        alloc.freeMem(_data);

        // Extents
        extents = exts;
        if (Dim >= 1) {
            width = exts[0];
            height = 1;
            depth = 1;
            others = 1;
        }
        if (Dim >= 2) {
            height = exts[1];
        }
        if (Dim >= 3)
            depth = exts[2];

        if (Dim >= 4) {
            for (int i = 3; i < Dim; ++i)
                others *= exts[i];
        }

        // Allocation
        _data = alloc.allocMem(width, height, depth * others, pitch);

        // Extents with pitch
        extents_pitch = extents;
        extents_pitch[0] = pitch;

        // Strides and lengths
        compute_strides(extents_strides, extents_pitch);

        // Total lengths
        length_in_bytes = pitch * height * depth * others;
        length = width * height * depth * others;
    }

    DEVICE_HOST void gpuAllocMem() {
        _data = gpu_alloc.allocMem(width, height, depth * others, pitch);
    }

    void gpuFreeMem() {
        gpu_alloc.freeMem(_data);
    }

    void gpuResize(int w, int h) {
        extents_type exts;
        exts[0] = w;
        exts[1] = h;
        gpuResize(exts);
    }

    void gpuResize(extents_type& exts) {
        gpu_alloc.freeMem(_data);

        // Extents
        extents = exts;
        if (Dim >= 1) {
            width = exts[0];
            height = 1;
            depth = 1;
            others = 1;
        }
        if (Dim >= 2) {
            height = exts[1];
        }
        if (Dim >= 3)
            depth = exts[2];

        if (Dim >= 4) {
            for (int i = 3; i < Dim; ++i)
                others *= exts[i];
        }

        // Allocation
        _data = gpu_alloc.allocMem(width, height, depth*others, pitch);

        // Extents with pitch
        extents_pitch = extents;
        extents_pitch[0] = pitch;

        // Strides
        compute_strides(extents_strides, extents_pitch);

        // Total lengths
        length_in_bytes = pitch * height * depth * others;
        length = width * height * depth * others;
    }

    //! HW 08/04/15 : The cudaMemset2D function only allows int values as follows:
    //! extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
#ifdef CUDA_CODE
    void gpuMemSet(int value) {
        checkCudaErrors(cudaMemset2D(_data,
                                     pitch,
                                     value,
                                     sizeof(Node) * width,
                                     height*depth*others));
    }

    //! HW 04/05/15 : modif
    void gpuMemSet(Node const& value) {
        this->gpuResetValue(value);
    }
#else
    void gpuMemSet(Node const& value) {
        this->resetValue(value);
    }
#endif

    DEVICE_HOST MultiGrid& gpuAssign(MultiGrid const& g2) {
#ifdef CUDA_CODE
        cudaMemcpy(_data, g2._data, pitch * height * depth * others, cudaMemcpyDeviceToDevice);
#else
        std::memcpy(_data, g2._data, pitch * height);
#endif
        return *this;
    }

    MultiGrid& assign(MultiGrid const& g2) {
        std::memcpy(_data, g2._data, pitch * height * depth * others);
        return *this;
    }

    //! HOST (*this) to DEVICE (gpuGrid)
    void gpuCopyHostToDevice(MultiGrid<Node, Dim> & gpuGrid) {
#ifdef CUDA_CODE
        cudaMemcpy2D(gpuGrid._data,
                     gpuGrid.pitch,
                     _data,
                     pitch,
                     sizeof(Node) * width,
                     height * depth * others,
                     cudaMemcpyHostToDevice);
#else
        // simulation
        memcpy(gpuGrid._data, _data, pitch * height);
#endif
    }

    //! DEVICE (gpuGrid) TO HOST (*this)
    void gpuCopyDeviceToHost(MultiGrid<Node, Dim> & gpuGrid) {
#ifdef CUDA_CODE
        cudaMemcpy2D(_data,
                     pitch,
                     gpuGrid._data,
                     gpuGrid.pitch,
                     sizeof(Node) * width,
                     height * depth * others,
                     cudaMemcpyDeviceToHost);
#else
        // simulation
        memcpy(_data, gpuGrid._data, pitch * height);
#endif
    }

    //! HW 18.05.15:  add copy from device to device
    //! DEVICE (*this) TO DEVICE (gpuGrid)
    void gpuCopyDeviceToDevice(MultiGrid<Node, Dim> & gpuGrid) {
#ifdef CUDA_CODE
        cudaMemcpy2D(gpuGrid._data,
                     gpuGrid.pitch,
                     _data,
                     pitch,
                     sizeof(Node) * width,
                     height * depth * others,
                     cudaMemcpyDeviceToDevice);
#else
        // simulation
        memcpy(gpuGrid._data, _data, pitch * height);
#endif
    }

    //! HW 18.05.15:  add copy from device to host with only the first element
    //! DEVICE (gpuGrid) TO HOST (*this) with only the first element
    void gpuCopyDeviceToHostFirst(MultiGrid<Node,Dim> & gpuGrid) {
#ifdef CUDA_CODE
        cudaMemcpy2D(_data,
                     pitch,
                     gpuGrid._data,
                     gpuGrid.pitch,
                     sizeof(Node) * 1,
                     1,
                     cudaMemcpyDeviceToHost);
#else
        // simulation
        memcpy(_data, gpuGrid._data, pitch * height);
#endif
    }

    friend ostream& operator<<(ostream & o, MultiGrid & mat);
    friend ifstream& operator>>(ifstream& i, MultiGrid& mat);
    //! @}
};//MultiGrid

template <class Node>
struct Grid : MultiGrid<Node,2> {
    typedef MultiGrid<Node,2> super_type;
  public:
    typedef typename super_type::index_type index_type;
    typedef typename super_type::extents_type extents_type;
    typedef typename super_type::coord_type coord_type;
    typedef typename super_type::iterator_type iterator_type;

    DEVICE_HOST explicit Grid() : MultiGrid() {}
    DEVICE_HOST explicit Grid(int w, int h) : MultiGrid(w, h) {}

    DEVICE_HOST void resetValue(Node const& node) {
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                *(((Node*)((char*)_data + _y * pitch)) + _x) = node;
            }
        }
    }

    //! @brief ResetValue method on GPU side
    GLOBAL void gpuResetValue(Node const& node) {
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_resetValue _KER_CALL_(b, t) ((*this), node);
    }

//    //! @brief Get coordinate for loop only
//    DEVICE_HOST inline Node* operator[](std::size_t y) {
//        return ((Node*)((char*)_data + y * pitch));
//    }

    //! @brief Affectation of a value (idem reset)
    DEVICE_HOST Grid& operator=(Node const& node) {
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                *(((Node*)((char*)_data + _y * pitch)) + _x) = node;
            }
        }
        return *this;
    }
    //! @brief Get coordinate for loop only
    //! mirror out-of-range position
    DEVICE_HOST inline Node const& fetchIntCoor(int x, int y) const {
        if (x < 0) x = abs(x + 1);
        if (y < 0) y = abs(y + 1);
        if (x >= width) x = width * 2 - x - 1;
        if (y >= height) y = height * 2 - y - 1;
        return *((Node*)((char*)_data + y * pitch) + x);
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline Node const& get(std::size_t const x, std::size_t const y) const {
        return *((Node*)((char*)_data + y * pitch) + x);
    }

    //! @brief Set coordinatev for loop only
    DEVICE_HOST inline void set(std::size_t const x, std::size_t const y, Node const& value) {
        *((Node*)((char*)_data + y * pitch) + x) = value;
    }

    //! @brief Auto Addition
    DEVICE_HOST Grid& operator+=(Grid const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) += g2.get(_x, _y);
            }
        }
        return *this;
    }

    //! @brief Auto Difference
    DEVICE_HOST Grid& operator-=(Grid const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) -= g2.get(_x, _y);
            }
        }
        return *this;
    }

    //! @brief Auto Addition
    DEVICE_HOST Grid& operator+=(Node const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) += g2;
            }
        }
        return *this;
    }

    //! @brief Auto Difference
    DEVICE_HOST Grid& operator-=(Node const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) -= g2;
            }
        }
        return *this;
    }

    //! @brief Auto Mult
    DEVICE_HOST Grid& operator*=(Grid const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) *= g2.get(_x, _y);
            }
        }
        return *this;
    }

#if INTERPOLATION
    //! @brief Get coordinate for loop only
    //! mirror out-of-range position
    //! read from arbitrary position within image using bilinear interpolation
    DEVICE_HOST inline Node const& fetchFloatCoor(GLfloat x, GLfloat y) const {
        // integer parts in floating point format
        GLfloat intPartX, intPartY;
        // get fractional parts of coordinates
        GLfloat dx = fabsf(modff(x, &intPartX));
        GLfloat dy = fabsf(modff(y, &intPartY));
        // assume pixels are squares
        // one of the corners
        int ix0 = (int)intPartX;
        int iy0 = (int)intPartY;
        // mirror out-of-range position
        if (ix0 < 0) ix0 = abs(ix0 + 1);
        if (iy0 < 0) iy0 = abs(iy0 + 1);
        if (ix0 >= width) ix0 = width * 2 - ix0 - 1;
        if (iy0 >= height) iy0 = height * 2 - iy0 - 1;
        // corner which is opposite to (ix0, iy0)
        int ix1 = ix0 + 1;
        int iy1 = iy0 + 1;
        if (ix1 >= width) ix1 = width * 2 - ix1 - 1;
        if (iy1 >= height) iy1 = height * 2 - iy1 - 1;
        GLfloat a = (1.0f - dx) * (1.0f - dy);
        GLfloat b = dx * (1.0f - dy);
        GLfloat c = (1.0f - dx) * dy;
        GLfloat d = dx * dy;
//        Node res = (this->get(ix0, iy0) * a);
//        res += (this->get(ix1, iy0) * b);
//        res += (this->get(ix0, iy1) * c);
//        res += (this->get(ix1, iy1) * d);
        Node res = (this->get(ix0, iy0) * a)
                + (this->get(ix1, iy0) * b)
                + (this->get(ix0, iy1) * c)
                + (this->get(ix1, iy1) * d);
        return res;
    }
#endif
    // Input/Ouput
    friend ostream& operator<<(ostream & o, MultiGrid & mat) {
        if (!o)
            return(o);

        size_t Dimension = mat.getDimension();
        if (Dimension >= 4)
            o << "Extents = " << mat.extents;
        else if (Dimension >= 1)
            o << "Width = " << mat.width << " ";
        if (Dimension >= 2)
            o << "Height = " << mat.height << " ";
        if (Dimension >= 3)
            o << "Depth = " << mat.depth << " ";
        o << endl;

        if (!o)
            return(o);

        MultiGrid::index_type idx;
        mat.iterInit(idx);
        while (mat.iterNext(idx)) {
            o << mat(idx, mat.extents_strides) << " ";
            if (idx[0] == mat.width - 1)
                o << endl;
        }
        return o;
    }

    friend ifstream& operator>>(ifstream& i, MultiGrid& mat) {
        char str[256];

        if (!i)
            return(i);

        mat.freeMem();

        MultiGrid::extents_type exts;
        size_t Dimension = mat.getDimension();
        if (Dimension >= 4) {
            i >> str >> str;
            i >> exts;
        }
        else if (Dimension >= 1)
            i >> str >> str >> exts[0];
        if (Dimension >= 2)
            i >> str >> str >> exts[1];
        if (Dimension >= 3)
            i >> str >> str >> exts[2];

        mat.resize(exts);

        MultiGrid::index_type idx;
        mat.iterInit(idx);
        while (mat.iterNext(idx)) {
            i >> mat(idx, mat.extents_strides);
        }
        return i;
    }

#if TEST_CODE
    // Input/Ouput
    friend ostream& operator<<(ostream & o, Grid const & mat) {
        if (!o)
            return(o);

        o << "Width = " << mat.width << " ";
        o << "Height = " << mat.height << " " << endl;

        if (!o)
            return(o);

        for (int _y = 0; _y < mat.height; _y++) {
            for (int _x = 0; _x < mat.width; _x++) {
                o << mat._data[_x + _y * mat.width] << " ";
            }
            o << endl;
        }
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Grid& mat) {
        char str[256];

        if (!i)
            return(i);

        mat.freeMem();

        i >> str >> str >> mat.width;
        i >> str >> str >> mat.height;

        mat.allocMem();

        for (int _y = 0; _y < mat.height; _y++) {
            for (int _x = 0; _x < mat.width; _x++) {
                i >> mat._data[_x + _y * mat.width];
            }
        }
        return i;
    }
#endif
    // C fashion Ouput
    DEVICE_HOST void printInt() {
        printf("Width = %d, Height = %d\n", width, height);
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*this)[_y][_x].printInt();
                printf("  ");
            }
            printf("\n");
        }
    }

};

//typedef MultiGrid<Point2D, 2> Grid2D;
//typedef MultiGrid<Point3D, 3> Grid3D;

typedef Grid<GLint> MatDensity;
typedef Grid<GLint> GridDensity;

typedef Grid<Point2D> Mat2DPoints;
typedef Grid<Point3D> Mat3DPoints;

typedef Mat2DPoints  Grid2DPoints;
typedef Mat3DPoints  Grid3DPoints;

typedef Grid<Point3D> MatPixels;
typedef Grid<Point3D> Image;

typedef Grid<GLdouble> MatObjectVal;
typedef Grid<GLfloat> MatDisparity;
typedef Grid<Point2D> MatMotion;

/*! \brief Vecteur de nodes
 */
template <class Node>
class LineOfNodes : public vector<Node> {};
template <class Node>
class SetOfLines : public vector<vector<Node> > {};

//! @}

#if TEST_CODE
//! Test program
class Test {
public:
    void run() {
        cout << "debut test Grid<> ..." << endl;
        Grid2DPoints gd(5, 5);
        cout << "... debut test GLint Grid..." << endl;
        gd.resize(10, 10);

        cout << "... debut test GLint Grid..." << endl;
        Grid<GLint> g1(10, 10), g2(10, 10), g3(10, 10);
        cout << "... debut test GLint Grid 0 ..." << endl;
        g1 = 0;
        cout << "... debut test GLint Grid 1 ..." << endl;
        g2 = g3 = 10;
        cout << "... debut test GLint Grid 2 ..." << endl;
        g1 = g2 = g3;
        cout << "... debut test GLint Grid 3 ..." << endl;
        g2 += g1 -= g3;

        cout << "... debut test Grid<Point2D> ..." << endl;

        Grid<Point2D> g4(10, 10), g5(10, 10), g6(10, 10);
        g4 = g5 = g6 = Point2D(10,10);
        g4 -= g5 - g6;
        Grid<Point2D> g7(5, 5);
        g7 += g6 - g5;

        cout << "... debut test Grid<Point3D> ..." << endl;

        // Somme of squared difference (squared l2 norm)
        Grid<Point3D> gd1(10,10), gd2(10,10);

        Grid<Point3D> ggd(5, 5);
        ggd.resize(10, 10);
        ggd = Point3D(2,3,4);
        ggd += gd1 - gd2;
        ggd += ggd * ggd;

        cout << "... debut test sum of square diff ..." << endl;
        Grid<GLfloat> gr(10, 10);
        for (int _y = 0; _y < gr.getHeight(); ++_y) {
            for (int _x = 0; _x < gr.getWidth(); ++_x) {
//                 gr.getData()[_x + _y * gr.getStride()] =
//                          ggd.getData()[_x + _y * gr.getStride()][0]
//                        + ggd.getData()[_x + _y * gr.getStride()][1]
//                        + ggd.getData()[_x + _y * gr.getStride()][2];
                gr[_y][_x] = ggd[_y][_x][0] + ggd[_y][_x][1] + ggd[_y][_x][2];
            }
        }
        cout << "... debut test ooperator [] ..." << endl;
        Grid<GLfloat> grr(10, 10);
        for (int _y = 0; _y < grr.getHeight(); ++_y) {
            for (int _x = 0; _x < grr.getWidth(); ++_x) {
                  grr[_y][_x] = ggd[_y][_x][0]
                        + ggd[_y][_x][1]
                        + ggd[_y][_x][2];
            }
        }
        cout << "fin de test Grid<> ..." << endl;
    }
};
#endif

}//namespace components

#endif // MULTI_GRID_H
